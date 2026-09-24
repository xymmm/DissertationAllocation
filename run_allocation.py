"""
run_allocation.py
=================
Run one allocation round by editing the settings below and pressing Run.

This is the file to open in PyCharm. Nothing here needs the command line, and
nothing needs to be typed twice: change the few lines in SETTINGS, press the
green arrow, and the script prints a plain summary in the console and writes an
output folder containing a report you can open by double-clicking it, a
spreadsheet, and the underlying CSVs.

Nothing is committed to the session by simply running this. The round becomes
real only when COMMIT is set to True, or by running commit_round.py against the
folder that was produced. That is the difference between a proposal and a
decision, and it lets an allocation be produced on Thursday and approved on
Friday without anything being recalculated in between.

The order of work across a session:

    1. set up_the session once          python setup_session.py
    2. produce a round                  this file, with COMMIT = False
    3. read the report, fix what needs fixing, run it again
    4. commit it                        this file with COMMIT = True,
                                        or python commit_round.py
    5. repeat for the next cohort; capacity carries across automatically
"""

from __future__ import annotations

import sys
import time
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

import matching_core as mc
import report as rp
import store as stx
import text_analysis as ta

# ===========================================================================
# SETTINGS — this is the only part you need to change
# ===========================================================================

SETTINGS = {
    # Where the session lives. One file per academic session; it holds the
    # supervisor register, every committed round and the audit trail.
    "session_file": "session_2026-27.json",

    # The student file for this round.
    "students_file": "sample_students.csv",

    # Declared conflicts of interest, or None.
    "conflicts_file": "sample_conflicts.csv",

    # Which cohort this round covers. Leave "programmes" empty for every
    # programme at this level.
    "level": "PGT",
    "programmes": ["MSc Business Analytics", "MSc Finance"],

    # How strict to be.
    #   minimum_fit       below this a pairing is never considered at all
    #   good_match        at or above this a pairing needs no further thought
    #   tolerance         share of pairings allowed below that standard
    "minimum_fit": 0.15,
    "good_match": 0.50,
    "tolerance": 0.10,

    # Capacity held back for cohorts not yet allocated.
    #   "proportional"  shares it out in proportion to the students still to place
    #   "use_all"       lets this round take everything, right for the last round
    "reserve": "proportional",
    "reserve_floor": 0,

    # Where the output folder goes.
    "output_folder": "runs",

    # True writes the round into the session and reduces everybody's remaining
    # capacity. Leave it False while you are still looking at the result.
    "commit": False,

    # Open the report in a browser as soon as it is written.
    "open_report": True,

    # Also solve the problem a second way and confirm both agree. Slower, and
    # worth doing once before a round goes out.
    "cross_check": False,

}

# ===========================================================================
# Everything below runs the round; you should not need to edit it
# ===========================================================================


def line(char: str = "-", width: int = 72) -> None:
    print(char * width)


def heading(text: str) -> None:
    print()
    print(text)
    line()


def fail(message: str) -> None:
    print()
    print("Stopped: " + message)
    sys.exit(1)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        fail(f"cannot find {path}. Check the file name in SETTINGS at the top of this file.")
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p, dtype=str).fillna("")
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str).fillna("")


def main() -> None:
    s = SETTINGS
    store = stx.SessionStore(s["session_file"])
    register = store.register_dataframe()
    if not len(register):
        fail(f"the register in {s['session_file']} is empty. Run setup_session.py first.")

    students_all = read_table(s["students_file"])
    committed = store.committed_students()

    heading(f"Session {store.session}, register version {store.version}")
    print(f"{len(register)} supervisors on the register")
    print(f"{len(students_all)} students in {s['students_file']}, "
          f"{len(committed & set(students_all['student_id']))} of them already placed "
          "in an earlier round")

    mask = np.ones(len(students_all), dtype=bool)
    if "level" in students_all.columns:
        mask &= (students_all["level"].astype(str) == s["level"]).to_numpy()
    if s["programmes"]:
        mask &= students_all["programme"].astype(str).isin(s["programmes"]).to_numpy()
    if committed:
        mask &= ~students_all["student_id"].astype(str).isin(committed).to_numpy()

    idx = np.where(mask)[0]
    if not len(idx):
        fail("no students match this round. Check the level and programmes in SETTINGS.")
    students = students_all.iloc[idx].reset_index(drop=True)

    outstanding = (~students_all["student_id"].astype(str).isin(committed)).to_numpy() \
        if committed else np.ones(len(students_all), dtype=bool)
    upcoming = int((outstanding & ~mask).sum())

    heading("This round")
    print(f"level {s['level']}, "
          f"{', '.join(s['programmes']) if s['programmes'] else 'every programme'}")
    print(f"{len(students)} students to place now, {upcoming} still to come afterwards")

    # --- scoring ---------------------------------------------------------
    print("\nWorking out how well every student fits every supervisor ...")
    started = time.time()
    score_all, components_all = mc.build_score_matrix(students_all, register)
    score = score_all[idx, :]
    components = {
        k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
        for k, v in components_all.items()
        if k in ("area", "method", "text", "phrase", "student_phrases",
                 "phrase_evidence", "student_areas", "student_methods")
    }
    components["supervisor_areas"] = components_all["supervisor_areas"]
    components["supervisor_methods"] = components_all["supervisor_methods"]
    print(f"done in {time.time() - started:.1f} seconds")

    best = score.max(axis=1)
    hopeless = int((best < s["minimum_fit"]).sum())
    if hopeless:
        print(f"{hopeless} students have nobody above the minimum standard of fit. They will "
              "come out unplaced, and the usual cause is a proposal too vague to match.")

    # --- capacity ---------------------------------------------------------
    level_capacity = store.capacity_for_round(register, s["level"])
    capacity = stx.reserve_capacity(level_capacity, len(students), upcoming,
                                    mode=s["reserve"],
                                    floor_per_supervisor=int(s["reserve_floor"]))
    heading("Capacity")
    print(f"{int(level_capacity.sum())} places available at {s['level']} across the register")
    print(f"{int(capacity.sum())} of them opened to this round under the "
          f"'{s['reserve']}' policy")
    if capacity.sum() < len(students):
        print("that is fewer places than students, so some will be left unplaced")

    # --- constraints ------------------------------------------------------
    sup_index = {str(v): j for j, v in enumerate(register["supervisor_id"])}
    locked = {}
    if "locked_supervisor_id" in students.columns:
        for i, v in enumerate(students["locked_supervisor_id"]):
            j = sup_index.get(str(v).strip())
            if j is not None:
                locked[i] = j
    if locked:
        print(f"{len(locked)} pairings are fixed by hand and will be kept")

    conflicts = set()
    if s["conflicts_file"]:
        cdf = read_table(s["conflicts_file"])
        stu_pos = {str(v): i for i, v in enumerate(students["student_id"])}
        for _, row in cdf.iterrows():
            i = stu_pos.get(str(row["student_id"]).strip())
            j = sup_index.get(str(row["supervisor_id"]).strip())
            if i is not None and j is not None:
                conflicts.add((i, j))
        print(f"{len(conflicts)} declared conflicts of interest apply to this round")

    unavailable = [j for j, a in enumerate(register["available"]) if int(a) == 0]
    if unavailable:
        print(f"{len(unavailable)} supervisors are unavailable this session")

    eligibility = None
    if "programmes" in register.columns and "programme" in students.columns:
        elig = np.ones((len(students), len(register)), dtype=bool)
        restricted = 0
        stu_progs = [str(v).strip().lower() for v in students["programme"]]
        for j, allowed in enumerate([mc.split_tags(v) for v in register["programmes"]]):
            if not allowed:
                continue
            restricted += 1
            for i, pg in enumerate(stu_progs):
                elig[i, j] = pg in allowed
        if restricted:
            eligibility = elig
            print(f"{restricted} supervisors take only certain programmes")

    # --- allocate ---------------------------------------------------------
    print("\nAllocating ...")
    candidates = mc.build_candidates(score, hard_floor=s["minimum_fit"], top_k=25,
                                     locked=locked, unavailable=unavailable,
                                     blocked_pairs=conflicts, eligibility=eligibility)
    params = mc.AllocationParams(good_threshold=s["good_match"], tolerance=s["tolerance"])
    register_run = register.copy()
    register_run["workload"] = capacity
    result = mc.allocate(students, register_run, score, candidates, params, capacity,
                         components=components)

    heading("Result")
    print(f"placed                        {result.n_assigned}")
    print(f"left without a supervisor     {result.n_unassigned}")
    print(f"for a second look             {result.n_below_good} "
          f"({result.tolerance_used:.0%} against a tolerance of {s['tolerance']:.0%})")
    print(f"average strength of match     "
          f"{result.total_score / max(result.n_assigned, 1):.3f}")
    print(f"solved in                     {result.solve_seconds:.2f} seconds")

    if s["cross_check"]:
        print("\nSolving a second way to confirm ...")
        exact = mc.allocate(students, register_run, score, candidates,
                            mc.AllocationParams(**{**params.__dict__, "engine": "milp"}),
                            capacity)
        if abs(exact.total_score - result.total_score) < 1e-6:
            print("both methods agree, the result is optimal")
        else:
            print(f"the two methods disagree: {exact.total_score:.4f} against "
                  f"{result.total_score:.4f}. Do not send this out; tell whoever maintains "
                  "this tool.")

    # --- output -----------------------------------------------------------
    diagnostics = mc.diagnose_allocation(result, score, candidates, students,
                                         register_run, capacity)
    requirements = ta.requirements_table(students)
    scope = {"level": s["level"], "programmes": s["programmes"]}
    run_params = {"hard_floor": s["minimum_fit"], "good_threshold": s["good_match"],
                  "tolerance": s["tolerance"], "reservation": s["reserve"],
                  "engine": result.engine, "register_version": store.version,
                  "produced_by": "run_allocation.py"}

    folder = Path(s["output_folder"]) / f"{time.strftime('%Y-%m-%d_%H%M')}_{s['level']}"
    written = rp.build_pack(folder, result, students, register, diagnostics, requirements,
                            store.capacity_table(register), scope, store.session,
                            store.version, run_params)

    heading("Written")
    for label, path in [("report, open this one", written["report"]),
                        ("spreadsheet", written["workbook"]),
                        ("allocation as CSV", written["allocation_csv"])]:
        print(f"{label:<26} {path}")

    if s["commit"]:
        rnd = store.commit_rows(
            [r for r in __import__("json").loads(written["draft"].read_text())["rows"]],
            f"{s['level']} · {', '.join(s['programmes']) or 'all'} · {result.n_assigned}",
            s["level"], s["programmes"], run_params)
        table = store.capacity_table(register)
        heading("Committed")
        print(f"round {rnd['round_id']} written into {s['session_file']}")
        print(f"{table['units_used'].sum():.0f} supervision units now used, "
              f"{table['units_left'].sum():.0f} left for the rest of the session")
    else:
        heading("Not committed")
        print("Nothing has been written into the session file and no supervisor's remaining")
        print("capacity has changed. When you are happy with the result, either set")
        print("  commit = True   in SETTINGS and run this file again,")
        print("or run:")
        print(f"  python commit_round.py {folder}")

    if s["open_report"]:
        webbrowser.open(written["report"].resolve().as_uri())


if __name__ == "__main__":
    main()
