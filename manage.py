"""
manage.py
=========
The occasional jobs: see where the session stands, undo a round, free a
supervisor's students, export everything for the record.

Run it in PyCharm and a numbered menu appears in the console; type a number and
press Enter. There is nothing to memorise and no arguments to get right.

    1  Where the session stands
    2  Undo a round
    3  A supervisor has withdrawn
    4  Free particular students
    5  Export everything
    6  Check a student file before allocating
    0  Quit

Everything it does is recorded in the session's audit trail, so a change made
here can be traced afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import matching_core as mc
import store as stx
import text_analysis as ta

SESSION_FILE = "session_2026-27.json"
STUDENTS_FILE = "sample_students.csv"      # used by the file check only


def rule(char: str = "-", width: int = 72) -> None:
    print(char * width)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def confirm(prompt: str) -> bool:
    return ask(f"{prompt} Type yes to go ahead: ").lower() in ("yes", "y")


# ---------------------------------------------------------------------------

def where_things_stand(store: stx.SessionStore) -> None:
    register = store.register_dataframe()
    if not len(register):
        print("The register is empty. Run setup_session.py first.")
        return
    table = store.capacity_table(register)
    print()
    rule("=")
    print(f"Session {store.session}   ·   register version {store.version}")
    rule("=")
    print(f"{len(register)} supervisors")
    print(f"{table['allowance_units'].sum():.0f} supervision units agreed in total, "
          f"{table['units_used'].sum():.0f} used, {table['units_left'].sum():.0f} left")
    for level, weight in sorted(store.level_weights.items()):
        placed = int(table[f"{level}_assigned"].sum())
        room = int(store.capacity_for_round(register, level).sum())
        print(f"  {level}: {placed} students placed so far, room for about {room} more "
              f"(each costs {weight} units)")

    rounds = store.rounds_summary()
    print()
    if not len(rounds):
        print("No rounds committed yet.")
    else:
        print("Rounds committed so far")
        rule()
        print(rounds.to_string(index=False))

    busiest = table.sort_values("units_left").head(8)
    print()
    print("Closest to full")
    rule()
    print(busiest[["name", "group", "allowance_units", "units_used", "units_left"]]
          .to_string(index=False))

    idle = table[table["units_used"] == 0]
    if len(idle):
        print(f"\n{len(idle)} supervisors have taken nobody at all this session.")


def undo_a_round(store: stx.SessionStore) -> None:
    rounds = store.data.get("rounds", [])
    if not rounds:
        print("There is nothing to undo.")
        return
    print()
    for i, r in enumerate(rounds, start=1):
        print(f"  {i}  {r['label']}")
        print(f"     committed {r['timestamp']}, {len(r['rows'])} students, id {r['round_id']}")
    choice = ask("\nWhich number? (Enter to cancel) ")
    if not choice.isdigit() or not (1 <= int(choice) <= len(rounds)):
        print("Cancelled.")
        return
    target = rounds[int(choice) - 1]
    print(f"\nThis will remove {len(target['rows'])} pairings and give the capacity back.")
    if not confirm("The students involved will have no supervisor again."):
        print("Cancelled.")
        return
    store.rollback(target["round_id"])
    print(f"Undone. {target['label']} is no longer in the session.")


def supervisor_withdrew(store: stx.SessionStore) -> None:
    register = store.register_dataframe()
    name = ask("Supervisor's name or id: ")
    if not name:
        print("Cancelled.")
        return
    matches = register[
        register["name"].str.contains(name, case=False, na=False)
        | (register["supervisor_id"].str.lower() == name.lower())]
    if not len(matches):
        print("Nobody of that name is on the register.")
        return
    if len(matches) > 1:
        print("\nMore than one match:")
        for _, row in matches.iterrows():
            print(f"  {row['supervisor_id']}  {row['name']}  ({row['group']})")
        print("Run this again with the id.")
        return

    row = matches.iloc[0]
    print(f"\n{row['name']} ({row['supervisor_id']}, {row['group']})")
    if not confirm("Free all of their students and mark them unavailable?"):
        print("Cancelled.")
        return
    affected = store.release_supervisor(row["supervisor_id"])
    store.update_supervisor(row["supervisor_id"], {"available": 0})
    print(f"\n{len(affected)} students are now without a supervisor:")
    for sid in affected:
        print(f"  {sid}")
    if affected:
        print("\nPut these student numbers into a small CSV of their own and run "
              "run_allocation.py against it to place them, leaving everybody else alone.")


def free_students(store: stx.SessionStore) -> None:
    raw = ask("Student numbers, separated by commas: ")
    ids = [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]
    if not ids:
        print("Cancelled.")
        return
    removed = store.release_students(ids)
    print(f"{removed} pairings removed. Those students have no supervisor again.")


def export_everything(store: stx.SessionStore) -> None:
    out = Path(ask("Folder to write into (Enter for 'exports'): ") or "exports")
    out.mkdir(parents=True, exist_ok=True)
    register = store.register_dataframe()

    rows = store.committed_rows()
    if len(rows):
        rows.to_csv(out / "all_allocations.csv", index=False, encoding="utf-8-sig")
    if len(register):
        store.capacity_table(register).to_csv(out / "register_remaining.csv", index=False,
                                              encoding="utf-8-sig")
    summary = store.rounds_summary()
    if len(summary):
        summary.to_csv(out / "rounds.csv", index=False, encoding="utf-8-sig")
    trail = store.audit_trail()
    if len(trail):
        trail.to_csv(out / "audit_trail.csv", index=False, encoding="utf-8-sig")
    import shutil
    shutil.copy2(store.path, out / Path(store.path).name)
    print(f"\nWritten to {out.resolve()}:")
    for f in sorted(out.iterdir()):
        print(f"  {f.name}")


def check_student_file(store: stx.SessionStore) -> None:
    path = ask(f"Student file (Enter for '{STUDENTS_FILE}'): ") or STUDENTS_FILE
    p = Path(path)
    if not p.exists():
        print(f"Cannot find {path}.")
        return
    students = pd.read_excel(p, dtype=str).fillna("") if p.suffix.lower() in (".xlsx", ".xls") \
        else pd.read_csv(p, encoding="utf-8-sig", dtype=str).fillna("")

    register = store.register_dataframe()
    probe = register.copy()
    probe["workload"] = 0
    problems = [i for i in mc.validate_inputs(students, probe) if "capacity" not in i.lower()]

    print()
    rule("=")
    print(f"{path}: {len(students)} rows")
    rule("=")
    already = len(store.committed_students() & set(students["student_id"]))
    print(f"{already} of them already have a supervisor from an earlier round")
    if "level" in students.columns and "programme" in students.columns:
        print()
        print(students.groupby(["level", "programme"]).size()
              .reset_index(name="students").to_string(index=False))
    if problems:
        print("\nWorth fixing before you allocate:")
        for msg in problems:
            print(f"  - {msg}")
    else:
        print("\nNo problems with the file itself.")

    req = ta.requirements_table(students)
    print()
    print(f"{int(req['ethics_review_likely'].sum())} proposals look like they need ethics "
          "approval")
    print(f"{int(req['needs_data_access_check'].sum())} depend on data the student may not "
          "have yet")
    thin = req[req["word_count"] < 25]
    if len(thin):
        print(f"{len(thin)} proposals are under 25 words, which is too little to match on. "
              "These are the ones that come out unplaced.")

    trim = mc.tag_limit_report(register)
    if len(trim):
        print(f"\n{len(trim)} supervisors have more tags than agreed; the extras are ignored.")


MENU = [
    ("Where the session stands", where_things_stand),
    ("Undo a round", undo_a_round),
    ("A supervisor has withdrawn", supervisor_withdrew),
    ("Free particular students", free_students),
    ("Export everything", export_everything),
    ("Check a student file before allocating", check_student_file),
]


def main() -> None:
    store = stx.SessionStore(SESSION_FILE)
    while True:
        print()
        rule("=")
        print(f"Dissertation allocation  ·  session {store.session}  ·  {SESSION_FILE}")
        rule("=")
        for i, (label, _) in enumerate(MENU, start=1):
            print(f"  {i}  {label}")
        print("  0  Quit")
        choice = ask("\nNumber: ")
        if choice in ("0", "q", "quit", ""):
            return
        if not choice.isdigit() or not (1 <= int(choice) <= len(MENU)):
            print("Type one of the numbers listed.")
            continue
        try:
            store.ensure_fresh()
            MENU[int(choice) - 1][1](store)
        except stx.ConcurrentEditError as exc:
            print(f"\n{exc}")
        except KeyboardInterrupt:
            print("\nCancelled.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
