"""
app.py
======
Streamlit front end for dissertation allocation, undergraduate and taught
postgraduate, written for a teaching operations office rather than for an
academic. Every stage produces a downloadable artefact, every parameter that
affects the outcome is visible on screen, and the allocation itself is
deterministic, so rerunning with the same inputs reproduces the same result
when a student queries their supervisor.

State for the session lives in a JSON store held by the application rather than
in whichever spreadsheet was uploaded most recently. Committing a round writes
back to the supervisor register immediately, so a supervisor who has taken six
masters students has that much less capacity when the undergraduate round runs
a fortnight later.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

import matching_core as mc
import elm_client as elm
import store as stx
import text_analysis as ta

st.set_page_config(page_title="Dissertation allocation", layout="wide", page_icon="🎓")

DEFAULT_AREA_VOCAB = [
    "supply chain", "operations", "logistics", "marketing analytics",
    "consumer behaviour", "finance", "banking", "accounting", "hrm",
    "entrepreneurship", "strategy", "digital economy", "healthcare management",
    "sustainability", "retail",
]
DEFAULT_METHOD_VOCAB = [
    "regression", "econometrics", "machine learning", "optimisation",
    "simulation", "survey", "interview", "case study", "text mining",
    "experiment", "network analysis", "systematic review",
]

for key, default in [
    ("students", None), ("score", None), ("components", None),
    ("candidates", None), ("result", None), ("extracted", None),
    ("audit", None), ("diagnostics", None), ("conflicts", None),
    ("round_scope", None), ("requirements", None), ("taxonomy_raw", None),
    ("store_path", "session_2026-27.json"),
    ("session_label", "2026/27"), ("store", None),
    ("pseudo", elm.Pseudonymiser()),
]:
    st.session_state.setdefault(key, default)


def read_table(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload, encoding="utf-8-sig", dtype=str).fillna("")
    return pd.read_excel(upload, dtype=str).fillna("")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def get_store() -> stx.SessionStore:
    """Return the session store, reloading it if another user has saved."""
    path = st.session_state.store_path
    store = st.session_state.store
    if store is None or str(store.path) != str(path):
        store = stx.SessionStore(path, session=st.session_state.session_label)
        st.session_state.store = store
    else:
        store.ensure_fresh()
    return store


def guarded(fn, *args, **kwargs):
    """Run a store mutation and report a clash rather than losing the edit."""
    try:
        return fn(*args, **kwargs), None
    except stx.ConcurrentEditError as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Academic session")
    st.session_state.store_path = st.text_input(
        "Session file", value=st.session_state.store_path,
        help="Holds the supervisor register, every committed round and the audit trail "
             "for one academic session. Start a new file at the beginning of each session.")
    store = get_store()
    st.caption(f"Session {store.session} · register version {store.version} · "
               f"{len(store.data['supervisors'])} supervisors · "
               f"{len(store.data['rounds'])} rounds committed")

    st.divider()
    st.header("ELM")
    st.caption("Request an API key from inside ELM. Enter the base URL up to the segment "
               "before /chat/completions.")
    base_url = st.text_input("Base URL", value=elm.DEFAULT_BASE_URL,
                             placeholder="https://<elm-endpoint>/v1")
    api_key = st.text_input("API key", value="", type="password")
    model_name = st.text_input("Model", value=elm.DEFAULT_MODEL)
    max_workers = st.slider("Concurrent requests", 1, 12, 6)
    use_cache = st.checkbox("Cache model responses", value=True,
                            help="Identical inputs are not sent twice, which saves both time "
                                 "and quota when parameters are revised and the run repeated")
    elm_cfg = elm.ELMConfig(base_url=base_url, api_key=api_key, model=model_name,
                            max_workers=max_workers, use_cache=use_cache)
    elm_ready = bool(base_url and api_key)
    if elm_ready:
        st.success("ELM configured")
    else:
        st.info("ELM is not configured. Rule-based scoring and the optimisation still work.")

    st.divider()
    st.header("Solver")
    engine = st.radio("Engine", ["lp", "milp"], index=0,
                      format_func=lambda x: {"lp": "Transportation LP with bisection (fast)",
                                             "milp": "Exact MILP (cross-check)"}[x])
    milp_solver = st.selectbox("MILP solver",
                               ["auto", "PULP_CBC_CMD", "HiGHS_CMD", "GUROBI_CMD"],
                               index=0, disabled=(engine != "milp"))
    st.caption("Both routes should return the same objective at the same tolerance. Run the "
               "MILP once before signing off a round as a check.")

st.title("Dissertation allocation")
st.caption("Teaching Operations Office · undergraduate and taught postgraduate")

tab_reg, tab_students, tab_score, tab_alloc, tab_audit, tab_rounds, tab_export = st.tabs(
    ["① Register", "② Students", "③ Matching", "④ Allocation round",
     "⑤ AI review", "⑥ Rounds", "⑦ Export"])

# ---------------------------------------------------------------------------
# Tab 1: supervisor register
# ---------------------------------------------------------------------------

with tab_reg:
    store = get_store()
    st.subheader("Supervisor register")
    st.caption("The register is the authoritative record for this session. A spreadsheet seeds "
               "it and later merges changes into it, but never replaces it, because a "
               "supervisor already carrying students must not be able to vanish from the "
               "register through a stale upload.")

    c1, c2 = st.columns(2)
    with c1:
        f_sup = st.file_uploader("Upload or refresh the register", type=["csv", "xlsx"],
                                 key="sup_up")
        update_allowances = st.checkbox(
            "Take allowances from the upload as well as profiles", value=True,
            help="Untick when allowances have been edited in the application and the "
                 "spreadsheet is out of date. Profiles are still refreshed.")
        if st.button("Load the sample register"):
            sample = pd.read_csv("sample_supervisors.csv", dtype=str).fillna("")
            report, err = guarded(store.sync_from_dataframe, sample)
            if err:
                st.error(err)
            else:
                st.success(f"Added {len(report['added'])}, updated {len(report['updated'])}")
                if report.get("tags_trimmed"):
                    st.warning(f"{len(report['tags_trimmed'])} tag lists were longer than agreed "
                               "and have been trimmed; see the table below.")
                    st.session_state.trim_report = pd.DataFrame(report["tags_trimmed"])
        if f_sup is not None and st.button("Merge the uploaded file"):
            report, err = guarded(store.sync_from_dataframe, read_table(f_sup),
                                  update_allowances=update_allowances)
            if err:
                st.error(err)
            else:
                st.success(f"Added {len(report['added'])}, updated {len(report['updated'])}, "
                           f"unchanged {len(report['unchanged'])}")
                if report.get("tags_trimmed"):
                    st.session_state.trim_report = pd.DataFrame(report["tags_trimmed"])
                    st.warning(f"{len(report['tags_trimmed'])} tag lists exceeded the agreed "
                               "limits and have been trimmed in the order given.")
                if report["absent_from_upload"]:
                    st.warning(
                        f"{len(report['absent_from_upload'])} supervisors in the register were "
                        "not in the upload and have been left untouched: "
                        + ", ".join(report["absent_from_upload"][:12])
                        + ("…" if len(report["absent_from_upload"]) > 12 else ""))
    with c2:
        st.markdown("**Tag limits**")
        st.caption(f"At most {mc.MAX_AREA_TAGS} research areas and {mc.MAX_METHOD_TAGS} methods "
                   "per supervisor, the same limits the data collection form should impose. "
                   "Order matters and is kept: a supervisor who lists inventory control first "
                   "and stochastic programming fourth is telling the office something, and the "
                   "score reflects it. Longer lists are trimmed from the end rather than "
                   "rejected, and what was dropped is reported so it can be discussed rather "
                   "than lost.")
        if st.session_state.get("trim_report") is not None:
            st.dataframe(st.session_state.trim_report, width="stretch", height=160)

        st.markdown("**Supervision units**")
        st.caption("An allowance is held in units rather than in a headcount, because a masters "
                   "dissertation is not the same amount of work as an undergraduate one. "
                   "Twelve units at the weights below buys twelve undergraduates, or eight "
                   "masters students, or any mixture of the two.")
        weights = store.level_weights
        wc = st.columns(len(weights))
        new_weights = {}
        for k, (lvl, w) in enumerate(sorted(weights.items())):
            new_weights[lvl] = wc[k].number_input(f"{lvl} weight", 0.1, 5.0, float(w), 0.1,
                                                  key=f"w_{lvl}")
        if st.button("Save weights"):
            _, err = guarded(store.set_level_weights, new_weights)
            st.error(err) if err else st.success("Weights updated")

    reg = store.register_dataframe()
    if not len(reg):
        st.info("The register is empty. Seed it from a spreadsheet above.")
    else:
        st.divider()
        st.markdown("**Current register and remaining capacity**")
        level_pick = st.selectbox("Show remaining places for level",
                                  sorted(store.level_weights.keys()))
        cap_table = store.capacity_table(reg, level=level_pick)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Supervisors", len(reg))
        c2.metric("Allowance (units)", f"{cap_table['allowance_units'].sum():.0f}")
        c3.metric("Units used", f"{cap_table['units_used'].sum():.0f}")
        c4.metric(f"Places left at {level_pick}",
                  int(cap_table[f"places_left_{level_pick}"].sum()))
        st.dataframe(cap_table, width="stretch", height=280)
        st.download_button("Download the register as CSV", to_csv_bytes(cap_table),
                           "supervisor_register.csv", "text/csv")

        st.divider()
        st.markdown("**Edit an individual supervisor**")
        st.caption("Use this when somebody agrees to take two more, goes on leave, or asks for "
                   "a ceiling on one level. Edits are written straight to the register and "
                   "recorded in the audit trail.")
        pick = st.selectbox("Supervisor",
                            [f"{r['name']} ({r['supervisor_id']})" for _, r in reg.iterrows()])
        sid = pick.rsplit("(", 1)[1].rstrip(")")
        current = store.data["supervisors"][sid]
        e1, e2, e3, e4 = st.columns(4)
        new_allow = e1.number_input("Allowance (units)", 0.0, 60.0,
                                    float(current.get("allowance_units", 0)), 0.5)
        new_min = e2.number_input("Minimum load", 0, 20, int(current.get("min_load", 0)))
        new_avail = e3.selectbox("Available this session", [1, 0],
                                 index=0 if int(current.get("available", 1)) == 1 else 1,
                                 format_func=lambda x: "Yes" if x else "No")
        caps = current.get("level_caps") or {}
        new_cap = e4.number_input(f"Cap at {level_pick}, 0 for none", 0, 40,
                                  int(caps.get(level_pick, 0) or 0))
        if st.button("Save this supervisor"):
            changes = {"allowance_units": float(new_allow), "min_load": int(new_min),
                       "available": int(new_avail)}
            if new_cap:
                changes[f"cap_{level_pick}"] = int(new_cap)
            _, err = guarded(store.update_supervisor, sid, changes)
            st.error(err) if err else st.success(f"{current.get('name', sid)} updated")

        with st.expander("Audit trail"):
            st.dataframe(store.audit_trail().tail(200), width="stretch", height=240)

# ---------------------------------------------------------------------------
# Tab 2: students and conflicts
# ---------------------------------------------------------------------------

with tab_students:
    store = get_store()
    st.subheader("Student submissions")
    st.caption("Required columns: student_id, project_title, abstract. Strongly recommended: "
               "level, being UG or PGT, and programme, since rounds are organised around them. "
               "Optional: project_name, areas, methods, references, preferred_supervisor_id, "
               "locked_supervisor_id.")
    f_stu = st.file_uploader("Upload the student file", type=["csv", "xlsx"], key="stu_up")
    if st.button("Load the sample student file"):
        st.session_state.students = pd.read_csv("sample_students.csv", dtype=str).fillna("")
    if f_stu is not None:
        st.session_state.students = read_table(f_stu)

    st.divider()
    st.subheader("Conflicts of interest")
    st.caption("Two columns, student_id and supervisor_id, plus a reason for the record. A "
               "declared family relationship, a previous appeal, a seat on the student's "
               "discipline panel: these are questions of eligibility rather than of fit, so "
               "they are excluded outright rather than folded into the match score, where a "
               "generous tolerance could quietly override them.")
    f_conf = st.file_uploader("Upload the conflicts file", type=["csv", "xlsx"], key="conf_up")
    if st.button("Load the sample conflicts file"):
        st.session_state.conflicts = pd.read_csv("sample_conflicts.csv", dtype=str).fillna("")
    if f_conf is not None:
        st.session_state.conflicts = read_table(f_conf)
    if st.session_state.conflicts is not None:
        st.dataframe(st.session_state.conflicts, width="stretch")

    stu = st.session_state.students
    reg = store.register_dataframe()
    if stu is not None and len(reg):
        st.divider()
        committed = store.committed_students()
        c1, c2, c3 = st.columns(3)
        c1.metric("Students in file", len(stu))
        c2.metric("Already committed", len(committed & set(stu["student_id"])))
        c3.metric("Still to place", len(set(stu["student_id"]) - committed))
        if "level" in stu.columns and "programme" in stu.columns:
            st.dataframe(stu.groupby(["level", "programme"]).size()
                         .reset_index(name="students"), width="stretch")
        check = reg.copy()
        check["workload"] = 0
        issues = [i for i in mc.validate_inputs(stu, check) if "capacity" not in i.lower()]
        for msg in issues:
            st.warning(msg)
        if not issues:
            st.success("The student file passes its checks")
        st.divider()
        st.subheader("What the proposals will require")
        st.caption("Read straight from the titles and abstracts, with no model involved. "
                   "Primary data collection and human participants point to ethics approval, "
                   "and restricted data or company access points to something the office will "
                   "have to confirm. February is a considerably better time to discover either "
                   "than June.")
        if st.button("Scan the proposals"):
            st.session_state.requirements = ta.requirements_table(stu)
        req = st.session_state.requirements
        if req is not None and len(req):
            c1, c2, c3 = st.columns(3)
            c1.metric("Ethics review likely", int(req["ethics_review_likely"].sum()))
            c2.metric("Data access to confirm", int(req["needs_data_access_check"].sum()))
            c3.metric("Median proposal length (words)", int(req["word_count"].median()))
            st.dataframe(req[req["requirements"] != ""], width="stretch", height=240)

        with st.expander("View the student file"):
            st.dataframe(stu, width="stretch", height=300)

# ---------------------------------------------------------------------------
# Tab 3: matching scores
# ---------------------------------------------------------------------------

with tab_score:
    store = get_store()
    reg = store.register_dataframe()
    if st.session_state.students is None or not len(reg):
        st.info("Seed the register in ① and load students in ②")
    else:
        stu = st.session_state.students

        st.subheader("A. Standardise submissions with ELM, optional but recommended")
        st.caption("Students describe the same method in a dozen different ways, and most "
                   "mismatch originates there rather than in the optimisation. This step maps "
                   "free text onto a controlled vocabulary and rates how specific each topic "
                   "is. Names and student numbers never enter a prompt: only a session token, "
                   "the title and the abstract are sent.")
        area_vocab = st.text_area("Area vocabulary", ";".join(DEFAULT_AREA_VOCAB), height=80)
        method_vocab = st.text_area("Method vocabulary", ";".join(DEFAULT_METHOD_VOCAB), height=80)

        with st.expander("Exactly what leaves this machine", expanded=False):
            st.caption("Students are identified by a sequence number issued here, not by their "
                       "student number, and supervisors by a code rather than by name. Titles "
                       "and abstracts are scanned for matriculation numbers and email addresses "
                       "first. The mapping from code back to person stays in this process and "
                       "is never written to disk.")
            preview_pseudo = elm.Pseudonymiser()
            demo_student = elm.student_payload(stu.iloc[0].to_dict(), preview_pseudo)
            demo_sup = elm.supervisor_payload(reg.iloc[0].to_dict(), preview_pseudo)
            st.code(json.dumps({"student": demo_student, "supervisor": demo_sup},
                               indent=2, ensure_ascii=False), language="json")
            mapping = st.session_state.pseudo.mapping_table()
            if mapping:
                st.download_button("Download the code mapping for your own records",
                                   to_csv_bytes(pd.DataFrame(mapping)),
                                   "code_mapping.csv", "text/csv")

        if st.button("Run ELM extraction", disabled=not elm_ready):
            client = elm.ELMClient(elm_cfg)
            pseudo = st.session_state.pseudo
            submissions = [elm.student_payload(row.to_dict(), pseudo)
                           for _, row in stu.iterrows()]
            bar = st.progress(0.0, text="Extracting")
            out = client.map_parallel(
                lambda s: elm.extract_profile(client, s, mc.split_tags(area_vocab),
                                              mc.split_tags(method_vocab)),
                submissions,
                progress=lambda d, t: bar.progress(d / t, text=f"Extracting {d}/{t}"))
            bar.empty()
            st.session_state.extracted = pd.DataFrame([{
                "student_id": pseudo.real(o.get("student_code", "")) or "",
                "llm_areas": ";".join(o.get("areas", []) or []),
                "llm_methods": ";".join(o.get("methods", []) or []),
                "clarity": o.get("clarity"),
                "method_confidence": o.get("method_confidence", ""),
                "flags": ";".join(o.get("feasibility_flags", []) or []),
                "summary": o.get("one_line_summary", ""),
            } for o in out if isinstance(o, dict)])
            st.success(f"{len(st.session_state.extracted)} submissions processed")

        if st.session_state.extracted is not None:
            ext = st.session_state.extracted
            st.dataframe(ext, width="stretch", height=240)
            vague = ext[pd.to_numeric(ext["clarity"], errors="coerce") <= 2]
            if len(vague):
                st.warning(f"{len(vague)} submissions are too vague to supervise as written. "
                           "Returning these for more detail before the round runs will do more "
                           "for the outcome than any adjustment to the model.")
            if st.button("Merge the extracted tags into the student file"):
                merged = stu.merge(ext[["student_id", "llm_areas", "llm_methods"]],
                                   on="student_id", how="left")
                merged["areas"] = np.where(merged["llm_areas"].fillna("") != "",
                                           merged["llm_areas"], merged.get("areas", ""))
                merged["methods"] = np.where(merged["llm_methods"].fillna("") != "",
                                             merged["llm_methods"], merged.get("methods", ""))
                st.session_state.students = merged.drop(columns=["llm_areas", "llm_methods"])
                st.success("Merged. Recalculate the score matrix below.")

        st.divider()
        st.subheader("B. Method taxonomy")
        st.caption("Undergraduates in particular can often place their project as qualitative "
                   "or quantitative and go no further, while supervisors describe themselves in "
                   "specific techniques. Without a hierarchy the two vocabularies never "
                   "intersect and every such student scores zero against everybody, which the "
                   "optimiser then reports as a shortage of supervisors that does not exist. A "
                   "student who names a technique still scores above one who names only the "
                   "paradigm, so precision is rewarded without vagueness being fatal.")
        default_tax = "\n".join(f"{k}: {'; '.join(v)}"
                                for k, v in mc.DEFAULT_METHOD_TAXONOMY.items())
        tax_raw = st.text_area("One line per paradigm, in the form paradigm: method; method",
                               st.session_state.taxonomy_raw or default_tax, height=120)
        st.session_state.taxonomy_raw = tax_raw
        tax_map = {}
        for line in tax_raw.splitlines():
            if ":" in line:
                head, tail = line.split(":", 1)
                tax_map[head.strip().lower()] = mc.split_tags(tail)
        taxonomy = mc.MethodTaxonomy(tax_map or None)

        st.divider()
        st.subheader("C. Rule-based scoring")
        c1, c2, c3 = st.columns(3)
        w_area = c1.slider("Research area", 0.0, 1.0, 0.32, 0.02)
        w_method = c2.slider("Method", 0.0, 1.0, 0.24, 0.02)
        w_text = c3.slider("Text similarity", 0.0, 1.0, 0.16, 0.02)
        c4, c5, c6 = st.columns(3)
        w_phrase = c4.slider("Proposal key phrases", 0.0, 1.0, 0.18, 0.02,
                             help="How much of the student's own distinctive wording the "
                                  "supervisor's profile contains")
        w_pref = c5.slider("Student preference", 0.0, 1.0, 0.05, 0.05)
        w_group = c6.slider("Same group", 0.0, 1.0, 0.05, 0.05)
        latent_share = st.slider("Share of text similarity taken from the latent space",
                                 0.0, 1.0, 0.5, 0.1,
                                 help="Surface overlap misses a proposal that says stock levels "
                                      "where a profile says inventory. Reducing the "
                                      "term-document matrix recovers much of that.")

        syn_raw = st.text_area(
            "Synonyms, one per line, in the form variant=canonical",
            "ml=machine learning\nai=machine learning\nstats=regression\nor=optimisation",
            height=100)
        synonyms = {}
        for line in syn_raw.splitlines():
            if "=" in line:
                a, b = line.split("=", 1)
                synonyms[a.strip().lower()] = b.strip().lower()

        if st.button("Calculate the score matrix", type="primary"):
            weights = mc.ScoreWeights(w_area, w_method, w_text, w_phrase, w_pref, w_group)
            score, comps = mc.build_score_matrix(
                st.session_state.students, reg, weights=weights, synonyms=synonyms,
                taxonomy=taxonomy, latent_share=latent_share)
            st.session_state.score = score
            st.session_state.components = comps
            st.success(f"Done: {score.shape[0]} students by {score.shape[1]} supervisors")

        if st.session_state.score is not None:
            best = st.session_state.score.max(axis=1)
            c1, c2, c3 = st.columns(3)
            c1.metric("Median best match", f"{np.median(best):.3f}")
            c2.metric("Lowest best match", f"{best.min():.3f}")
            c3.metric("Students whose best match is below 0.3", int((best < 0.3).sum()))
            st.caption("A student whose best available match is very low is a structural problem "
                       "rather than an allocation one, since nobody in the register works in "
                       "that area. That is a matter for recruitment or for steering topics, and "
                       "no tolerance setting will repair it.")
            st.bar_chart(np.histogram(best, bins=20, range=(0, 1))[0])

            comps = st.session_state.components
            if comps and comps.get("student_phrases"):
                st.markdown("**Key phrases taken from each proposal**")
                st.caption("These drive part of the score and, more usefully, they appear "
                           "against each pairing as the evidence for it, so that a decision can "
                           "be explained in the student's own words rather than as a number.")
                phrase_table = pd.DataFrame({
                    "student_id": st.session_state.students["student_id"],
                    "project_title": st.session_state.students["project_title"],
                    "key_phrases": ["; ".join(p) for p in comps["student_phrases"]],
                })
                st.dataframe(phrase_table, width="stretch", height=240)
                empty = int(sum(1 for p in comps["student_phrases"] if not p))
                if empty:
                    st.warning(f"{empty} proposals yielded no distinctive phrases at all, which "
                               "usually means the text is too short or too generic to match on.")

# ---------------------------------------------------------------------------
# Tab 4: allocation round
# ---------------------------------------------------------------------------

with tab_alloc:
    store = get_store()
    reg = store.register_dataframe()
    if st.session_state.score is None or not len(reg):
        st.info("Calculate the score matrix in ③ first")
    else:
        stu_all = st.session_state.students
        score_all = st.session_state.score

        st.subheader("Scope of this round")
        st.caption("A round covers one level and any number of programmes within it. Holding a "
                   "round to a single level is what allows a unit allowance to be converted into "
                   "a clean headcount ceiling, which is in turn what keeps the relaxation "
                   "integral and the solve fast.")
        levels = sorted(store.level_weights.keys())
        has_level = "level" in stu_all.columns
        c1, c2, c3 = st.columns([1, 2, 1])
        level = c1.selectbox("Level", levels)
        if has_level and "programme" in stu_all.columns:
            progs = sorted(stu_all.loc[stu_all["level"].astype(str) == level, "programme"]
                           .astype(str).unique())
        else:
            progs = []
            c2.info(f"No level column in the student file, so every student is treated as {level}")
        chosen = c2.multiselect("Programmes in this round", progs, default=progs)
        skip_committed = c3.checkbox("Exclude students already committed", value=True)

        mask = np.ones(len(stu_all), dtype=bool)
        if has_level:
            mask &= (stu_all["level"].astype(str) == level).to_numpy()
        if chosen:
            mask &= stu_all["programme"].astype(str).isin(chosen).to_numpy()
        committed = store.committed_students()
        if skip_committed and committed:
            mask &= ~stu_all["student_id"].astype(str).isin(committed).to_numpy()
        idx = np.where(mask)[0]
        stu = stu_all.iloc[idx].reset_index(drop=True)
        score = score_all[idx, :]

        if committed:
            outstanding = (~stu_all["student_id"].astype(str).isin(committed)).to_numpy()
        else:
            outstanding = np.ones(len(stu_all), dtype=bool)
        upcoming = int((outstanding & ~mask).sum())

        st.subheader("Capacity")
        c1, c2, c3 = st.columns(3)
        reservation = c1.radio("Reserve for later rounds", ["proportional", "use_all"], index=0,
                               format_func=lambda x: {"proportional": "In proportion to students",
                                                      "use_all": "Let this round use everything"}[x],
                               help="Whoever runs first takes the best-matched supervisors. "
                                    "Reserving protects the cohort allocated last.")
        floor_reserve = c2.number_input("Minimum places kept per supervisor", 0, 10, 0)
        c3.metric(f"Units per {level} dissertation", store.level_weights.get(level, 1.0))

        level_capacity = store.capacity_for_round(reg, level)
        capacity = stx.reserve_capacity(level_capacity, len(stu), upcoming, mode=reservation,
                                        floor_per_supervisor=int(floor_reserve))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Students in this round", len(stu))
        c2.metric("Still to place afterwards", upcoming)
        c3.metric(f"Places available at {level}", int(level_capacity.sum()))
        c4.metric("Places opened to this round", int(capacity.sum()))
        if capacity.sum() < len(stu):
            st.warning("Fewer places are open than there are students in the round, so some "
                       "will be left unassigned by construction.")

        with st.expander("Group ceilings, to stop one group absorbing a popular cohort"):
            use_group_caps = st.checkbox("Apply group ceilings", value=False)
            group_caps = {}
            if use_group_caps:
                groups = sorted(reg["group"].astype(str).unique())
                cols = st.columns(min(3, len(groups)))
                for k, g in enumerate(groups):
                    default = int(math.ceil(len(stu) / max(len(groups), 1) * 1.4))
                    group_caps[g] = cols[k % len(cols)].number_input(
                        g, 0, max(len(stu), 1), default, key=f"gc_{level}_{g}")
                st.caption("Supervisors partition into groups, so these ceilings nest inside the "
                           "individual ones and the model remains a flow network. Integrality "
                           "and solve time are unaffected.")

        st.subheader("Thresholds and tolerance")
        c1, c2, c3 = st.columns(3)
        hard_floor = c1.slider("Hard floor", 0.0, 1.0, 0.15, 0.01,
                               help="Pairings below this score never enter the model at all")
        good_threshold = c2.slider("Good-match threshold", 0.0, 1.0, 0.50, 0.01)
        tolerance = c3.slider("Mismatch tolerance", 0.0, 1.0, 0.10, 0.01,
                              help="Share of allocations permitted to fall below the "
                                   "good-match threshold")
        c4, c5, c6 = st.columns(3)
        top_k = c4.slider("Candidate supervisors kept per student", 5, 60, 25)
        restrict_group = c5.checkbox("Confine students to their stated group", value=False)
        unassigned_penalty = c6.slider("Penalty for leaving a student unassigned",
                                       1.0, 20.0, 5.0, 0.5,
                                       help="High values prefer a weak match to none, low values "
                                            "prefer to hand the student to a human")

        sup_index = {str(s): j for j, s in enumerate(reg["supervisor_id"])}
        locked = {}
        if "locked_supervisor_id" in stu.columns:
            for i, v in enumerate(stu["locked_supervisor_id"]):
                j = sup_index.get(str(v).strip())
                if j is not None:
                    locked[i] = j
            if locked:
                st.info(f"{len(locked)} pairings are locked by the office and will be preserved")

        unavailable = [j for j, a in enumerate(reg["available"]) if int(a) == 0]
        if unavailable:
            st.info(f"{len(unavailable)} supervisors are unavailable this session and have been "
                    "removed from the candidate pool")

        conflict_pairs = set()
        conf = st.session_state.conflicts
        if conf is not None and len(conf):
            stu_pos = {str(s): i for i, s in enumerate(stu["student_id"])}
            for _, row in conf.iterrows():
                i = stu_pos.get(str(row["student_id"]).strip())
                j = sup_index.get(str(row["supervisor_id"]).strip())
                if i is not None and j is not None:
                    conflict_pairs.add((i, j))
            if conflict_pairs:
                st.info(f"{len(conflict_pairs)} declared conflicts apply to this round")

        eligibility = None
        if "programmes" in reg.columns and "programme" in stu.columns:
            elig = np.ones((len(stu), len(reg)), dtype=bool)
            restricted = 0
            stu_progs = [str(v).strip().lower() for v in stu["programme"]]
            for j, allowed in enumerate([mc.split_tags(v) for v in reg["programmes"]]):
                if not allowed:
                    continue
                restricted += 1
                for i, pg in enumerate(stu_progs):
                    elig[i, j] = pg in allowed
            if restricted:
                eligibility = elig
                st.info(f"{restricted} supervisors take only certain programmes, and candidates "
                        "have been filtered accordingly")

        if st.button("Run this round", type="primary"):
            cand = mc.build_candidates(
                score, hard_floor=hard_floor, top_k=top_k,
                restrict_to_group=restrict_group,
                student_groups=stu.get("preferred_group"),
                supervisor_groups=reg["group"],
                locked=locked, unavailable=unavailable,
                blocked_pairs=conflict_pairs, eligibility=eligibility)
            params = mc.AllocationParams(
                good_threshold=good_threshold, tolerance=tolerance,
                unassigned_penalty=unassigned_penalty,
                engine=engine, milp_solver=milp_solver)
            reg_for_run = reg.copy()
            reg_for_run["workload"] = capacity
            with st.spinner("Solving"):
                comps_all = st.session_state.components
                comps_round = None
                if comps_all:
                    comps_round = {
                        k: (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
                        for k, v in comps_all.items()
                        if k in ("area", "method", "text", "phrase", "student_phrases",
                                 "phrase_evidence", "student_areas", "student_methods")
                    }
                    comps_round["supervisor_areas"] = comps_all["supervisor_areas"]
                    comps_round["supervisor_methods"] = comps_all["supervisor_methods"]
                result = mc.allocate(stu, reg_for_run, score, cand, params, capacity,
                                     group_caps if use_group_caps else None,
                                     components=comps_round)
            st.session_state.candidates = cand
            st.session_state.result = result
            st.session_state.audit = None
            st.session_state.diagnostics = mc.diagnose_allocation(
                result, score, cand, stu, reg_for_run, capacity)
            st.session_state.round_scope = {
                "level": level, "programmes": chosen, "students": len(stu),
                "params": {"hard_floor": hard_floor, "good_threshold": good_threshold,
                           "tolerance": tolerance, "top_k": top_k,
                           "reservation": reservation, "engine": engine,
                           "group_caps": group_caps if use_group_caps else {},
                           "register_version": store.version},
            }

        res = st.session_state.result
        if res is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Assigned", res.n_assigned)
            c2.metric("Unassigned", res.n_unassigned)
            c3.metric("Below the good-match threshold", res.n_below_good)
            c4.metric("Mean match", f"{res.total_score / max(res.n_assigned, 1):.3f}")
            c5.metric("Solve time", f"{res.solve_seconds:.2f}s")
            for line in res.log:
                st.caption(line)

            st.markdown("**Allocation**")
            st.dataframe(res.assignment, width="stretch", height=320)
            st.markdown("**Load, measured against the places opened to this round rather than "
                        "the annual allowance**")
            st.dataframe(res.loads.sort_values("spare"), width="stretch", height=240)

            diag = st.session_state.diagnostics
            if diag is not None and len(diag):
                st.markdown("**Prepared answers for queries**")
                st.caption("A wasteful pairing is one where a better-suited supervisor still had "
                           "room, which happens legitimately when a minimum load or a group "
                           "ceiling binds. A blocking pairing is one where the better-suited "
                           "supervisor was full but is carrying a weaker match, which is what "
                           "two students notice as soon as they compare notes. Both are normal "
                           "products of the model, listed here so that the office can explain "
                           "them rather than meet them cold.")
                st.dataframe(diag, width="stretch", height=220)

            st.divider()
            if st.button("Trace the tolerance frontier"):
                params = mc.AllocationParams(good_threshold=good_threshold, tolerance=tolerance,
                                             unassigned_penalty=unassigned_penalty, engine="lp")
                reg_for_run = reg.copy()
                reg_for_run["workload"] = capacity
                frontier = mc.tolerance_frontier(stu, reg_for_run, score,
                                                 st.session_state.candidates, params,
                                                 capacity=capacity)
                st.dataframe(frontier, width="stretch")
                st.line_chart(frontier.set_index("tolerance")[["mean_score"]])

# ---------------------------------------------------------------------------
# Tab 5: AI review
# ---------------------------------------------------------------------------

with tab_audit:
    store = get_store()
    res = st.session_state.result
    if res is None:
        st.info("Run a round in ④ first")
    else:
        reg = store.register_dataframe()
        stu_all = st.session_state.students
        scope = st.session_state.round_scope or {}
        level = scope.get("level", "UG")

        st.subheader("AI review of the completed round")
        st.caption("The model reviews rather than decides. It reads pairings that are already "
                   "settled and returns ok, query or reject with a reason, and the office "
                   "decides what to act on. The standard applied differs by level, since a "
                   "taught masters topic is expected to need real methodological depth from the "
                   "supervisor where an undergraduate one often does not.")
        scope_choice = st.radio("What to review",
                                ["Only pairings below the good-match threshold",
                                 "The N lowest-scoring pairings", "Every pairing"], index=0)
        n_lowest = st.number_input("N", 10, 500, 50, step=10,
                                   disabled=(scope_choice != "The N lowest-scoring pairings"))

        assigned = res.assignment.dropna(subset=["supervisor_id"])
        if scope_choice.startswith("Only"):
            target = assigned[assigned["below_good"]]
        elif scope_choice.startswith("The N"):
            target = assigned.nsmallest(int(n_lowest), "score")
        else:
            target = assigned
        send_keywords = st.checkbox(
            "Include supervisor keywords in the review", value=False,
            help="Off by default: a profile written as a list of publication titles identifies "
                 "the person as surely as their name does. The research areas and methods alone "
                 "are usually enough for the model to judge fit.")
        st.write(f"{len(target)} pairings selected")

        if st.button("Run the review", type="primary",
                     disabled=not elm_ready or len(target) == 0):
            client = elm.ELMClient(elm_cfg)
            pseudo = st.session_state.pseudo
            stu_by_id = stu_all.set_index("student_id")
            sup_by_id = reg.set_index("supervisor_id")
            pairs = []
            for _, row in target.iterrows():
                srow = stu_by_id.loc[row["student_id"]].to_dict()
                srow["student_id"] = row["student_id"]
                trow = sup_by_id.loc[row["supervisor_id"]].to_dict()
                trow["supervisor_id"] = row["supervisor_id"]
                pairs.append({
                    **elm.student_payload(srow, pseudo, level=level),
                    "supervisor": elm.supervisor_payload(trow, pseudo,
                                                         include_keywords=send_keywords),
                    "rule_score": round(float(row["score"]), 4),
                    "evidence": str(row.get("evidence", "")),
                })
            bar = st.progress(0.0, text="Reviewing")
            reviews = elm.audit_allocation(
                client, pairs, batch_size=8,
                progress=lambda d, t: bar.progress(d / t, text=f"Reviewing {d}/{t}"))
            bar.empty()
            audit_df = pd.DataFrame(reviews)
            if len(audit_df):
                audit_df["student_id"] = audit_df["student_code"].map(pseudo.real)
                audit_df = audit_df.merge(
                    assigned[["student_id", "supervisor_name", "score", "project_title"]],
                    on="student_id", how="left")
                audit_df = audit_df[["student_id", "project_title", "supervisor_name",
                                     "score", "verdict", "confidence", "reason"]]
            st.session_state.audit = audit_df

        audit_df = st.session_state.audit
        if audit_df is not None and len(audit_df):
            counts = audit_df["verdict"].value_counts()
            c1, c2, c3 = st.columns(3)
            c1.metric("ok", int(counts.get("ok", 0)))
            c2.metric("query", int(counts.get("query", 0)))
            c3.metric("reject", int(counts.get("reject", 0)))
            st.dataframe(audit_df.sort_values("verdict"), width="stretch", height=380)
            st.caption("Treat a reject as a prompt rather than a ruling. The model cannot see a "
                       "supervisor's full record, and the cost of a wrong rejection falls on the "
                       "office, so this column is best used to order the manual check.")

# ---------------------------------------------------------------------------
# Tab 6: rounds
# ---------------------------------------------------------------------------

with tab_rounds:
    store = get_store()
    reg = store.register_dataframe()

    st.subheader("Commit this round")
    st.caption("Committing writes the pairings into the session file and decrements the "
               "supervisors' remaining allowance immediately, so that the next round, at either "
               "level, sees what is genuinely left. Until then the result is a draft and can be "
               "rerun as often as needed.")
    res = st.session_state.result
    scope = st.session_state.round_scope
    if res is None or scope is None:
        st.info("There is no result from ④ to commit")
    else:
        default_label = (f"{scope['level']} · "
                         + (", ".join(scope["programmes"]) if scope["programmes"] else "all")
                         + f" · {res.n_assigned} students")
        label = st.text_input("Label for this round", value=default_label)
        c1, c2 = st.columns([1, 3])
        if c1.button("Commit", type="primary"):
            _, err = guarded(store.commit_round, res, label, scope["level"],
                             scope["programmes"], scope["params"])
            if err:
                st.error(err)
            else:
                st.session_state.result = None
                st.session_state.round_scope = None
                st.success("Committed. Remaining allowances have been updated.")
                st.rerun()
        c2.caption(f"{res.n_assigned} pairings will be written. The {res.n_unassigned} unassigned "
                   "students are not committed and stay in the pool for a later round or for "
                   "manual handling.")

    st.divider()
    st.subheader("Session position")
    if not len(reg):
        st.info("The register is empty")
    else:
        cap = store.capacity_table(reg)
        c1, c2, c3 = st.columns(3)
        c1.metric("Allowance (units)", f"{cap['allowance_units'].sum():.0f}")
        c2.metric("Used", f"{cap['units_used'].sum():.0f}")
        c3.metric("Left", f"{cap['units_left'].sum():.0f}")
        st.dataframe(cap.sort_values("units_left"), width="stretch", height=260)
        summary = store.rounds_summary()
        if len(summary):
            st.markdown("**Committed rounds**")
            st.dataframe(summary, width="stretch")

    st.divider()
    st.subheader("Amendments")
    st.caption("These are the things that always happen mid-session: a supervisor goes on sick "
               "leave or resigns, a student changes topic, two students ask to swap, a round was "
               "run on the wrong parameters. None of them requires the rest of the session to be "
               "unpicked.")
    if len(store.data.get("rounds", [])):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Undo a whole round**")
            options = {f"{r['label']} · {r['timestamp']} · {len(r['rows'])}": r["round_id"]
                       for r in store.data["rounds"]}
            pick = st.selectbox("Round", list(options.keys()))
            if st.button("Undo this round"):
                _, err = guarded(store.rollback, options[pick])
                if err:
                    st.error(err)
                else:
                    st.success("Undone. The capacity has been returned to the register.")
                    st.rerun()
        with c2:
            st.markdown("**Supervisor withdraws**")
            st.caption("Releases their students only. Every other pairing stands, and the "
                       "released students can be rerun on their own in ④.")
            if len(reg):
                sup_map = {f"{r['name']} ({r['supervisor_id']})": r["supervisor_id"]
                           for _, r in reg.iterrows()}
                pick_sup = st.selectbox("Supervisor", list(sup_map.keys()), key="rel_sup")
                if st.button("Release their students"):
                    affected, err = guarded(store.release_supervisor, sup_map[pick_sup])
                    if err:
                        st.error(err)
                    else:
                        st.warning(f"{len(affected)} students returned to the pool: "
                                   + ", ".join(affected[:20])
                                   + ("…" if len(affected) > 20 else ""))

        st.markdown("**Release individual students**")
        ids_raw = st.text_area("student_id, separated by commas or new lines", height=70)
        if st.button("Release these students"):
            ids = [x.strip() for x in re.split(r"[,\s]+", ids_raw) if x.strip()]
            removed, err = guarded(store.release_students, ids)
            st.error(err) if err else st.success(f"{removed} pairings released")
    else:
        st.info("No rounds have been committed yet")

    st.divider()
    st.subheader("The price of running in sequence")
    st.caption("Cohort-by-cohort allocation is a greedy procedure: whoever runs first takes the "
               "best-matched supervisors and the last cohort pays for it. This compares the "
               "sequence against a single joint solve of the same students, which is what the "
               "office would achieve if every programme submitted on the same day.")
    if st.session_state.score is None or not len(reg) or st.session_state.students is None:
        st.info("A score matrix from ③ is needed")
    else:
        stu_all = st.session_state.students
        col = "programme" if "programme" in stu_all.columns else "level"
        values = sorted(stu_all[col].astype(str).unique())
        order = st.multiselect(f"Order of {col}s, first to last", values, default=values)
        mode = st.radio("Reservation", ["proportional", "use_all"], index=0, horizontal=True,
                        format_func=lambda x: {"proportional": "In proportion to students",
                                               "use_all": "Each round uses everything"}[x],
                        key="seqmode")
        lvl_for_cost = st.selectbox("Capacity basis", sorted(store.level_weights.keys()),
                                    key="costlevel")
        if st.button("Estimate the cost of sequencing") and order:
            reg_cost = reg.copy()
            cap_basis = store.capacity_for_round(reg, lvl_for_cost)
            reg_cost["workload"] = cap_basis
            with st.spinner("Estimating"):
                cost = stx.sequencing_cost(
                    stu_all, reg_cost, st.session_state.score,
                    mc.AllocationParams(good_threshold=0.5, tolerance=0.1, engine="lp"),
                    col, order, capacity=cap_basis, reservation=mode)
            st.dataframe(cost, width="stretch")
            st.caption("A positive gap means that cohort loses out under the sequence. Showing "
                       "this to the programme director scheduled last is more persuasive than "
                       "any assurance.")

# ---------------------------------------------------------------------------
# Tab 7: export
# ---------------------------------------------------------------------------

with tab_export:
    store = get_store()
    reg = store.register_dataframe()
    res = st.session_state.result

    st.subheader("This round")
    if res is None:
        st.info("Nothing from ④ to export")
    else:
        st.download_button("Allocation (CSV)", to_csv_bytes(res.assignment),
                           "allocation_round.csv", "text/csv")
        st.download_button("Supervisor load (CSV)", to_csv_bytes(res.loads),
                           "round_loads.csv", "text/csv")
        if st.session_state.audit is not None and len(st.session_state.audit):
            st.download_button("AI review (CSV)", to_csv_bytes(st.session_state.audit),
                               "ai_review.csv", "text/csv")
        if st.session_state.diagnostics is not None and len(st.session_state.diagnostics):
            st.download_button("Prepared answers (CSV)",
                               to_csv_bytes(st.session_state.diagnostics),
                               "query_diagnostics.csv", "text/csv")
        if st.session_state.extracted is not None:
            st.download_button("ELM extraction (CSV)", to_csv_bytes(st.session_state.extracted),
                               "extracted_profiles.csv", "text/csv")

    st.divider()
    st.subheader("Whole session")
    all_rows = store.committed_rows()
    if len(all_rows):
        st.dataframe(all_rows, width="stretch", height=260)
        st.download_button("All committed allocations (CSV)", to_csv_bytes(all_rows),
                           f"allocations_{store.session.replace('/', '-')}.csv", "text/csv")
        if len(reg):
            st.download_button("Register with remaining capacity (CSV)",
                               to_csv_bytes(store.capacity_table(reg)),
                               "register_remaining.csv", "text/csv")
        trail = store.audit_trail()
        if len(trail):
            st.download_button("Audit trail (CSV)", to_csv_bytes(trail),
                               "audit_trail.csv", "text/csv")
    else:
        st.info("No rounds have been committed in this session yet")

    st.divider()
    st.subheader("Provenance")
    st.caption("Kept so that a query about any student can be answered by rerunning the same "
               "round against the same register version and obtaining the same result.")
    if res is not None and st.session_state.round_scope:
        provenance = {
            "session": store.session,
            "register_version": st.session_state.round_scope["params"]["register_version"],
            "level": st.session_state.round_scope["level"],
            "programmes": st.session_state.round_scope["programmes"],
            "engine": res.engine,
            "lagrange_multiplier": res.lagrange_lambda,
            "assigned": res.n_assigned,
            "unassigned": res.n_unassigned,
            "below_good": res.n_below_good,
            "realised_mismatch_rate": round(res.tolerance_used, 4),
            "total_score": round(res.total_score, 4),
            "solve_seconds": round(res.solve_seconds, 3),
            "parameters": st.session_state.round_scope["params"],
        }
        st.json(provenance)
        st.download_button("Provenance (JSON)",
                           json.dumps(provenance, ensure_ascii=False, indent=2).encode("utf-8"),
                           "run_provenance.json", "application/json")
