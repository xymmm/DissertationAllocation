"""
setup_session.py
================
Start an academic session and load the supervisor register into it.

Run this once at the beginning of the session, and again whenever the register
spreadsheet changes. It never replaces the register wholesale: supervisors
already carrying students cannot be removed by a stale spreadsheet, and anybody
missing from the upload is reported and left alone.

Open it in PyCharm, adjust the few lines below, and press Run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import matching_core as mc
import store as stx

SETTINGS = {
    "session_file": "session_2026-27.json",
    "session_name": "2026/27",

    # The register spreadsheet. Required columns: supervisor_id, name, group,
    # research_areas, methods, allowance_units. Optional: keywords, min_load,
    # available, programmes, cap_UG, cap_PGT.
    "register_file": "sample_supervisors.csv",

    # Supervision units per dissertation, by level. A masters dissertation is
    # more work than an undergraduate one, so an allowance of twelve units buys
    # twelve undergraduates, or eight masters students, or a mixture.
    "level_weights": {"UG": 1.0, "PGT": 1.5},

    # Leave allowances alone and refresh only names, areas and methods. Set this
    # to True once the office has started editing allowances in the session and
    # the spreadsheet has fallen behind.
    "keep_existing_allowances": False,
}


def main() -> None:
    s = SETTINGS
    path = Path(s["register_file"])
    if not path.exists():
        sys.exit(f"Cannot find {path}. Check register_file at the top of this file.")

    store = stx.SessionStore(s["session_file"], session=s["session_name"])
    if store.level_weights != s["level_weights"]:
        store.set_level_weights(s["level_weights"])

    df = pd.read_excel(path, dtype=str).fillna("") if path.suffix.lower() in (".xlsx", ".xls") \
        else pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")

    report = store.sync_from_dataframe(
        df, update_allowances=not s["keep_existing_allowances"])

    print(f"Session {store.session}, file {store.path}, register version {store.version}")
    print(f"  added    {len(report['added'])}")
    print(f"  updated  {len(report['updated'])}")
    print(f"  the same {len(report['unchanged'])}")
    if report["absent_from_upload"]:
        print(f"  in the register but not in the spreadsheet, left untouched: "
              f"{len(report['absent_from_upload'])}")
        print("    " + ", ".join(report["absent_from_upload"][:15]))

    if report["tags_trimmed"]:
        print(f"\n{len(report['tags_trimmed'])} tag lists were longer than agreed and have "
              f"been trimmed to {mc.MAX_AREA_TAGS} areas and {mc.MAX_METHOD_TAGS} methods, "
              "keeping the order given. Worth going back to these people rather than losing "
              "the rest quietly:")
        print(pd.DataFrame(report["tags_trimmed"]).to_string(index=False))

    register = store.register_dataframe()
    table = store.capacity_table(register)
    print(f"\n{len(register)} supervisors, {table['allowance_units'].sum():.0f} supervision "
          f"units in total, {table['units_used'].sum():.0f} already used, "
          f"{table['units_left'].sum():.0f} left")
    for level, weight in sorted(store.level_weights.items()):
        places = store.capacity_for_round(register, level)
        print(f"  room for {int(places.sum())} more students at {level} "
              f"(one of them costs {weight} units)")

    unavailable = int((register["available"].astype(int) == 0).sum())
    if unavailable:
        print(f"  {unavailable} supervisors are marked unavailable this session")


if __name__ == "__main__":
    main()
