"""
commit_round.py
===============
Turn a draft round into a decision.

    python commit_round.py runs/2026-09-24_1530_PGT

Reads the folder produced by run_allocation.py, checks that it is still valid,
and writes it into the session file, reducing everybody's remaining capacity.
The check matters: if somebody has changed an allowance or committed another
round since the draft was produced, the capacity the draft was computed against
no longer exists, and committing it anyway would overbook people. In that case
this refuses and asks for the round to be run again.

Pass --force only if you are certain the draft is still right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import store as stx

SESSION_FILE = "session_2026-27.json"


def main(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    force = "--force" in argv
    if not args:
        sys.exit("Give the folder to commit, for example:\n"
                 "  python commit_round.py runs/2026-09-24_1530_PGT")

    folder = Path(args[0])
    draft_file = folder / "draft.json"
    if not draft_file.exists():
        sys.exit(f"No draft.json in {folder}. Is that the folder run_allocation.py made?")
    draft = json.loads(draft_file.read_text(encoding="utf-8"))

    store = stx.SessionStore(SESSION_FILE)
    if draft["session"] != store.session:
        sys.exit(f"This draft belongs to session {draft['session']} but {SESSION_FILE} is "
                 f"{store.session}. Refusing to mix sessions.")

    drafted_at = draft["params"].get("register_version")
    if drafted_at != store.version and not force:
        sys.exit(f"The register has moved from version {drafted_at} to {store.version} since "
                 "this draft was produced, so an allowance changed or another round was "
                 "committed. Run the allocation again so it uses the capacity that actually "
                 "exists. Add --force only if you are certain the draft is still right.")

    already = store.committed_students()
    clash = [r["student_id"] for r in draft["rows"] if r["student_id"] in already]
    if clash:
        sys.exit(f"{len(clash)} of these students already have a supervisor, for example "
                 f"{', '.join(clash[:5])}. Run the allocation again.")

    rnd = store.commit_rows(draft["rows"], draft["label"], draft["level"],
                            draft["programmes"], draft["params"])
    table = store.capacity_table(store.register_dataframe())
    print(f"Committed {len(rnd['rows'])} pairings as round {rnd['round_id']}")
    print(f"{table['units_used'].sum():.0f} supervision units now used, "
          f"{table['units_left'].sum():.0f} left for the rest of the session")
    print(f"\nWhat is in the session now:")
    print(store.rounds_summary().to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
