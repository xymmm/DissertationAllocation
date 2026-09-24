"""
store.py
========
Persistent state for one academic session.

Everything that has to survive between runs lives in a single JSON file scoped
to an academic session, for example ``session_2026-27.json``:

* the **supervisor register**, which is the authoritative record of who is
  supervising this session and how much each of them has agreed to take;
* the **round ledger**, which records every allocation the office has
  committed, with the parameters that produced it;
* an **audit trail** of edits, so a change to an allowance can be traced.

The register matters more than it first appears. A supervisor commonly takes
both undergraduate and postgraduate taught dissertations, and those two
cohorts are allocated weeks apart. If each round started from a freshly
uploaded spreadsheet, the second round would have no idea what the first one
had already committed, and the most popular supervisors would quietly be
booked twice. Here the register is read and written by the application itself:
committing a PGT round decrements the same allowance that the UG round will
later draw on, with no spreadsheet passing between the two.

Allowances are held in **supervision units** rather than in a headcount,
because a taught masters dissertation is not the same amount of work as an
undergraduate one. Each level carries a weight, so an allowance of twelve
units at weights of one for UG and one and a half for PGT means twelve
undergraduates, or eight masters students, or any combination in between.

Rounds are expected to be single-level. That is not an arbitrary restriction:
within one level every student consumes the same number of units, so the unit
allowance converts cleanly into a headcount ceiling and every coefficient in
the capacity constraints stays equal to one, which is what keeps the linear
relaxation integral. A mixed-level round would turn the capacity rows into
knapsack constraints and the model into a generalised assignment problem,
which is why the application routes such a round to the exact solver and warns
the office rather than pretending nothing has changed.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matching_core as mc

DEFAULT_LEVEL_WEIGHTS = {"UG": 1.0, "PGT": 1.5}

PROFILE_FIELDS = ["name", "group", "research_areas", "methods", "keywords",
                  "programmes", "notes"]
CAPACITY_FIELDS = ["allowance_units", "min_load", "available"]


class ConcurrentEditError(RuntimeError):
    """Raised when the file on disk has moved on since it was read."""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class SessionStore:
    """The supervisor register and the round ledger for one academic session."""

    def __init__(self, path: str | Path, session: str = "2026/27",
                 level_weights: Optional[Dict[str, float]] = None):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "session": session,
                "version": 0,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "level_weights": dict(level_weights or DEFAULT_LEVEL_WEIGHTS),
                "supervisors": {},
                "rounds": [],
                "audit": [],
            }
            self.save(force=True)

    # -- basics ------------------------------------------------------------

    @property
    def session(self) -> str:
        return self.data.get("session", "")

    @property
    def version(self) -> int:
        return int(self.data.get("version", 0))

    @property
    def level_weights(self) -> Dict[str, float]:
        return dict(self.data.get("level_weights", DEFAULT_LEVEL_WEIGHTS))

    def set_level_weights(self, weights: Dict[str, float], actor: str = "office") -> None:
        self.data["level_weights"] = {str(k): float(v) for k, v in weights.items()}
        self._audit(actor, "set_level_weights", detail=json.dumps(weights))
        self.save()

    def disk_version(self) -> int:
        if not self.path.exists():
            return -1
        try:
            return int(json.loads(self.path.read_text(encoding="utf-8")).get("version", 0))
        except Exception:
            return -1

    def reload(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def ensure_fresh(self) -> bool:
        """Reload if another process has written since this copy was read."""
        if self.path.exists() and self.disk_version() != self.version:
            self.reload()
            return True
        return False

    def save(self, force: bool = False) -> None:
        """Write atomically, refusing to overwrite somebody else's edit.

        Two people in the same office can have the application open at once, so
        every write first checks that the file on disk is still at the version
        this copy was read at. A temporary file plus an atomic replace means a
        crash midway through cannot leave a half-written register behind.
        """
        if not force and self.path.exists():
            on_disk = self.disk_version()
            if on_disk != self.version:
                raise ConcurrentEditError(
                    f"The register on disk has moved to version {on_disk} while this "
                    f"copy is at version {self.version}, which means somebody else has "
                    f"saved a change. Reload the register and apply the edit again.")
        self.data["version"] = self.version + 1
        self.data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def _audit(self, actor: str, action: str, subject: str = "", detail: str = "") -> None:
        self.data.setdefault("audit", []).append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "actor": actor, "action": action, "subject": subject, "detail": detail,
        })

    def audit_trail(self) -> pd.DataFrame:
        return pd.DataFrame(self.data.get("audit", []))

    # -- supervisor register ----------------------------------------------

    def register_dataframe(self) -> pd.DataFrame:
        rows = []
        for sid, rec in self.data.get("supervisors", {}).items():
            row = {"supervisor_id": sid}
            row.update({k: rec.get(k, "") for k in PROFILE_FIELDS})
            row["allowance_units"] = float(rec.get("allowance_units", 0))
            row["min_load"] = int(rec.get("min_load", 0))
            row["available"] = int(rec.get("available", 1))
            for lvl, cap in (rec.get("level_caps") or {}).items():
                row[f"cap_{lvl}"] = cap
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["supervisor_id"] + PROFILE_FIELDS + CAPACITY_FIELDS)
        return pd.DataFrame(rows).sort_values("supervisor_id").reset_index(drop=True)

    def sync_from_dataframe(self, df: pd.DataFrame, actor: str = "office",
                            update_allowances: bool = True) -> Dict[str, list]:
        """Merge an uploaded register into the stored one.

        A merge rather than a replacement, deliberately. Once a round has been
        committed, wholesale replacement from a spreadsheet would be capable of
        deleting a supervisor who is already carrying students, so a supervisor
        who is absent from the upload is reported and left alone rather than
        removed. Allowances can be held back from the merge when the office has
        been editing them in the application and the spreadsheet is stale.
        """
        report = {"added": [], "updated": [], "absent_from_upload": [], "unchanged": [],
                  "tags_trimmed": []}
        supervisors = self.data.setdefault("supervisors", {})
        seen = set()

        for _, row in df.iterrows():
            sid = str(row.get("supervisor_id", "")).strip()
            if not sid:
                continue
            seen.add(sid)
            existing = supervisors.get(sid)
            record = dict(existing) if existing else {"level_caps": {}}

            changed = False
            for field in PROFILE_FIELDS:
                if field not in df.columns:
                    continue
                value = "" if pd.isna(row[field]) else str(row[field]).strip()
                # Supervisors are asked for at most four areas and six methods.
                # A longer list is trimmed rather than rejected, in the order the
                # supervisor gave, and the discarded tags are reported so that the
                # office can go back to them rather than silently losing the tail.
                if field in ("research_areas", "methods"):
                    limit = mc.MAX_AREA_TAGS if field == "research_areas" else mc.MAX_METHOD_TAGS
                    tags = mc.split_tags(value)
                    if len(tags) > limit:
                        report["tags_trimmed"].append({
                            "supervisor_id": sid,
                            "field": field,
                            "kept": "; ".join(tags[:limit]),
                            "dropped": "; ".join(tags[limit:]),
                        })
                    value = "; ".join(tags[:limit])
                if record.get(field, "") != value:
                    record[field] = value
                    changed = True

            if update_allowances:
                if "allowance_units" in df.columns:
                    new_allowance = float(pd.to_numeric(row["allowance_units"], errors="coerce") or 0)
                elif "workload" in df.columns:
                    new_allowance = float(pd.to_numeric(row["workload"], errors="coerce") or 0)
                else:
                    new_allowance = float(record.get("allowance_units", 0))
                if float(record.get("allowance_units", -1)) != new_allowance:
                    record["allowance_units"] = new_allowance
                    changed = True
                for field, caster in (("min_load", int), ("available", int)):
                    if field in df.columns:
                        value = caster(pd.to_numeric(row[field], errors="coerce") or 0)
                        if record.get(field) != value:
                            record[field] = value
                            changed = True
                caps = dict(record.get("level_caps") or {})
                for col in df.columns:
                    if col.startswith("cap_"):
                        lvl = col[4:]
                        raw = pd.to_numeric(row[col], errors="coerce")
                        if not pd.isna(raw):
                            caps[lvl] = int(raw)
                if caps != (record.get("level_caps") or {}):
                    record["level_caps"] = caps
                    changed = True

            record.setdefault("allowance_units", 0.0)
            record.setdefault("min_load", 0)
            record.setdefault("available", 1)
            record.setdefault("level_caps", {})
            record["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            if existing is None:
                supervisors[sid] = record
                report["added"].append(sid)
            elif changed:
                supervisors[sid] = record
                report["updated"].append(sid)
            else:
                report["unchanged"].append(sid)

        report["absent_from_upload"] = [sid for sid in supervisors if sid not in seen]
        self._audit(actor, "sync_register",
                    detail=f"added {len(report['added'])}, updated {len(report['updated'])}, "
                           f"absent {len(report['absent_from_upload'])}, "
                           f"tag lists trimmed {len(report['tags_trimmed'])}")
        self.save()
        return report

    def update_supervisor(self, supervisor_id: str, changes: dict,
                          actor: str = "office") -> None:
        rec = self.data["supervisors"].get(str(supervisor_id))
        if rec is None:
            raise KeyError(f"Unknown supervisor: {supervisor_id}")
        for key, value in changes.items():
            if key.startswith("cap_"):
                rec.setdefault("level_caps", {})[key[4:]] = int(value)
            else:
                rec[key] = value
        rec["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._audit(actor, "update_supervisor", subject=str(supervisor_id),
                    detail=json.dumps(changes, ensure_ascii=False, default=str))
        self.save()

    # -- committed allocations --------------------------------------------

    def committed_rows(self) -> pd.DataFrame:
        rows = []
        for r in self.data.get("rounds", []):
            for row in r.get("rows", []):
                rows.append({**row, "round_id": r["round_id"], "round_label": r["label"],
                             "level": r.get("level", ""),
                             "programmes": ", ".join(r.get("programmes", []))})
        if not rows:
            return pd.DataFrame(columns=["student_id", "supervisor_id", "supervisor_name",
                                         "score", "level", "round_id", "round_label"])
        return pd.DataFrame(rows)

    def committed_students(self) -> set:
        rows = self.committed_rows()
        return set(rows["student_id"].astype(str)) if len(rows) else set()

    def used_units(self, supervisor_ids: Sequence[str]) -> np.ndarray:
        weights = self.level_weights
        pos = {str(s): j for j, s in enumerate(supervisor_ids)}
        used = np.zeros(len(supervisor_ids))
        for r in self.data.get("rounds", []):
            w = float(weights.get(r.get("level", ""), 1.0))
            for row in r.get("rows", []):
                j = pos.get(str(row["supervisor_id"]))
                if j is not None:
                    used[j] += w
        return used

    def used_headcount(self, supervisor_ids: Sequence[str],
                       level: Optional[str] = None) -> np.ndarray:
        pos = {str(s): j for j, s in enumerate(supervisor_ids)}
        used = np.zeros(len(supervisor_ids))
        for r in self.data.get("rounds", []):
            if level is not None and r.get("level") != level:
                continue
            for row in r.get("rows", []):
                j = pos.get(str(row["supervisor_id"]))
                if j is not None:
                    used[j] += 1
        return used

    def capacity_for_round(self, supervisors: pd.DataFrame, level: str) -> np.ndarray:
        """Headcount each supervisor may still take at this level.

        The unit allowance left over is divided by the weight of this level and
        rounded down, then trimmed by any cap the supervisor has set for the
        level itself, and finally by their availability.
        """
        ids = [str(s) for s in supervisors["supervisor_id"]]
        weight = float(self.level_weights.get(level, 1.0))
        allowance = np.array([float(self.data["supervisors"].get(sid, {})
                                    .get("allowance_units", 0)) for sid in ids])
        remaining_units = np.maximum(allowance - self.used_units(ids), 0.0)
        headcount = np.floor(remaining_units / max(weight, 1e-9))

        level_caps = np.array([
            float((self.data["supervisors"].get(sid, {}).get("level_caps") or {})
                  .get(level, np.inf)) for sid in ids])
        used_at_level = self.used_headcount(ids, level)
        headcount = np.minimum(headcount, np.maximum(level_caps - used_at_level, 0))

        available = np.array([int(self.data["supervisors"].get(sid, {}).get("available", 1))
                              for sid in ids])
        return np.where(available == 1, headcount, 0.0)

    def capacity_table(self, supervisors: Optional[pd.DataFrame] = None,
                       level: Optional[str] = None) -> pd.DataFrame:
        reg = self.register_dataframe() if supervisors is None else supervisors
        ids = [str(s) for s in reg["supervisor_id"]]
        weights = self.level_weights
        out = reg[["supervisor_id", "name", "group"]].copy()
        out["allowance_units"] = [float(self.data["supervisors"].get(s, {})
                                        .get("allowance_units", 0)) for s in ids]
        out["units_used"] = self.used_units(ids).round(2)
        out["units_left"] = (out["allowance_units"] - out["units_used"]).clip(lower=0).round(2)
        for lvl in weights:
            out[f"{lvl}_assigned"] = self.used_headcount(ids, lvl).astype(int)
        if level:
            out[f"places_left_{level}"] = self.capacity_for_round(reg, level).astype(int)
        return out

    # -- rounds ------------------------------------------------------------

    def commit_round(self, result: mc.AllocationResult, label: str, level: str,
                     programmes: Sequence[str], params: dict,
                     actor: str = "office") -> dict:
        rows = []
        for _, row in result.assignment.iterrows():
            if pd.isna(row["supervisor_id"]):
                continue
            rows.append({
                "student_id": str(row["student_id"]),
                "supervisor_id": str(row["supervisor_id"]),
                "supervisor_name": str(row["supervisor_name"]),
                "score": float(row["score"]),
                "below_good": bool(row["below_good"]),
                "locked": bool(row["locked"]),
            })
        rnd = {
            "round_id": uuid.uuid4().hex[:8],
            "label": label,
            "level": level,
            "programmes": list(programmes),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
            "rows": rows,
        }
        self.data.setdefault("rounds", []).append(rnd)
        self._audit(actor, "commit_round", subject=rnd["round_id"],
                    detail=f"{label}: {len(rows)} pairings at level {level}")
        self.save()
        return rnd

    def rollback(self, round_id: str, actor: str = "office") -> bool:
        before = len(self.data.get("rounds", []))
        self.data["rounds"] = [r for r in self.data.get("rounds", [])
                               if r["round_id"] != round_id]
        changed = len(self.data["rounds"]) < before
        if changed:
            self._audit(actor, "rollback_round", subject=round_id)
            self.save()
        return changed

    def release_students(self, student_ids: Sequence[str], actor: str = "office") -> int:
        targets = {str(s) for s in student_ids}
        removed = 0
        for r in self.data.get("rounds", []):
            keep = [row for row in r["rows"] if row["student_id"] not in targets]
            removed += len(r["rows"]) - len(keep)
            r["rows"] = keep
        if removed:
            self._audit(actor, "release_students", detail=f"{removed} pairings released")
            self.save()
        return removed

    def release_supervisor(self, supervisor_id: str, actor: str = "office") -> List[str]:
        """Free every student of one supervisor after illness or departure."""
        affected = []
        for r in self.data.get("rounds", []):
            keep = []
            for row in r["rows"]:
                if str(row["supervisor_id"]) == str(supervisor_id):
                    affected.append(row["student_id"])
                else:
                    keep.append(row)
            r["rows"] = keep
        if affected:
            self._audit(actor, "release_supervisor", subject=str(supervisor_id),
                        detail=f"{len(affected)} students returned to the pool")
            self.save()
        return affected

    def rounds_summary(self) -> pd.DataFrame:
        rows = []
        for r in self.data.get("rounds", []):
            scores = [x["score"] for x in r["rows"]]
            rows.append({
                "round_id": r["round_id"],
                "label": r["label"],
                "level": r.get("level", ""),
                "programmes": ", ".join(r.get("programmes", [])) or "all",
                "timestamp": r["timestamp"],
                "assigned": len(r["rows"]),
                "mean_score": round(float(np.mean(scores)), 4) if scores else None,
                "below_good": sum(1 for x in r["rows"] if x.get("below_good")),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reserving capacity for rounds that have not run yet
# ---------------------------------------------------------------------------

def reserve_capacity(
    remaining: np.ndarray,
    current_students: int,
    upcoming_students: int,
    mode: str = "proportional",
    floor_per_supervisor: int = 0,
) -> np.ndarray:
    """Decide how much of the remaining capacity this round may consume.

    ``use_all`` lets the current round take everything it can, which is right
    for the final round of a session and reckless for the first. ``proportional``
    gives the round a share equal to its share of the students still to be
    placed, so that the cohort allocated last still finds supervisors who work
    on its topics. The result is always large enough in total to place the
    current cohort, since a reservation that manufactured unassigned students
    would be worse than the problem it solves.
    """
    remaining = np.asarray(remaining, dtype=float)
    if mode == "use_all" or upcoming_students <= 0:
        return remaining.copy()

    share = current_students / max(current_students + upcoming_students, 1)
    raw = remaining * share

    allowed = np.floor(raw)
    leftover = int(round(raw.sum() - allowed.sum()))
    if leftover > 0:
        order = np.argsort(-(raw - allowed))
        for j in order[:leftover]:
            if allowed[j] < remaining[j]:
                allowed[j] += 1

    if floor_per_supervisor:
        allowed = np.maximum(allowed, np.minimum(remaining, floor_per_supervisor))

    deficit = current_students - allowed.sum()
    if deficit > 0:
        order = np.argsort(-(remaining - allowed))
        for j in order:
            take = min(deficit, remaining[j] - allowed[j])
            allowed[j] += take
            deficit -= take
            if deficit <= 0:
                break
    return np.minimum(allowed, remaining)


# ---------------------------------------------------------------------------
# What does running in sequence actually cost?
# ---------------------------------------------------------------------------

def sequencing_cost(
    students: pd.DataFrame,
    supervisors: pd.DataFrame,
    score: np.ndarray,
    params: mc.AllocationParams,
    cohort_column: str,
    order: Sequence[str],
    capacity: Optional[np.ndarray] = None,
    hard_floor: float = 0.15,
    top_k: int = 25,
    reservation: str = "proportional",
) -> pd.DataFrame:
    """Compare cohort-by-cohort allocation against a single joint allocation.

    The joint solve is the benchmark: it is what the office would achieve if
    every programme submitted on the same day. The difference is the price of
    the calendar, and it is worth knowing before promising one programme an
    early answer.
    """
    if capacity is None:
        capacity = supervisors["workload"].astype(float).to_numpy()
    capacity = np.asarray(capacity, dtype=float)

    joint_cand = mc.build_candidates(score, hard_floor, top_k)
    joint = mc.allocate(students, supervisors, score, joint_cand, params, capacity)
    joint_by_cohort = (joint.assignment.assign(
        cohort=students.set_index("student_id")[cohort_column]
        .reindex(joint.assignment["student_id"]).values)
        .dropna(subset=["supervisor_id"])
        .groupby("cohort")["score"].agg(["count", "mean"]))

    remaining = capacity.copy()
    rows = []
    for idx, cohort in enumerate(order):
        mask = students[cohort_column].astype(str) == str(cohort)
        sub = students[mask].reset_index(drop=True)
        if not len(sub):
            continue
        upcoming = int((students[cohort_column].astype(str).isin(
            [str(c) for c in order[idx + 1:]])).sum())
        allowed = reserve_capacity(remaining, len(sub), upcoming, mode=reservation)

        sub_score = score[np.where(mask)[0], :]
        cand = mc.build_candidates(sub_score, hard_floor, top_k)
        res = mc.allocate(sub, supervisors, sub_score, cand, params, allowed)

        used = np.zeros(len(supervisors))
        pos = {str(s): j for j, s in enumerate(supervisors["supervisor_id"])}
        for sid in res.assignment["supervisor_id"].dropna():
            used[pos[str(sid)]] += 1
        remaining = np.maximum(remaining - used, 0)

        seq_mean = res.total_score / max(res.n_assigned, 1)
        jrow = joint_by_cohort.loc[str(cohort)] if str(cohort) in joint_by_cohort.index else None
        rows.append({
            "order": idx + 1,
            "cohort": cohort,
            "students": len(sub),
            "sequential_assigned": res.n_assigned,
            "sequential_mean": round(seq_mean, 4),
            "joint_assigned": int(jrow["count"]) if jrow is not None else None,
            "joint_mean": round(float(jrow["mean"]), 4) if jrow is not None else None,
            "gap": round(float(jrow["mean"]) - seq_mean, 4) if jrow is not None else None,
        })
    return pd.DataFrame(rows)
