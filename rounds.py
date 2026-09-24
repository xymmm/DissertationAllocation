"""
rounds.py
=========
Round ledger for running the allocation in stages.

The office rarely allocates a whole cohort in one sitting. Business Management
closes first, Accounting a fortnight later, a handful of late submissions and
resit students arrive in August, and somewhere in the middle a supervisor goes
on unexpected leave and their students have to be moved. All of that is the
same underlying problem solved repeatedly against a shrinking pool of capacity,
so the state that has to persist between runs is small: which students are
already committed, and how much workload each supervisor has left.

The ledger keeps that state in a single JSON file, which the office can copy,
archive and email. Every committed round records the parameters that produced
it, so a result can be reproduced or rolled back without guesswork.

A warning the ledger is designed to make visible rather than hide: allocating
programme by programme is a greedy procedure. Whoever runs first takes the best
supervisors, and the last programme pays for it. ``sequencing_cost`` measures
that penalty against a joint allocation of the same students, and
``reserve_capacity`` is the mitigation, holding back capacity for the rounds
that have not run yet.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import matching_core as mc


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

@dataclass
class Round:
    round_id: str
    label: str
    programmes: List[str]
    timestamp: str
    params: dict
    rows: List[dict]              # student_id, supervisor_id, score, locked

    def to_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "label": self.label,
            "programmes": self.programmes,
            "timestamp": self.timestamp,
            "params": self.params,
            "rows": self.rows,
        }


class Ledger:
    """Committed allocations plus whatever capacity is left."""

    def __init__(self, path: str | Path = "allocation_ledger.json"):
        self.path = Path(path)
        self.rounds: List[Round] = []
        if self.path.exists():
            self.load()

    # -- persistence -------------------------------------------------------

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.rounds = [Round(**r) for r in data.get("rounds", [])]

    def save(self) -> None:
        payload = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "rounds": [r.to_dict() for r in self.rounds]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # -- state -------------------------------------------------------------

    def committed_rows(self) -> pd.DataFrame:
        rows = []
        for r in self.rounds:
            for row in r.rows:
                rows.append({**row, "round_id": r.round_id, "round_label": r.label})
        if not rows:
            return pd.DataFrame(columns=["student_id", "supervisor_id", "score",
                                         "round_id", "round_label"])
        return pd.DataFrame(rows)

    def committed_students(self) -> set:
        return set(self.committed_rows().get("student_id", pd.Series(dtype=str)).astype(str))

    def used_capacity(self, supervisors: pd.DataFrame) -> np.ndarray:
        used = np.zeros(len(supervisors))
        pos = {str(s): j for j, s in enumerate(supervisors["supervisor_id"])}
        rows = self.committed_rows()
        for sid in rows.get("supervisor_id", pd.Series(dtype=str)):
            j = pos.get(str(sid))
            if j is not None:
                used[j] += 1
        return used

    def remaining_capacity(self, supervisors: pd.DataFrame) -> np.ndarray:
        annual = supervisors["workload"].astype(float).to_numpy()
        return np.maximum(annual - self.used_capacity(supervisors), 0.0)

    def capacity_table(self, supervisors: pd.DataFrame) -> pd.DataFrame:
        used = self.used_capacity(supervisors)
        out = supervisors[["supervisor_id", "name", "group", "workload"]].copy()
        out["already_assigned"] = used.astype(int)
        out["remaining"] = (out["workload"].astype(float) - used).clip(lower=0).astype(int)
        return out

    # -- mutation ----------------------------------------------------------

    def commit(self, result: mc.AllocationResult, label: str,
               programmes: Sequence[str], params: dict) -> Round:
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
        rnd = Round(
            round_id=uuid.uuid4().hex[:8],
            label=label,
            programmes=list(programmes),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            params=params,
            rows=rows,
        )
        self.rounds.append(rnd)
        self.save()
        return rnd

    def rollback(self, round_id: str) -> bool:
        before = len(self.rounds)
        self.rounds = [r for r in self.rounds if r.round_id != round_id]
        self.save()
        return len(self.rounds) < before

    def release_students(self, student_ids: Sequence[str]) -> int:
        """Free specific students from every round, for a topic change or a swap."""
        targets = {str(s) for s in student_ids}
        removed = 0
        for r in self.rounds:
            keep = [row for row in r.rows if row["student_id"] not in targets]
            removed += len(r.rows) - len(keep)
            r.rows = keep
        self.save()
        return removed

    def release_supervisor(self, supervisor_id: str) -> List[str]:
        """Free every student of one supervisor, for illness or resignation.

        Returns the affected student ids so the office can re-run only those
        while every other pairing stays locked, which keeps the disruption of a
        mid-cycle departure confined to the people actually affected.
        """
        affected = []
        for r in self.rounds:
            keep = []
            for row in r.rows:
                if str(row["supervisor_id"]) == str(supervisor_id):
                    affected.append(row["student_id"])
                else:
                    keep.append(row)
            r.rows = keep
        self.save()
        return affected

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "round_id": r.round_id,
            "label": r.label,
            "programmes": ", ".join(r.programmes) if r.programmes else "全部",
            "timestamp": r.timestamp,
            "assigned": len(r.rows),
            "mean_score": round(float(np.mean([x["score"] for x in r.rows])), 4) if r.rows else None,
            "below_good": sum(1 for x in r.rows if x.get("below_good")),
        } for r in self.rounds])


# ---------------------------------------------------------------------------
# Capacity reservation for the rounds that have not run yet
# ---------------------------------------------------------------------------

def reserve_capacity(
    remaining: np.ndarray,
    current_students: int,
    upcoming_students: int,
    mode: str = "proportional",
    floor_per_supervisor: int = 0,
) -> np.ndarray:
    """Decide how much of the remaining capacity this round may consume.

    ``use_all`` lets the current round take everything it can, which is correct
    for the final round and dangerous for the first. ``proportional`` gives the
    current round a share equal to its share of the students still to be
    placed, so the last programme in the queue still finds supervisors who work
    on its topics. The result is always large enough in total to place the
    current cohort, otherwise the reservation would manufacture unassigned
    students out of nothing.
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

    # never reserve the current round into infeasibility
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
    programme_column: str,
    order: Sequence[str],
    hard_floor: float = 0.15,
    top_k: int = 25,
    reservation: str = "proportional",
) -> pd.DataFrame:
    """Compare programme-by-programme allocation against one joint allocation.

    The joint solve is the benchmark: it is what the office would achieve if
    every programme submitted on the same day. The difference is the price of
    the calendar, and it is worth knowing before promising one programme an
    early answer.
    """
    capacity = supervisors["workload"].astype(float).to_numpy()

    joint_cand = mc.build_candidates(score, hard_floor, top_k)
    joint = mc.allocate(students, supervisors, score, joint_cand, params, capacity)
    joint_by_prog = (joint.assignment.assign(
        programme=students.set_index("student_id")[programme_column]
        .reindex(joint.assignment["student_id"]).values)
        .dropna(subset=["supervisor_id"])
        .groupby("programme")["score"].agg(["count", "mean"]))

    remaining = capacity.copy()
    rows = []
    for idx, prog in enumerate(order):
        mask = students[programme_column].astype(str) == str(prog)
        sub = students[mask].reset_index(drop=True)
        if not len(sub):
            continue
        upcoming = int((students[programme_column].astype(str).isin(
            [str(p) for p in order[idx + 1:]])).sum())
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
        jrow = joint_by_prog.loc[str(prog)] if str(prog) in joint_by_prog.index else None
        rows.append({
            "order": idx + 1,
            "programme": prog,
            "students": len(sub),
            "sequential_assigned": res.n_assigned,
            "sequential_mean": round(seq_mean, 4),
            "joint_assigned": int(jrow["count"]) if jrow is not None else None,
            "joint_mean": round(float(jrow["mean"]), 4) if jrow is not None else None,
            "gap": round(float(jrow["mean"]) - seq_mean, 4) if jrow is not None else None,
        })
    return pd.DataFrame(rows)
