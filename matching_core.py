"""
matching_core.py
================
Scoring and optimisation core for undergraduate dissertation allocation.

The module is deliberately independent of Streamlit and of any LLM so that the
allocation itself stays deterministic, auditable and reproducible: given the
same two CSV files and the same parameters it returns exactly the same
allocation, which is what a teaching operations office needs when a student
queries the outcome.

Two solution engines are provided.

1. ``lp``  - the transportation-LP path. Every constraint except the mismatch
   tolerance is a pure assignment/capacity constraint, so the constraint matrix
   is totally unimodular and the LP vertex returned by HiGHS is integral
   without any branching. The tolerance is enforced by Lagrangian bisection on
   a penalty applied to below-threshold pairs, so the whole procedure is a
   handful of LP solves and stays fast at faculty scale.

2. ``milp`` - the exact path. The tolerance becomes a genuine cardinality
   constraint and the model is handed to CBC, HiGHS or Gurobi through PuLP.
   Slower in principle, still trivial at this size, and useful as a cross-check
   on the bisection result.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

# ---------------------------------------------------------------------------
# Expected columns
# ---------------------------------------------------------------------------

SUPERVISOR_COLUMNS = {
    "supervisor_id": True,      # required
    "name": True,
    "group": True,              # one of the six subject groups
    "research_areas": True,     # semicolon separated tags
    "methods": True,            # semicolon separated tags
    "keywords": False,          # free text, publications, past topics
    "workload": True,           # number of dissertations for THIS cycle
    "min_load": False,          # lower bound on number supervised
    "available": False,         # 1/0, sabbatical or leave sets this to 0
}

STUDENT_COLUMNS = {
    "student_id": True,         # target ID, the only identifier that travels
    "name": False,
    "programme": False,
    "project_title": True,
    "abstract": True,
    "project_name": False,      # student's own working name for the project
    "areas": False,             # tags, either self-declared or LLM-extracted
    "methods": False,
    "references": False,        # free text reference list
    "preferred_supervisor_id": False,
    "preferred_group": False,
    "locked_supervisor_id": False,   # office override, forced assignment
}

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SPLIT_RE = re.compile(r"[;；,，/|]+")


# ---------------------------------------------------------------------------
# Tag handling
# ---------------------------------------------------------------------------

def split_tags(value) -> List[str]:
    """Split a semicolon separated tag cell into a clean lowercase list."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parts = SPLIT_RE.split(str(value))
    out = []
    for p in parts:
        t = p.strip().lower()
        if t:
            out.append(t)
    return out


def expand_with_synonyms(tags: Sequence[str], synonyms: Dict[str, str]) -> set:
    """Map每个 tag 到其规范形式, unknown tags are kept as they are."""
    return {synonyms.get(t, t) for t in tags}


def weighted_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def coverage(a: set, b: set) -> float:
    """Share of the student's tags covered by the supervisor's profile.

    Jaccard punishes a supervisor with a broad profile, which is the wrong
    incentive here: a generalist who can cover every method the student needs
    is a good match even if most of their own profile is unused.
    """
    if not a:
        return 0.0
    return len(a & b) / len(a)


# ---------------------------------------------------------------------------
# Score matrix
# ---------------------------------------------------------------------------

@dataclass
class ScoreWeights:
    area: float = 0.40
    method: float = 0.30
    text: float = 0.20
    preference: float = 0.05
    same_group: float = 0.05

    def normalised(self) -> "ScoreWeights":
        total = self.area + self.method + self.text + self.preference + self.same_group
        if total <= 0:
            raise ValueError("权重之和必须为正")
        return ScoreWeights(
            self.area / total,
            self.method / total,
            self.text / total,
            self.preference / total,
            self.same_group / total,
        )


def _text_similarity(student_texts: List[str], supervisor_texts: List[str]) -> np.ndarray:
    """TF-IDF cosine similarity between free text profiles.

    The analyser is chosen from the corpus itself so that a bilingual intake
    behaves sensibly: character n-grams for Chinese, word n-grams otherwise.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return _fallback_similarity(student_texts, supervisor_texts)

    corpus = student_texts + supervisor_texts
    joined = " ".join(corpus)
    cjk_ratio = len(CJK_RE.findall(joined)) / max(len(joined), 1)
    if cjk_ratio > 0.05:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1)
    else:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                              stop_words="english", sublinear_tf=True)
    try:
        matrix = vec.fit_transform(corpus)
    except ValueError:
        return _fallback_similarity(student_texts, supervisor_texts)

    ns = len(student_texts)
    S = matrix[:ns]
    T = matrix[ns:]
    sim = (S @ T.T).toarray()
    return np.clip(sim, 0.0, 1.0)


def _fallback_similarity(student_texts: List[str], supervisor_texts: List[str]) -> np.ndarray:
    stoks = [set(re.findall(r"\w+", t.lower())) for t in student_texts]
    ttoks = [set(re.findall(r"\w+", t.lower())) for t in supervisor_texts]
    out = np.zeros((len(stoks), len(ttoks)))
    for i, a in enumerate(stoks):
        for j, b in enumerate(ttoks):
            out[i, j] = weighted_jaccard(a, b)
    return out


def build_score_matrix(
    students: pd.DataFrame,
    supervisors: pd.DataFrame,
    weights: ScoreWeights = ScoreWeights(),
    synonyms: Optional[Dict[str, str]] = None,
    llm_scores: Optional[np.ndarray] = None,
    llm_blend: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Return the n_students x n_supervisors score matrix in [0, 1].

    ``llm_scores`` is an optional matrix in [0, 1] produced by the ELM screening
    stage; ``llm_blend`` is the share of the final score it carries, so that the
    rule-based score remains the backbone and the model only tilts it.
    """
    synonyms = synonyms or {}
    w = weights.normalised()

    s_areas = [expand_with_synonyms(split_tags(v), synonyms) for v in students.get("areas", pd.Series([""] * len(students)))]
    s_methods = [expand_with_synonyms(split_tags(v), synonyms) for v in students.get("methods", pd.Series([""] * len(students)))]
    t_areas = [expand_with_synonyms(split_tags(v), synonyms) for v in supervisors["research_areas"]]
    t_methods = [expand_with_synonyms(split_tags(v), synonyms) for v in supervisors["methods"]]

    n, m = len(students), len(supervisors)
    area = np.zeros((n, m))
    method = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            area[i, j] = 0.5 * coverage(s_areas[i], t_areas[j]) + 0.5 * weighted_jaccard(s_areas[i], t_areas[j])
            method[i, j] = 0.5 * coverage(s_methods[i], t_methods[j]) + 0.5 * weighted_jaccard(s_methods[i], t_methods[j])

    def _cat(df, cols):
        out = []
        for _, row in df.iterrows():
            parts = [str(row[c]) for c in cols if c in df.columns and pd.notna(row.get(c))]
            out.append(" ".join(parts))
        return out

    student_text = _cat(students, ["project_title", "project_name", "abstract", "areas", "methods", "references"])
    supervisor_text = _cat(supervisors, ["research_areas", "methods", "keywords"])
    text = _text_similarity(student_text, supervisor_text)
    if text.size and text.max() > 0:
        text = text / text.max()      # rescale, TF-IDF cosines are small in absolute terms

    pref = np.zeros((n, m))
    sup_index = {str(sid): j for j, sid in enumerate(supervisors["supervisor_id"])}
    if "preferred_supervisor_id" in students.columns:
        for i, v in enumerate(students["preferred_supervisor_id"]):
            j = sup_index.get(str(v).strip())
            if j is not None:
                pref[i, j] = 1.0

    same_group = np.zeros((n, m))
    if "preferred_group" in students.columns:
        sup_groups = [str(g).strip().lower() for g in supervisors["group"]]
        for i, g in enumerate(students["preferred_group"]):
            gs = str(g).strip().lower()
            if not gs or gs == "nan":
                continue
            for j, sg in enumerate(sup_groups):
                if sg == gs:
                    same_group[i, j] = 1.0

    score = (w.area * area + w.method * method + w.text * text
             + w.preference * pref + w.same_group * same_group)

    if llm_scores is not None and llm_blend > 0:
        mask = ~np.isnan(llm_scores)
        blended = score.copy()
        blended[mask] = (1 - llm_blend) * score[mask] + llm_blend * llm_scores[mask]
        score = blended

    components = {"area": area, "method": method, "text": text,
                  "preference": pref, "same_group": same_group}
    return np.clip(score, 0.0, 1.0), components


# ---------------------------------------------------------------------------
# Candidate sparsification
# ---------------------------------------------------------------------------

@dataclass
class Candidates:
    """Sparse pair list: the model only ever sees admissible pairs."""
    rows: np.ndarray            # student index per pair
    cols: np.ndarray            # supervisor index per pair
    scores: np.ndarray          # match score per pair
    n_students: int = 0
    n_supervisors: int = 0
    forced: Dict[int, int] = field(default_factory=dict)   # student -> supervisor

    def __len__(self) -> int:
        return len(self.rows)


def build_candidates(
    score: np.ndarray,
    hard_floor: float,
    top_k: int = 25,
    restrict_to_group: bool = False,
    student_groups: Optional[Sequence[str]] = None,
    supervisor_groups: Optional[Sequence[str]] = None,
    locked: Optional[Dict[int, int]] = None,
    unavailable: Optional[Sequence[int]] = None,
    blocked_pairs: Optional[set] = None,
    eligibility: Optional[np.ndarray] = None,
) -> Candidates:
    """Keep only pairs that clear the hard floor, within the top_k per student.

    This is where the problem size is actually decided: with 900 students and
    150 supervisors the dense matrix has 135,000 pairs, while top_k = 25 leaves
    22,500 columns, which any solver disposes of instantly.

    ``blocked_pairs`` is a set of (student_index, supervisor_index) that must
    never be assigned, which is how a conflict of interest or a declared
    personal relationship is handled. ``eligibility`` is an optional boolean
    matrix for supervisors who only take certain programmes.
    """
    n, m = score.shape
    locked = locked or {}
    blocked = set(unavailable or [])
    blocked_pairs = blocked_pairs or set()

    rows, cols, vals = [], [], []
    for i in range(n):
        if i in locked:
            j = locked[i]
            rows.append(i); cols.append(j); vals.append(score[i, j])
            continue
        row = score[i].copy()
        for j in blocked:
            row[j] = -1.0
        for (bi, bj) in blocked_pairs:
            if bi == i:
                row[bj] = -1.0
        if eligibility is not None:
            row = np.where(eligibility[i], row, -1.0)
        if restrict_to_group and student_groups is not None and supervisor_groups is not None:
            g = str(student_groups[i]).strip().lower()
            if g and g != "nan":
                for j in range(m):
                    if str(supervisor_groups[j]).strip().lower() != g:
                        row[j] = -1.0
        eligible = np.where(row >= hard_floor)[0]
        if len(eligible) > top_k:
            order = eligible[np.argsort(-row[eligible])][:top_k]
        else:
            order = eligible
        for j in order:
            rows.append(i); cols.append(int(j)); vals.append(float(score[i, j]))

    return Candidates(np.array(rows, dtype=int), np.array(cols, dtype=int),
                      np.array(vals, dtype=float), n, m, dict(locked))


# ---------------------------------------------------------------------------
# Optimisation parameters
# ---------------------------------------------------------------------------

@dataclass
class AllocationParams:
    good_threshold: float = 0.55      # 达到这个分数视为 good match
    tolerance: float = 0.10           # 允许低于 good_threshold 的比例
    unassigned_penalty: float = 5.0   # 宁可勉强分配也不要空着的程度
    min_load_penalty: float = 2.0
    engine: str = "lp"                # lp | milp
    milp_solver: str = "auto"         # auto | CBC | HiGHS | GUROBI
    time_limit: int = 120


@dataclass
class AllocationResult:
    assignment: pd.DataFrame
    total_score: float
    n_assigned: int
    n_unassigned: int
    n_below_good: int
    tolerance_used: float
    engine: str
    solve_seconds: float
    lagrange_lambda: Optional[float] = None
    loads: Optional[pd.DataFrame] = None
    log: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine 1: transportation LP with Lagrangian bisection on the tolerance
# ---------------------------------------------------------------------------

def _solve_transport_lp(
    cand: Candidates,
    capacity: np.ndarray,
    min_load: np.ndarray,
    params: AllocationParams,
    lam: float,
    group_index: Optional[Sequence[int]] = None,
    group_caps: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float]:
    """One transportation LP with penalty ``lam`` on below-threshold pairs.

    Columns: one per candidate pair, one unassigned slack per student, one
    shortfall slack per supervisor min-load. The matrix is a node-arc incidence
    matrix and therefore totally unimodular, so the simplex vertex is integral.
    """
    n, m = cand.n_students, cand.n_supervisors
    npairs = len(cand)

    bad = (cand.scores < params.good_threshold).astype(float)
    c_pairs = -(cand.scores) + lam * bad
    c_unassigned = np.full(n, params.unassigned_penalty)
    c_shortfall = np.full(m, params.min_load_penalty)
    c = np.concatenate([c_pairs, c_unassigned, c_shortfall])

    # each student assigned exactly once, or explicitly left unassigned
    eq_rows = np.concatenate([cand.rows, np.arange(n)])
    eq_cols = np.concatenate([np.arange(npairs), npairs + np.arange(n)])
    eq_data = np.ones(len(eq_rows))
    A_eq = coo_matrix((eq_data, (eq_rows, eq_cols)), shape=(n, npairs + n + m)).tocsr()
    b_eq = np.ones(n)

    # capacity, then min load with a shortfall slack, then optional group caps.
    # Supervisors partition into groups, so the capacity family is laminar and
    # the whole system is still a flow network of the form
    # student -> supervisor -> group -> sink, which keeps integrality intact.
    ncols_total = npairs + n + m
    ub_rows = [cand.cols, m + cand.cols, m + np.arange(m)]
    ub_cols = [np.arange(npairs), np.arange(npairs), npairs + n + np.arange(m)]
    ub_data = [np.ones(npairs), -np.ones(npairs), -np.ones(m)]
    b_parts = [capacity, -min_load]
    n_rows = 2 * m

    if group_caps is not None and group_index is not None and len(group_caps):
        gi = np.asarray(group_index)[cand.cols]
        ub_rows.append(n_rows + gi)
        ub_cols.append(np.arange(npairs))
        ub_data.append(np.ones(npairs))
        b_parts.append(np.asarray(group_caps, dtype=float))
        n_rows += len(group_caps)

    A_ub = coo_matrix((np.concatenate(ub_data),
                       (np.concatenate(ub_rows), np.concatenate(ub_cols))),
                      shape=(n_rows, ncols_total)).tocsr()
    b_ub = np.concatenate(b_parts)

    bounds = [(0, 1)] * npairs + [(0, 1)] * n + [(0, None)] * m
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP 求解失败: {res.message}")
    x = np.asarray(res.x[:npairs])
    return x, float(res.fun)


def solve_lp(cand: Candidates, capacity: np.ndarray, min_load: np.ndarray,
             params: AllocationParams,
             group_index: Optional[Sequence[int]] = None,
             group_caps: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, Dict]:
    """Bisect the multiplier until the share of poor matches meets tolerance."""
    t0 = time.time()
    log = []
    limit = int(math.floor(params.tolerance * cand.n_students))

    x, _ = _solve_transport_lp(cand, capacity, min_load, params, 0.0, group_index, group_caps)
    bad = int(round(float(((cand.scores < params.good_threshold) * x).sum())))
    log.append(f"λ=0 时低于 good threshold 的分配数 = {bad}，容差上限 = {limit}")
    if bad <= limit:
        return x, {"lambda": 0.0, "log": log, "seconds": time.time() - t0}

    lo, hi = 0.0, 1.0
    for _ in range(20):
        x_hi, _ = _solve_transport_lp(cand, capacity, min_load, params, hi, group_index, group_caps)
        bad_hi = int(round(float(((cand.scores < params.good_threshold) * x_hi).sum())))
        if bad_hi <= limit:
            break
        hi *= 2
        if hi > 64:
            log.append("即便惩罚极大也无法满足容差，说明候选池本身供给不足")
            return x_hi, {"lambda": hi, "log": log, "seconds": time.time() - t0}

    best_x = x_hi
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        x_mid, _ = _solve_transport_lp(cand, capacity, min_load, params, mid, group_index, group_caps)
        bad_mid = int(round(float(((cand.scores < params.good_threshold) * x_mid).sum())))
        if bad_mid <= limit:
            hi, best_x = mid, x_mid
        else:
            lo = mid
        if hi - lo < 1e-4:
            break
    log.append(f"二分收敛，乘子 λ ≈ {hi:.4f}")
    return best_x, {"lambda": hi, "log": log, "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# Engine 2: exact MILP through PuLP
# ---------------------------------------------------------------------------

def solve_milp(cand: Candidates, capacity: np.ndarray, min_load: np.ndarray,
               params: AllocationParams,
               group_index: Optional[Sequence[int]] = None,
               group_caps: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, Dict]:
    import pulp

    t0 = time.time()
    npairs = len(cand)
    n, m = cand.n_students, cand.n_supervisors
    limit = int(math.floor(params.tolerance * n))

    prob = pulp.LpProblem("dissertation_allocation", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{k}", cat="Binary") for k in range(npairs)]
    u = [pulp.LpVariable(f"u_{i}", cat="Binary") for i in range(n)]
    s = [pulp.LpVariable(f"s_{j}", lowBound=0) for j in range(m)]

    prob += (pulp.lpSum(cand.scores[k] * x[k] for k in range(npairs))
             - params.unassigned_penalty * pulp.lpSum(u)
             - params.min_load_penalty * pulp.lpSum(s))

    by_student: Dict[int, List[int]] = {}
    by_supervisor: Dict[int, List[int]] = {}
    for k in range(npairs):
        by_student.setdefault(int(cand.rows[k]), []).append(k)
        by_supervisor.setdefault(int(cand.cols[k]), []).append(k)

    for i in range(n):
        prob += pulp.lpSum(x[k] for k in by_student.get(i, [])) + u[i] == 1
    for j in range(m):
        ks = by_supervisor.get(j, [])
        prob += pulp.lpSum(x[k] for k in ks) <= float(capacity[j])
        if min_load[j] > 0:
            prob += pulp.lpSum(x[k] for k in ks) + s[j] >= float(min_load[j])

    if group_caps is not None and group_index is not None and len(group_caps):
        gi = np.asarray(group_index)
        for g, cap_g in enumerate(group_caps):
            ks = [k for k in range(npairs) if gi[int(cand.cols[k])] == g]
            if ks:
                prob += pulp.lpSum(x[k] for k in ks) <= float(cap_g)

    bad_idx = [k for k in range(npairs) if cand.scores[k] < params.good_threshold]
    if bad_idx:
        prob += pulp.lpSum(x[k] for k in bad_idx) <= limit

    solver = _pick_solver(params)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    xv = np.array([v.value() or 0.0 for v in x])
    return xv, {"log": [f"MILP 状态: {status}"], "seconds": time.time() - t0,
                "lambda": None}


def _pick_solver(params: AllocationParams):
    import pulp
    name = (params.milp_solver or "auto").upper()
    available = pulp.listSolvers(onlyAvailable=True)
    order = [name] if name != "AUTO" else ["GUROBI_CMD", "HiGHS_CMD", "PULP_CBC_CMD"]
    for cand_name in order:
        for avail in available:
            if cand_name.upper().startswith(avail.split("_")[0].upper()) or avail.upper().startswith(cand_name.upper()):
                return pulp.getSolver(avail, msg=False, timeLimit=params.time_limit)
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=params.time_limit)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def allocate(
    students: pd.DataFrame,
    supervisors: pd.DataFrame,
    score: np.ndarray,
    cand: Candidates,
    params: AllocationParams,
    capacity: Optional[np.ndarray] = None,
    group_caps: Optional[Dict[str, float]] = None,
) -> AllocationResult:
    """Run one allocation.

    ``capacity`` overrides the workload column, which is how a later round uses
    the remaining capacity left by earlier rounds rather than the annual figure.
    ``group_caps`` puts a ceiling on how many students a whole subject group
    takes, which stops one popular group absorbing an entire cohort.
    """
    if capacity is None:
        capacity = supervisors["workload"].astype(float).to_numpy()
    else:
        capacity = np.asarray(capacity, dtype=float)
    if "min_load" in supervisors.columns:
        min_load = supervisors["min_load"].fillna(0).astype(float).to_numpy()
    else:
        min_load = np.zeros(len(supervisors))
    min_load = np.minimum(min_load, capacity)

    group_index = group_cap_vector = None
    if group_caps:
        names = list(group_caps.keys())
        lookup = {g: k for k, g in enumerate(names)}
        group_index = [lookup.get(str(g), len(names)) for g in supervisors["group"]]
        if any(idx == len(names) for idx in group_index):
            names.append("__other__")
            group_caps = {**group_caps, "__other__": float(len(students))}
        group_cap_vector = [float(group_caps[g]) for g in names]

    if params.engine == "milp":
        x, meta = solve_milp(cand, capacity, min_load, params, group_index, group_cap_vector)
    else:
        x, meta = solve_lp(cand, capacity, min_load, params, group_index, group_cap_vector)

    chosen = np.where(x > 0.5)[0]
    records = []
    assigned_students = set()
    for k in chosen:
        i, j = int(cand.rows[k]), int(cand.cols[k])
        assigned_students.add(i)
        records.append({
            "student_id": students.iloc[i]["student_id"],
            "project_title": students.iloc[i].get("project_title", ""),
            "supervisor_id": supervisors.iloc[j]["supervisor_id"],
            "supervisor_name": supervisors.iloc[j].get("name", ""),
            "group": supervisors.iloc[j].get("group", ""),
            "score": round(float(cand.scores[k]), 4),
            "below_good": bool(cand.scores[k] < params.good_threshold),
            "locked": i in cand.forced,
            "rank_for_student": int((score[i] > score[i, j]).sum()) + 1,
        })
    for i in range(len(students)):
        if i not in assigned_students:
            records.append({
                "student_id": students.iloc[i]["student_id"],
                "project_title": students.iloc[i].get("project_title", ""),
                "supervisor_id": None,
                "supervisor_name": "未分配",
                "group": "",
                "score": float("nan"),
                "below_good": True,
                "locked": False,
                "rank_for_student": None,
            })

    assignment = pd.DataFrame(records).sort_values(
        ["supervisor_name", "score"], ascending=[True, False]).reset_index(drop=True)

    loads = (assignment.dropna(subset=["supervisor_id"])
             .groupby(["supervisor_id", "supervisor_name", "group"])
             .agg(assigned=("student_id", "count"), mean_score=("score", "mean"))
             .reset_index())
    cap_df = supervisors[["supervisor_id"]].copy()
    cap_df["workload"] = capacity
    loads = loads.merge(cap_df, on="supervisor_id", how="right")
    loads["assigned"] = loads["assigned"].fillna(0).astype(int)
    loads["spare"] = loads["workload"].astype(int) - loads["assigned"]

    n_unassigned = int(assignment["supervisor_id"].isna().sum())
    n_assigned = len(assignment) - n_unassigned
    n_below = int(assignment.loc[assignment["supervisor_id"].notna(), "below_good"].sum())

    return AllocationResult(
        assignment=assignment,
        total_score=float(assignment["score"].sum(skipna=True)),
        n_assigned=n_assigned,
        n_unassigned=n_unassigned,
        n_below_good=n_below,
        tolerance_used=n_below / max(n_assigned, 1),
        engine=params.engine,
        solve_seconds=meta["seconds"],
        lagrange_lambda=meta.get("lambda"),
        loads=loads,
        log=meta.get("log", []),
    )


def tolerance_frontier(
    students: pd.DataFrame,
    supervisors: pd.DataFrame,
    score: np.ndarray,
    cand: Candidates,
    params: AllocationParams,
    grid: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0),
    capacity: Optional[np.ndarray] = None,
    group_caps: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Trace the trade-off between tolerance and achievable total match quality."""
    rows = []
    for t in grid:
        p = AllocationParams(**{**params.__dict__, "tolerance": t})
        r = allocate(students, supervisors, score, cand, p, capacity, group_caps)
        rows.append({
            "tolerance": t,
            "total_score": r.total_score,
            "mean_score": r.total_score / max(r.n_assigned, 1),
            "below_good": r.n_below_good,
            "unassigned": r.n_unassigned,
            "seconds": round(r.solve_seconds, 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation helpers for the teaching operations office
# ---------------------------------------------------------------------------

def validate_inputs(students: pd.DataFrame, supervisors: pd.DataFrame) -> List[str]:
    issues = []
    for col, required in SUPERVISOR_COLUMNS.items():
        if required and col not in supervisors.columns:
            issues.append(f"导师表缺少必填列: {col}")
    for col, required in STUDENT_COLUMNS.items():
        if required and col not in students.columns:
            issues.append(f"学生表缺少必填列: {col}")
    if issues:
        return issues

    if supervisors["supervisor_id"].duplicated().any():
        issues.append("导师表存在重复的 supervisor_id")
    if students["student_id"].duplicated().any():
        issues.append("学生表存在重复的 student_id")

    total_capacity = supervisors["workload"].fillna(0).sum()
    if total_capacity < len(students):
        issues.append(
            f"总容量 {int(total_capacity)} 小于学生人数 {len(students)}，必然出现未分配学生")
    empty_abstract = students["abstract"].fillna("").str.strip().eq("").sum()
    if empty_abstract:
        issues.append(f"{empty_abstract} 位学生的 abstract 为空，文本相似度对这些学生不起作用")
    return issues


# ---------------------------------------------------------------------------
# Diagnostics for handling student queries
# ---------------------------------------------------------------------------

def diagnose_allocation(
    result: AllocationResult,
    score: np.ndarray,
    cand: Candidates,
    students: pd.DataFrame,
    supervisors: pd.DataFrame,
    capacity: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Flag the two patterns students actually complain about.

    A *wasteful* pair is a student who would score strictly higher with a
    supervisor who still has spare capacity, which happens legitimately when a
    min-load floor or a group cap binds, and which the office needs to be able
    to explain rather than discover in a meeting.

    A *blocking* pair is a student who would score higher with a supervisor who
    is full but is carrying someone with a weaker fit, which is the classical
    instability and the thing that looks unfair when two students compare notes.
    """
    if capacity is None:
        capacity = supervisors["workload"].astype(float).to_numpy()

    sup_pos = {str(s): j for j, s in enumerate(supervisors["supervisor_id"])}
    stu_pos = {str(s): i for i, s in enumerate(students["student_id"])}

    assigned_of: Dict[int, int] = {}
    for _, row in result.assignment.iterrows():
        if pd.isna(row["supervisor_id"]):
            continue
        assigned_of[stu_pos[str(row["student_id"])]] = sup_pos[str(row["supervisor_id"])]

    load = np.zeros(len(supervisors))
    worst = np.full(len(supervisors), np.inf)
    for i, j in assigned_of.items():
        load[j] += 1
        worst[j] = min(worst[j], score[i, j])
    spare = capacity - load

    records = []
    for k in range(len(cand)):
        i, j2 = int(cand.rows[k]), int(cand.cols[k])
        own = assigned_of.get(i)
        current = score[i, own] if own is not None else -1.0
        if score[i, j2] <= current + 1e-9:
            continue
        if spare[j2] > 0:
            kind = "wasteful"
        elif worst[j2] < score[i, j2] - 1e-9:
            kind = "blocking"
        else:
            continue
        records.append({
            "student_id": students.iloc[i]["student_id"],
            "current_supervisor": supervisors.iloc[own]["name"] if own is not None else "未分配",
            "current_score": round(float(current), 4) if own is not None else None,
            "better_supervisor": supervisors.iloc[j2]["name"],
            "better_score": round(float(score[i, j2]), 4),
            "type": kind,
        })

    df = pd.DataFrame(records)
    if len(df):
        df = df.sort_values(["type", "better_score"], ascending=[True, False])
        df = df.drop_duplicates("student_id", keep="first")
    return df
