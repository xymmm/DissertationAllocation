"""Benchmark the two engines at realistic and pessimistic sizes."""
import time
import numpy as np
import pandas as pd
import matching_core as mc

rng = np.random.default_rng(7)
AREAS = [f"area{i}" for i in range(15)]
METHODS = [f"method{i}" for i in range(10)]


def synth(n, m):
    sup = pd.DataFrame([{
        "supervisor_id": f"S{j:04d}", "name": f"Supervisor {j}", "group": f"group{j % 6}",
        "research_areas": ";".join(rng.choice(AREAS, size=3, replace=False)),
        "methods": ";".join(rng.choice(METHODS, size=3, replace=False)),
        "keywords": "empirical UK firms panel", "workload": 0, "min_load": 0,
    } for j in range(m)])
    sup["workload"] = int(np.ceil(n / m)) + 2
    stu = pd.DataFrame([{
        "student_id": f"s{i:05d}",
        "project_title": f"A study of {rng.choice(AREAS)} using {rng.choice(METHODS)}",
        "abstract": "This project examines a business question with firm level data.",
        "areas": str(rng.choice(AREAS)), "methods": str(rng.choice(METHODS)),
    } for i in range(n)])
    return stu, sup


print(f"{'students':>9} {'supervisors':>12} {'pairs':>8} {'LP (s)':>9} {'MILP (s)':>9} {'objectives agree':>18}")
for n, m in [(300, 90), (570, 121), (900, 150), (2000, 200)]:
    stu, sup = synth(n, m)
    score, _ = mc.build_score_matrix(stu, sup)
    cand = mc.build_candidates(score, hard_floor=0.15, top_k=25)
    p = mc.AllocationParams(good_threshold=float(np.quantile(score, 0.97)), tolerance=0.08)
    a = mc.allocate(stu, sup, score, cand, mc.AllocationParams(**{**p.__dict__, "engine": "lp"}))
    b = mc.allocate(stu, sup, score, cand, mc.AllocationParams(**{**p.__dict__, "engine": "milp"}))
    agree = "yes" if abs(a.total_score - b.total_score) < 1e-6 else f"no ({a.total_score:.3f} vs {b.total_score:.3f})"
    print(f"{n:>9} {m:>12} {len(cand):>8} {a.solve_seconds:>9.2f} {b.solve_seconds:>9.2f} {agree:>18}")
