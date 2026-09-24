"""
app.py
======
Streamlit front end for undergraduate dissertation allocation, written for a
teaching operations office rather than for an academic: every stage produces a
downloadable artefact, every parameter that affects the outcome is on screen,
and the allocation itself is deterministic so that a rerun with the same inputs
reproduces the same result.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import matching_core as mc
import elm_client as elm
import rounds as rd

st.set_page_config(page_title="毕业论文导师分配", layout="wide", page_icon="🎓")

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

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in [
    ("supervisors", None), ("students", None), ("score", None),
    ("components", None), ("candidates", None), ("result", None),
    ("llm_scores", None), ("extracted", None), ("audit", None),
    ("pseudo", elm.Pseudonymiser()),
    ("ledger_path", "allocation_ledger.json"), ("ledger", None),
    ("blocked_pairs", None), ("round_scope", None), ("diagnostics", None),
]:
    st.session_state.setdefault(key, default)


def read_table(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload, encoding="utf-8-sig", dtype=str).fillna("")
    return pd.read_excel(upload, dtype=str).fillna("")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def get_ledger() -> rd.Ledger:
    path = st.session_state.ledger_path
    ledger = st.session_state.ledger
    if ledger is None or str(ledger.path) != str(path):
        ledger = rd.Ledger(path)
        st.session_state.ledger = ledger
    return ledger


# ---------------------------------------------------------------------------
# Sidebar: ELM configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("ELM 设置")
    st.caption("API key 在 ELM 平台内部申请，base URL 以 ELM 文档给出的为准，"
               "填到 /chat/completions 之前的那一段。")
    base_url = st.text_input("Base URL", value=elm.DEFAULT_BASE_URL,
                             placeholder="https://<elm-endpoint>/v1")
    api_key = st.text_input("API key", value="", type="password")
    model_name = st.text_input("模型", value=elm.DEFAULT_MODEL)
    max_workers = st.slider("并发请求数", 1, 12, 6)
    use_cache = st.checkbox("缓存模型响应", value=True,
                            help="同样的输入不重复调用，改参数重跑时省时间也省额度")

    elm_cfg = elm.ELMConfig(base_url=base_url, api_key=api_key, model=model_name,
                            max_workers=max_workers, use_cache=use_cache)
    elm_ready = bool(base_url and api_key)
    st.success("ELM 已配置") if elm_ready else st.info("未配置 ELM，规则打分与优化仍可正常使用")

    st.divider()
    st.header("轮次台账")
    st.session_state.ledger_path = st.text_input(
        "台账文件", value=st.session_state.ledger_path,
        help="所有已定分配与剩余容量都存在这个文件里，可复制、可归档、可回滚")
    _ledger = get_ledger()
    st.caption(f"已提交 {len(_ledger.rounds)} 轮，已定 {len(_ledger.committed_students())} 位学生")

    st.divider()
    st.header("求解引擎")
    engine = st.radio("引擎", ["lp", "milp"], index=0,
                      format_func=lambda x: {"lp": "运输 LP + 拉格朗日二分（快）",
                                             "milp": "精确 MILP（交叉验证）"}[x])
    milp_solver = st.selectbox("MILP 求解器", ["auto", "PULP_CBC_CMD", "HiGHS_CMD", "GUROBI_CMD"],
                               index=0, disabled=(engine != "milp"))
    st.caption("两条路径在同样的容差下应当给出同样的目标值，"
               "定稿前用 MILP 跑一次作为对照即可。")

st.title("本科毕业论文导师分配")
st.caption("Teaching Operations Office 操作台")

tab_data, tab_score, tab_alloc, tab_audit, tab_rounds, tab_export = st.tabs(
    ["① 数据", "② 匹配打分", "③ 分配优化", "④ AI 复核", "⑤ 轮次台账", "⑥ 导出"])

# ---------------------------------------------------------------------------
# Tab 1: data
# ---------------------------------------------------------------------------

with tab_data:
    col_sup, col_stu = st.columns(2)

    with col_sup:
        st.subheader("导师池")
        st.caption("必填列：supervisor_id, name, group, research_areas, methods, workload。"
                   "workload 就是这一届该老师认领的论文数上限，逐年重填。")
        f_sup = st.file_uploader("上传导师表", type=["csv", "xlsx"], key="sup_up")
        if st.button("载入示例导师表"):
            st.session_state.supervisors = pd.read_csv("sample_supervisors.csv", dtype=str).fillna("")
        if f_sup is not None:
            st.session_state.supervisors = read_table(f_sup)

    with col_stu:
        st.subheader("学生提交")
        st.caption("必填列：student_id, project_title, abstract。"
                   "可选 project_name, areas, methods, references, "
                   "preferred_supervisor_id, locked_supervisor_id。")
        f_stu = st.file_uploader("上传学生表", type=["csv", "xlsx"], key="stu_up")
        if st.button("载入示例学生表"):
            st.session_state.students = pd.read_csv("sample_students.csv", dtype=str).fillna("")
        if f_stu is not None:
            st.session_state.students = read_table(f_stu)

    st.divider()
    st.subheader("利益冲突禁配表（可选）")
    st.caption("两列 student_id 与 supervisor_id，外加 reason 备查。"
               "亲属关系、申诉当事人、纪律委员会成员这类情况放在这里，"
               "它们是硬性排除而不是扣分，不应该混进匹配分里。")
    f_block = st.file_uploader("上传禁配表", type=["csv", "xlsx"], key="blk_up")
    if st.button("载入示例禁配表"):
        st.session_state.blocked_pairs = pd.read_csv("sample_blocked_pairs.csv", dtype=str).fillna("")
    if f_block is not None:
        st.session_state.blocked_pairs = read_table(f_block)
    if st.session_state.blocked_pairs is not None:
        st.dataframe(st.session_state.blocked_pairs, width="stretch")

    sup, stu = st.session_state.supervisors, st.session_state.students
    if sup is not None and stu is not None:
        sup = sup.copy()
        sup["workload"] = pd.to_numeric(sup["workload"], errors="coerce").fillna(0).astype(int)
        if "min_load" in sup.columns:
            sup["min_load"] = pd.to_numeric(sup["min_load"], errors="coerce").fillna(0).astype(int)
        if "available" in sup.columns:
            sup["available"] = pd.to_numeric(sup["available"], errors="coerce").fillna(1).astype(int)
        st.session_state.supervisors = sup

        issues = mc.validate_inputs(stu, sup)
        if issues:
            for msg in issues:
                st.warning(msg)
        else:
            st.success("数据检查通过")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("学生人数", len(stu))
        c2.metric("导师人数", len(sup))
        c3.metric("总容量", int(sup["workload"].sum()))
        c4.metric("容量盈余", int(sup["workload"].sum()) - len(stu))

        st.markdown("**各组容量对比**")
        by_group = sup.groupby("group").agg(导师数=("supervisor_id", "count"),
                                            总容量=("workload", "sum")).reset_index()
        st.dataframe(by_group, width="stretch")

        with st.expander("查看导师表"):
            st.dataframe(sup, width="stretch")
        with st.expander("查看学生表"):
            st.dataframe(stu, width="stretch")

# ---------------------------------------------------------------------------
# Tab 2: scoring
# ---------------------------------------------------------------------------

with tab_score:
    if st.session_state.supervisors is None or st.session_state.students is None:
        st.info("先在①载入两张表")
    else:
        sup = st.session_state.supervisors
        stu = st.session_state.students

        st.subheader("A. 用 ELM 标准化学生提交（可选但建议）")
        st.caption("学生自己填的 areas 和 methods 往往词汇混乱，同一个方法能写出十种说法。"
                   "这一步把自由文本映射到受控词表，绝大部分 mismatch 是在这里消掉的，"
                   "而不是在优化模型里。学生的姓名与学号不会进入请求，只送匿名化之后的题目与摘要。")
        area_vocab = st.text_area("研究领域词表", ";".join(DEFAULT_AREA_VOCAB), height=80)
        method_vocab = st.text_area("研究方法词表", ";".join(DEFAULT_METHOD_VOCAB), height=80)

        if st.button("运行 ELM 抽取", disabled=not elm_ready):
            client = elm.ELMClient(elm_cfg)
            pseudo = st.session_state.pseudo
            submissions = []
            for _, row in stu.iterrows():
                submissions.append({
                    "anon_id": pseudo.token(row["student_id"]),
                    "project_title": row.get("project_title", ""),
                    "project_name": row.get("project_name", ""),
                    "abstract": row.get("abstract", ""),
                    "references": row.get("references", ""),
                })
            bar = st.progress(0.0, text="正在抽取")
            def _cb(done, total):
                bar.progress(done / total, text=f"正在抽取 {done}/{total}")
            out = client.map_parallel(
                lambda s: elm.extract_profile(client, s,
                                              mc.split_tags(area_vocab),
                                              mc.split_tags(method_vocab)),
                submissions, progress=_cb)
            bar.empty()

            extracted = pd.DataFrame([{
                "student_id": pseudo.real(o.get("anon_id", "")) or "",
                "llm_areas": ";".join(o.get("areas", []) or []),
                "llm_methods": ";".join(o.get("methods", []) or []),
                "clarity": o.get("clarity"),
                "flags": ";".join(o.get("feasibility_flags", []) or []),
                "summary": o.get("one_line_summary", ""),
            } for o in out if isinstance(o, dict)])
            st.session_state.extracted = extracted
            st.success(f"完成 {len(extracted)} 条")

        if st.session_state.extracted is not None:
            ext = st.session_state.extracted
            st.dataframe(ext, width="stretch", height=240)
            vague = ext[pd.to_numeric(ext["clarity"], errors="coerce") <= 2]
            if len(vague):
                st.warning(f"{len(vague)} 位学生的选题清晰度不足，建议在分配之前退回补充，"
                           "这些人无论算法怎么跑都会产生 mismatch")
            if st.button("把抽取结果并入学生表"):
                merged = stu.merge(ext[["student_id", "llm_areas", "llm_methods"]],
                                   on="student_id", how="left")
                merged["areas"] = np.where(merged["llm_areas"].fillna("") != "",
                                           merged["llm_areas"], merged.get("areas", ""))
                merged["methods"] = np.where(merged["llm_methods"].fillna("") != "",
                                             merged["llm_methods"], merged.get("methods", ""))
                st.session_state.students = merged.drop(columns=["llm_areas", "llm_methods"])
                st.success("已并入，下面重新计算分数")

        st.divider()
        st.subheader("B. 规则打分")
        c1, c2, c3, c4, c5 = st.columns(5)
        w_area = c1.slider("研究领域", 0.0, 1.0, 0.40, 0.05)
        w_method = c2.slider("研究方法", 0.0, 1.0, 0.30, 0.05)
        w_text = c3.slider("文本相似", 0.0, 1.0, 0.20, 0.05)
        w_pref = c4.slider("学生意向", 0.0, 1.0, 0.05, 0.05)
        w_group = c5.slider("同组加分", 0.0, 1.0, 0.05, 0.05)

        syn_raw = st.text_area(
            "同义词映射，一行一条，格式 原词=规范词",
            "ml=machine learning\nai=machine learning\nstats=regression\n运筹=optimisation",
            height=100)
        synonyms = {}
        for line in syn_raw.splitlines():
            if "=" in line:
                a, b = line.split("=", 1)
                synonyms[a.strip().lower()] = b.strip().lower()

        if st.button("计算分数矩阵", type="primary"):
            weights = mc.ScoreWeights(w_area, w_method, w_text, w_pref, w_group)
            score, comps = mc.build_score_matrix(
                st.session_state.students, st.session_state.supervisors,
                weights=weights, synonyms=synonyms,
                llm_scores=st.session_state.llm_scores,
                llm_blend=0.3 if st.session_state.llm_scores is not None else 0.0)
            st.session_state.score = score
            st.session_state.components = comps
            st.success(f"完成，矩阵规模 {score.shape[0]} × {score.shape[1]}")

        if st.session_state.score is not None:
            score = st.session_state.score
            best = score.max(axis=1)
            c1, c2, c3 = st.columns(3)
            c1.metric("最佳匹配分中位数", f"{np.median(best):.3f}")
            c2.metric("最佳匹配分最低值", f"{best.min():.3f}")
            c3.metric("最佳分低于 0.3 的学生", int((best < 0.3).sum()))
            st.caption("最佳匹配分很低的学生是真正的结构性问题，"
                       "说明导师池里没有人做这个方向，这是招聘或课题引导的问题，不是分配算法的问题。")
            hist = pd.DataFrame({"每位学生的最佳匹配分": best})
            st.bar_chart(np.histogram(best, bins=20, range=(0, 1))[0])

# ---------------------------------------------------------------------------
# Tab 3: allocation
# ---------------------------------------------------------------------------

with tab_alloc:
    if st.session_state.score is None:
        st.info("先在②计算分数矩阵")
    else:
        sup = st.session_state.supervisors
        stu_all = st.session_state.students
        score_all = st.session_state.score
        ledger = get_ledger()

        st.subheader("本轮范围")
        st.caption("按 programme 分批时，后跑的批次面对的是前面剩下的容量，"
                   "所以下面的容量来源和预留策略决定了这一批会不会把好导师吃干净。")

        prog_col = "programme" if "programme" in stu_all.columns else None
        committed = ledger.committed_students()
        c1, c2 = st.columns([2, 1])
        if prog_col:
            all_progs = sorted(stu_all[prog_col].astype(str).unique())
            chosen_progs = c1.multiselect("本轮要安排的 programme", all_progs, default=all_progs)
        else:
            chosen_progs = []
            c1.info("学生表没有 programme 列，本轮按全体学生处理")
        skip_committed = c2.checkbox("排除台账中已定的学生", value=True)

        mask = np.ones(len(stu_all), dtype=bool)
        if prog_col and chosen_progs:
            mask &= stu_all[prog_col].astype(str).isin(chosen_progs).to_numpy()
        if skip_committed and committed:
            mask &= ~stu_all["student_id"].astype(str).isin(committed).to_numpy()
        idx = np.where(mask)[0]
        stu = stu_all.iloc[idx].reset_index(drop=True)
        score = score_all[idx, :]

        upcoming = 0
        if prog_col and chosen_progs:
            rest = ~stu_all[prog_col].astype(str).isin(chosen_progs)
            if skip_committed and committed:
                rest &= ~stu_all["student_id"].astype(str).isin(committed)
            upcoming = int(rest.sum())

        st.subheader("容量")
        c1, c2, c3 = st.columns(3)
        cap_source = c1.radio("容量来源", ["台账剩余", "年度 workload"], index=0,
                              help="分批运行时用台账剩余，重跑整届时用年度 workload")
        reservation = c2.radio("为后续批次预留", ["proportional", "use_all"], index=0,
                               format_func=lambda x: {"proportional": "按人数比例预留",
                                                      "use_all": "本轮用尽"}[x],
                               help="本轮之后还有学生未安排时，按比例预留可以避免最后一批无人可配")
        floor_reserve = c3.number_input("每位导师至少保留", 0, 10, 0,
                                        help="预留时给每位导师保底留出的名额")

        annual = sup["workload"].astype(float).to_numpy()
        remaining = ledger.remaining_capacity(sup) if cap_source == "台账剩余" else annual
        capacity = rd.reserve_capacity(remaining, len(stu), upcoming,
                                       mode=reservation,
                                       floor_per_supervisor=int(floor_reserve))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("本轮学生", len(stu))
        c2.metric("后续待安排", upcoming)
        c3.metric("可用容量", int(remaining.sum()))
        c4.metric("本轮开放容量", int(capacity.sum()))
        if capacity.sum() < len(stu):
            st.warning("本轮开放容量小于本轮学生数，必然出现未分配")

        with st.expander("组级上限，用于避免某一个组被热门选题淹没"):
            use_group_caps = st.checkbox("启用组级上限", value=False)
            group_caps = {}
            if use_group_caps:
                groups = sorted(sup["group"].astype(str).unique())
                cols = st.columns(min(3, len(groups)))
                for k, g in enumerate(groups):
                    default = int(math.ceil(len(stu) / len(groups) * 1.4))
                    group_caps[g] = cols[k % len(cols)].number_input(
                        g, 0, len(stu), default, key=f"gc_{g}")
                st.caption("组级约束与导师容量是嵌套的层状结构，仍然构成网络流，"
                           "所以加上它不会破坏整数性，求解速度也不受影响。")

        st.subheader("容差与门槛")
        c1, c2, c3 = st.columns(3)
        hard_floor = c1.slider("硬下限 hard floor", 0.0, 1.0, 0.15, 0.01,
                               help="低于此分的配对根本不进入模型，宁可让学生留待人工处理")
        good_threshold = c2.slider("good match 门槛", 0.0, 1.0, 0.50, 0.01,
                                   help="达到此分视为合格匹配")
        tolerance = c3.slider("mismatch tolerance", 0.0, 1.0, 0.10, 0.01,
                              help="允许低于合格门槛的分配占比上限")

        c4, c5, c6 = st.columns(3)
        top_k = c4.slider("每位学生保留候选导师数", 5, 60, 25)
        restrict_group = c5.checkbox("限定在学生指定的组内分配", value=False)
        unassigned_penalty = c6.slider("未分配惩罚", 1.0, 20.0, 5.0, 0.5,
                                       help="调高则宁可勉强配也不留空，调低则宁可留空交人工")

        sup_index = {str(s_): j for j, s_ in enumerate(sup["supervisor_id"])}
        locked = {}
        if "locked_supervisor_id" in stu.columns:
            for i, v in enumerate(stu["locked_supervisor_id"]):
                j = sup_index.get(str(v).strip())
                if j is not None:
                    locked[i] = j
            if locked:
                st.info(f"检测到 {len(locked)} 条人工锁定，这些配对被强制保留")

        unavailable = []
        if "available" in sup.columns:
            unavailable = [j for j, a in enumerate(sup["available"]) if int(a) == 0]
            if unavailable:
                st.info(f"{len(unavailable)} 位导师本届不可用，已从候选中剔除")

        blocked_pairs = set()
        bp = st.session_state.blocked_pairs
        if bp is not None and len(bp):
            stu_pos = {str(s_): i for i, s_ in enumerate(stu["student_id"])}
            for _, row in bp.iterrows():
                i = stu_pos.get(str(row["student_id"]).strip())
                j = sup_index.get(str(row["supervisor_id"]).strip())
                if i is not None and j is not None:
                    blocked_pairs.add((i, j))
            st.info(f"{len(blocked_pairs)} 条利益冲突禁配在本轮生效")

        eligibility = None
        if "programmes" in sup.columns and prog_col:
            elig = np.ones((len(stu), len(sup)), dtype=bool)
            sup_progs = [mc.split_tags(v) for v in sup["programmes"]]
            stu_progs = [str(v).strip().lower() for v in stu[prog_col]]
            restricted = 0
            for j, allowed_progs in enumerate(sup_progs):
                if not allowed_progs:
                    continue
                restricted += 1
                for i, pg in enumerate(stu_progs):
                    elig[i, j] = pg in allowed_progs
            if restricted:
                eligibility = elig
                st.info(f"{restricted} 位导师限定了可带的 programme，已按此过滤候选")

        if st.button("运行分配", type="primary"):
            cand = mc.build_candidates(
                score, hard_floor=hard_floor, top_k=top_k,
                restrict_to_group=restrict_group,
                student_groups=stu.get("preferred_group"),
                supervisor_groups=sup["group"],
                locked=locked, unavailable=unavailable,
                blocked_pairs=blocked_pairs, eligibility=eligibility)
            params = mc.AllocationParams(
                good_threshold=good_threshold, tolerance=tolerance,
                unassigned_penalty=unassigned_penalty,
                engine=engine, milp_solver=milp_solver)
            with st.spinner("求解中"):
                result = mc.allocate(stu, sup, score, cand, params, capacity,
                                     group_caps if use_group_caps else None)
            st.session_state.candidates = cand
            st.session_state.result = result
            st.session_state.audit = None
            st.session_state.diagnostics = mc.diagnose_allocation(
                result, score, cand, stu, sup, capacity)
            st.session_state.round_scope = {
                "programmes": chosen_progs,
                "students": len(stu),
                "params": {"hard_floor": hard_floor, "good_threshold": good_threshold,
                           "tolerance": tolerance, "top_k": top_k,
                           "capacity_source": cap_source, "reservation": reservation,
                           "group_caps": group_caps if use_group_caps else {},
                           "engine": engine},
                "student_ids": stu["student_id"].astype(str).tolist(),
            }

        res = st.session_state.result
        if res is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("已分配", res.n_assigned)
            c2.metric("未分配", res.n_unassigned)
            c3.metric("低于合格门槛", res.n_below_good)
            c4.metric("平均匹配分", f"{res.total_score / max(res.n_assigned, 1):.3f}")
            c5.metric("求解耗时", f"{res.solve_seconds:.2f}s")
            for line in res.log:
                st.caption(line)

            st.markdown("**分配结果**")
            st.dataframe(res.assignment, width="stretch", height=320)

            st.markdown("**导师负荷，workload 一列是本轮开放的容量而非年度总量**")
            st.dataframe(res.loads.sort_values("spare"), width="stretch", height=260)

            diag = st.session_state.diagnostics
            if diag is not None and len(diag):
                st.markdown("**申诉预案**")
                st.caption("wasteful 是本可以配给更合适且仍有余量的导师，通常由组级上限或下限约束造成；"
                           "blocking 是更合适的导师已满但带着匹配更弱的学生，这是两个学生一对话就会发现的不公平感来源。"
                           "两类都属于模型的正常产物，列出来是为了 office 在被问到时答得出所以然。")
                st.dataframe(diag, width="stretch", height=220)

            st.divider()
            st.subheader("容差前沿")
            if st.button("计算前沿曲线"):
                params = mc.AllocationParams(
                    good_threshold=good_threshold, tolerance=tolerance,
                    unassigned_penalty=unassigned_penalty, engine="lp")
                frontier = mc.tolerance_frontier(
                    stu, sup, score, st.session_state.candidates, params, capacity=capacity)
                st.dataframe(frontier, width="stretch")
                st.line_chart(frontier.set_index("tolerance")[["mean_score"]])

# ---------------------------------------------------------------------------
# Tab 4: AI double check
# ---------------------------------------------------------------------------

with tab_audit:
    if st.session_state.result is None:
        st.info("先在③跑出分配结果")
    else:
        res = st.session_state.result
        sup = st.session_state.supervisors
        stu = st.session_state.students

        st.subheader("AI 复核")
        st.caption("模型在这里只做审核，不做决定：它读已经定下来的配对，"
                   "对每一条给出 ok / query / reject 与理由，由 office 决定是否人工干预。"
                   "reject 的条目可以写进 locked 之外的黑名单，调整之后回到③重跑。")

        scope = st.radio("复核范围",
                         ["只看低于合格门槛的配对", "分数最低的 N 条", "全部配对"],
                         index=0)
        n_lowest = st.number_input("N", 10, 500, 50, step=10,
                                   disabled=(scope != "分数最低的 N 条"))

        assigned = res.assignment.dropna(subset=["supervisor_id"])
        if scope == "只看低于合格门槛的配对":
            target = assigned[assigned["below_good"]]
        elif scope == "分数最低的 N 条":
            target = assigned.nsmallest(int(n_lowest), "score")
        else:
            target = assigned
        st.write(f"待复核 {len(target)} 条")

        if st.button("运行复核", type="primary", disabled=not elm_ready or len(target) == 0):
            client = elm.ELMClient(elm_cfg)
            pseudo = st.session_state.pseudo
            stu_by_id = stu.set_index("student_id")
            sup_by_id = sup.set_index("supervisor_id")

            pairs = []
            for _, row in target.iterrows():
                s = stu_by_id.loc[row["student_id"]]
                t = sup_by_id.loc[row["supervisor_id"]]
                pairs.append({
                    "anon_id": pseudo.token(row["student_id"]),
                    "title": elm.strip_identifiers(str(s.get("project_title", ""))),
                    "abstract": elm.strip_identifiers(str(s.get("abstract", "")))[:1200],
                    "supervisor_areas": str(t.get("research_areas", "")),
                    "supervisor_methods": str(t.get("methods", "")),
                    "supervisor_keywords": str(t.get("keywords", ""))[:300],
                    "rule_score": float(row["score"]),
                })
            bar = st.progress(0.0, text="复核中")
            def _cb(done, total):
                bar.progress(done / total, text=f"复核中 {done}/{total}")
            reviews = elm.audit_allocation(client, pairs, batch_size=8, progress=_cb)
            bar.empty()

            audit_df = pd.DataFrame(reviews)
            if len(audit_df):
                audit_df["student_id"] = audit_df["anon_id"].map(pseudo.real)
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
            st.caption("模型的 reject 是提示而不是裁决，"
                       "它看不到导师的完整履历，误判的成本落在 office 身上，"
                       "所以这一列请当作人工复查的排序依据来用。")

# ---------------------------------------------------------------------------
# Tab 5: round ledger
# ---------------------------------------------------------------------------

with tab_rounds:
    ledger = get_ledger()
    sup = st.session_state.supervisors

    st.subheader("提交本轮")
    st.caption("提交之后这一轮的配对写入台账，导师剩余容量相应扣减，下一批学生面对的就是扣减后的池子。"
               "提交前的结果只是草稿，可以反复重跑。")
    res = st.session_state.result
    scope = st.session_state.round_scope
    if res is None or scope is None:
        st.info("③还没有可提交的结果")
    else:
        label = st.text_input("本轮名称",
                              value=("、".join(scope["programmes"]) if scope["programmes"] else "全体")
                              + f"  {len(res.assignment) - res.n_unassigned} 人")
        c1, c2 = st.columns([1, 3])
        if c1.button("提交入台账", type="primary"):
            ledger.commit(res, label, scope["programmes"], scope["params"])
            st.session_state.result = None
            st.session_state.round_scope = None
            st.success("已提交，导师剩余容量已更新")
            st.rerun()
        c2.caption(f"待提交 {res.n_assigned} 条配对，未分配 {res.n_unassigned} 人不会写入台账，"
                   "他们留在池子里等下一轮或人工处理")

    st.divider()
    st.subheader("台账状态")
    if sup is None:
        st.info("先在①载入导师表")
    else:
        cap_table = ledger.capacity_table(sup)
        c1, c2, c3 = st.columns(3)
        c1.metric("年度总容量", int(cap_table["workload"].sum()))
        c2.metric("已占用", int(cap_table["already_assigned"].sum()))
        c3.metric("剩余", int(cap_table["remaining"].sum()))
        st.dataframe(cap_table.sort_values("remaining"), width="stretch", height=260)
        st.download_button("导出剩余容量表 CSV", to_csv_bytes(cap_table),
                           "remaining_capacity.csv", "text/csv")

        summary = ledger.summary()
        if len(summary):
            st.markdown("**已提交轮次**")
            st.dataframe(summary, width="stretch")

    st.divider()
    st.subheader("改动与回滚")
    st.caption("这里处理的是学期中途一定会发生的事：导师病假或离职、学生换题、两位学生申请对调、"
               "某一轮跑错参数需要整体撤销。每一种都不需要推翻其他人的安排。")

    if len(ledger.rounds):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**撤销整轮**")
            options = {f"{r.label}｜{r.timestamp}｜{len(r.rows)} 人": r.round_id
                       for r in ledger.rounds}
            pick = st.selectbox("选择轮次", list(options.keys()))
            if st.button("撤销这一轮"):
                ledger.rollback(options[pick])
                st.success("已撤销，容量释放回池子")
                st.rerun()
        with c2:
            st.markdown("**导师退出**")
            st.caption("释放该导师名下全部学生，其余配对不动，随后回到③只重跑这些学生")
            if sup is not None:
                sup_map = {f"{r['name']}（{r['supervisor_id']}）": r["supervisor_id"]
                           for _, r in sup.iterrows()}
                pick_sup = st.selectbox("选择导师", list(sup_map.keys()))
                if st.button("释放该导师的学生"):
                    affected = ledger.release_supervisor(sup_map[pick_sup])
                    st.warning(f"释放 {len(affected)} 位学生：{', '.join(affected[:20])}"
                               + ("…" if len(affected) > 20 else ""))

        st.markdown("**释放个别学生**")
        st.caption("学生换题、休学、申请更换导师时用，多个 ID 用逗号或换行分隔")
        ids_raw = st.text_area("student_id", height=70, key="release_ids")
        if st.button("释放这些学生"):
            ids = [x.strip() for x in re.split(r"[,，\s]+", ids_raw) if x.strip()]
            removed = ledger.release_students(ids)
            st.success(f"释放 {removed} 条配对，这些学生回到未分配池")
    else:
        st.info("台账为空")

    st.divider()
    st.subheader("分批的代价")
    st.caption("把按 programme 顺序分批的结果与一次性联合求解的结果放在一起比。"
               "联合求解是所有 programme 同一天交材料时能达到的上界，两者之差就是日程安排的价格。"
               "先跑的批次通常占便宜，最后一批吃亏，预留策略是用来压缩这个差距的。")
    if (st.session_state.score is None or sup is None
            or "programme" not in (st.session_state.students.columns if st.session_state.students is not None else [])):
        st.info("需要学生表带 programme 列，并且已在②算好分数矩阵")
    else:
        stu_all = st.session_state.students
        all_progs = sorted(stu_all["programme"].astype(str).unique())
        order = st.multiselect("批次顺序，从先到后", all_progs, default=all_progs)
        mode = st.radio("预留策略", ["proportional", "use_all"], index=0, horizontal=True,
                        format_func=lambda x: {"proportional": "按人数比例预留",
                                               "use_all": "本轮用尽"}[x],
                        key="seqmode")
        if st.button("测算分批代价") and order:
            with st.spinner("测算中"):
                cost = rd.sequencing_cost(
                    stu_all, sup, st.session_state.score,
                    mc.AllocationParams(good_threshold=0.5, tolerance=0.1, engine="lp"),
                    "programme", order, reservation=mode)
            st.dataframe(cost, width="stretch")
            st.caption("gap 为正表示该 programme 在分批安排下吃亏，"
                       "把它拿给排在最后的项目主任看比任何口头保证都有用。")


# ---------------------------------------------------------------------------
# Tab 5: export
# ---------------------------------------------------------------------------

with tab_export:
    res = st.session_state.result
    if res is None:
        st.info("还没有可导出的结果")
    else:
        st.subheader("导出")
        st.download_button("分配结果 CSV", to_csv_bytes(res.assignment),
                           "allocation.csv", "text/csv")
        st.download_button("导师负荷 CSV", to_csv_bytes(res.loads),
                           "supervisor_loads.csv", "text/csv")
        if st.session_state.audit is not None and len(st.session_state.audit):
            st.download_button("AI 复核报告 CSV", to_csv_bytes(st.session_state.audit),
                               "ai_audit.csv", "text/csv")
        if st.session_state.extracted is not None:
            st.download_button("ELM 抽取结果 CSV", to_csv_bytes(st.session_state.extracted),
                               "extracted_profiles.csv", "text/csv")
        if st.session_state.diagnostics is not None and len(st.session_state.diagnostics):
            st.download_button("申诉预案 CSV", to_csv_bytes(st.session_state.diagnostics),
                               "diagnostics.csv", "text/csv")

        st.divider()
        st.subheader("全届汇总，跨所有已提交轮次")
        _ledger = get_ledger()
        all_rows = _ledger.committed_rows()
        if len(all_rows):
            st.dataframe(all_rows, width="stretch", height=240)
            st.download_button("全届分配结果 CSV", to_csv_bytes(all_rows),
                               "allocation_all_rounds.csv", "text/csv")
            if st.session_state.supervisors is not None:
                cap = _ledger.capacity_table(st.session_state.supervisors)
                nxt = st.session_state.supervisors.copy()
                nxt["workload"] = cap["remaining"].values
                st.download_button(
                    "更新 workload 后的导师表 CSV，可直接作为下一轮输入",
                    to_csv_bytes(nxt), "supervisors_remaining.csv", "text/csv")
        else:
            st.info("台账为空，还没有已提交的轮次")

        st.divider()
        st.subheader("本次运行的参数存档")
        st.caption("留档的意义在于，学生若来质询分配结果，office 可以用同一份参数重跑得到同一份结果。")
        provenance = {
            "engine": res.engine,
            "lagrange_lambda": res.lagrange_lambda,
            "assigned": res.n_assigned,
            "unassigned": res.n_unassigned,
            "below_good": res.n_below_good,
            "realised_mismatch_rate": round(res.tolerance_used, 4),
            "total_score": round(res.total_score, 4),
            "solve_seconds": round(res.solve_seconds, 3),
        }
        st.json(provenance)
        st.download_button("参数存档 JSON",
                           json.dumps(provenance, ensure_ascii=False, indent=2).encode("utf-8"),
                           "run_provenance.json", "application/json")
