"""
report.py
=========
Turns a finished allocation into something a teaching operations office can
actually use: an HTML page that opens by double-clicking it, an Excel workbook
with one sheet per question they will ask, and the plain CSVs underneath for
anyone who wants the raw rows.

The page is deliberately not a dashboard. The office does not need a chart of
the score distribution; it needs to know which students still have nobody,
which pairings somebody should look at before the list goes out, and where a
given student ended up. So the page opens with what has to be done, puts the
numbers underneath that, and gives the full list a search box, because "find
me s2400312" is the single commonest thing anybody will do with it.

Colour carries status and nothing else. Amber means a human should look, slate
means settled, and a red left rule means nobody was found. Nothing on the page
is coloured for decoration.

No network access is needed to open the report, no fonts are fetched, and no
JavaScript library is loaded. It is one file that works from a memory stick in
ten years' time.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__version__ = "2.0.0"

# --- design tokens ---------------------------------------------------------
# Ink is a deep slate rather than a tinted black; the two status colours are the
# only saturated things on the page and both of them mean something.
INK = "#16202B"
INK_SOFT = "#5A6675"
RULE = "#D8DEE6"
PAPER = "#FFFFFF"
SHADE = "#F4F6F8"
ATTENTION = "#B26B00"      # a person should look at this
SETTLED = "#0F6B63"        # nothing to do
MISSING = "#A3322B"        # nobody was found


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _esc(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return html.escape(str(value))


def _stat(value, label: str, tone: str = "") -> str:
    cls = f" class=\"{tone}\"" if tone else ""
    return (f'<div class="stat"><span class="stat-value"{cls}>{_esc(value)}</span>'
            f'<span class="stat-label">{_esc(label)}</span></div>')


def _rows(df: pd.DataFrame, columns: List[str], row_class=None) -> str:
    out = []
    for _, row in df.iterrows():
        cls = row_class(row) if row_class else ""
        cells = "".join(f"<td>{_esc(row.get(c, ''))}</td>" for c in columns)
        out.append(f'<tr class="{cls}">{cells}</tr>')
    return "\n".join(out)


CSS = f"""
:root {{
  --ink: {INK}; --ink-soft: {INK_SOFT}; --rule: {RULE};
  --paper: {PAPER}; --shade: {SHADE};
  --attention: {ATTENTION}; --settled: {SETTLED}; --missing: {MISSING};
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  font-size: 17px; line-height: 1.55;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 48px 32px 96px; }}
p, li {{ max-width: 68ch; }}

header.masthead {{ border-bottom: 2px solid var(--ink); padding-bottom: 18px; }}
.masthead h1 {{ font-size: 34px; line-height: 1.15; margin: 0 0 6px; font-weight: 600; }}
.masthead .meta {{ color: var(--ink-soft); font-size: 15px; margin: 0; }}
.masthead .meta span {{ margin-right: 22px; }}

h2 {{ font-size: 23px; margin: 52px 0 6px; font-weight: 600; }}
h2 + p.lede {{ margin: 0 0 18px; color: var(--ink-soft); }}

.todo {{ border-left: 3px solid var(--attention); padding: 2px 0 2px 20px; margin: 28px 0 8px; }}
.todo.none {{ border-left-color: var(--settled); }}
.todo h2 {{ margin-top: 0; }}
.todo ul {{ margin: 10px 0 0; padding-left: 20px; }}
.todo li {{ margin-bottom: 8px; }}
.todo a {{ color: var(--ink); text-decoration-color: var(--rule); }}

.stats {{ margin: 22px 0 4px; padding: 18px 0;
         border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }}
.stat {{ display: inline-block; vertical-align: top; margin-right: 46px; }}
.stat-value, .stat-label {{ display: block; }}
.stat-value {{ font-size: 30px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.stat-value.attention {{ color: var(--attention); }}
.stat-value.missing {{ color: var(--missing); }}
.stat-label {{ font-size: 14px; color: var(--ink-soft); }}

table {{ width: 100%; border-collapse: collapse; margin-top: 14px;
        font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 14.5px; }}
th {{ text-align: left; font-weight: 600; border-bottom: 1.5px solid var(--ink);
     padding: 8px 10px; white-space: nowrap; }}
td {{ padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
tbody tr:hover {{ background: var(--shade); }}
td:nth-child(n+3) {{ font-variant-numeric: tabular-nums; }}
tr.flag td:first-child {{ box-shadow: inset 3px 0 0 var(--attention); }}
tr.gap td:first-child {{ box-shadow: inset 3px 0 0 var(--missing); }}
.scroll {{ overflow-x: auto; }}

.chip {{ display: inline-block; padding: 1px 8px; border-radius: 3px; font-size: 12.5px;
        border: 1px solid var(--rule); color: var(--ink-soft); white-space: nowrap; }}
.chip.flag {{ color: var(--attention); border-color: var(--attention); }}
.chip.gap {{ color: var(--missing); border-color: var(--missing); }}

.search {{ margin: 18px 0 0; display: flex; gap: 12px; align-items: center; }}
.search input {{ font: inherit; font-size: 15px; padding: 8px 12px; width: 320px;
                border: 1px solid var(--rule); border-radius: 3px; background: var(--paper);
                color: var(--ink); }}
.search input:focus {{ outline: 2px solid var(--ink); outline-offset: 1px; }}
.search .count {{ color: var(--ink-soft); font-size: 14px; }}

footer {{ margin-top: 64px; padding-top: 16px; border-top: 1px solid var(--rule);
         color: var(--ink-soft); font-size: 14px; }}

@media (max-width: 760px) {{
  .wrap {{ padding: 28px 18px 64px; }}
  .masthead h1 {{ font-size: 27px; }}
  .stats {{ gap: 24px; }}
}}
@media print {{
  .search {{ display: none; }}
  body {{ font-size: 11pt; }}
  .wrap {{ max-width: none; padding: 0; }}
  tr {{ break-inside: avoid; }}
  h2 {{ break-after: avoid; }}
}}
"""

SEARCH_JS = """
(function () {
  var box = document.getElementById('find');
  var count = document.getElementById('found');
  var rows = Array.prototype.slice.call(
    document.querySelectorAll('#allocation tbody tr'));
  if (!box) return;
  function apply() {
    var q = box.value.trim().toLowerCase();
    var shown = 0;
    rows.forEach(function (tr) {
      var hit = !q || tr.textContent.toLowerCase().indexOf(q) !== -1;
      tr.style.display = hit ? '' : 'none';
      if (hit) shown++;
    });
    count.textContent = q ? shown + ' of ' + rows.length + ' shown' : rows.length + ' pairings';
  }
  box.addEventListener('input', apply);
  apply();
})();
"""


def build_html(result, students: pd.DataFrame, register: pd.DataFrame,
               diagnostics: pd.DataFrame, requirements: pd.DataFrame,
               scope: dict, session: str, register_version: int,
               params: dict) -> str:
    assignment = result.assignment.copy()
    assigned = assignment.dropna(subset=["supervisor_id"]).copy()
    unassigned = assignment[assignment["supervisor_id"].isna()].copy()
    titles = students.set_index("student_id")["project_title"].astype(str)

    assigned["score"] = assigned["score"].astype(float).round(3)
    weakest = assigned[assigned["below_good"]].sort_values("score")

    loads = result.loads.copy()
    full = loads[loads["spare"] <= 0]
    unused = loads[(loads["assigned"] == 0) & (loads["workload"] > 0)]

    ethics = int(requirements["ethics_review_likely"].sum()) if len(requirements) else 0
    access = int(requirements["needs_data_access_check"].sum()) if len(requirements) else 0

    todo: List[str] = []
    if len(unassigned):
        todo.append(f"<strong>{len(unassigned)} students have no supervisor.</strong> "
                    "No member of staff cleared the minimum standard of fit, which in "
                    "practice nearly always means the proposal is too vague to place. "
                    "They are listed under Students still without a supervisor; the usual "
                    "next step is to ask them for a clearer topic and run the round again.")
    if len(weakest):
        todo.append(f"<strong>{len(weakest)} pairings are worth a second look.</strong> "
                    "These sit below the standard the office set for a good match but were "
                    "allocated anyway, within the tolerance. Each row says what the two "
                    "have in common, so a quick read is usually enough to accept or move it.")
    if ethics:
        todo.append(f"<strong>{ethics} proposals look like they need ethics approval.</strong> "
                    "They mention collecting data from people directly. Better to start that "
                    "now than in the last month of the project.")
    if access:
        todo.append(f"<strong>{access} proposals depend on data the student may not have.</strong> "
                    "They name a paid database or access to a company, which is worth "
                    "confirming before the work gets under way.")
    if len(unused):
        todo.append(f"<strong>{len(unused)} supervisors took nobody in this round.</strong> "
                    "That is expected when their areas are unlike the topics submitted, but "
                    "it is worth a glance in case somebody's record is out of date.")
    if not todo:
        todo.append("Nothing needs a decision. Every student has a supervisor, every pairing "
                    "met the standard, and no proposal raised a flag.")

    programmes = ", ".join(scope.get("programmes") or []) or "all programmes"
    mean = result.total_score / max(result.n_assigned, 1)

    alloc_view = assigned.assign(
        project=assigned["student_id"].map(titles).fillna(""),
        status=np.where(assigned["below_good"], "second look", "settled"),
    )[["student_id", "project", "supervisor_name", "group", "score", "status", "evidence"]]

    unassigned_view = unassigned.assign(
        project=unassigned["student_id"].map(titles).fillna(""))[["student_id", "project"]]

    weak_view = weakest.assign(
        project=weakest["student_id"].map(titles).fillna("")
    )[["student_id", "project", "supervisor_name", "score", "evidence"]]

    by_sup = (assigned.groupby(["supervisor_name", "group"])
              .agg(students=("student_id", "count"),
                   mean_score=("score", "mean"),
                   second_look=("below_good", "sum"))
              .reset_index().sort_values(["group", "supervisor_name"]))
    by_sup["mean_score"] = by_sup["mean_score"].round(3)

    parts = [f"""<!doctype html>
<html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Allocation, {_esc(scope.get('level', ''))}, {_esc(session)}</title>
<style>{CSS}</style></head><body><div class="wrap">

<header class="masthead">
  <h1>Dissertation allocation, {_esc(scope.get('level', ''))}</h1>
  <p class="meta">
    <span>Session {_esc(session)}</span>
    <span>{_esc(programmes)}</span>
    <span>Produced {time.strftime('%d %B %Y at %H:%M')}</span>
    <span>Register version {_esc(register_version)}</span>
  </p>
</header>

<div class="todo{'' if len(todo) > 1 or len(unassigned) or len(weakest) else ' none'}">
<h2>What needs doing</h2>
<ul>{''.join(f'<li>{t}</li>' for t in todo)}</ul>
</div>

<div class="stats">
  {_stat(result.n_assigned, 'students placed')}
  {_stat(len(unassigned), 'without a supervisor', 'missing' if len(unassigned) else '')}
  {_stat(len(weakest), 'for a second look', 'attention' if len(weakest) else '')}
  {_stat(f'{mean:.2f}', 'average strength of match')}
  {_stat(assigned['supervisor_id'].nunique(), 'supervisors used')}
</div>
"""]

    if len(unassigned_view):
        parts.append(f"""
<h2>Students still without a supervisor</h2>
<p class="lede">Nobody in the register was a close enough fit to allocate automatically.</p>
<div class="scroll"><table>
<thead><tr><th>Student</th><th>Project</th></tr></thead>
<tbody>{_rows(unassigned_view, ['student_id', 'project'], lambda r: 'gap')}</tbody>
</table></div>
""")

    if len(weak_view):
        parts.append(f"""
<h2>Pairings worth a second look</h2>
<p class="lede">Allocated within the tolerance the office set, but below the standard for a
good match. What the two have in common is given in the last column.</p>
<div class="scroll"><table>
<thead><tr><th>Student</th><th>Project</th><th>Supervisor</th><th>Match</th>
<th>What they have in common</th></tr></thead>
<tbody>{_rows(weak_view, ['student_id', 'project', 'supervisor_name', 'score', 'evidence'],
              lambda r: 'flag')}</tbody>
</table></div>
""")

    parts.append(f"""
<h2>Everyone, in one list</h2>
<p class="lede">Type a name, a student number or a topic to filter the list.</p>
<div class="search">
  <label for="find" class="visually-hidden"></label>
  <input id="find" type="search" placeholder="Find a student or a supervisor"
         aria-label="Find a student or a supervisor">
  <span class="count" id="found"></span>
</div>
<div class="scroll"><table id="allocation">
<thead><tr><th>Student</th><th>Project</th><th>Supervisor</th><th>Group</th>
<th>Match</th><th>Status</th><th>What they have in common</th></tr></thead>
<tbody>{_rows(alloc_view,
              ['student_id', 'project', 'supervisor_name', 'group', 'score', 'status', 'evidence'],
              lambda r: 'flag' if r['status'] == 'second look' else '')}</tbody>
</table></div>

<h2>Load by supervisor</h2>
<p class="lede">Counted for this round only. Remaining capacity across the whole session is in
the workbook and in the register.</p>
<div class="scroll"><table>
<thead><tr><th>Supervisor</th><th>Group</th><th>Students</th><th>Average match</th>
<th>For a second look</th></tr></thead>
<tbody>{_rows(by_sup, ['supervisor_name', 'group', 'students', 'mean_score', 'second_look'],
              lambda r: 'flag' if r['second_look'] else '')}</tbody>
</table></div>

<h2>How this allocation was produced</h2>
<p>Every student was matched against every available supervisor on research area, method,
the wording of the proposal and the phrases that distinguish it. The allocation itself
maximises the total strength of match subject to the capacity each supervisor agreed to,
so it is not first-come-first-served and it does not depend on the order the students
applied in. Running it again on the same inputs gives the same answer.</p>
<div class="scroll"><table>
<thead><tr><th>Setting</th><th>Value</th><th>What it means</th></tr></thead>
<tbody>
<tr><td>Minimum standard of fit</td><td>{_esc(params.get('hard_floor'))}</td>
<td>Below this, a pairing was never considered at all</td></tr>
<tr><td>Standard for a good match</td><td>{_esc(params.get('good_threshold'))}</td>
<td>At or above this, a pairing needs no further thought</td></tr>
<tr><td>Tolerance</td><td>{_esc(params.get('tolerance'))}</td>
<td>The share of pairings allowed to fall below that standard</td></tr>
<tr><td>Capacity reserved for later rounds</td><td>{_esc(params.get('reservation'))}</td>
<td>How much was held back for cohorts not yet allocated</td></tr>
<tr><td>Solver</td><td>{_esc(result.engine)}</td>
<td>Finished in {result.solve_seconds:.2f} seconds</td></tr>
</tbody></table></div>

<footer>
Produced by the dissertation allocation tool, version {__version__}. This page is a single
file and needs nothing installed to open it. Alongside it you will find the same information
as a spreadsheet, and the underlying rows as CSV.
</footer>

</div><script>{SEARCH_JS}</script></body></html>
""")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Excel workbook
# ---------------------------------------------------------------------------

def build_workbook(path: Path, result, students: pd.DataFrame, register: pd.DataFrame,
                   diagnostics: pd.DataFrame, requirements: pd.DataFrame,
                   capacity_table: Optional[pd.DataFrame], scope: dict,
                   session: str, params: dict) -> None:
    """One sheet per question the office actually asks.

    The summary sheet counts with formulas rather than with numbers typed in by
    the script, so that if somebody swaps a supervisor by hand on the Allocation
    sheet, which they will, the totals follow them instead of quietly becoming
    wrong. Excel recalculates these when the file is opened.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    head_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="16202B")
    body_font = Font(name="Arial", size=10)
    title_font = Font(name="Arial", size=14, bold=True)
    note_font = Font(name="Arial", size=9, italic=True, color="5A6675")
    flag_fill = PatternFill("solid", fgColor="FDF1DC")
    gap_fill = PatternFill("solid", fgColor="FBE6E4")
    thin = Side(style="thin", color="D8DEE6")
    border = Border(bottom=thin)

    wb = Workbook()
    titles = students.set_index("student_id")["project_title"].astype(str)
    assignment = result.assignment.copy()
    assigned = assignment.dropna(subset=["supervisor_id"]).copy()
    assigned["project_title"] = assigned["student_id"].map(titles).fillna("")
    assigned["status"] = np.where(assigned["below_good"], "second look", "settled")
    unassigned = assignment[assignment["supervisor_id"].isna()].copy()
    unassigned["project_title"] = unassigned["student_id"].map(titles).fillna("")

    def write_sheet(ws, df: pd.DataFrame, columns: List[str], widths: List[int],
                    highlight=None) -> None:
        for c, name in enumerate(columns, start=1):
            cell = ws.cell(row=1, column=c, value=name.replace("_", " ").capitalize())
            cell.font = head_font
            cell.fill = head_fill
            cell.alignment = Alignment(vertical="center")
            ws.column_dimensions[get_column_letter(c)].width = widths[c - 1]
        for r, (_, row) in enumerate(df.iterrows(), start=2):
            fill = highlight(row) if highlight else None
            for c, name in enumerate(columns, start=1):
                value = row.get(name, "")
                if isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = round(float(value), 4)
                elif isinstance(value, (np.bool_, bool)):
                    value = bool(value)
                cell = ws.cell(row=r, column=c, value=value)
                cell.font = body_font
                cell.border = border
                cell.alignment = Alignment(vertical="top", wrap_text=(widths[c - 1] > 40))
                if fill:
                    cell.fill = fill
        ws.freeze_panes = "A2"
        if len(df):
            ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{len(df) + 1}"

    # --- Allocation -------------------------------------------------------
    ws_alloc = wb.active
    ws_alloc.title = "Allocation"
    alloc_cols = ["student_id", "project_title", "supervisor_id", "supervisor_name", "group",
                  "score", "status", "evidence"]
    write_sheet(ws_alloc, assigned[alloc_cols], alloc_cols, [14, 44, 14, 22, 26, 9, 13, 50],
                highlight=lambda r: flag_fill if r["status"] == "second look" else None)
    n_alloc = len(assigned)

    # --- Summary, counted from the Allocation sheet ------------------------
    ws_sum = wb.create_sheet("Summary", 0)
    ws_sum["A1"] = f"Dissertation allocation, {scope.get('level', '')}, session {session}"
    ws_sum["A1"].font = title_font
    ws_sum["A2"] = (f"{', '.join(scope.get('programmes') or []) or 'all programmes'} · "
                    f"produced {time.strftime('%d %B %Y at %H:%M')}")
    ws_sum["A2"].font = note_font
    ws_sum["A4"] = ("The figures below count the Allocation sheet, so if you move a student "
                    "to a different supervisor there, they update when you reopen the file.")
    ws_sum["A4"].font = note_font

    last = max(n_alloc + 1, 2)
    rows = [
        ("Students placed", f"=COUNTA(Allocation!A2:A{last})"),
        ("Students without a supervisor", len(unassigned)),
        ("Pairings for a second look",
         f'=COUNTIF(Allocation!G2:G{last},"second look")'),
        ("Average strength of match", f"=IFERROR(AVERAGE(Allocation!F2:F{last}),0)"),
        ("Weakest match allocated", f"=IFERROR(MIN(Allocation!F2:F{last}),0)"),
        # distinct supervisors, counted on the identifier rather than the name,
        # because two members of staff can share a name and one of them would
        # otherwise disappear from the count
        ("Supervisors used", f"=SUMPRODUCT(1/COUNTIF(Allocation!C2:C{last},"
                             f"Allocation!C2:C{last}))"),
    ]
    for i, (label, value) in enumerate(rows, start=6):
        ws_sum.cell(row=i, column=1, value=label).font = Font(name="Arial", size=10, bold=True)
        cell = ws_sum.cell(row=i, column=2, value=value)
        cell.font = body_font
        if "Average" in label or "Weakest" in label:
            cell.number_format = "0.000"

    ws_sum.cell(row=13, column=1, value="Settings used").font = Font(name="Arial", size=11,
                                                                    bold=True)
    explain = {
        "hard_floor": "Below this, a pairing was never considered",
        "good_threshold": "At or above this, a pairing needs no further thought",
        "tolerance": "Share of pairings allowed below that standard",
        "reservation": "How much capacity was held back for later rounds",
        "engine": "Which solver produced the result",
        "register_version": "Version of the supervisor register used",
    }
    r = 14
    for key, note in explain.items():
        if key in params:
            ws_sum.cell(row=r, column=1, value=key.replace("_", " ")).font = body_font
            ws_sum.cell(row=r, column=2, value=params[key]).font = body_font
            ws_sum.cell(row=r, column=3, value=note).font = note_font
            r += 1
    for col, width in zip("ABC", (34, 18, 58)):
        ws_sum.column_dimensions[col].width = width

    # --- Without a supervisor ---------------------------------------------
    if len(unassigned):
        ws = wb.create_sheet("Without a supervisor")
        cols = ["student_id", "project_title"]
        write_sheet(ws, unassigned[cols], cols, [14, 70], highlight=lambda r: gap_fill)
        ws.cell(row=len(unassigned) + 3, column=1,
                value=("Nobody cleared the minimum standard of fit for these students, which "
                       "usually means the proposal is too vague to place rather than that "
                       "capacity ran out.")).font = note_font

    # --- By supervisor -----------------------------------------------------
    ws = wb.create_sheet("By supervisor")
    by_sup = (assigned.groupby(["supervisor_id", "supervisor_name", "group"])
              .agg(students=("student_id", "count"), mean_score=("score", "mean"),
                   second_look=("below_good", "sum")).reset_index())
    by_sup["mean_score"] = by_sup["mean_score"].round(3)
    if capacity_table is not None and len(capacity_table):
        keep = [c for c in ("supervisor_id", "allowance_units", "units_used", "units_left")
                if c in capacity_table.columns]
        by_sup = by_sup.merge(capacity_table[keep], on="supervisor_id", how="left")
    cols = [c for c in ["supervisor_name", "group", "students", "mean_score", "second_look",
                        "allowance_units", "units_used", "units_left"] if c in by_sup.columns]
    write_sheet(ws, by_sup.sort_values(["group", "supervisor_name"]), cols,
                [24, 28, 10, 12, 13, 14, 12, 12],
                highlight=lambda r: flag_fill if r.get("second_look") else None)

    # --- Second look --------------------------------------------------------
    weak = assigned[assigned["below_good"]].sort_values("score")
    if len(weak):
        ws = wb.create_sheet("Second look")
        cols = ["student_id", "project_title", "supervisor_name", "score", "evidence"]
        write_sheet(ws, weak[cols], cols, [14, 44, 22, 9, 56], highlight=lambda r: flag_fill)

    # --- Requirements -------------------------------------------------------
    if len(requirements):
        ws = wb.create_sheet("Ethics and data")
        flagged = requirements[(requirements["requirements"] != "")]
        cols = ["student_id", "requirements", "ethics_review_likely",
                "needs_data_access_check", "word_count"]
        write_sheet(ws, flagged[cols], cols, [14, 46, 18, 20, 12],
                    highlight=lambda r: flag_fill if r["ethics_review_likely"] else None)

    # --- Queries ------------------------------------------------------------
    if len(diagnostics):
        ws = wb.create_sheet("If a student asks")
        cols = [c for c in diagnostics.columns]
        write_sheet(ws, diagnostics, cols, [14, 22, 12, 22, 12, 14])
        ws.cell(row=len(diagnostics) + 3, column=1,
                value=("Blocking means a better-suited supervisor was already full. Wasteful "
                       "means one still had room, which happens when a minimum load or a group "
                       "ceiling applies. Both are normal, and both have an answer.")
                ).font = note_font

    wb.save(path)


# ---------------------------------------------------------------------------
# The whole pack
# ---------------------------------------------------------------------------

def build_pack(outdir: Path, result, students: pd.DataFrame, register: pd.DataFrame,
               diagnostics: pd.DataFrame, requirements: pd.DataFrame,
               capacity_table: Optional[pd.DataFrame], scope: dict, session: str,
               register_version: int, params: dict) -> Dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    written["report"] = outdir / "report.html"
    written["report"].write_text(
        build_html(result, students, register, diagnostics, requirements, scope,
                   session, register_version, params), encoding="utf-8")

    written["workbook"] = outdir / "allocation.xlsx"
    build_workbook(written["workbook"], result, students, register, diagnostics,
                   requirements, capacity_table, scope, session, params)

    result.assignment.to_csv(outdir / "allocation.csv", index=False, encoding="utf-8-sig")
    result.loads.to_csv(outdir / "supervisor_loads.csv", index=False, encoding="utf-8-sig")
    written["allocation_csv"] = outdir / "allocation.csv"
    if len(diagnostics):
        diagnostics.to_csv(outdir / "query_diagnostics.csv", index=False, encoding="utf-8-sig")
    if len(requirements):
        requirements.to_csv(outdir / "requirements.csv", index=False, encoding="utf-8-sig")

    rows = [{
        "student_id": str(r["student_id"]),
        "supervisor_id": str(r["supervisor_id"]),
        "supervisor_name": str(r["supervisor_name"]),
        "score": float(r["score"]),
        "below_good": bool(r["below_good"]),
        "locked": bool(r["locked"]),
    } for _, r in result.assignment.iterrows() if pd.notna(r["supervisor_id"])]
    written["draft"] = outdir / "draft.json"
    written["draft"].write_text(json.dumps({
        "session": session,
        "label": f"{scope.get('level', '')} · "
                 f"{', '.join(scope.get('programmes') or []) or 'all'} · {len(rows)} students",
        "level": scope.get("level", ""),
        "programmes": scope.get("programmes") or [],
        "params": params,
        "produced_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return written
