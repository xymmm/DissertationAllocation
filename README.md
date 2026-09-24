# Dissertation allocation

Allocates business school dissertations to supervisors, undergraduate and taught
postgraduate, for a teaching operations office. It runs on one computer, keeps its records in
one file, and produces a report that somebody who has never seen the code can read.

There is no web server, no account to sign in to and no AI service involved. Nothing is sent
anywhere: the matching is arithmetic on the text and tags you supply, so the same inputs always
give the same answer and every pairing can be explained by pointing at what the two have in
common. Everything stays on the machine it runs on, including after it has run.

## What you get out of a run

Every round writes a dated folder containing:

| File | What it is for |
|---|---|
| `report.html` | **Open this one.** A single page: what needs doing, who has nobody, which pairings are worth a second look, and a searchable list of everyone. Double-click it; it needs nothing installed and works offline. |
| `allocation.xlsx` | The same information as a spreadsheet, one sheet per question: Summary, Allocation, Without a supervisor, By supervisor, Second look, Ethics and data, If a student asks. |
| `allocation.csv`, `supervisor_loads.csv` and the rest | The plain rows, for anyone who wants to work with them directly. |
| `draft.json` | How the round is committed later. Not for reading. |

The report opens with what has to be done rather than with statistics, because that is the
first question anybody has. Colour means one thing only: amber for a pairing somebody should
look at, red for a student nobody was found for. Nothing is coloured for decoration.

The workbook's summary counts the Allocation sheet with formulas rather than with numbers
typed in by the script, so if somebody moves a student to a different supervisor by hand,
which they will, the totals follow them instead of quietly becoming wrong.

## The four scripts

Open these in PyCharm and press Run. Each has a short `SETTINGS` block at the top, and that
is the only part anybody needs to change.

**`setup_session.py`** — run once at the start of the year, and again whenever the register
spreadsheet changes. It creates the session file and loads the supervisors into it. It never
replaces the register wholesale: a supervisor already carrying students cannot be removed by
a stale spreadsheet, and anybody missing from the upload is reported and left alone. Tag
lists longer than agreed are trimmed and the discarded entries are printed, so they can be
discussed rather than lost.

**`run_allocation.py`** — the main one. Set the level, the programmes and how strict to be,
press Run, and read the console summary and then the report. Nothing is committed by running
it: the round becomes real only when you set `commit` to `True` or run `commit_round.py`.
That separation is the difference between a proposal and a decision, and it lets a round be
produced on Thursday and approved on Friday without anything being recalculated in between.

**`commit_round.py`** — writes a round into the session and reduces everybody's remaining
capacity.

```bash
python commit_round.py runs/2026-09-24_1530_PGT
```

If the register has changed since the draft was produced, it refuses and says so, because the
capacity the draft was computed against no longer exists and committing anyway would
overbook people.

**`manage.py`** — the occasional jobs, as a numbered menu in the console: where the session
stands, undo a round, a supervisor has withdrawn, free particular students, export
everything, check a student file before allocating. Nothing to memorise and no arguments to
get right.

## How a session fits together

```
setup_session.py            once, then whenever the register changes
       |
run_allocation.py           per cohort: PGT in January, UG in March, stragglers in June
       |                    commit = False while you are still looking
   read report.html
       |
commit_round.py             capacity comes off everybody's allowance
       |
   next cohort              which now sees what is genuinely left
```

Everything that survives between runs lives in one JSON file for one academic session, for
example `session_2026-27.json`: the supervisor register, every committed round with the
settings that produced it, and an audit trail of changes. Copy it, archive it, email it.

This matters most where the same person supervises at both levels. A supervisor commonly
takes undergraduate and masters students in the same year, weeks apart. If each round started
from a freshly uploaded spreadsheet the second would know nothing of the first, and the most
popular supervisors would quietly be booked twice. Here, committing a round writes straight
back to the register, so the March round already reflects the January one without anybody
editing a cell.

Two people can work at once: every write checks that the file on disk has not moved since it
was read, and reports a clash rather than overwriting somebody else's change. Keep the session
file on a managed drive and both of you will be working on the same record.

## Supervision units and levels

An allowance is recorded in supervision units rather than in a headcount, because a masters
dissertation is not the same amount of work as an undergraduate one. Each level carries a
weight, one for UG and one and a half for PGT by default. An allowance of twelve units buys
twelve undergraduates, or eight masters students, or any combination between. A supervisor
can also cap one level, which is how somebody willing to take masters students but wanting no
more than three undergraduates records that.

Rounds are single-level by design. Within one level every student costs the same number of
units, so the allowance converts cleanly into a number of places, which is what keeps the
underlying calculation fast and exact.

## Data formats

**Register**, `sample_supervisors.csv`

| Column | Required | Notes |
|---|---|---|
| supervisor_id | yes | Unique identifier |
| name, group | yes | |
| research_areas | yes | Semicolon-separated, at most four, most important first |
| methods | yes | Semicolon-separated, at most six, most important first |
| keywords | no | Free text: publications, past topics, anything that helps the match |
| allowance_units | yes | Supervision units agreed for the session |
| cap_UG, cap_PGT | no | Ceiling on one level |
| min_load | no | Lower bound |
| available | no | 0 for leave or absence |
| programmes | no | Restricts to named programmes; blank means all |

**Students**, `sample_students.csv`

| Column | Required | Notes |
|---|---|---|
| student_id | yes | The only identifier that ever leaves the process |
| level | recommended | UG or PGT |
| programme | recommended | Used to scope a round |
| project_title, abstract | yes | |
| areas, methods | no | A method may be a technique or simply `quantitative`, `qualitative` or `both` |
| preferred_supervisor_id | no | A bonus, never a constraint |
| locked_supervisor_id | no | Forces a pairing the office has already decided |

**Conflicts**, `sample_conflicts.csv`: `student_id`, `supervisor_id`, `reason`. A declared
family relationship, a previous appeal, a seat on a discipline panel. These are questions of
eligibility rather than of fit, so they are excluded outright rather than folded into the
score where a generous tolerance could override them.

## Tag limits, and why order matters

At most four research areas and six methods per supervisor, and the data collection form
should say the same. A profile of twelve areas matches everything and therefore distinguishes
nothing, and the person who writes it ends up supervising work they have no interest in.

Order is kept and treated as meaningful, so a supervisor who lists inventory control first and
stochastic programming fourth scores higher on the first. Longer lists are trimmed from the
end at upload and everything dropped is reported.

## Method vocabulary at two levels

Undergraduates in particular can often say only that their project is qualitative or
quantitative, while supervisors describe themselves in specific techniques. Treated as flat
text those vocabularies never meet, every such student scores zero against everybody, and the
result looks like a shortage of supervisors that does not exist.

Methods are therefore held as a two-level taxonomy, paradigm above technique. A student who
names a technique the supervisor also names scores 1; one who names only the paradigm scores
0.8, rising to 0.9 where the supervisor offers two or more techniques within it; one who names
a different technique from the same paradigm scores 0.45. Precision is rewarded without
vagueness being fatal. On the sample cohort, where 134 of 570 submissions give nothing more
specific than a paradigm:

| | Median best match for those students | Left unplaced | Of which paradigm-only |
|---|---|---|---|
| Flat method text | 0.324 | 126 | 77 |
| Two-level taxonomy | 0.535 | 41 | 0 |

## Using the title and the proposal

Three things are read from the title and abstract, with no model involved and no network
access, in `text_analysis.py`.

**Key phrases**: the terms that distinguish one proposal from the rest of the cohort. They
carry weight in the score and, more usefully, they appear against every pairing as the
evidence for it, so the report says *areas: finance; methods: optimisation; proposal terms:
optimisation finance* rather than *0.61*. That column is what makes a decision explainable to
the student it concerns.

**Latent semantic similarity**: surface overlap misses a proposal that says "stock levels"
where a profile says "inventory". Reducing the term-document matrix recovers much of that.

**Requirements**: whether the proposal implies primary data collection, human participants,
company access, paid data or heavy computation. The first two point to ethics approval and the
others to something to confirm, and February is a far better time to discover either than
June. The reference list is deliberately excluded, since it contributes author surnames rather
than subject matter.

## How the allocation is decided

Every student is scored against every available supervisor on research area, method, the
wording of the proposal and its distinguishing phrases. The allocation then maximises the
total strength of match subject to the capacity each supervisor agreed to, so it is not
first-come-first-served and does not depend on the order students applied in. The same inputs
and settings always give the same answer.

Three settings control strictness. The **minimum standard of fit** keeps a pairing out of
consideration entirely; below it, a student is better handed to a person than forced onto
somebody unsuitable. The **standard for a good match** defines what needs no further thought.
The **tolerance** is the share of pairings allowed to fall below that standard: at zero the
calculation will leave students unplaced rather than force a weak pairing, and loosening it
trades average quality for fewer gaps.

Underneath, with the tolerance set aside, the problem is a transportation problem, whose
structure guarantees that the fast method already returns a whole-number answer with no
searching. The tolerance is the only thing that complicates that, and it is handled two ways:
quickly, by penalising weak pairings and tuning the penalty until the tolerance is met, and
exactly, by stating it as a constraint and solving the integer program. Set `cross_check` to
`True` and both run, and the run tells you whether they agree. Measured on this code:

| Students | Supervisors | Fast method | Exact method | Agree |
|---|---|---|---|---|
| 300 | 90 | 0.61 s | 0.92 s | yes |
| 570 | 121 | 0.10 s | 2.73 s | yes |
| 900 | 150 | 0.16 s | 5.61 s | yes |
| 2,000 | 200 | 0.45 s | 23.19 s | yes |

`python benchmark.py` reproduces it. A commercial solver is unnecessary at this size.

## Situations this is built to handle

| Situation | What to do |
|---|---|
| One programme needs an answer before the others | Scope the round to it; capacity is reserved for the rest automatically |
| The same supervisor takes UG and PGT students | One allowance in units, reduced by whichever round commits first |
| Willing to take masters students but only three undergraduates | A cap on one level in the register |
| Late, deferred and resitting students | Run another round; students already placed are excluded automatically |
| A supervisor goes on leave or resigns | `manage.py`, option 3: frees their students only, everything else stands |
| A student changes topic or withdraws | `manage.py`, option 4 |
| A declared conflict of interest | The conflicts file |
| One group is being swamped by a fashionable topic | Group ceilings |
| A round was run on the wrong settings | `manage.py`, option 2, before or after committing |
| Two people working at once | Writes are version-checked and a clash is reported |
| A student asks what the pairing was based on | The evidence column, in the report and the workbook |
| A student asks why they did not get somebody better | The *If a student asks* sheet |
| Reproducing a result months later | The settings are recorded with the round, including the register version |

## Not included

- Second supervisors and second markers. Structurally the same calculation run again with the
  first supervisor excluded, and not difficult to add.
- A year-on-year fairness ledger, so that somebody who took more than their share last year
  automatically takes fewer. For now that judgement belongs to the office and goes into the
  allowance.
- Supervisors ranking students as well as the other way round.
- Any link to EUCLID or Learn. Files in, files out.

## Installing

```bash
pip install -r requirements.txt
```

In PyCharm, open the folder, let it create an interpreter, then run `pip install -r
requirements.txt` in the terminal it provides. After that, every script runs from the green
arrow.

## Files

- `setup_session.py`, `run_allocation.py`, `commit_round.py`, `manage.py` — the four things
  anybody needs to open
- `matching_core.py` — scoring, the method taxonomy, the allocation itself and the diagnostics
- `store.py` — the session file: register, rounds, capacity and audit trail
- `text_analysis.py` — key phrases, semantic similarity, ethics and data flags
- `report.py` — the HTML report and the workbook
- `benchmark.py` — reproduces the timing table
- `sample_supervisors.csv`, `sample_students.csv`, `sample_conflicts.csv` — 121 supervisors and
  570 students across three undergraduate and four masters programmes
