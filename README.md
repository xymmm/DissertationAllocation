# Dissertation allocation

Allocation of business school dissertations to supervisors, undergraduate and taught
postgraduate, operated by the teaching operations office. Two files go in, a supervisor
register and a set of student submissions, and a reproducible, explainable allocation comes
out, together with an AI review of the pairings and a written answer prepared for the
queries that follow.

ELM is used for extracting information and for reviewing the result. The allocation itself
is decided by a deterministic optimisation model, so the same inputs and the same parameters
always produce the same outcome, and the office has something defensible to show a student
who asks why they were given the supervisor they were given.

## Installing and running

```bash
pip install -r requirements.txt
streamlit run app.py
```

The sample files load from buttons inside the application, which is the quickest way to see
the whole sequence work before real data is involved.

## How a session works

Everything that has to survive between runs is held in one JSON file for one academic
session, for example `session_2026-27.json`. That file holds the supervisor register, every
committed round together with the parameters that produced it, and an audit trail of edits.
A spreadsheet seeds the register at the start of the session and merges changes into it
later, but it never replaces it, because a supervisor already carrying students must not be
able to disappear through a stale upload.

This matters most where the same person supervises at both levels. A supervisor commonly
takes undergraduate and masters students in the same session, and those cohorts are
allocated weeks apart. If each round began from a freshly uploaded spreadsheet, the second
round would know nothing of what the first had committed, and the most popular supervisors
would quietly be booked twice. Here, committing a round writes straight back to the
register, so the undergraduate round in March already reflects the masters students taken in
January without anybody editing a cell.

## Supervision units and levels

An allowance is recorded in supervision units rather than in a headcount, because a taught
masters dissertation is not the same amount of work as an undergraduate one. Each level
carries a weight, one for UG and one and a half for PGT by default, both adjustable in the
application. An allowance of twelve units therefore buys twelve undergraduates, or eight
masters students, or any combination between. A supervisor can additionally cap a single
level, which is how somebody content to take masters students but wanting no more than three
undergraduates records that.

Rounds are single-level by design. Within one level every student consumes the same number of
units, so the unit allowance converts cleanly into a headcount ceiling and every coefficient
in the capacity constraints stays equal to one, which is exactly what keeps the linear
relaxation integral. A mixed-level round would turn those rows into knapsack constraints and
the model into a generalised assignment problem, so the restriction buys a great deal of
speed for very little inconvenience.

## Data formats

**Supervisor register**, `sample_supervisors.csv`

| Column | Required | Notes |
|---|---|---|
| supervisor_id | yes | Unique identifier |
| name | yes | |
| group | yes | Subject group |
| research_areas | yes | Semicolon-separated tags, at most four, most important first |
| methods | yes | Semicolon-separated tags, at most six, most important first |
| keywords | no | Free text: publications, previous topics, anything that feeds text similarity |
| allowance_units | yes | Total supervision units agreed for the session |
| cap_UG, cap_PGT | no | Optional ceiling on a single level |
| min_load | no | Lower bound, to guarantee everybody takes some |
| available | no | 0 for research leave or any other absence |
| programmes | no | Restricts the supervisor to named programmes; blank means all |

**Student submissions**, `sample_students.csv`

| Column | Required | Notes |
|---|---|---|
| student_id | yes | Target ID, the only identifier that ever leaves the process |
| level | recommended | UG or PGT; rounds are organised around it |
| programme | recommended | Used to scope a round and to reserve capacity |
| project_title | yes | |
| abstract | yes | |
| project_name | no | The student's own working name for the project |
| areas, methods | no | Self-declared tags, or written back by the ELM extraction. A method may be a specific technique or simply `quantitative`, `qualitative` or `both` |
| references | no | Feeds text similarity |
| preferred_supervisor_id | no | Treated as a bonus, never as a constraint |
| locked_supervisor_id | no | Office override; the pairing is forced |

**Conflicts of interest**, `sample_conflicts.csv`: `student_id`, `supervisor_id`, `reason`. A
declared family relationship, a previous appeal, a seat on the student's discipline panel.
These are questions of eligibility rather than of fit, so they are excluded outright rather
than folded into the match score, where a generous tolerance could quietly override them.

## Tag limits and why order matters

Supervisors are asked for at most four research areas and six methods, and the data
collection form should impose the same limits. The constraint is not administrative
tidiness. A profile of twelve areas matches everything and therefore discriminates between
nothing, and the supervisor who writes it ends up carrying students whose work they have no
particular interest in. Four and six force a statement about what the person actually wants
to supervise.

Order is preserved and treated as meaningful, so a supervisor who lists inventory control
first and stochastic programming fourth scores higher on the former. A longer list is
trimmed from the end at the point of upload rather than rejected, and everything dropped is
reported, so the office can go back to the person rather than lose the tail silently.

## Method vocabulary at two levels

Undergraduates in particular can often place their project as qualitative or quantitative
and go no further, while supervisors describe themselves in specific techniques. Treated as
flat strings those two vocabularies simply never intersect, every such student scores zero
against everybody, and the optimiser reports a shortage of supervisors that does not exist.

Methods are therefore held as a two-level taxonomy, paradigm above technique, editable in
the application. A student who names a technique the supervisor also names scores one; a
student who names only the paradigm scores 0.8, rising to 0.9 where the supervisor offers
two or more techniques within it; a student who names a technique from the same paradigm as
the supervisor's scores 0.45. Precision is rewarded without vagueness being fatal.

On the sample cohort, in which 134 of 570 submissions give nothing more specific than a
paradigm, the effect is large:

| | Median best match for paradigm-only students | Unassigned overall | Of which paradigm-only |
|---|---|---|---|
| Flat method strings | 0.324 | 126 | 77 |
| Two-level taxonomy | 0.535 | 41 | 0 |

## Using the title and the proposal

Three things are read out of the title and abstract, with no model involved and no network
access required, in `text_analysis.py`.

**Key phrases.** The terms that distinguish one proposal from the rest of the cohort, taken
from term frequency weighted against the corpus, with multi-word phrases preferred over the
single words inside them. These carry a weight in the score, and they are also attached to
every pairing as the evidence for it, so the allocation table reads `areas: finance;
methods: optimisation; proposal terms: optimisation finance` rather than `0.61`. That column
is what makes a decision explainable to the student it concerns.

**Latent semantic similarity.** Surface overlap misses a proposal that says "stock levels"
where a profile says "inventory". Reducing the term-document matrix to a smaller number of
latent dimensions places those two in nearly the same direction, because they keep company
with the same other words across the cohort, and the share of the text score taken from that
reduced space is adjustable.

**Requirements.** Whether the proposal implies primary data collection, human participants,
company access, restricted or paid data, or heavy computation. The first two point to ethics
approval and the others to something the office has to confirm, and February is a far better
time to discover either than June. The reference list is deliberately kept out of the text
model, since it contributes author surnames rather than subject matter.

## The optimisation model

Let $s_{ij}$ be the match score between student $i$ and supervisor $j$, $x_{ij} \in \{0,1\}$
the allocation decision, $u_i \in \{0,1\}$ an indicator that student $i$ is left unassigned,
$w_j$ the places open to supervisor $j$ in this round, $\underline{w}_j$ a lower bound,
$\theta$ the good-match threshold, $\tau$ the mismatch tolerance and $\mathcal{A}$ the
candidate set surviving the hard floor and the top-k filter.

$$\max \sum_{(i,j)\in\mathcal{A}} s_{ij}x_{ij} - M_u\sum_i u_i - M_\ell \sum_j \sigma_j$$

$$\sum_{j:(i,j)\in\mathcal{A}} x_{ij} + u_i = 1 \quad \forall i$$
$$\sum_{i:(i,j)\in\mathcal{A}} x_{ij} \le w_j \quad \forall j$$
$$\sum_{i:(i,j)\in\mathcal{A}} x_{ij} + \sigma_j \ge \underline{w}_j \quad \forall j$$
$$\sum_{(i,j)\in\mathcal{A}\,:\, s_{ij}<\theta} x_{ij} \le \lfloor \tau n \rfloor$$

Without the last constraint the matrix is the node-arc incidence matrix of a transportation
problem, which is totally unimodular, so the vertex returned by the linear relaxation is
already integral and no branching is needed. Group ceilings do not change that, since
supervisors partition into groups and the capacity family stays laminar, which is to say the
model remains a flow network of the form student to supervisor to group to sink.

The tolerance constraint is the only thing that breaks the property, so there are two routes:

- `lp` penalises below-threshold pairings in the objective and bisects on the multiplier,
  solving nothing but transportation linear programmes throughout;
- `milp` states the cardinality constraint honestly and hands the model to CBC, HiGHS or
  Gurobi through PuLP.

Measured on this code with candidate lists capped at 25 per student:

| Students | Supervisors | Variables | LP with bisection | CBC MILP | Objectives agree |
|---|---|---|---|---|---|
| 300 | 90 | 7,500 | 0.61 s | 0.92 s | yes |
| 570 | 121 | 14,250 | 0.10 s | 2.73 s | yes |
| 900 | 150 | 22,500 | 0.16 s | 5.61 s | yes |
| 2,000 | 200 | 50,000 | 0.45 s | 23.19 s | yes |

Gurobi is unnecessary at this size. The interface is there and `auto` prefers it when a
licence is present. Run `python benchmark.py` to reproduce the table.

## The three thresholds

- **Hard floor.** Pairings below this score never enter the model. It is a backstop: better to
  hand a student to a human than to give somebody working on qualitative organisational
  behaviour to a supervisor who does nothing but stochastic optimisation.
- **Good-match threshold.** At or above this score a pairing counts as acceptable.
- **Mismatch tolerance.** The share of allocations permitted to fall below that threshold. At
  zero the model leaves students unassigned rather than force a weak pairing; loosen it and
  the unassigned count falls while average quality drops. The tolerance frontier traces that
  exchange rate in one click, which is a more productive thing to take into a meeting with
  programme directors than any single result.

## What is sent to ELM

Students are identified by a sequence number issued inside the application, `STU-0001` and
onwards, never by their matriculation or student number, and supervisors appear as codes
such as `SV-001` rather than by name. Titles and abstracts are scanned for matriculation
numbers and email addresses before they leave. Supervisor keywords are excluded by default,
since a profile written as a list of publication titles identifies the person as surely as
their name does; the research areas and methods alone are normally enough for the model to
judge fit. A complete payload therefore looks like this:

```json
{
  "student": {"student_code": "STU-0001", "level": "UG",
              "title": "...", "abstract": "..."},
  "supervisor": {"supervisor_code": "SV-001",
                 "research_areas": "finance; banking",
                 "methods": "regression; survey"}
}
```

The application shows this preview before anything is sent, and the mapping from code back
to person is held in memory for the life of the session and never written to disk. It can be
downloaded if the office wants its own copy.

## ELM

Three jobs, and no others:

1. **Extraction and standardisation.** Free-text titles and abstracts are mapped onto a
   controlled vocabulary, and each submission is rated for how specific it is, judged against
   its own level rather than in the abstract. Most mismatch originates in students describing
   the same method in a dozen different ways, so this is where the largest gain is available.
   The model is instructed to fall back to the paradigm rather than invent a technique the
   student never implied, and it reports whether it found a specific method or only the broad
   approach. Submissions rated one or two for clarity should be returned for more detail
   before the round runs.
2. **Screening.** A rating of shortlisted pairings, blended into the rule-based score at a
   small weight so that it tilts the result rather than deciding it.
3. **Review of the finished round.** Each committed pairing is read and returned as ok, query
   or reject with a reason. The two levels are held to different standards: at PGT the
   supervisor should be able to supervise the stated method at depth, while at UG a sound
   grasp of the area and general competence in the method is enough. A reject is a prompt for
   a manual check rather than a ruling, since the model cannot see a supervisor's full record
   and the cost of a wrong rejection falls on the office.

## Running in sequence, and what it costs

Allocating cohort by cohort is a greedy procedure. Whoever runs first takes the best-matched
supervisors and the cohort allocated last pays for it. The application measures this rather
than hiding it: the sequence is compared against a single joint solve of the same students,
which is what the office would achieve if every programme submitted on the same day, and the
difference is the price of the calendar.

Two mitigations are available. Reserving capacity in proportion to the students still to be
placed holds something back for the rounds that have not yet run, at a small cost to the
round currently running. Group ceilings stop one popular subject group absorbing an entire
cohort.

## Situations the application is built to handle

| Situation | How it is handled |
|---|---|
| One programme needs an answer before the others | Scope the round to that programme and reserve capacity for the rest |
| The same supervisor takes UG and PGT students | A single unit allowance, decremented by whichever round commits first |
| A supervisor will take masters students but only three undergraduates | A per-level cap in the register |
| Late, deferred and resitting students | Run another round; committed students are excluded automatically |
| A supervisor goes on sick leave or resigns | Release their students only; every other pairing stands, and the released students are rerun alone |
| A supervisor is away for the whole session | Set `available` to 0 |
| A student changes topic or withdraws | Release that student back into the pool |
| A declared conflict of interest | The conflicts file, excluded outright |
| A supervisor takes only certain programmes | The `programmes` column |
| One group is being swamped by a fashionable topic | Group ceilings |
| Everybody should take at least two | `min_load` |
| A pairing must follow a human decision | `locked_supervisor_id` |
| A round was run on the wrong parameters | Undo the round; capacity returns to the register |
| Two members of staff have the application open at once | Writes are version-checked, and a clash is reported rather than silently overwriting |
| A student asks why they did not get somebody better suited | The prepared answers table lists wasteful and blocking pairings with reasons |
| A student asks what the pairing was based on | The evidence column names the shared areas, methods and proposal terms |
| A supervisor returns twelve keywords | Trimmed to four and six in the order given, with the remainder reported |
| A student knows only that the work is quantitative | The method taxonomy scores the paradigm rather than returning zero |
| A proposal implies interviews or a questionnaire | Flagged for ethics review when the proposals are scanned |
| An allowance changes mid-session | Edit the supervisor in the register; the change is logged in the audit trail |
| The outcome has to be reproduced months later | The provenance record names the register version and every parameter |

## Not included

- Allocation of second supervisors and second markers. Structurally this is the same model run
  again with a constraint forbidding the first supervisor, and it would not be difficult to
  add.
- A cross-session equity ledger, meaning somebody who took more than their share last year
  automatically taking fewer this year. At present that judgement belongs to the office and is
  expressed in the allowance.
- Two-sided stable matching, where supervisors also rank students. The current model uses a
  single match score, and stability appears only as a diagnostic rather than as a constraint.
- Integration with EUCLID or Learn. The application reads and writes CSV.

Before the first live run, the vocabulary lists are worth circulating to the six groups so
that each confirms the areas and methods used to describe its own staff, since the quality of
that vocabulary affects the outcome far more than any parameter in the model. The data
protection position is also worth confirming with the School: only a session token, the
project title and the abstract are sent to ELM, with names, student numbers and email
addresses stripped beforehand, and the mapping from token to real identifier stays in the
local process.

## Files

- `app.py` — the Streamlit application
- `matching_core.py` — scoring, the method taxonomy, optimisation and diagnostics,
  independent of Streamlit and of ELM so that it can be driven from a script for batch
  experiments
- `text_analysis.py` — key phrases, latent semantic similarity and requirement detection
- `store.py` — the session store: supervisor register, round ledger, capacity reservation and
  the sequencing cost calculation
- `elm_client.py` — the ELM client, OpenAI-compatible, so moving platform means changing a
  base URL
- `benchmark.py` — reproduces the performance table above
- `sample_supervisors.csv`, `sample_students.csv`, `sample_conflicts.csv` — six groups, 121
  supervisors and 570 students across three undergraduate and four masters programmes, 134
  of whom give no method beyond a paradigm
