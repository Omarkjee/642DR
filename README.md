# Smart Thermostat Energy Saver

A CSCE 642 course project exploring whether a reinforcement-learning controller
can anticipate occupancy and electricity-price changes better than a reactive
thermostat, while balancing household comfort and energy cost.

The project is being built by a team of two. The initial implementation is a
simulated, single-zone home with a resistor-capacitor (RC) thermal model, a
tabular Q-learning agent, and controller baselines.

## Project status

The repository includes the project proposal and literature survey, a seeded
Gymnasium single-zone environment, three reference controllers, tabular
Q-learning, fair controller evaluation, and reproducible experiment scripts.
`python -m thermostat_rl` prints configuration; use the scripts below to run
baselines and training.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
cp .env.example .env               # Windows PowerShell: Copy-Item .env.example .env
set -a && source .env && set +a    # Linux/macOS: export the example settings
python -m thermostat_rl            # display the active experiment settings
python -m pytest
python -m black --check src tests scripts
```

The `.env` file contains local, non-secret experiment settings. Do not commit
`.env`; `.env.example` is the version-controlled template. If you change values
in `.env`, reload it in your shell before running Python.

## Repository layout

```text
.
├── .github/
│   └── ISSUE_TEMPLATE/       # GitHub issue form for project tasks
├── configs/                  # Reserved for checked-in experiment configs
├── docs/
│   ├── ITINERARY.md          # Week-by-week task list through Nov 29, 2026
│   ├── LITERATURE_SURVEY.md  # Editable, cited literature-survey source
│   ├── PROJECT_GUIDE.md      # Finalized model, state, test, and result conventions
│   └── FINAL_REPORT.md       # Report-ready methods, results, and limitations draft
├── scripts/
│   ├── run_baselines.py
│   ├── run_experiments.py
│   └── render_literature_survey.py
├── results/                  # Reproducible baseline and experiment artifacts
├── src/thermostat_rl/
│   ├── config.py             # Validated model and experiment settings
│   ├── controllers.py        # Random, deadband, and scheduled-setback policies
│   ├── env.py                # Gymnasium single-zone thermal simulator
│   ├── evaluation.py         # Seed-controlled metrics and summaries
│   ├── q_learning.py         # Sparse tabular Q-learning
│   ├── state.py              # Tabular state discretization and size estimate
│   └── __main__.py           # Configuration check; not a training runner
├── tests/
│   ├── test_config.py
│   ├── test_controllers.py
│   ├── test_env.py
│   ├── test_evaluation.py
│   └── test_q_learning.py
├── .env.example
├── Course Project details.txt
├── CSCE 642_ Course Project.pdf
└── CSCE642_LitSurvey.pdf
```

The source documents are retained in the repository: [Course Project
details.txt](Course%20Project%20details.txt) is the course handout,
[CSCE 642_ Course Project.pdf](CSCE%20642_%20Course%20Project.pdf) is the team
proposal, [docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md) is the
editable survey source, and [CSCE642_LitSurvey.pdf](CSCE642_LitSurvey.pdf) is
the current survey PDF. See the [project guide](docs/PROJECT_GUIDE.md) for the
implemented model, state key, test conventions, and report expectations. See the
[itinerary](docs/ITINERARY.md) for dated work items and completion status.
The [final report draft](docs/FINAL_REPORT.md) summarizes the measured
experiment and lists team-only submission steps still outstanding.

## Simulator

`ThermostatEnv` implements the one-hour RC model and five discrete HVAC
actions. It follows Gymnasium's `reset`/`step` API, returns a seven-day episode,
and reports per-step energy, cost, comfort, and mode-change metrics in `info`.
The shared model parameters and state-space estimate are documented in the
[project guide](docs/PROJECT_GUIDE.md). A small smoke run is:

```python
from thermostat_rl import ThermostatEnv

env = ThermostatEnv()
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`python scripts/run_baselines.py` evaluates all three rule-based controllers
over 30 common seeded episodes by default and writes raw runs, summaries, and a
sanity plot. `python scripts/run_experiments.py` trains Q-learning for 1,000
episodes at three comfort-reward weights with five independent training seeds
and 20 held-out evaluation seeds per policy. It saves seed-level metrics,
coverage, configuration, learning curves, and the cost–comfort plot under
`results/`.
To preview a faster run, use
`python scripts/run_experiments.py --episodes 25 --training-seeds 2 --evaluation-seeds 3`;
use the defaults for reportable results. The experiment also reruns baselines
on the same held-out seeds.

Run `python -m pytest` for unit and Gymnasium API tests. Regenerate the survey
PDF from its Markdown source with `python scripts/render_literature_survey.py`.
Format Python with `python -m black src tests scripts`.

## Experiment settings

Settings are read from the `THERMOSTAT_` environment variables documented in
`.env.example`. `src/thermostat_rl/config.py` validates their ranges and reports
invalid settings with an error. Values there are starting points for
implementation, not experimentally selected hyperparameters; document and test
any values used in the final report. The example sourcing command above is for
Linux/macOS; on Windows PowerShell, set the documented variables in the active
terminal before running Python.

## Course deliverables

The course handout lists the final report deadline as **Friday, December 4,
2026, at 11:59 p.m.** It calls for source code and a runnable README, a detailed
conference-style report, a five-minute unlisted video, at least ten fully
identified references, an experiment with a baseline comparison, and an
individual contribution summary uploaded by each teammate. Confirm deadlines
and submission instructions on the [course project
page](https://people.engr.tamu.edu/guni/csce642/project.html). The itinerary
ends Nov 29 and leaves the remaining days for final report/video packaging and
submission.

## Collaboration and integrity

Follow the course's team-of-two and pair-programming guidance. Review external
code and ideas carefully, cite sources in the report, and distinguish reused
material from the team's contributions. Each teammate submits their own
individual contribution summary, even though the team submits one project
report.
