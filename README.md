# Smart Thermostat Energy Saver

A CSCE 642 course project exploring whether a reinforcement-learning controller
can anticipate occupancy and electricity-price changes better than a reactive
thermostat, while balancing household comfort and energy cost.

The project is being built by a team of two. The initial implementation is a
simulated, single-zone home with a resistor-capacitor (RC) thermal model, a
tabular Q-learning agent, and controller baselines.

## Project status

The repository includes the project proposal and an editable literature survey,
the documented simulation assumptions, a seeded Gymnasium single-zone
thermostat environment, and validated experiment settings. Baseline controllers,
Q-learning, and the experiment runner remain later milestones; `python -m
thermostat_rl` currently prints settings and does not start training.

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
│   └── PROJECT_GUIDE.md      # Finalized model, state, test, and result conventions
├── scripts/
│   └── render_literature_survey.py
├── results/                  # Generated artifacts (kept out of Git)
├── src/thermostat_rl/
│   ├── config.py             # Validated model and experiment settings
│   ├── env.py                # Gymnasium single-zone thermal simulator
│   ├── state.py              # Tabular state discretization and size estimate
│   └── __main__.py           # Configuration check; not a training runner
├── tests/
│   ├── test_config.py
│   └── test_env.py
├── .env.example
├── Course Project details.txt
├── CSCE 642_ Course Project.pdf
└── litsurvey.pdf
```

The source documents are retained in the repository: [Course Project
details.txt](Course%20Project%20details.txt) is the course handout,
[CSCE 642_ Course Project.pdf](CSCE%20642_%20Course%20Project.pdf) is the team
proposal, [docs/LITERATURE_SURVEY.md](docs/LITERATURE_SURVEY.md) is the
editable survey source, and [litsurvey.pdf](litsurvey.pdf) is its rendered PDF.
See the [project guide](docs/PROJECT_GUIDE.md) for the implemented model,
state key, test conventions, and report expectations. See the
[itinerary](docs/ITINERARY.md) for dated work items and completion status.

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

Run `python -m pytest` to validate configuration, dynamics, schedules, rewards,
determinism, state discretization, and the Gymnasium interface. Regenerate the
survey PDF from its Markdown source with
`python scripts/render_literature_survey.py`.
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
