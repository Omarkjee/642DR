# Smart Thermostat Energy Saver

A CSCE 642 course project exploring whether a reinforcement-learning controller
can anticipate occupancy and electricity-price changes better than a reactive
thermostat, while balancing household comfort and energy cost.

The project is being built by a team of two. The initial implementation is
planned as a simulated, single-zone home with a resistor-capacitor (RC) thermal
model, a tabular Q-learning agent, and controller baselines.

## Project status

This repository currently contains the proposal, literature survey, project
guidance, an experiment-configuration scaffold, and the work itinerary. The
thermal simulator, controllers, learner, and experiment runner are planned work
items; there is not yet a training command.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env               # Windows PowerShell: Copy-Item .env.example .env
set -a && source .env && set +a    # Linux/macOS: export the example settings
python -m thermostat_rl            # display the active experiment settings
python -m unittest discover -s tests
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
│   └── PROJECT_GUIDE.md      # Modeling, implementation, experiment, and writing guide
├── results/                  # Generated artifacts (kept out of Git)
├── src/thermostat_rl/
│   ├── config.py             # Validated experiment settings
│   └── __main__.py           # Configuration check; not a training runner
├── tests/
│   └── test_config.py
├── .env.example
├── Course Project details.txt
├── CSCE 642_ Course Project.pdf
└── litsurvey.pdf
```

The source documents are retained in the repository: [Course Project
details.txt](Course%20Project%20details.txt) is the course handout,
[CSCE 642_ Course Project.pdf](CSCE%20642_%20Course%20Project.pdf) is the team
proposal, and [litsurvey.pdf](litsurvey.pdf) is the literature-survey draft.
See the [project guide](docs/PROJECT_GUIDE.md) for the proposed model,
implementation conventions, and report expectations. See the
[itinerary](docs/ITINERARY.md) for dated work items that can be copied into
GitHub Issues.

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
