# Smart Thermostat Energy Saver — Results Draft

**CSCE 642 — Reinforcement Learning course project**

Jessica Salguero and Omar Khan

This is a report-ready technical draft, not a claim that the course submission
is complete. The team must verify the course requirements, add its repository
and video links, proofread the final report, and submit it. Each teammate must
also write their own accurate contribution statement.

## Research question

Can tabular Q-learning use occupancy and time-of-use price information to
balance weekly electricity cost and occupied comfort in a one-zone simulated
home? How does changing the comfort-violation reward weight change the learned
policy's cost–comfort trade-off?

## Environment and method

The Gymnasium environment simulates 168 one-hour transitions with an explicit
Euler, single-zone RC thermal model. The checked-in configuration uses
`R = 6 deg F/kW`, `C = 5 kWh/deg F`, heating/cooling powers of 4 and 8 kW,
an occupied comfort band of 68–76 deg F, a deterministic daily outdoor
temperature profile, weekday absence from 08:00 to 18:00, and peak electricity
prices from 16:00 to 21:00. Initial indoor temperature and per-step noise are
seeded. Electrical input is assumed equal to thermal HVAC power (COP 1), so
this is a controlled teaching model, not a calibrated building or real-bill
forecast.

The observation includes binned indoor/outdoor temperatures and the exact
hour, occupancy, weekday/weekend, HVAC mode, and price tier. The agent chooses
among off, low/high heat, and low/high cool. Its per-step reward is the
negative weighted sum of energy cost, occupied comfort violation in
degree-hours, and a mode-change indicator. Cost and switching weights were
held at 1.0 and 0.1; the comfort weight was swept across 2.5, 5, and 10.

For each weight, tabular Q-learning ran for 1,000 episodes on each of five
training seeds (42–46), with 168 steps per episode. The learning rate was 0.1,
discount factor 0.99, and epsilon decayed from 1.0 by 0.995 per episode to a
floor of 0.05. Evaluation was greedy (epsilon 0) on 20 held-out seeds
(2000–2019). Every policy and the random, deadband, and scheduled-setback
baselines used the same held-out seeds. The exploratory pilot used seeds
1000–1019; those episodes are not included in the final experiment.

Reported outcomes are weekly energy cost, occupied comfort degree-hours,
percentage of occupied hours in the comfort band, HVAC mode changes, arrival
temperature deficit, and energy consumed in the two-hour pre-transition
windows. Windows overlapping one another are deduplicated. The scripts also
record per-episode learning curves and state/state-action coverage.

## Results

All dollar values are **per simulated week**. Baseline variation is sample
standard deviation across 20 evaluation episodes. Q-learning variation is
standard deviation across the five independent training-seed means; each such
mean averages that policy's 20 held-out episodes. These different sources of
variation are shown explicitly and should not be read as a significance test.

| Controller | Comfort weight | Weekly cost ($) | Comfort violation (degree-hours) | Occupied hours in band (%) | Mode changes |
|---|---:|---:|---:|---:|---:|
| Random | — | 136.78 ± 8.44 | 1746.2 ± 326.3 | 3.5 ± 2.4 | 107.2 ± 6.0 |
| Deadband | — | 67.92 ± 1.12 | 63.1 ± 5.7 | 21.8 ± 1.0 | 56.9 ± 2.9 |
| Scheduled setback | — | 69.26 ± 1.19 | 174.2 ± 7.0 | 12.8 ± 0.9 | 33.5 ± 2.0 |
| Q-learning | 2.5 | 124.28 ± 9.48 | 171.0 ± 178.1 | 72.0 ± 14.5 | 88.3 ± 7.6 |
| Q-learning | 5 | 118.15 ± 16.69 | 61.4 ± 115.3 | 84.9 ± 12.8 | 75.4 ± 8.3 |
| Q-learning | 10 | 114.53 ± 17.27 | 30.8 ± 41.5 | 91.3 ± 7.5 | 82.9 ± 5.7 |

Increasing the comfort weight reduced mean occupied violation and increased
the share of occupied time in band in this run, but it did not reduce cost:
Q-learning cost remained roughly 69–83% above deadband. The strongest comfort
result (weight 10) also had greater cost and more HVAC mode changes than
deadband. The weight-2.5 policy was highly variable across training seeds.
Thus, this experiment does **not** show Q-learning dominating the simple
thermostat baselines. It does show a reward-weight trade-off and a substantial
comfort improvement for some learned policies at additional energy cost.

The deadband controller's mean arrival deficit was 0.05 deg F, compared with
0.74, 1.38, and 0.56 deg F for Q-learning weights 2.5, 5, and 10,
respectively. Mean pre-transition energy was 92 kWh for deadband and 198, 185,
and 189 kWh for the three Q-learning policies. The additional energy therefore
did not reliably translate into better arrival temperatures.

After training, each policy visited about 3.1–3.2% of the 74,880
schedule-aware state upper bound and 2.94–3.04% of the 374,400 state-action
upper bound. State coverage increased only slightly from the lower to higher
comfort weights. Sparse tabular coverage, high between-training-seed
variability, and the synthetic model limit the conclusions. The learning
curves and coverage records are retained so the team can inspect whether the
training budget is adequate; results should not be presented as evidence of
convergence or real-building performance.

## Limitations and next work

- The thermal model, deterministic weather, occupancy, tariff, and COP=1
  assumption simplify real buildings and equipment.
- A state-space upper bound of 74,880 states is large for a tabular learner;
  only a small fraction was visited. Coarser or factored state representations,
  function approximation, or a more focused action/state design may improve
  learning.
- Only five training seeds and 20 held-out weekly scenarios were used. The
  reported Q-learning standard deviations across training seeds are large;
  further runs and uncertainty analysis are needed before strong claims.
- The reward is a design choice, not a direct measure of resident preference.
  Reward-weight selection and a more principled evaluation of comfort/cost
  trade-offs remain open.
- The present model assumes perfect schedule knowledge and does not model
  humidity, equipment dynamics, occupancy uncertainty, or measured weather.

## Reproducibility and handoff

From the repository root, install the editable package and run:

```bash
python -m pip install -e ".[dev,docs]"
python -m pytest
python scripts/run_baselines.py
python scripts/run_experiments.py
```

The final experiment writes its settings, seed-level runs, summaries, learning
curves, coverage, and figures under `results/`. The `experiment_metadata.json`
file records its configuration and exact seed lists. The editable literature
survey is [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md); the project model and
units are described in [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Team completion before submission

- Add the repository URL and unlisted five-minute video URL required by the
  course.
- Proofread this draft, verify references against the report, and check the
  current Canvas deadline and submission format.
- Each teammate independently writes and submits an accurate contribution
  summary. Do not copy or infer another teammate's statement.
- Record the final report and video submission with the team; these external
  actions cannot be completed by repository changes alone.
