# Team itinerary through Nov 29, 2026

This is an issue-ready plan for the two-person project team. Dates assume the
current course cycle is 2026; verify all course deadlines on the official
[project page](https://people.engr.tamu.edu/guni/csce642/project.html). The
course handout lists the literature survey deadline as **Nov 9** and the final
report deadline as **Dec 4**. This itinerary stops on Nov 29 and reserves the
remaining days for final packaging and submission.

## Working agreement

- Pair-program the environment, reward, learner, and shared evaluation
  interfaces; switch driver and reviewer at each session.
- Turn each milestone below into a GitHub Issue using the **Project task**
  form. Add labels such as `setup`, `model`, `tests`, `baseline`, `agent`,
  `experiment`, or `writing`, and assign the pair together.
- Keep each issue's definition of done visible in its checklist. Link the
  relevant code, test output, plot, or draft before closing it.
- If a milestone slips, protect the core model, baseline comparison, and one
  well-analyzed experiment. Defer stretch goals rather than skipping
  validation.

## Week 1 — Setup and scope (Oct 28–Nov 1)

### Issue: Lock down the model and repository conventions
**Target date:** Nov 1 · **Suggested labels:** `setup`, `model`

- [ ] Review the proposal and survey against `PROJECT_GUIDE.md`.
- [ ] Decide and document thermal-model units, timestep, comfort band,
      occupancy schedule, price schedule, and what each HVAC action means.
- [ ] Resolve the proposed state discretization and define a state key suitable
      for a Q table; estimate the number of reachable states.
- [ ] Agree on test, formatting, configuration, and results-file conventions.
- [ ] Record any changes from the proposal for the literature survey/report.

### Issue: Finish and proofread the literature survey
**Target date:** Nov 8 (submit by Nov 9) · **Suggested labels:** `writing`

- [ ] Replace the survey's unfinished planning placeholders with final text.
- [ ] Check that the summary describes the current project, not an earlier plan.
- [ ] Verify at least 10 complete references and the technical comparison for
      each related-work item.
- [ ] Confirm every cited work appears in the references and vice versa.
- [ ] Proofread and submit according to the course instructions.

## Week 2 — Environment and tests (Nov 2–8)

### Issue: Implement and validate the single-zone simulator
**Target date:** Nov 8 · **Suggested labels:** `model`, `tests`

- [ ] Implement the RC temperature update with documented, dimensionally
      consistent parameters.
- [ ] Add occupancy, outdoor-temperature, and time-of-use price schedules.
- [ ] Define energy use/cost, comfort violation, and mode-switch calculations.
- [ ] Test heating, cooling, passive drift, comfort, switching, and seeded
      reproducibility with deterministic cases.
- [ ] Add the Gymnasium interface and test `reset`/`step` observations, reward,
      termination, and `info` metrics.

## Week 3 — Baselines and metrics (Nov 9–15)

### Issue: Establish fair controller baselines and evaluation metrics
**Target date:** Nov 15 · **Suggested labels:** `baseline`, `experiment`, `tests`

- [ ] Implement random-action and deadband controllers.
- [ ] Implement scheduled setback if the agreed core is complete; otherwise
      record it as deferred and keep the first comparison focused.
- [ ] Run each baseline in the same scenarios with the same evaluation seeds.
- [ ] Calculate energy cost, comfort degree-hours, occupied comfort
      percentage, and HVAC mode changes.
- [ ] Add arrival-temperature deficit and pre-transition energy metrics if
      their definitions are stable.
- [ ] Save a small baseline results table and at least one sanity-check plot.

## Week 4 — Q-learning (Nov 16–22)

### Issue: Train and evaluate the tabular Q-learning controller
**Target date:** Nov 22 · **Suggested labels:** `agent`, `tests`, `experiment`

- [ ] Implement the Q-learning update and epsilon-greedy selection.
- [ ] Add a documented exploration schedule and load hyperparameters from the
      experiment configuration.
- [ ] Seed environment and agent randomness; test deterministic short runs.
- [ ] Track learning curves and ensure evaluation runs do not explore.
- [ ] Compare the trained policy with deadband and random controllers on shared
      scenarios and seeds.
- [ ] Investigate state-table coverage and report if discretization prevents
      useful learning.

## Week 5 — Experiments, analysis, and report-ready evidence (Nov 23–29)

### Issue: Run the primary controlled experiment
**Target date:** Nov 26 · **Suggested labels:** `experiment`

- [ ] Select one focused question, such as changing the energy/comfort reward
      weights or insulation quality.
- [ ] Predefine scenarios, training settings, and held-out evaluation seeds.
- [ ] Run enough seeds to report variability; retain machine-readable results.
- [ ] Produce a controller comparison and a trade-off plot with clear labels
      and units.
- [ ] Inspect both improvements and failures; do not omit negative results.

### Issue: Freeze reproducible results and assemble the final handoff
**Target date:** Nov 29 · **Suggested labels:** `writing`, `experiment`

- [ ] Confirm a clean setup and rerun tests from the documented instructions.
- [ ] Ensure result tables/figures can be regenerated from saved settings and
      commands.
- [ ] Update the README and guide to match the implemented commands and
      measured behavior.
- [ ] Draft report sections for method, experiment, results, limitations, and
      individual contribution summaries.
- [ ] Record remaining work for Nov 30–Dec 4: final report, five-minute video,
      repository link, proofreading, and Canvas submission.

## GitHub issue tracking

This file is the canonical dated checklist until the tasks are entered as
Issues. A repository maintainer can create the issues from these headings
using `.github/ISSUE_TEMPLATE/project-task.yml`, then replace each checklist
with its matching issue link. No remote issues are claimed to have been
created by this setup.
