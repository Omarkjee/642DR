# Project guide

## What we are studying

**Smart Thermostat Energy Saver** asks whether a reinforcement-learning (RL)
controller can learn to anticipate occupancy and time-of-use price changes,
balancing resident comfort against electricity cost better than a reactive
deadband thermostat. The initial scope is one simulated home zone and a
tabular, epsilon-greedy Q-learning agent.

The project is a sequential decision problem: a thermostat action affects the
indoor temperature and HVAC mode that the controller will observe later. The
main comparison should be narrow and interpretable rather than spread across
many stretch goals.

## Proposed model

The proposal describes a first-order RC-style thermal update:

```text
T_indoor[t+1] = T_indoor[t]
              + (dt / C) * ((T_outdoor[t] - T_indoor[t]) / R
                            + Q_HVAC(action[t]))
              + noise[t]
```

Before implementing the simulator, agree on units and parameter meanings so
that temperature, time step, thermal resistance/capacitance, and HVAC power
are dimensionally consistent. Keep the assumptions and chosen values in one
configuration source and explain how they were selected in the report.

The proposed observation includes indoor and outdoor temperature, time of day,
occupancy, weekday/weekend, current HVAC mode, and price tier. The proposed
discrete actions are off, low heat, high heat, low cool, and high cool. Start
with a documented, coarse state discretization for tabular learning; do not
silently combine impossible or ambiguous state bins.

The reward is the negative weighted sum of energy cost, occupant discomfort,
and HVAC mode switching. Define those three quantities and their units before
tuning weights. Comfort penalties should be occupancy-aware as proposed.
Include the previous HVAC mode in the state if switching cost depends on it.
Treat the values in `.env.example` as reproducible starting values, not
evidence that a particular weighting is best.

## Implementation sequence

1. Define and unit-test the thermal update and the state/action/reward meanings.
2. Wrap the simulator in Gymnasium's `reset`/`step` interface, including seeded
   randomness and explicit episode termination.
3. Implement random, deadband, and (if time permits) scheduled-setback
   controllers before training RL.
4. Implement tabular Q-learning with epsilon-greedy action selection and
   configurable, recorded hyperparameters.
5. Evaluate every controller using the same scenarios and random seeds.
6. Only consider DQN, historical weather, continuous control, multiple zones,
   or uncertain occupancy after the required experiment is reliable.

Each implementation step should have tests and a reproducible example before
the next dependent step begins.

## Evaluation protocol

At minimum, report:

- total simulated energy cost;
- comfort violation magnitude (degree-hours) and the percentage of occupied
  time inside the comfort band;
- HVAC mode changes;
- the indoor-temperature deficit at occupancy arrival and, if implemented,
  energy used before occupancy or peak-price transitions.

Compare Q-learning against the deadband baseline; retain a random policy as a
sanity floor. The literature survey also proposes scheduled setback as a
third baseline. Use identical weather, occupancy, price schedules, episode
lengths, and evaluation seeds across controllers. Report averages and a
statistical summary (for example, standard deviation or confidence intervals)
across multiple evaluation runs. Analyze at least one clear trade-off, such as
reward-weight changes or insulation quality, and include limitations and
negative results.

Keep training and evaluation separate. Do not select hyperparameters on the
same evaluation runs used to make the final comparison. Record the values
tried and how they were selected.

## Collaboration and source integrity

The course requires a team of two and encourages pair programming. Pair on
the model, reward, and evaluation interfaces; rotate driver/reviewer roles
between sessions. Keep issues small enough to finish and review together.

Credit papers, datasets, and external code in the report. Clearly identify
what was reused and what the team implemented. Do not present prior work or
unattributed code as a new contribution. Each teammate must independently
submit a short contribution summary; the project report is submitted once per
team.

## Report and submission checklist

- [ ] Describe the sequential problem and specify state, actions, transitions,
      reward, and assumptions.
- [ ] Summarize related work with at least 10 full bibliographic references and
      explain how each relates to this project.
- [ ] Include a baseline-comparison experiment, statistical evidence, and an
      honest discussion of strengths, limitations, and parameter selection.
- [ ] Cite all borrowed ideas, code, and data; proofread the final report.
- [ ] Include the repository URL and an unlisted five-minute video URL in the
      report.
- [ ] Each teammate prepares their own contribution statement for individual
      upload.
- [ ] Verify the Canvas deadline and submission details on the official
      [course project page](https://people.engr.tamu.edu/guni/csce642/project.html).

The course handout in this repository lists the literature survey deadline as
Nov 9, 2026, and the final report deadline as Dec 4, 2026. The itinerary covers
work through Nov 29 and intentionally reserves the final days for report,
video, and submission packaging.
