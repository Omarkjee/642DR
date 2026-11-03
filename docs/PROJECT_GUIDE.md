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

## Finalized initial model

The first implementation uses this explicit-Euler, one-zone update:

```text
T_indoor[t+1] = T_indoor[t]
              + (dt / C) * ((T_outdoor[t] - T_indoor[t]) / R
                            + Q_HVAC(action[t]))
              + noise[t]
```

The version-controlled starting values are in `.env.example` and validated by
`ExperimentConfig`:

- `dt = 1 hour`; indoor and outdoor temperatures are in degrees Fahrenheit.
- `R = 2 deg F/kW`; `C = 5 kWh/deg F`. Thus `dt/C` converts a kW heat flow to
  degrees Fahrenheit and the passive time constant is `R*C = 10 hours`.
- The signed HVAC thermal powers are `0`, `+3`, `+6`, `-3`, and `-6 kW` for
  off, low heat, high heat, low cool, and high cool. Positive heats the zone.
- Outdoor temperature repeats daily as `50 + 12*cos(2*pi*(hour-15)/24) deg F`.
  This is a controlled synthetic weather profile, not measured weather.
- The episode is one week (168 hourly steps). On weekdays, the home is away
  from hour 8 inclusive to hour 18 exclusive; it is occupied at all hours on
  weekends. Peak price is `$0.36/kWh` from hour 16 inclusive to hour 21
  exclusive; the off-peak price is `$0.12/kWh`.
- The occupied comfort band is `[68, 76] deg F`. Comfort is unpenalized while
  unoccupied. Initial temperature is sampled from `N(72, 1 deg F)` and the
  model adds seeded, per-step Gaussian noise with standard deviation `0.05 deg
  F`.
- Each HVAC action consumes `abs(power)*dt` kWh. Energy cost is energy times
  the interval's price. Comfort violation is distance outside the occupied
  band at the interval end times `dt`, in degree-hours. A mode change is a
  change among off/heating/cooling; changing low/high power within one mode
  does not count, but switching to or from off does.
- The per-step reward is the negative weighted sum of cost in dollars,
  occupied comfort violation in degree-hours, and a unit mode-change indicator.
  The weights in `.env.example` are starting values, not empirical conclusions.
- To keep the initial model small, electrical input is assumed equal to the
  absolute HVAC thermal power (coefficient of performance 1). This is an
  explicit simplification, not a realistic heat-pump efficiency model.

Model and reward assumptions are externalized through `THERMOSTAT_*` settings.
The timestep is intentionally fixed at one hour while occupancy and tariff
schedules are hour-based; other values are rejected rather than silently
advancing the clock inconsistently. Invalid physical, schedule, comfort, and
bin settings raise a `ValueError`.

## State and action conventions

The Gymnasium observation is a dictionary of indoor/outdoor temperature,
hour, occupancy, day type (weekday/weekend), current HVAC mode, and price tier.
The five discrete actions are off, low/high heat, and low/high cool. The
tabular key is the tuple `(indoor_bin, outdoor_bin, hour, occupancy, day_type,
hvac_mode, price_tier)`. Indoor temperature is binned in 2 deg F intervals
over `[40, 92)` (26 bins); outdoor temperature uses 5 deg F intervals over
`[-20, 80)` (20 bins). Values outside these ranges use the nearest edge bin.
Hour is exact (24 values), mode has three values, and occupancy and price are
retained in the key even though the deterministic schedules derive them from
hour and day type.

The naive product of feature cardinalities is `26*20*24*2*2*3*2 = 299,520`
states. Because occupancy and price are fixed by the 48 hour/day-type
combinations, the schedule-aware upper bound is `26*20*48*3 = 74,880` states
or 374,400 state-action values. Physical reachability and finite training
coverage are lower still; the agent must report observed table coverage rather
than treating this bound as the number of learned states.

## Code and result conventions

- Keep simulation, discretization, and validated settings in
  `src/thermostat_rl/`; add deterministic tests under `tests/`.
- Use `pytest` as the test runner. Pure dynamics/schedule/reward tests should
  disable temperature noise and set the initial temperature explicitly;
  stochastic tests should use explicit seeds.
- Format Python with Black using the repository's 88-character line limit.
- Keep shared starting settings and their units in `.env.example`; use ignored
  `.env` only for local overrides. Reward weights are experimental inputs, not
  selected results.
- Keep generated outputs under `results/`. Save machine-readable metrics and
  the exact seed/scenario/configuration alongside any plots used in the report.

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
