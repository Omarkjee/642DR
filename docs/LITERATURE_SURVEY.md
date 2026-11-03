# Reinforcement Learning for a Smart Thermostat Energy Saver

**CSCE 642 — Reinforcement Learning course project**  
Jessica Salguero and Omar Khan

## Summary of the current project

This project studies whether a tabular reinforcement-learning (RL) controller
can anticipate occupancy and time-of-use electricity prices in a simulated
single-zone home, and how the learned behavior changes as the energy-cost and
comfort weights change. The core comparison is an epsilon-greedy Q-learning
controller against a reactive deadband thermostat and a random-action
sanity-check policy. The initial simulator is a deliberately transparent,
first-order resistor-capacitor (RC) model implemented as a Gymnasium
environment. It is an educational, controlled experiment, not a calibrated
building model or a deployment claim.

The hourly observation contains indoor and outdoor temperature, hour,
occupancy, weekday/weekend, current HVAC mode, and a time-of-use price tier.
The five actions are off, low heat, high heat, low cool, and high cool. Heating
and cooling are represented by signed thermal powers. The reward is the
negative weighted sum of interval electricity cost, occupied comfort
violation in degree-hours, and a mode-change indicator. The current
configuration uses a one-hour time step, `R = 2 deg F/kW`, `C = 5 kWh/deg F`,
an occupied comfort band of 68–76 deg F, a deterministic daily outdoor
temperature profile, weekday work-hour absence, and an afternoon/evening peak
price. These are stated assumptions for reproducibility, not measurements or
tuned results; the [project guide](PROJECT_GUIDE.md) records their definitions
and units. For this initial model, electrical input equals absolute HVAC
thermal power (coefficient of performance 1), a simplifying assumption rather
than a realistic HVAC efficiency model.

The research question is whether a small tabular policy learns useful
pre-conditioning behavior from these signals, and how that behavior trades
comfort against cost. We will measure total energy cost, occupied comfort
degree-hours, percentage of occupied time in band, HVAC mode changes, and
the lower comfort-bound deficit at the first occupied step after an away
period. The held-out controller comparison must use the same weather,
schedules, episode lengths, and seeds. No performance result is claimed before
those experiments are run.

## Decisions since the proposal

The proposal left the hour representation, thermal units, and exact
occupancy/price schedules open. The implementation uses the exact hour (0–23)
and documents the corresponding Fahrenheit, kW, kWh, and hour-based RC units.
It fixes weekday absence to 08:00–18:00, keeps weekends occupied, and sets the
daily peak-price interval to 16:00–21:00. It also makes electrical input equal
to thermal HVAC power (COP 1) so that energy and cost are computable without
adding an unsupported equipment-efficiency model. The first experiment remains
tabular Q-learning with deadband and random baselines; deep or continuous
controllers remain stretch goals. Comfort is reported both as degree-hours and
occupied time in band, and the planned arrival metric is the lower-bound
temperature deficit rather than an unspecified “comfort” score.

## Related work

### RL foundations and algorithm choice

Watkins and Dayan established the tabular Q-learning update and its
convergence conditions: all relevant state-action pairs must be visited
repeatedly and the step sizes must satisfy suitable conditions [1]. Those
assumptions make state coverage an important practical limitation here, not a
guarantee that a finite training run will learn an effective thermostat.
Sutton and Barto provide the practical RL treatment underlying our
epsilon-greedy policy and reward formulation [2]. Our first agent uses a
table because the action set is small and the resulting policy is inspectable;
function approximation is not needed to answer the initial research question.

Mnih et al.'s deep Q-network (DQN) uses replay and a target network to stabilize
value learning with function approximation [3]. It is a useful possible
extension if the discrete table proves too sparse, but it introduces more
training choices than the core comparison requires. Lillicrap et al.'s deep
deterministic policy gradient (DDPG) method learns continuous actions with an
actor and critic [4], while Schulman et al.'s proximal policy optimization
(PPO) uses clipped policy updates [5]. Both address a continuous-control
setting more directly than the five-action project; neither is part of the
initial implementation. This distinction keeps the first result attributable
to the reward, state discretization, and simple thermal dynamics rather than
to a larger algorithm stack.

### RL for buildings and residential thermostats

Vázquez-Canteli and Nagy survey RL for demand response, comparing modeling and
algorithm choices across grid-interactive loads [6]. Wang and Hong review RL
for building controls and organize studies by state, action, reward, and
environment [7]. Together, these reviews motivate an explicit MDP definition
and careful separation between simulated evidence and field validation. Our
study narrows the domain to one residential zone and a deterministic,
reproducible price/occupancy schedule; it does not address deployment,
multi-building transfer, or grid-scale coordination.

Barrett and Linder present a residential HVAC architecture pairing Bayesian
occupancy prediction with Q-learning [8]. It is the closest methodological
comparison: both use tabular value learning for thermostat decisions, but our
initial agent receives the scheduled occupancy state directly and is compared
with a deadband controller under a synthetic time-of-use tariff. The proposed
experiment can therefore test whether anticipation emerges from the learned
policy under known schedules; it does not reproduce their occupancy
prediction method.

Ruelens et al. apply batch fitted Q-iteration to thermostatically controlled
loads for residential demand response [9]. Their offline regression over
collected transitions and forecast-aware policy differ from our online
tabular updates and small deterministic simulator. Their work makes the
price-responsive thermal-control question relevant, while the implementation
contrast clarifies that our project tests accessibility and interpretability,
not an alternative claim of algorithmic superiority.

### Thermal models and simulation

Bacher and Madsen describe selecting grey-box models for building heat
dynamics and estimating physically meaningful thermal parameters from data
[10]. Our one-node RC model is a much smaller special case: its resistance and
capacitance are hand-set assumptions, not identified from measurements. Its
advantage is that the dynamics and time constant are easy to inspect; its
limitation is that it omits solar gain, internal loads, humidity, equipment
efficiency, and zone coupling.

Crawley et al. describe EnergyPlus as a detailed whole-building energy
simulation program [11]. EnergyPlus is better suited to realistic building
loads and envelope behavior but is more involved than the inexpensive
equation used here. Towers et al. describe Gymnasium's standardized
reinforcement-learning environment interface [12]; we use its seeded
`reset`/`step` contract and separate termination flags so the environment can
be validated independently of the learner. Our simulator sacrifices fidelity
for speed, transparent assumptions, and deterministic test cases.

### Synthesis and limitations

Prior work establishes several relevant methods: tabular and deep value
learning, policy-gradient control, batch learning for thermostatic loads, and
physics-based building simulation [1–12]. The contribution here is narrower:
a small inspectable environment that puts occupancy, time-varying prices,
comfort, and switching in one reproducible experiment, followed by a direct
test of whether an inspectable Q-table learns anticipatory actions. It is not
a new RL algorithm and does not claim measured energy savings or real-home
validity.

Several limitations must remain explicit in the final report. The synthetic
weather and occupancy schedules are known in advance; the RC parameters and
HVAC powers are illustrative rather than calibrated; one-hour transitions and
linear heat flow omit important building behavior; the state discretization
can limit Q-table coverage; and chosen reward weights affect the policy.
Results will be reported across held-out seeds with variability and with
negative outcomes included. A high-fidelity simulator, uncertain occupancy,
continuous control, and transfer between buildings are possible future work,
not evidence available from this project.

## References

1. Watkins, C. J. C. H., and Dayan, P. “Q-learning.” *Machine Learning*,
   8(3–4), 279–292, 1992. https://doi.org/10.1007/BF00992698.
2. Sutton, R. S., and Barto, A. G. *Reinforcement Learning: An Introduction*,
   2nd ed. MIT Press, 2018. http://incompleteideas.net/book/the-book-2nd.html.
3. Mnih, V., Kavukcuoglu, K., Silver, D., et al. “Human-level control through
   deep reinforcement learning.” *Nature*, 518, 529–533, 2015.
   https://doi.org/10.1038/nature14236.
4. Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. “Continuous control with
   deep reinforcement learning.” *International Conference on Learning
   Representations*, 2016. https://arxiv.org/abs/1509.02971.
5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O.
   “Proximal policy optimization algorithms.” arXiv:1707.06347, 2017.
   https://arxiv.org/abs/1707.06347.
6. Vázquez-Canteli, J. R., and Nagy, Z. “Reinforcement learning for demand
   response: A review of algorithms and modeling techniques.” *Applied
   Energy*, 235, 1072–1089, 2019.
   https://doi.org/10.1016/j.apenergy.2018.11.002.
7. Wang, Z., and Hong, T. “Reinforcement learning for building controls: The
   opportunities and challenges.” *Applied Energy*, 269, article 115036, 2020.
   https://doi.org/10.1016/j.apenergy.2020.115036.
8. Barrett, E., and Linder, S. “Autonomous HVAC control, a reinforcement
   learning approach.” In *Machine Learning and Knowledge Discovery in
   Databases: European Conference, ECML PKDD 2015*, pp. 3–19. Springer, 2015.
   https://doi.org/10.1007/978-3-319-23461-8_1.
9. Ruelens, F., Claessens, B. J., Vandael, S., De Schutter, B., Babuška, R.,
   and Belmans, R. “Residential demand response of thermostatically
   controlled loads using batch reinforcement learning.” *IEEE Transactions
   on Smart Grid*, 8(5), 2149–2159, 2017.
   https://doi.org/10.1109/TSG.2016.2517211.
10. Bacher, P., and Madsen, H. “Identifying suitable models for the heat
    dynamics of buildings.” *Energy and Buildings*, 43(7), 1511–1522, 2011.
    https://doi.org/10.1016/j.enbuild.2011.02.005.
11. Crawley, D. B., Lawrie, L. K., Winkelmann, F. C., Buhl, W. F., Huang,
    Y. J., Pedersen, C. O., Strand, R. K., Liesen, R. J., Fisher, D. E.,
    Witte, M. J., and Glazer, J. “EnergyPlus: Creating a new-generation
    building energy simulation program.” *Energy and Buildings*, 33(4),
    319–331, 2001. https://doi.org/10.1016/S0378-7788(00)00114-6.
12. Towers, M., Kwiatkowski, A., Terry, J., et al. “Gymnasium: A standard
    interface for reinforcement learning environments.” arXiv:2407.17032,
    2024. https://arxiv.org/abs/2407.17032.
