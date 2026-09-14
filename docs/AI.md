# Citizen neural AI

Every civilization owns a trainable **society core**: a C++ feedforward network
with **82 inputs → 2048 tanh units → 2560 tanh units → 2560 tanh units → 12
linear outputs**, fitted from that civilization's world seed with a deterministic
synthetic curriculum. It has **12,002,316 trainable scalar parameters**,
including biases. Once per tick it reads the live population-average observation
vector and returns normalized advisory signals for the twelve intentions.

Each living citizen owns a smaller trainable personal network:
**94 inputs → 56 tanh units → 28 tanh units → 12 linear Q values**,
**7,264 parameters**. The 94 inputs are the citizen's own 82 live observations
plus the 12 advisory channels from its civilization's core. The personal network
evaluates one set of action values every game tick and learns from the
consequences of its own selected action.

This is an artificial-life controller, not a language model, a human-level
intelligence, or a model trained on data about real societies. Its relationships,
infrastructure and population arise from limited local agents interacting with
the simulated world.

## Observation contract

`InputCount` is 82. Every input is finite and normalized to `[0, 1]`; nonfinite
values become zero and values outside the interval are clamped. Indices 0–19
are retained from the first model so saved simulation assumptions and basic
survival semantics remain stable.

| Indices | Measurements |
| --- | --- |
| 0–4 | Hunger, thirst, fatigue (`1 - energy`), social need (`1 - social`), health deficit (`1 - health`) |
| 5–9 | Food inventory / 10, wood inventory / 10, cooperation trait, aggression trait, age / 6,000 ticks |
| 10–15 | Reachable proximity to forage, water, forest, home, farm and another citizen (`1 / (1 + path distance)`) |
| 16–19 | Current-tile fire, season yield, birth readiness, current-tile fertility |
| 20 | Most recent bounded reward mapped from `[-2, 2]` to `[0, 1]` |
| 21–27 | Population / 256; home capacity coverage (homes × 3 / population); farm capacity coverage (farms × 2 / population); communal food security (carried food / population × 4); mean wellbeing; mean cooperation; highest generation / 8 |
| 28–31 | One-hot season: spring, summer, autumn, winter |
| 32–36 | Current-tile food / 8, wood / 5, traffic, is-home, is-farm |
| 37–40 | One-hot terrain: sand, grass, forest, rock. Citizens never occupy water, so it does not use an input bit. |
| 41–51 | Radius-2 square means: food / 8, wood / 5, fertility, fire, traffic, home density, farm density, people-per-tile density, living-neighbour mean cooperation, mean aggression, Simpson clan diversity |
| 52–62 | The same eleven measurements across a radius-5 square |
| 63–67 | Nearest reachable living citizen's hunger, food / 10, cooperation, aggression and social need; all zero when none exists |
| 68–69 | Proximity to nearest reachable same-clan and different-clan citizen (`1 / (1 + Manhattan distance)`), zero when absent |
| 70–81 | One-hot previous action: Wander, Gather, Eat, Drink, Rest, Chop, Build, Farm, Share, Socialize, Reproduce, Attack |

The radius summaries give a citizen both immediate and neighbourhood-scale data:
it can distinguish a bare tile within a food-rich settlement from a food-rich
tile surrounded by danger, crowding or poor infrastructure. Global coverage,
security and generation measurements let its learned policy respond to the
state of the society instead of only its own needs. The previous-action bits
are state, not a command: the network can learn either to continue or abandon
an intention.

## Society advisory core

`CoreAdviceCount` is 12, matching one signal per intention. `BrainInputCount`
is 94 = the 82 observation bits plus 12 advisory channels. Each tick the
simulation measures the mean observation vector of the living population, runs
the society core on that average, clamps each raw output to `[-3.5, 3.5]` and
rescales to `[0, 1]`, and appends the twelve signals below the citizen's own
observation bits. `compose(observation, advice)` forms the full 94-wide brain
input used for every choice, learning update and mutation check.

`makeSocietyCore(seed)` trains one core per civilization: every `Simulation`
fits its own core from its world seed, so different seeds give different
civilizations distinct collective instincts while the same seed always
reproduces the identical core. The trained core is cached per seed, so a
restarted experiment or the observer's reference copy reuses the exact same
core, and the cache has a bounded size to keep a long-lived process stable.
Training uses Xavier initialization and a multi-output curriculum of 64 epochs
over two passes of a tribe-mean sample set at learning rates 0.05 and 0.025,
fitting all 12 outputs together. Because the core observes the flattened
population average, its signals let scattered personal policies coordinate
around society-wide crowding, food security, construction and conflict
pressure. The core is shared by all of a civilization's citizens and is
read-only after training; citizens never train it.

The two wide hidden layers dominate the forward pass cost and are split across
a fixed-size worker pool set once per process (`Config::threads`, default all
cores). Each output unit is an independent dot product over its own weight row,
so the parallel forward pass is bit-for-bit identical to the serial one; the
pool's chunk boundaries depend only on a constant row block size, never on the
worker count. Mean observations in `currentAdvice()` are likewise gathered in
parallel and folded in a fixed block order, which keeps the whole simulation
trace reproducible for any number of threads.

Founder brains are prepared with `makeFounderBrain(seed, core)`. The
founder curriculum cycles the core over twelve similarly-programmed
population-average scenarios so early networks learn to use the advisory
channels as well as their own 82 sensors. Advisory signals are live state, not
commands: the personal network still chooses among physically legal actions and
learns from the outcome of whatever it actually did.

## Intentions and execution

Outputs are ordered `Wander`, `Gather`, `Eat`, `Drink`, `Rest`, `Chop`, `Build`,
`Farm`, `Share`, `Socialize`, `Reproduce`, `Attack`. A value is a relative Q
estimate, not a probability or event score.

The controller chooses epsilon-greedily among physically legal actions: it
usually takes the legal action with the highest neural value and sometimes
samples a uniformly random legal action. The legality mask excludes impossible
actions from both paths. An empty mask is a caller error; the world always
offers a living citizen at least one legal action.

There is no hunger threshold, urgency rule, or scripted fallback in the choice
function. Movement, collision, resource use and interactions are world
mechanics after the neural decision. Choosing Drink may involve several ticks
of travel, for example, and the brain is evaluated again on every tick.

## Founder preparation and inheritance

`makeFounderBrain(seed, core)` uses Xavier initialization and a compact
deterministic synthetic curriculum over the founder's own sparse observation
space, keyed to its civilization core's advisory channels. It creates 3,072 stratified
observations (two passes of 1,536): ordinary life, scarcity, emergencies,
construction opportunities, farming, sharing, social contact, reproduction and
conflict. The examples use the full 82-bit observation layout plus the core's
advisory signals, including categorical seasons, terrain and prior actions,
correlated local and global conditions, and deliberately repeated rare
scenarios.

Each curriculum optimizer step evaluates the deep network once and fits all
12 output targets together. This multi-output update replaces the old costly
per-action bootstrap loop while still giving founders a useful survival and
settlement prior. Targets reward survival, fire avoidance, resource gathering,
construction, cultivation, support, social contact and viable reproduction;
cooperation, aggression, nearby citizens and clan context influence the social
targets. These targets exist only while preparing founders. Runtime selection
uses the network weights.

Founders copy the prepared policy and receive small individual mutations.
Children copy a parent's current learned policy and receive Gaussian mutations,
so both lifetime learning and inheritance change later generations. A brain's
digest includes every parameter bit and its update count; its fingerprint
includes every layer and bias. With the same build, seed, choices and mutation
stream, copied and mutated policies are deterministic.

## Online reinforcement learning

After an action executes, the citizen receives a bounded reward and updates its
own selected Q value with temporal-difference learning:

```text
target = reward + 0.85 × max Q(next composed input, legal next action)
```

Terminal transitions omit the future term. Rewards are bounded to `[-2, 2]`,
targets to `[-4, 4]`, and online learning uses a rate of `0.008`. `Brain::train`
computes the squared prediction error before modifying the model, clips the
output gradient to `[-2, 2]`, then backpropagates it through the 28-unit and
56-unit tanh layers using weights from the same forward pass. It updates the
selected output head and both shared hidden layers. Parameter values are bounded
to keep long autonomous runs finite.

Founder multi-output training records one optimizer update per observation.
The public `train` and `learn` APIs each record one update per call. Inference,
training and mutation use fixed-size arrays with no per-decision heap allocation,
which keeps the 256-citizen population practical at five game ticks per second.

The model has no replay buffer, target network, recurrent hidden state, language,
institution planner or persistent relationship embedding. Online learning can
interfere with earlier behaviour. A society can cooperate, fight, overbuild,
exhaust resources or die out; survival and growth are outcomes rather than
guarantees. Reproducibility is within a build: floating-point math and standard
normal-distribution implementations can differ across toolchains.

## Event importance is separate

Every recorded world event has an integer importance score from **0 to 100**.
That score describes simulated impact and is independent of neural Q values and
reinforcement rewards. Q values rank intentions; they do not rate events.
