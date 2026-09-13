# Citizen neural AI

Each citizen owns a small, trainable neural network written in C++ without a
machine learning dependency: **20 inputs → 24 tanh hidden units → 12 linear
action values**. These values rank intentions. The network selects every
intention during the simulation; the world then executes that intention using
ordinary movement, collision and resource rules.

This is an artificial-life controller with online reinforcement learning. It is
not a language model, a human-level intelligence, or a model trained on real
societies. Relationships, infrastructure and population emerge from interactions
between these limited individual controllers and the simulated environment.

## Observation contract

All inputs are finite numbers normalized to `[0, 1]` in this order:

| Index | Input |
| --- | --- |
| 0 | Hunger |
| 1 | Thirst |
| 2 | Fatigue (one minus energy) |
| 3 | Social need (one minus social satisfaction) |
| 4 | Health deficit |
| 5 | Carried food divided by 10 |
| 6 | Carried wood divided by 10 |
| 7 | Cooperation trait |
| 8 | Aggression trait |
| 9 | Age divided by 6,000 ticks |
| 10 | Proximity to forage |
| 11 | Proximity to water |
| 12 | Proximity to forest |
| 13 | Proximity to a home |
| 14 | Proximity to a farm |
| 15 | Proximity to another citizen |
| 16 | Fire on the current tile |
| 17 | Seasonal yield |
| 18 | Birth readiness |
| 19 | Fertility of the current tile |

Proximity is `1 / (1 + distance)` for an available target, and zero when no target
is found. A high proximity therefore means a nearby target. Values outside the
contract are clamped; nonfinite input values become zero.

## Intentions and execution

Output order is `Wander`, `Gather`, `Eat`, `Drink`, `Rest`, `Chop`, `Build`, `Farm`,
`Share`, `Socialize`, `Reproduce`, `Attack`. The controller uses an epsilon-greedy
choice: it usually selects the highest network value among legal intentions,
and sometimes explores a uniformly sampled legal intention.

The legality mask represents physical prerequisites, such as possessing food
to eat or enough timber to build. Illegal intentions are excluded from both
greedy and exploratory selection. An empty mask is a caller error; the world
always provides at least one legal intention. There is no hunger threshold or
script in the choice function that replaces the neural decision.

Movement toward an intention's target and resolving an action are world
mechanics. For example, choosing Drink can take several ticks of travel before
reaching water, and choosing Reproduce does not guarantee a birth. The network
is evaluated again each game tick, so an intention can change before arrival.

## Founder preparation and inheritance

`makeFounderBrain(seed)` first initializes Xavier-distributed weights and then
trains on 8,000 reproducible synthetic observations, with a target for each of
the 12 actions. Targets teach basic survival, avoiding fire, gathering supplies,
construction, farming, sharing, social contact and reproduction. Aggression and
cooperation affect their respective targets. The broad curriculum includes both
quiet daily life and emergencies.

These training targets are authored priors, not knowledge discovered from
scratch. They are evaluated only while preparing the founder model; subsequent
choices use the actual network weights. Founders share this prepared starting
policy with individual mutations. Children inherit a parent's learned policy
with Gaussian mutations, so behaviour can change through both lifetime learning
and inheritance. The random seed controls initialization and mutation.

## Learning during the simulation

After an intention executes, its citizen receives a reward reflecting the
result. The network is updated using the temporal-difference target:

```text
target = reward + 0.85 × max Q(next observation, legal next intention)
```

Terminal transitions omit the future term. Rewards are bounded to `[-2, 2]`,
targets to `[-4, 4]`, and the online learning rate is `0.008`. Backpropagation
updates the selected output's weights and bias and the shared hidden layer,
using the derivative of tanh and the original output weights from that forward
pass. The error gradient is clipped to `[-2, 2]`, and parameters are bounded to
keep long runs finite. `Brain::train` reports the squared prediction error
before its update. The update counter includes bootstrap training.

The model has no replay buffer, separate target network, recurrent memory,
language, institutions planner, or persistent relationship embedding. Online
learning with a small shared hidden layer can interfere with previous skills;
survival and population growth are not guaranteed. The society can cooperate,
fight, overbuild, exhaust resources or die out. Seeded runs are reproducible
within a build; floating-point libraries and standard random distributions may
produce different outcomes on different toolchains.

## Event importance is separate

Each recorded world event has an integer importance score from **0 to 100**.
That scale describes the event's simulated impact and is separate from a
citizen's neural Q values or reinforcement reward. Q values are relative
predictions used to select intentions, not probabilities or event ratings.
