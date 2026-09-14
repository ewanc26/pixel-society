# Pixel Society

**Set the starting conditions. Let the pixels decide what happens next.**

A native C++20 artificial-life simulation of an autonomous society. Seed a
landscape, choose its terrain shape, scale, population, resources, cooperation
and hazard level, then watch citizens forage, build, farm, share, form new
generations and struggle to survive. Once the simulation begins, the interface
only observes.

A shared **society core** (92 → 2048 → 2560 → 2560 → 13,
**12,025,357 trainable parameters**) reads the living population's average
sensor view once each tick and broadcasts one advisory signal per intention.
Every citizen combines those 13 advisory channels with its own live view and
runs a personal **105 → 56 → 28 → 13 neural network** (7,909 parameters) that
ranks intentions, then learns from the consequences. The 92 raw signals include
individual needs and inventory, the current tile, nearby resources and hazards,
settlement and neighbourhood patterns, seasonal and population context, plus
live clan reserve, storage-route and courier-skill data. The desktop world
advances at **five game ticks per second**. Every recorded event has an integer
importance score from **0 to 100**.

The terrain, structures, citizens, charts and bitmap lettering are drawn from
pixels. There are no downloaded graphics, fonts, models, AI services or API keys.

## Build and run

On macOS:

```sh
brew install cmake sdl2
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/pixel-society
```

You can also double-click `Launch Pixel Society.command` after installing those
dependencies. On Ubuntu or Debian, install `build-essential cmake libsdl2-dev`,
then use the same CMake commands. Requires CMake 3.20+, a C++20 compiler, and
SDL2 2.0.18+. Windows has not been verified.

Choose the initial conditions on the setup screen and begin. After that,
citizens decide what to do; there are no building tools, orders, speed controls,
or resource injections. Selecting a citizen and changing an observation layer
only changes what you see. Press **M** or **Esc** for the observer menu: resume,
return to world setup, open the guide or quit. The menu pauses the simulation
and does not alter its rules or state.

## A society, tick by tick

- Five terrain shapes — island, archipelago, inland sea, highlands and
  riverlands — are available at five scales from 48 × 32 to 192 × 128. Each
  is deterministic from the chosen seed and supplies distinct routes, coasts,
  forests and renewable food.
- Individual citizens experience hunger, thirst, fatigue, health and social needs.
- Neural outputs rank thirteen intentions: wander, gather, eat, drink, rest,
  chop, build, farm, share, socialize, reproduce, attack and haul.
- Citizens navigate the land and consume actual shared world resources. Homes
  and farms arise from their own work and timber inventories.
- Once a clan has shelter, a well-stocked builder can establish a **storehouse**.
  Citizens learn logistics from actual deliveries, following their clan's route
  field to stock food and wood reserves or provision themselves from them.
- Every civilization's own trained society core observes the population average
  every tick and advises its citizens, so personal policies coordinate around
  society-wide crowding, food security, construction and conflict pressure.
  Different civilization seeds train different cores.
- Cooperation, competition, inherited traits and learned policies influence
  how the population develops. Children inherit a mutated parental network.
- Seasonal production and environmental hazards change conditions over time.
- Fires are disturbance events: a stand-replacing burn scars the land and then
  regenerates as grass clearings with boosted ash fertility, reshaping the map
  season after season.
- Crowded, unhappy settlements can kindle density-driven **plagues**. Disease
  spreads between adjacent citizens, housed citizens self-quarantine and spread
  it far less, survivors gain temporary immunity, and immunity slowly wanes.
- Social contact doubles as **cultural transmission**: the less experienced
  citizen blends its policy toward the more seasoned one, so elders and
  veterans pass learned behaviour onward between generations.
- Every second the world marks each land tile with the **civilisation territory**
  that claims it: the clan whose nearest owned home, farm or resident wins the
  ground. The claim map reproduces exactly for a given seed and thread count,
  and the observer can sketch the resulting frontiers.
- The observer shows live population, wellbeing, neural action values, sensor
  group summaries and data quality, history, and scored events. The population
  is bounded at 256 for predictable performance.

This is a compact artificial-life society, with authored physical and social
rules and learned individual decisions. It does not simulate language, formal
governments, currency or market prices. It has no victory condition: survival,
growth, conflict and extinction are outcomes to observe.

## What the AI actually does

Every civilization trains its own **society core** (92 → 2048 → 2560 → 2560 →
13, **12,025,357 parameters**) from its world seed, so different seeds produce
genuinely different collective instincts. Each tick it reads the live
population-average observation and returns thirteen advisory signals. Each citizen
appends those signals to its own 92 live observations (its body and supplies,
local terrain and resources, reachable world features, nearby social conditions,
longer-running world context, clan reserve and courier context), and its personal
**105 → 56 → 28 → 13**
network learns with online backpropagation. Every intention comes from neural
action values or exploratory sampling of physically legal actions. The
simulation executes movement and interactions and returns a reward; the citizen
updates its own network.

The starting curriculum is an authored prior. The model is a small reinforcement
learning controller, and does not claim human intelligence. Read the
[AI design](docs/AI.md) for the inputs, training targets, learning rule and limits.

## Reproducible experiments

The headless runner uses the same C++ simulation without graphics. By default it
runs fixed ticks as fast as possible; use `--realtime` for five ticks per wall-clock
second. One simulated day is 300 ticks (60 seconds at normal pace). Heavy
vectorized work — society-core inference, mean observations and route maps — runs
on a fixed-size worker pool (`--threads N`, default: all cores). The pool's work
partition does not depend on the thread count, so the simulation trace and digest
are identical whether it runs with one thread or many.

```sh
./build/pixel-society --headless --seed 42 --ticks 6000 --events events.jsonl
./build/pixel-society --headless --seed 42 --ticks 50 --realtime
./build/pixel-society --headless --seed 123 --founders 72 --threads 2
./build/pixel-society --headless --seed 123 --advisor-every 5 --ticks 1200
./build/pixel-society --headless --seed 123 --shape archipelago --size large --ticks 1200
./build/pixel-society --seed 123 --founders 72 --fertility 0.8 --cooperation 0.9
./build/pixel-society --help
```

Headless output is a JSON summary including the selected terrain shape,
dimensions and advisor cadence, population, births, deaths, construction,
storehouses, food and wood reserves, deliveries, neural decisions, learning
updates, the worker count and a reproducibility digest. The optional JSONL event
log contains every recorded event, its tick, coordinates, category, text and
score. The desktop keeps a
bounded recent journal in memory. An event's importance is separate from neural
action values or learning rewards.

The shared society core refreshes its advice every tick by default. Headless
experiments can use `--advisor-every N` (1–300) to retain the most recent
collective advice for N ticks between refreshes. This is useful when comparing
long scenarios on smaller machines; it is part of the reproducibility contract,
so a different cadence produces a different world trace.

The same seed and settings reproduce the same trajectory within the same build,
independent of the worker count. Floating-point math and random distributions
can differ across platforms. Worlds are currently in-memory experiments; closing
the application discards the world, and there is no save/load feature.

The desktop's **BORDERS** observation layer tints every claimed tile with its
owning civilisation's colour and draws a bright frontier between regions claimed
differently (or not yet claimed). Territory is derived only for display and the
reproducibility digest; citizens never see it, so observing the map does not
change the experiment.

For machines without SDL:

```sh
cmake -S . -B build-core -DPIXEL_SOCIETY_GUI=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-core --parallel
./build-core/pixel-society --headless --ticks 3000
```

## Verification

```sh
ctest --test-dir build --output-on-failure
./build/pixel-society --smoke-test --screenshot build/observatory.bmp

cmake -S . -B build-sanitize -DPIXEL_SOCIETY_GUI=OFF \
  -DPIXEL_SOCIETY_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

Tests cover 92 finite normalized observations across individual, world and
economy change, a high-index sensor that changes learned action values, deeper
neural learning and decision changes, legal action masking, inheritance and
cultural imitation, five-tick timing with retained frame debt, seeded
determinism, autonomous construction, births and shared storehouses, all five
terrain shapes and world-size presets, the emergent disease, fire-succession and
transmission mechanics, deterministic civilisation territory and its visible
borders, resource and population invariants, event scores, and extreme starting
conditions. The society-core constructor is asserted to be over ten million
parameters, its training is deterministic across identical seeds, and a
dedicated check proves per-citizen policies react to advisory channels. The
interface smoke test exercises setup, the guide, layers and the pausing observer
menu, then compares the resulting world with an unattended simulation to detect
accidental player influence.

The CI workflow runs macOS and Linux builds, plus an address/undefined-behaviour
sanitizer build. See [verification evidence](docs/VERIFICATION.md).

## Source and license

| File | Responsibility |
| --- | --- |
| `src/simulation.*` | World generation, citizens, environment, interactions and events |
| `src/neural.*` | Society-core training, inference, backpropagation, founder curriculum and inheritance |
| `src/ticker.hpp` | Fixed 200 ms simulation clock |
| `src/ui.cpp` | Pixel rendering, starting conditions and observation controls |
| `src/main.cpp` | CLI and headless experiments |
| `tests/core_tests.cpp` | Learning, timing and society verification |

The simulation and neural network have no third-party dependencies. The desktop
uses SDL2 under its zlib license. The game's code and original procedural artwork
are open source under the [MIT license](LICENSE).
