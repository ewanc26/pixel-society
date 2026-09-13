# Pixel Society

**Set the starting conditions. Let the pixels decide what happens next.**

A native C++20 artificial-life simulation of an autonomous society. Seed an
island, choose its population, resources, cooperation and hazard level, then
watch its citizens forage, build, farm, share, form new generations and struggle
to survive. Once the simulation begins, the interface only observes.

Every citizen has an individual **20 → 24 → 12 neural network** that selects
their intentions and learns from the consequences. The desktop world advances
at **five game ticks per second**. Every recorded event has an integer
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
only changes what you see. Close the window to finish the experiment.

## A society, tick by tick

- A seeded 96 × 64 world supplies land, water, forests and renewable food.
- Individual citizens experience hunger, thirst, fatigue, health and social needs.
- Neural outputs rank twelve intentions: wander, gather, eat, drink, rest, chop,
  build, farm, share, socialize, reproduce and attack.
- Citizens navigate the land and consume actual shared world resources. Homes
  and farms arise from their own work and timber inventories.
- Cooperation, competition, inherited traits and learned policies influence
  how the population develops. Children inherit a mutated parental network.
- Seasonal production and environmental hazards change conditions over time.
- The observer shows live population, wellbeing, neural action values, history,
  and scored events. The population is bounded at 256 for predictable performance.

This is a compact artificial-life society, with authored physical and social
rules and learned individual decisions. It does not simulate language, formal
governments or a full human economy. It has no victory condition: survival,
growth, conflict and extinction are outcomes to observe.

## What the AI actually does

Each citizen runs inference and online backpropagation. A reproducible synthetic
curriculum prepares the founder policy with basic survival knowledge. During
play, every intention comes from neural action values or exploratory sampling
of physically legal actions. The simulation executes movement and interactions
and returns a reward; the citizen updates its own network.

The starting curriculum is an authored prior. The model is a small reinforcement
learning controller, and does not claim human intelligence. Read the
[AI design](docs/AI.md) for the inputs, training targets, learning rule and limits.

## Reproducible experiments

The headless runner uses the same C++ simulation without graphics. By default it
runs fixed ticks as fast as possible; use `--realtime` for five ticks per wall-clock
second. One simulated day is 300 ticks (60 seconds at normal pace).

```sh
./build/pixel-society --headless --seed 42 --ticks 6000 --events events.jsonl
./build/pixel-society --headless --seed 42 --ticks 50 --realtime
./build/pixel-society --seed 123 --founders 72 --fertility 0.8 --cooperation 0.9
./build/pixel-society --help
```

Headless output is a JSON summary including population, births, deaths,
construction, neural decisions, learning updates and a reproducibility digest.
The optional JSONL event log contains every recorded event, its tick, coordinates,
category, text and score. The desktop keeps a bounded recent journal in memory.
An event's importance is separate from neural action values or learning rewards.

The same seed and settings reproduce the same trajectory within the same build.
Floating-point math and random distributions can differ across platforms.
Worlds are currently in-memory experiments; closing the application discards
the world, and there is no save/load feature.

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

Tests cover real neural learning and decision changes, legal action masking,
inheritance, five-tick timing with retained frame debt, seeded determinism,
autonomous construction and births, resource and population invariants, event
scores, and extreme starting conditions. The interface smoke test injects SDL
input through setup and observation controls, and compares the resulting world
with an unattended simulation to detect accidental player influence.

The CI workflow runs macOS and Linux builds, plus an address/undefined-behaviour
sanitizer build. See [verification evidence](docs/VERIFICATION.md).

## Source and license

| File | Responsibility |
| --- | --- |
| `src/simulation.*` | World generation, citizens, environment, interactions and events |
| `src/neural.*` | Inference, backpropagation, founder curriculum and inheritance |
| `src/ticker.hpp` | Fixed 200 ms simulation clock |
| `src/ui.cpp` | Pixel rendering, starting conditions and observation controls |
| `src/main.cpp` | CLI and headless experiments |
| `tests/core_tests.cpp` | Learning, timing and society verification |

The simulation and neural network have no third-party dependencies. The desktop
uses SDL2 under its zlib license. The game's code and original procedural artwork
are open source under the [MIT license](LICENSE).
