# Verification record

This record describes the verification plan and the historical checks run for
the initial public release on macOS with AppleClang, CMake and SDL2. It is a
reproducibility aid, not a promise that every random starting condition will
thrive.

## Rich neural observation checks

The controller is deliberately tested through its public architecture
constants rather than only described that way. The core suite asserts the
**society core** is `82 → 2048 → 2560 → 2560 → 12` with
`SocietyCore::parameterCount() >= 10,000,000` (actual value: 12,002,316), and
that the personal network is `94 → 56 → 28 → 12` (7,264 parameters) over 82
observation bits plus 12 core advisory channels. It then exercises all 82
observation positions through a normal seeded world. It checks that every value
is finite and normalized, that substantial data exists in both the lower and
high-index halves at the start, and that broad individual/world sampling
activates and changes a large portion of the vector during autonomous ticks.
Empty buildings or the absence of a disaster are accepted as legitimate initial
conditions, so the check measures coverage rather than demanding every optional
condition at tick zero.

The deep-tail regression check creates two observations that differ only at
input 81. It trains a single action value toward opposite targets and requires
the resulting values to separate. This catches a disconnected final feature or
a shallow implementation that silently ignores the high-index observation
data. A companion advisory-channel check trains identical personal networks
against a core advice of 0.05 versus 0.95 on the farm intention and requires
the learned action values to separate, proving that a civilization core's
advisory signals really change per-citizen policy. The society core itself
asserts deterministic training across identical seeds and divergent cores (and
therefore divergent advice) across different civilization seeds. Simulations
with the same world seed reproduce exactly, and simulations seeded differently
diverge at tick zero.
The ordinary learning, action-mask, mutation, temporal-difference and long-world
checks still run alongside these.

The desktop observer now renders both network architectures from the same public
constants, five groups covering the ordered observation contract, and a live
finite/normalized sensor count. Its SDL smoke scenario verifies that this
inspector, selection, layers and guide remain observation-only after start.

Run the full current verification after changes with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/pixel-society --smoke-test --screenshot build/observatory.bmp

cmake -S . -B build-sanitize -DPIXEL_SOCIETY_GUI=OFF \
  -DPIXEL_SOCIETY_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure
```

## Initial public-release snapshot

## Automated checks

The regular desktop build completed:

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

5 / 5 tests passed in 112.45 seconds
```

Those tests cover neural inference and backpropagation, the ten-million-plus
parameter per-civilization society core and its deterministic training,
advice-channel sensitivity, action masking, mutation/inheritance, the fixed
200 ms clock and catch-up debt, seeded determinism, autonomous construction and
reproduction, score bounds, world invariants, invalid input and the SDL observer
flow.

The instrumented core build also completed:

```text
cmake -S . -B build-sanitize -DPIXEL_SOCIETY_GUI=OFF \
  -DPIXEL_SOCIETY_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure

4 / 4 tests passed in 97.78 seconds
```

The sanitizer configuration keeps the same test categories but uses a shorter
sustained society run; the release suite runs the 6,000-tick multigeneration
scenario. The CI workflow repeats the regular build on macOS and Linux and runs
the sanitizer configuration on Linux.

## Autonomous run

The following headless run executes the same simulation used by the desktop
observer:

```sh
./build/pixel-society --headless --seed 42 --ticks 6000 --events events.jsonl
```

It represents 1,200 simulated seconds (20 game days). On the release build it
finished with 256 citizens, 272 births, 64 deaths, generation 4, 94.74%
wellbeing, 1,355,831 neural decisions and 1,355,835 learning updates, and a
civilization seed-42 society core of 12,002,316 parameters. Every one of the
twelve intentions had a nonzero action count. The JSONL log contained 82,816
scored events; a range check found zero scores outside 0–100. Sharing and
conflict dominate the chronicle here as the advice-informed policies cooperate
and compete more actively than the original release.

Two independent runs with those same settings produced the same deterministic
world digest: `12328926939605385823`. Because each civilization now trains its
own core from its world seed, this snapshot's digest and population statistics
differ from the earlier shared-core release: the seed-42 civilization's core
and the world it spawns are part of a single deterministic seed contract. Wall-
clock duration is intentionally not part of the digest or reproducibility
claim.

## Native observer check

The native SDL smoke check ran the setup interaction, started the simulation,
selected a citizen, changed every observation layer, opened and closed the
guide, and tried setup keys after the world was live. It passed at 11 ticks with
an expected 11 ticks at five per second. It confirmed that the setup stays
frozen before start and that observer input preserves a same-tick reference
world digest.

```sh
./build/pixel-society --smoke-test --screenshot build/observatory.bmp
```
