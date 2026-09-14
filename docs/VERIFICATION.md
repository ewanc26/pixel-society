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

5 / 5 tests passed in 61.38 seconds
```

Those tests cover neural inference and backpropagation, the ten-million-plus
parameter per-civilization society core and its deterministic training,
advice-channel sensitivity, action masking, mutation/inheritance and social
imitation, the fixed 200 ms clock and catch-up debt, seeded determinism,
worker-count invariance, the emergent disease/fire/transmission mechanics,
autonomous construction and reproduction, score bounds, world invariants,
invalid input and the SDL observer flow.

The instrumented core build also completed:

```text
cmake -S . -B build-sanitize -DPIXEL_SOCIETY_GUI=OFF \
  -DPIXEL_SOCIETY_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure

4 / 4 tests passed in 93.59 seconds
```

The sanitizer configuration keeps the same test categories but uses a shorter
sustained society run; the release suite runs the 6,000-tick multigeneration
scenario. The sanitizer build also exercises the worker pool: simulations run
with real threads under AddressSanitizer and UBSan. The CI workflow repeats the
regular build on macOS and Linux and runs the sanitizer configuration on Linux.

## Autonomous run

The following headless run executes the same simulation used by the desktop
observer:

```sh
./build/pixel-society --headless --seed 42 --ticks 6000 --events events.jsonl
```

It represents 1,200 simulated seconds (20 game days). On the release build it
finished on this machine in 22.69 seconds wall-clock with ten worker threads,
256 citizens, 306 births, 98 deaths, generation 5, 92.39% wellbeing, 1,353,682
neural decisions and 1,353,688 learning updates, and a civilization seed-42
society core of 12,002,316 parameters. Every one of the twelve intentions had a
nonzero action count. The JSONL log contained 117,286 scored events; a range
check found zero scores outside 0–100. Sharing, fire, plague and recovery
events now shape the chronicle: the emergent mechanics raise death and event
volume above the earlier snapshot while the society keeps reproducing under
the added disease pressure.

Two independent runs with those same settings produced the same deterministic
world digest: `13830909894271019662`. The digest is independent of the worker
pool as well as of wall-clock time: the same world run with one, ten or 256
threads reported the identical digest. Because each civilization trains its own
core from its world seed, this snapshot's digest and population statistics
differ from the earlier shared-core release: the seed-42 civilization's core
and the world it spawns are part of a single deterministic seed contract.

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
