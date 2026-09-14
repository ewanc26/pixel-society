# Verification record

This record describes the verification plan and the historical checks run for
the initial public release on macOS with AppleClang, CMake and SDL2. It is a
reproducibility aid, not a promise that every random starting condition will
thrive.

## Rich neural observation checks

The controller is deliberately tested through its public architecture
constants rather than only described that way. The core suite asserts the
**society core** is `92 → 2048 → 2560 → 2560 → 13` with
`SocietyCore::parameterCount() >= 10,000,000` (actual value: 12,025,357), and
that the personal network is `105 → 56 → 28 → 13` (7,909 parameters) over 92
observation bits plus 13 core advisory channels. It then exercises all 92
observation positions through a normal seeded world. It checks that every value
is finite and normalized, that substantial data exists in both the lower and
high-index halves at the start, and that broad individual/world sampling
activates and changes a large portion of the vector during autonomous ticks.
Empty buildings or the absence of a disaster are accepted as legitimate initial
conditions, so the check measures coverage rather than demanding every optional
condition at tick zero.

The deep-tail regression check creates two observations that differ only at
input 91. It trains a single action value toward opposite targets and requires
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
constants, six groups covering the ordered observation contract, and a live
finite/normalized sensor count. Its SDL smoke scenario verifies that the
inspector, selection, layers, guide and pausing observer menu remain
observation-only after start.

The core suite also runs a seeded autonomous economy scenario. It requires
builders to establish storehouses, the neural `Haul` intention to make real
deliveries, clan reserves to remain within per-storehouse capacity, and headline
statistics, tile structures and clan ledgers to agree. It then verifies that the
nine economy inputs and courier experience become live rather than remaining
declared but disconnected fields.

## Landscape and scale checks

The setup surface and headless runner expose five deterministic terrain shapes:
Island, Archipelago, Inland Sea, Highlands and Riverlands. Each can run at Tiny
(48 × 32), Small (64 × 40), Classic (96 × 64), Large (128 × 80) or Huge
(192 × 128) scale. The core suite creates every size and verifies that tile
storage, bounds and usable land match the advertised dimensions. It also builds
each terrain shape twice from the same seed, requires matching digests, and
checks its coastline and terrain makeup. Highlands must contain rocky ridges;
seascapes must leave meaningful open water.

The desktop smoke flow changes founders, terrain shape and world size before it
starts the experiment. It then verifies that the resulting draft is frozen and
that the observer still preserves an unattended reference world's digest.

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

## Current validation

The current Release build completed all six CTest entries in **32.76 seconds**.
That includes the core suite, CLI help and invalid-input checks, a headless
run, and the native SDL observer smoke flow. The latter confirmed selection,
all four map layers, the guide, the new pause/resume menu, frozen menu time and
an untouched reference-world digest:

```text
UI smoke: PASS; ticks=12; expected_at_5Hz=12; setup=1; setup_frozen=1;
setup_locked=1; selection=1; layers=1; guide=1; menu=1; pause_frozen=1;
resumed=1; observer_preserved_world=1
```

The AddressSanitizer and UndefinedBehaviorSanitizer configuration completed all
five non-GUI CTest entries in **56.28 seconds**. Its 160-tick seeded economy
case established four storehouses, retained real food and wood reserves, and
executed deliveries through the `Haul` action. The headless summary now reports
these values directly alongside the existing construction and neural fields.

## Historical pre-economy baseline

The following measurements are retained from the earlier 82-input / 12-action
release for comparison. They are not assertions about the current 92-input,
13-action build.

## Automated checks

The regular desktop build completed:

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

6 / 6 tests passed in 34.81 seconds
```

Those tests cover neural inference and backpropagation, the ten-million-plus
parameter per-civilization society core and its deterministic training,
advice-channel sensitivity, action masking, mutation/inheritance and social
imitation, the fixed 200 ms clock and catch-up debt, seeded determinism,
worker-count invariance, the emergent disease/fire/transmission mechanics,
deterministic civilisation territory and its visible borders, autonomous
construction and reproduction, every terrain shape and world-size preset,
score bounds, world invariants, invalid input and the SDL observer flow.

The instrumented core build also completed:

```text
cmake -S . -B build-sanitize -DPIXEL_SOCIETY_GUI=OFF \
  -DPIXEL_SOCIETY_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure

5 / 5 tests passed in 58.87 seconds
```

The sanitizer configuration keeps the same test categories but uses a shorter
sustained society run. Long invariant coverage samples the still-real
12-million-parameter society core at a coarser configured cadence, while the
focused disease/fire scenario and UI smoke retain the normal every-tick
cadence. The sanitizer build also exercises the worker pool: simulations run
with real threads under AddressSanitizer and UBSan. The CI workflow repeats the
regular build on macOS and Linux and runs the sanitizer configuration on Linux.

## Autonomous run

The following headless run executes the same simulation used by the desktop
observer:

```sh
./build/pixel-society --headless --seed 42 --ticks 6000 --events events.jsonl
```

It represents 1,200 simulated seconds (20 game days). On the release build it
finished on this machine in 24.638 seconds wall-clock with ten worker threads,
256 citizens, 270 births, 62 deaths, generation 4, 94.57% wellbeing, 1,353,768
neural decisions and 1,353,776 learning updates, and a civilization seed-42
society core of 12,002,316 parameters. Every one of the twelve intentions had a
nonzero action count. The JSONL log contained 137,615 scored events; a range
check found zero scores outside 0–100. Sharing, fire, plague and recovery
events now shape the chronicle: the emergent mechanics raise death and event
volume above the earlier snapshot while the society keeps reproducing under
the added disease pressure.

Two independent runs with those same settings produced the same deterministic
world digest: `18428606461124059589`. The digest is independent of the worker
pool as well as of wall-clock time: the same world run with one, ten or 256
threads reported the identical digest. The territory labels now contribute to
the digest (and therefore shift it from the mechanics-only snapshot), while
the population statistics remain unchanged because the claim map is derived
for display only. Because each civilization trains its own core from its world
seed, this snapshot's digest and population statistics differ from the earlier
shared-core release: the seed-42 civilization's core and the world it spawns
are part of a single deterministic seed contract. The selected landscape shape
and size are also part of that contract.

## Expanded-world spot check

The headless runner also completed 300 ticks of a Huge (192 × 128)
Archipelago at four workers in 0.907 seconds wall-clock. The resulting society
held 94 residents, had 215 homes and 175 farms, and reported the same
12,002,316 core parameters as the Classic world. This is an evidence point for
the larger map path rather than a claim that every terrain/size/seed combination
will thrive equally.

## Native observer check

The native SDL smoke check changed founders, terrain shape and world size in
setup, started the simulation, selected a citizen, changed every observation
layer, opened and closed the guide, and tried setup keys after the world was
live. It passed at 11 ticks with an expected 11 ticks at five per second. It
confirmed that the setup stays frozen before start and that observer input
preserves a same-tick reference world digest.

```sh
./build/pixel-society --smoke-test --screenshot build/observatory.bmp
```
