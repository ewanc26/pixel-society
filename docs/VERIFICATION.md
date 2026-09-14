# Verification record

This record describes the checks run for the initial public release on macOS
with AppleClang, CMake and SDL2. It is a reproducibility aid, not a promise that
every random starting condition will thrive.

## Automated checks

The regular desktop build completed:

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure

5 / 5 tests passed in 7.11 seconds
```

Those tests cover neural inference and backpropagation, action masking,
mutation/inheritance, the fixed 200 ms clock and catch-up debt, seeded
determinism, autonomous construction and reproduction, score bounds, world
invariants, invalid input and the SDL observer flow.

The instrumented core build also completed:

```text
cmake -S . -B build-sanitize -DPIXEL_SOCIETY_GUI=OFF \
  -DPIXEL_SOCIETY_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel
ctest --test-dir build-sanitize --output-on-failure

4 / 4 tests passed in 46.65 seconds
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
finished with 256 citizens, 222 births, 14 deaths, generation 4, 97.09%
wellbeing, 1,372,594 neural decisions and 1,372,607 learning updates. Every one
of the twelve intentions had a nonzero action count. The JSONL log contained
7,638 scored events; a range check found zero scores outside 0–100.

Two independent runs with those same settings produced the same deterministic
world digest: `771264245566295619`. Wall-clock duration is intentionally not
part of the digest or reproducibility claim.

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
