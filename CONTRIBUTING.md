# Contributing to Pixel Society

Pixel Society is a native C++20 artificial-life simulation of an autonomous society: you set the starting conditions, and the pixels decide what happens next. A shared society core learns to advise every citizen, each of whom runs a small personal network that learns from consequences. Thank you for wanting to help!

## Getting started

```sh
brew install cmake sdl2            # macOS (Ubuntu: build-essential cmake libsdl2-dev)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/pixel-society
```

Requires CMake 3.20+, a C++20 compiler, and SDL2 2.0.18+. Windows has not been verified.

## Notes for this project

- Build with CMake and run CTest before submitting a change; include a seed and the starting conditions when reporting an issue or a PR, because terrain, structures and outcomes are all deterministic from the seed.
- **Once the simulation begins, the interface only observes.** There are no building tools, orders, speed controls or resource injections, and the observer menu must not alter the rules or the world state. Contributions that grant the player power over citizens cut against the design.
- Everything is drawn from pixels — no downloaded graphics, fonts, models, AI services or API keys. Keep it that way.
- The neural architecture is deliberate and small (a society core of ~12M parameters advising citizens with ~8K-parameter personal networks), learning from live consequences; keep it hand-rolled and inspected, not swapped for an opaque off-the-shelf model.

## Before opening a PR

- The documented build passes and CTest is green, and you ran them yourself with a known seed.
- Conventional commits; one focused PR per change.
- AI-assisted PRs are welcome; stand behind every line.

## Licence

MIT (see [LICENSE](LICENSE)). Contributions are accepted under the same terms.