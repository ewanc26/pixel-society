#include "simulation.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pixels {
namespace {
constexpr int Area = WorldWidth * WorldHeight;
constexpr int Directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
constexpr std::size_t EventLimit = 8192;
constexpr std::size_t HistoryLimit = 600;
constexpr int ClanCount = 4;
static_assert(InputCount == 82, "Simulation observation producer follows the 82-feature contract");
float unit(float value) { return std::clamp(value, 0.0f, 1.0f); }
int indexOf(int x, int y) { return y * WorldWidth + x; }
bool inBounds(int x, int y) { return x >= 0 && y >= 0 && x < WorldWidth && y < WorldHeight; }
float randomUnit(std::mt19937& random) {
    // Using raw engine bits also avoids implementation-specific real distributions.
    return static_cast<float>(random() >> 8) / 16777216.0f;
}
int distance(int x, int y, int a, int b) { return std::abs(x - a) + std::abs(y - b); }
bool buildable(const Tile& tile) {
    return tile.structure == Structure::None && tile.fire < .15f &&
           (tile.terrain == Terrain::Grass || tile.terrain == Terrain::Sand || tile.terrain == Terrain::Forest);
}
float comfort(const Citizen& citizen) {
    return unit(.35f * citizen.health + .2f * (1 - citizen.hunger) +
                .2f * (1 - citizen.thirst) + .15f * citizen.energy + .1f * citizen.social);
}
bool birthReady(const Citizen& citizen, const Config& config) {
    return citizen.age >= 600 && citizen.age < 6600 && citizen.birthCooldown == 0 &&
           citizen.food >= 3 && citizen.health > .6f && citizen.energy > .35f &&
           citizen.hunger < .65f && config.fertility > 0;
}
}

Simulation::Simulation(Config config) : core_(makeSocietyCore(config.seed)), config_(config), rng_(config.seed), tiles_(Area) {
    if (config.founders < 2 || config.founders > PopulationLimit ||
        !std::isfinite(config.fertility) || config.fertility < 0 || config.fertility > 1 ||
        !std::isfinite(config.cooperation) || config.cooperation < 0 || config.cooperation > 1 ||
        !std::isfinite(config.hazards) || config.hazards < 0 || config.hazards > 1) {
        throw std::invalid_argument("Founders must be 2..256 and fertility, cooperation, hazards must be finite values in [0,1]");
    }
    citizens_.reserve(PopulationLimit);
    for (int kind = 0; kind < 5; ++kind) {
        destinations_[kind].resize(Area, -1);
        distances_[kind].resize(Area, Area);
    }
    generate();
    rebuildDestinations();
    refreshStatistics();
    history_.push_back({0, stats_.population, stats_.wellbeing});
}

const Tile& Simulation::tile(int x, int y) const {
    if (!inBounds(x, y)) throw std::out_of_range("Tile coordinates outside world");
    return tiles_[indexOf(x, y)];
}

bool Simulation::walkable(int x, int y) const {
    return inBounds(x, y) && tiles_[indexOf(x, y)].terrain != Terrain::Water &&
           tiles_[indexOf(x, y)].terrain != Terrain::Rock;
}

void Simulation::generate() {
    const float phase = randomUnit(rng_) * 6.2831853f;
    for (int y = 0; y < WorldHeight; ++y) {
        const float river = WorldWidth * .48f + 8 * std::sin(y * .12f + phase);
        for (int x = 0; x < WorldWidth; ++x) {
            Tile& ground = tiles_[indexOf(x, y)];
            const float n = randomUnit(rng_);
            const float wetness = std::abs(x - river);
            const float lakeA = std::pow((x - 17.0f) / 7.0f, 2) + std::pow((y - 16.0f) / 5.0f, 2);
            const float lakeB = std::pow((x - 78.0f) / 7.0f, 2) + std::pow((y - 48.0f) / 5.0f, 2);
            if (wetness < 1.35f || lakeA < 1 || lakeB < 1) {
                ground.terrain = Terrain::Water;
                continue;
            }
            ground.fertility = unit(.3f + .55f * config_.fertility + .2f * randomUnit(rng_) - .005f * wetness);
            if (wetness < 2.6f || lakeA < 1.35f || lakeB < 1.35f) ground.terrain = Terrain::Sand;
            else if (n < .025f) ground.terrain = Terrain::Rock;
            else if (n < .34f) ground.terrain = Terrain::Forest;
            else ground.terrain = Terrain::Grass;
            if (ground.terrain != Terrain::Rock) {
                ground.food = ground.fertility * (1.0f + 2.0f * randomUnit(rng_));
                ground.wood = ground.terrain == Terrain::Forest ? 2.5f + 2.5f * randomUnit(rng_) : 0;
            }
        }
    }
    landComponents_.assign(Area, -1);
    int component = 0;
    for (int start = 0; start < Area; ++start) {
        if (!walkable(start % WorldWidth, start / WorldWidth) || landComponents_[start] >= 0) continue;
        std::array<int, Area> queue{};
        int read = 0, write = 0;
        queue[write++] = start; landComponents_[start] = component;
        while (read < write) {
            const int cell = queue[read++];
            for (const auto& direction : Directions) {
                const int nx = cell % WorldWidth + direction[0], ny = cell / WorldWidth + direction[1];
                if (!walkable(nx, ny)) continue;
                const int next = indexOf(nx, ny);
                if (landComponents_[next] >= 0) continue;
                landComponents_[next] = component;
                queue[write++] = next;
            }
        }
        ++component;
    }
    const Brain founderPolicy = makeFounderBrain(config_.seed, *core_);
    const std::array<std::pair<int, int>, ClanCount> camps = {{{30, 22}, {63, 24}, {24, 43}, {70, 43}}};
    for (int i = 0; i < config_.founders; ++i) {
        Citizen citizen;
        citizen.id = nextId_++;
        citizen.clan = i % ClanCount;
        citizen.age = 600 + static_cast<int>(rng_() % 1200);
        citizen.cooperation = unit(config_.cooperation + (randomUnit(rng_) - .5f) * .3f);
        citizen.aggression = unit(.05f + (1 - citizen.cooperation) * .6f + randomUnit(rng_) * .1f);
        citizen.hunger = .1f + randomUnit(rng_) * .3f;
        citizen.thirst = .1f + randomUnit(rng_) * .3f;
        citizen.energy = .65f + randomUnit(rng_) * .3f;
        citizen.social = .3f + randomUnit(rng_) * .5f;
        citizen.food = 3 + randomUnit(rng_) * 3;
        citizen.wood = randomUnit(rng_) * 2;
        auto camp = camps[static_cast<std::size_t>(citizen.clan)];
        bool placed = false;
        for (int trial = 0; trial < Area && !placed; ++trial) {
            citizen.x = std::clamp(camp.first + static_cast<int>(rng_() % 15) - 7, 0, WorldWidth - 1);
            citizen.y = std::clamp(camp.second + static_cast<int>(rng_() % 15) - 7, 0, WorldHeight - 1);
            placed = walkable(citizen.x, citizen.y);
        }
        if (!placed) {
            for (int cell = 0; cell < Area; ++cell) {
                if (walkable(cell % WorldWidth, cell / WorldWidth)) {
                    citizen.x = cell % WorldWidth; citizen.y = cell / WorldWidth; break;
                }
            }
        }
        citizen.brain = founderPolicy;
        citizen.brain.mutate(rng_, .012f);
        citizens_.push_back(std::move(citizen));
    }
    emit("founding", std::to_string(config_.founders) + " founders enter an unbuilt world", .65f);
}

void Simulation::rebuildDestinations() {
    for (int kind = 0; kind < 5; ++kind) {
        auto& destinations = destinations_[kind];
        auto& distances = distances_[kind];
        std::fill(destinations.begin(), destinations.end(), -1);
        std::fill(distances.begin(), distances.end(), Area);
        std::array<int, Area> queue{};
        int read = 0, write = 0;
        for (int i = 0; i < Area; ++i) {
            const Tile& ground = tiles_[i];
            if (!walkable(i % WorldWidth, i / WorldWidth)) continue;
            bool source = (kind == 0 && ground.food >= .2f) ||
                          (kind == 2 && ground.wood >= .2f) ||
                          (kind == 3 && ground.structure == Structure::Home) ||
                          (kind == 4 && ground.structure == Structure::Farm);
            int target = i;
            if (kind == 1) {
                for (const auto& direction : Directions) {
                    const int nx = i % WorldWidth + direction[0], ny = i / WorldWidth + direction[1];
                    if (inBounds(nx, ny) && tiles_[indexOf(nx, ny)].terrain == Terrain::Water) {
                        target = indexOf(nx, ny); source = true; break;
                    }
                }
            }
            if (source) {
                distances[i] = kind == 1 ? 1 : 0;
                destinations[i] = target;
                queue[write++] = i;
            }
        }
        while (read < write) {
            const int cell = queue[read++];
            for (const auto& direction : Directions) {
                const int nx = cell % WorldWidth + direction[0], ny = cell / WorldWidth + direction[1];
                if (!walkable(nx, ny)) continue;
                const int next = indexOf(nx, ny);
                if (destinations[next] >= 0) continue;
                distances[next] = distances[cell] + 1;
                destinations[next] = destinations[cell];
                queue[write++] = next;
            }
        }
    }
}

int Simulation::nearest(int x, int y, int kind, int radius, int exclude) const {
    if (!inBounds(x, y)) return -1;
    if (kind >= 0 && kind < 5) {
        const int cell = indexOf(x, y);
        return distances_[kind][cell] <= radius ? destinations_[kind][cell] : -1;
    }
    if (kind == 5 || kind == 7) {
        int closest = -1, best = radius + 1;
        for (std::size_t i = 0; i < citizens_.size(); ++i) {
            const Citizen& other = citizens_[i];
            if (!other.alive || other.id == exclude) continue;
            if (landComponents_[indexOf(other.x, other.y)] != landComponents_[indexOf(x, y)]) continue;
            if (kind == 7 && (other.age < 600 || other.age >= 6600 || other.health <= .6f)) continue;
            const int d = distance(x, y, other.x, other.y);
            if (d < best) { closest = static_cast<int>(i); best = d; }
        }
        return closest;
    }
    if (kind == 6) {
        for (int ring = 0; ring <= radius; ++ring) {
            for (int dx = -ring; dx <= ring; ++dx) {
                const int dy = ring - std::abs(dx);
                for (int sign : {-1, 1}) {
                    const int nx = x + dx, ny = y + dy * sign;
                    if (walkable(nx, ny) && buildable(tiles_[indexOf(nx, ny)]) &&
                        landComponents_[indexOf(nx, ny)] == landComponents_[indexOf(x, y)]) return indexOf(nx, ny);
                }
            }
        }
    }
    return -1;
}

Observation Simulation::observe(const Citizen& citizen) const {
    Observation state{};
    // [0,19] are the original controller inputs. Keep their calculations in
    // place because the founder curriculum and existing learned policies use
    // this prefix as their stable survival vocabulary.
    state[0] = citizen.hunger; state[1] = citizen.thirst; state[2] = 1 - citizen.energy;
    state[3] = 1 - citizen.social; state[4] = 1 - citizen.health;
    state[5] = citizen.food / 10; state[6] = citizen.wood / 10;
    state[7] = citizen.cooperation; state[8] = citizen.aggression;
    state[9] = citizen.age / 6000.0f;
    int closest = -1;
    if (inBounds(citizen.x, citizen.y)) {
        const int cell = indexOf(citizen.x, citizen.y);
        for (int kind = 0; kind < 5; ++kind) {
            if (destinations_[kind][cell] >= 0) state[10 + kind] = 1.0f / (1 + distances_[kind][cell]);
        }
        closest = nearest(citizen.x, citizen.y, 5, WorldWidth + WorldHeight, citizen.id);
        if (closest >= 0) state[15] = 1.0f / (1 + distance(citizen.x, citizen.y, citizens_[closest].x, citizens_[closest].y));
        state[16] = tiles_[cell].fire;
        state[19] = tiles_[cell].fertility;
    }
    state[17] = seasonYield();
    state[18] = birthReady(citizen, config_) ? 1.0f : 0.0f;

    // [20,27] are slow-changing personal and society-wide context. Statistics
    // are refreshed at the end of every tick; construction, births and deaths
    // also adjust their relevant counters immediately, yielding a cheap,
    // deterministic global snapshot while citizens take their turns.
    const float population = static_cast<float>(std::max(0, stats_.population));
    state[20] = (citizen.lastReward + 2.0f) * .25f;
    state[21] = population / static_cast<float>(PopulationLimit);
    if (population > 0.0f) {
        state[22] = static_cast<float>(stats_.homes) * 3.0f / population;
        state[23] = static_cast<float>(stats_.farms) * 2.0f / population;
        state[24] = stats_.food / (population * 4.0f);
    }
    state[25] = stats_.wellbeing;
    state[26] = stats_.cooperation;
    state[27] = static_cast<float>(std::max(0, stats_.generation)) / 8.0f;

    // [28,31] provide a categorical season rather than requiring the network
    // to infer a phase from the season-yield scalar alone.
    const int season = static_cast<int>((tick_ / (TicksPerDay * 4)) % 4);
    state[28 + season] = 1.0f;

    if (inBounds(citizen.x, citizen.y)) {
        const int cell = indexOf(citizen.x, citizen.y);
        const Tile& ground = tiles_[cell];

        // [32,40] are the resources, built use and terrain directly under the
        // citizen. They let identical needs lead to different intentions on a
        // cultivated field, a forest, or a hazardous/trafficked route.
        state[32] = ground.food / 8.0f;
        state[33] = ground.wood / 5.0f;
        state[34] = ground.traffic;
        state[35] = ground.structure == Structure::Home ? 1.0f : 0.0f;
        state[36] = ground.structure == Structure::Farm ? 1.0f : 0.0f;
        state[37] = ground.terrain == Terrain::Sand ? 1.0f : 0.0f;
        state[38] = ground.terrain == Terrain::Grass ? 1.0f : 0.0f;
        state[39] = ground.terrain == Terrain::Forest ? 1.0f : 0.0f;
        state[40] = ground.terrain == Terrain::Rock ? 1.0f : 0.0f;

        struct Neighborhood {
            float food = 0, wood = 0, fertility = 0, fire = 0, traffic = 0;
            float homes = 0, farms = 0, peopleDensity = 0;
            float cooperation = 0, aggression = 0, clanDiversity = 0;
        };
        auto summarize = [&](int radius) {
            Neighborhood summary;
            const int minX = std::max(0, citizen.x - radius);
            const int maxX = std::min(WorldWidth - 1, citizen.x + radius);
            const int minY = std::max(0, citizen.y - radius);
            const int maxY = std::min(WorldHeight - 1, citizen.y + radius);
            int tiles = 0;
            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    const Tile& nearby = tiles_[indexOf(x, y)];
                    ++tiles;
                    summary.food += unit(nearby.food / 8.0f);
                    summary.wood += unit(nearby.wood / 5.0f);
                    summary.fertility += unit(nearby.fertility);
                    summary.fire += unit(nearby.fire);
                    summary.traffic += unit(nearby.traffic);
                    summary.homes += nearby.structure == Structure::Home ? 1.0f : 0.0f;
                    summary.farms += nearby.structure == Structure::Farm ? 1.0f : 0.0f;
                }
            }
            const float inverseTiles = tiles > 0 ? 1.0f / static_cast<float>(tiles) : 0.0f;
            summary.food *= inverseTiles;
            summary.wood *= inverseTiles;
            summary.fertility *= inverseTiles;
            summary.fire *= inverseTiles;
            summary.traffic *= inverseTiles;
            summary.homes *= inverseTiles;
            summary.farms *= inverseTiles;

            // Citizen aggregation uses the same clipped square as the tile
            // scan. It stays bounded by PopulationLimit and stores counts on
            // the stack, avoiding per-observation allocation in the hot path.
            int people = 0;
            std::array<int, ClanCount> clans{};
            for (const Citizen& nearby : citizens_) {
                if (!nearby.alive || std::abs(nearby.x - citizen.x) > radius ||
                    std::abs(nearby.y - citizen.y) > radius) continue;
                ++people;
                summary.cooperation += nearby.cooperation;
                summary.aggression += nearby.aggression;
                ++clans[std::clamp(nearby.clan, 0, ClanCount - 1)];
            }
            summary.peopleDensity = tiles > 0 ? static_cast<float>(people) / static_cast<float>(tiles) : 0.0f;
            if (people > 0) {
                const float inversePeople = 1.0f / static_cast<float>(people);
                summary.cooperation *= inversePeople;
                summary.aggression *= inversePeople;
                // Normalized Simpson diversity is zero for one represented
                // clan and one for a perfectly balanced four-clan neighborhood.
                float concentration = 0.0f;
                for (int count : clans) {
                    const float share = static_cast<float>(count) * inversePeople;
                    concentration += share * share;
                }
                summary.clanDiversity = (1.0f - concentration) / (1.0f - 1.0f / static_cast<float>(ClanCount));
            }
            return summary;
        };

        // [41,51] and [52,62] are respectively radius-two and radius-five
        // square summaries: resources, hazards, construction, population and
        // social temperament. They are sampled from the live tile/citizen
        // state, never placeholders.
        const Neighborhood near = summarize(2);
        const Neighborhood broad = summarize(5);
        auto writeNeighborhood = [&](int start, const Neighborhood& summary) {
            state[start] = summary.food;
            state[start + 1] = summary.wood;
            state[start + 2] = summary.fertility;
            state[start + 3] = summary.fire;
            state[start + 4] = summary.traffic;
            state[start + 5] = summary.homes;
            state[start + 6] = summary.farms;
            state[start + 7] = summary.peopleDensity;
            state[start + 8] = summary.cooperation;
            state[start + 9] = summary.aggression;
            state[start + 10] = summary.clanDiversity;
        };
        writeNeighborhood(41, near);
        writeNeighborhood(52, broad);

        // [63,69] retain the nearest reachable citizen's needs and traits,
        // then distinguish nearby allies from members of other clans. A
        // positive proximity is also a presence signal; zero means absent.
        if (closest >= 0) {
            const Citizen& neighbour = citizens_[static_cast<std::size_t>(closest)];
            state[63] = neighbour.hunger;
            state[64] = neighbour.food / 10.0f;
            state[65] = neighbour.cooperation;
            state[66] = neighbour.aggression;
            state[67] = 1.0f - neighbour.social;
        }
        const int component = landComponents_[cell];
        int sameClanDistance = Area;
        int differentClanDistance = Area;
        for (const Citizen& neighbour : citizens_) {
            if (!neighbour.alive || neighbour.id == citizen.id ||
                landComponents_[indexOf(neighbour.x, neighbour.y)] != component) continue;
            const int separation = distance(citizen.x, citizen.y, neighbour.x, neighbour.y);
            if (neighbour.clan == citizen.clan) sameClanDistance = std::min(sameClanDistance, separation);
            else differentClanDistance = std::min(differentClanDistance, separation);
        }
        if (sameClanDistance < Area) state[68] = 1.0f / (1.0f + static_cast<float>(sameClanDistance));
        if (differentClanDistance < Area) state[69] = 1.0f / (1.0f + static_cast<float>(differentClanDistance));
    }

    // [70,81] encode the action selected on the prior decision. At the start
    // of a tick Citizen::action is the last completed intention; after an
    // action resolves it becomes the prior-action input for the next state.
    const int previousAction = static_cast<int>(citizen.action);
    if (previousAction >= 0 && previousAction < ActionCount) state[70 + previousAction] = 1.0f;
    for (float& value : state) value = std::isfinite(value) ? unit(value) : 0;
    return state;
}

Values Simulation::currentAdvice() const {
    // The society core reads the population's mean observation, so its advice
    // summarizes the whole settlement rather than any single resident. Every
    // citizen combines that shared advice with its own live observation.
    CoreInput mean{};
    std::size_t living = 0;
    for (std::size_t i = 0; i < citizens_.size(); ++i) {
        const Citizen& resident = citizens_[i];
        if (!resident.alive) continue;
        const Observation view = observe(resident);
        for (int k = 0; k < InputCount; ++k) mean[static_cast<std::size_t>(k)] += view[static_cast<std::size_t>(k)];
        ++living;
    }
    if (living > 0) {
        const float inverse = 1.0f / static_cast<float>(living);
        for (int k = 0; k < InputCount; ++k) mean[static_cast<std::size_t>(k)] *= inverse;
    }
    return core_->advise(mean);
}

ActionMask Simulation::legalActions(const Citizen& citizen) const {
    ActionMask legal{};
    if (!citizen.alive) return legal;
    const int other = nearest(citizen.x, citizen.y, 5, 24, citizen.id);
    const int site = nearest(citizen.x, citizen.y, 6, 4);
    legal[static_cast<int>(Action::Wander)] = true;
    legal[static_cast<int>(Action::Gather)] = citizen.food < 10 && nearest(citizen.x, citizen.y, 0, Area) >= 0;
    legal[static_cast<int>(Action::Eat)] = citizen.food >= .25f;
    legal[static_cast<int>(Action::Drink)] = nearest(citizen.x, citizen.y, 1, Area) >= 0;
    legal[static_cast<int>(Action::Rest)] = true;
    legal[static_cast<int>(Action::Chop)] = citizen.wood < 10 && nearest(citizen.x, citizen.y, 2, Area) >= 0;
    legal[static_cast<int>(Action::Build)] = citizen.wood >= 4 && site >= 0;
    legal[static_cast<int>(Action::Farm)] = nearest(citizen.x, citizen.y, 4, Area) >= 0 || (citizen.wood >= 1 && site >= 0);
    legal[static_cast<int>(Action::Share)] = citizen.food > .5f && other >= 0;
    legal[static_cast<int>(Action::Socialize)] = other >= 0;
    legal[static_cast<int>(Action::Reproduce)] = birthReady(citizen, config_) && nearest(citizen.x, citizen.y, 7, 24, citizen.id) >= 0 &&
        citizens_.size() < PopulationLimit;
    legal[static_cast<int>(Action::Attack)] = other >= 0;
    return legal;
}

bool Simulation::moveToward(Citizen& citizen, int target) {
    if (target < 0 || target >= Area) return false;
    const int start = indexOf(citizen.x, citizen.y);
    const int tx = target % WorldWidth, ty = target / WorldWidth;
    const bool exact = walkable(tx, ty);
    auto reached = [&](int cell) {
        return exact ? cell == target : distance(cell % WorldWidth, cell / WorldWidth, tx, ty) == 1;
    };
    if (reached(start)) return false;
    // Social and construction targets are nearby. A deterministic local step
    // gives each actor cheap, continuous navigation; the cached breadth-first
    // fields handle long trips to resources. Choosing a side step at an
    // obstacle lets citizens work around rocks without a costly full-map path
    // search on every social interaction.
    int next = -1;
    int best = std::numeric_limits<int>::max();
    for (const auto& direction : Directions) {
        const int nx = citizen.x + direction[0], ny = citizen.y + direction[1];
        if (!walkable(nx, ny)) continue;
        const int candidate = distance(nx, ny, tx, ty);
        if (candidate < best) {
            best = candidate;
            next = indexOf(nx, ny);
        }
    }
    if (next < 0) return false;
    citizen.x = next % WorldWidth; citizen.y = next / WorldWidth;
    tiles_[next].traffic = unit(tiles_[next].traffic + .045f);
    citizen.energy = unit(citizen.energy - .0015f);
    return true;
}

bool Simulation::moveAlongField(Citizen& citizen, int kind) {
    if (kind < 0 || kind >= 5) return false;
    const int start = indexOf(citizen.x, citizen.y);
    const auto& distances = distances_[kind];
    if (distances[start] >= Area) return false;
    int next = -1;
    int best = distances[start];
    // The multi-source breadth-first field was rebuilt before the citizens
    // act. One local gradient step follows the same shortest route as a fresh
    // path search, without doing a full-world search for every food or water
    // trip. Tie order stays stable for deterministic runs.
    for (const auto& direction : Directions) {
        const int nx = citizen.x + direction[0], ny = citizen.y + direction[1];
        if (!walkable(nx, ny)) continue;
        const int candidate = indexOf(nx, ny);
        if (distances[candidate] < best) {
            best = distances[candidate];
            next = candidate;
        }
    }
    if (next < 0) return false;
    citizen.x = next % WorldWidth;
    citizen.y = next / WorldWidth;
    tiles_[next].traffic = unit(tiles_[next].traffic + .045f);
    citizen.energy = unit(citizen.energy - .0015f);
    return true;
}

void Simulation::emit(std::string kind, std::string text, float impact, int x, int y) {
    const int score = static_cast<int>(std::lround(unit(std::isfinite(impact) ? impact : 0) * 100));
    events_.push_back({tick_, score, std::move(kind), std::move(text), x, y});
    ++stats_.eventCount;
    while (events_.size() > EventLimit) events_.pop_front();
}

float Simulation::act(Citizen& citizen, Action action) {
    const int cell = indexOf(citizen.x, citizen.y);
    Tile& ground = tiles_[cell];
    auto travel = [&](int target) { return moveToward(citizen, target) ? .015f : -.015f; };
    switch (action) {
    case Action::Wander: {
        const int first = static_cast<int>(rng_() % 4);
        for (int attempt = 0; attempt < 4; ++attempt) {
            const auto& direction = Directions[(first + attempt) % 4];
            const int nx = citizen.x + direction[0], ny = citizen.y + direction[1];
            if (!walkable(nx, ny)) continue;
            const float escape = ground.fire - tiles_[indexOf(nx, ny)].fire;
            citizen.x = nx; citizen.y = ny;
            tiles_[indexOf(nx, ny)].traffic = unit(tiles_[indexOf(nx, ny)].traffic + .025f);
            return .25f * escape - .005f;
        }
        return -.02f;
    }
    case Action::Gather: {
        const int target = nearest(citizen.x, citizen.y, 0, Area);
        if (target != cell) return moveAlongField(citizen, 0) ? .015f : -.015f;
        const float amount = std::min({2.0f, ground.food, 10 - citizen.food});
        const float demand = .15f + .65f * (1 - citizen.food / 10) + .2f * citizen.hunger;
        citizen.food += amount; ground.food -= amount;
        return .3f * amount * demand;
    }
    case Action::Eat: {
        const float consumed = std::min(citizen.food, 1.0f);
        const float before = citizen.hunger;
        citizen.food -= consumed; citizen.hunger = unit(citizen.hunger - .42f * consumed);
        return .9f * (before - citizen.hunger) - .025f * consumed;
    }
    case Action::Drink: {
        const int target = nearest(citizen.x, citizen.y, 1, Area);
        if (target < 0) return -.05f;
        if (distance(citizen.x, citizen.y, target % WorldWidth, target / WorldWidth) > 1)
            return moveAlongField(citizen, 1) ? .015f : -.015f;
        const float before = citizen.thirst;
        citizen.thirst = unit(citizen.thirst - .7f);
        return .9f * (before - citizen.thirst) - .006f;
    }
    case Action::Rest: {
        const float before = citizen.energy;
        const bool housed = ground.structure == Structure::Home;
        citizen.energy = unit(citizen.energy + (housed ? .23f : .12f));
        if (housed && citizen.hunger < .7f && citizen.thirst < .7f) citizen.health = unit(citizen.health + .008f);
        return .75f * (citizen.energy - before) - .008f;
    }
    case Action::Chop: {
        const int target = nearest(citizen.x, citizen.y, 2, Area);
        if (target != cell) return moveAlongField(citizen, 2) ? .015f : -.015f;
        const float demand = 1 - citizen.wood / 10;
        const float amount = std::min({1.5f, ground.wood, 10 - citizen.wood});
        citizen.wood += amount; ground.wood -= amount;
        citizen.energy = unit(citizen.energy - .01f);
        return .25f * amount * demand;
    }
    case Action::Build: {
        const int site = nearest(citizen.x, citizen.y, 6, 4);
        if (site < 0 || citizen.wood < 4) return -.05f;
        if (site != cell) return travel(site);
        ground.structure = Structure::Home; ground.owner = citizen.clan;
        ground.terrain = Terrain::Grass; ground.wood = 0; ground.food = 0;
        citizen.wood -= 4;
        ++stats_.homes;
        // Shelter becomes progressively less valuable once this community has
        // enough nearby places to rest. The policy still decides whether to
        // build; this reward makes limitless construction a bad experience.
        const float need = unit(1 - stats_.homes * 3.0f / std::max(1, stats_.population));
        emit("home", "Citizen " + std::to_string(citizen.id) + " builds a home", .3f + .25f * need, citizen.x, citizen.y);
        return .90f * need - .25f;
    }
    case Action::Farm: {
        const int target = nearest(citizen.x, citizen.y, 4, Area);
        const bool nearby = target >= 0 && distance(citizen.x, citizen.y, target % WorldWidth, target / WorldWidth) <= 2;
        if (nearby) {
            if (target != cell) return travel(target);
            const float stockBefore = citizen.food;
            const float amount = std::min({2.0f, ground.food, 10 - citizen.food});
            citizen.food += amount; ground.food -= amount;
            ground.fertility = unit(ground.fertility + .025f);
            ground.food = std::min(8.0f, ground.food + .12f * seasonYield() * ground.fertility);
            citizen.energy = unit(citizen.energy - .006f);
            return .3f * amount * (1 - stockBefore / 10) + (ground.food < 6 ? .02f : -.02f);
        }
        if (citizen.wood >= 1) {
            const int site = nearest(citizen.x, citizen.y, 6, 4);
            if (site >= 0) {
                if (site != cell) return travel(site);
                ground.structure = Structure::Farm; ground.owner = citizen.clan;
                ground.terrain = Terrain::Grass; ground.wood = 0;
                ground.food = std::max(ground.food, 1.0f);
                ground.fertility = std::max(ground.fertility, .45f);
                citizen.wood -= 1;
                ++stats_.farms;
                // A field yields enough food for roughly two citizens under
                // ordinary conditions. Plenty of fields and carried food make
                // a further conversion of wild land an unattractive action.
                const float fieldCoverage = stats_.farms * 2.0f / std::max(1, stats_.population);
                const float foodSecurity = unit(stats_.food / std::max(1.0f, stats_.population * 4.0f));
                const float need = unit(.8f * (1 - fieldCoverage) + .2f * (1 - foodSecurity));
                emit("farm", "Citizen " + std::to_string(citizen.id) + " plants a communal field", .4f, citizen.x, citizen.y);
                return .80f * need - .18f;
            }
        }
        return target >= 0 && moveAlongField(citizen, 4) ? .015f : -.015f;
    }
    case Action::Share:
    case Action::Socialize:
    case Action::Reproduce:
    case Action::Attack: {
        const int otherIndex = nearest(citizen.x, citizen.y, action == Action::Reproduce ? 7 : 5, 24, citizen.id);
        if (otherIndex < 0) return -.02f;
        Citizen& other = citizens_[static_cast<std::size_t>(otherIndex)];
        if (distance(citizen.x, citizen.y, other.x, other.y) > 1) return travel(indexOf(other.x, other.y));
        if (action == Action::Share) {
            const float amount = std::min({1.0f, citizen.food, 10 - other.food});
            const float need = unit(other.hunger + (3 - other.food) * .2f);
            citizen.food -= amount; other.food += amount;
            citizen.social = unit(citizen.social + .035f); other.social = unit(other.social + .05f);
            if (amount > .1f && need > .3f) emit("sharing", "Food shared between " + std::to_string(citizen.id) + " and " + std::to_string(other.id), .12f + .3f * need, citizen.x, citizen.y);
            return amount * (.5f * citizen.cooperation * need - .04f);
        }
        if (action == Action::Socialize) {
            const float before = citizen.social;
            citizen.social = unit(citizen.social + .22f); other.social = unit(other.social + .12f);
            citizen.cooperation = unit(citizen.cooperation + .0008f * other.cooperation);
            return .5f * (citizen.social - before) - .007f;
        }
        if (action == Action::Reproduce) {
            if (citizens_.size() >= PopulationLimit || !birthReady(citizen, config_)) return -.03f;
            if (randomUnit(rng_) >= .35f * config_.fertility + .03f) return -.004f;
            Citizen child;
            child.id = nextId_++; child.x = citizen.x; child.y = citizen.y;
            child.clan = citizen.clan; child.generation = std::max(citizen.generation, other.generation) + 1;
            child.cooperation = unit((citizen.cooperation + other.cooperation) * .5f + (randomUnit(rng_) - .5f) * .12f);
            child.aggression = unit((citizen.aggression + other.aggression) * .5f + (randomUnit(rng_) - .5f) * .1f);
            child.food = 2; child.wood = 0; child.birthCooldown = 600;
            child.brain = citizen.brain;
            child.brain.mutate(rng_, .025f);
            citizen.food -= 3; citizen.energy = unit(citizen.energy - .18f);
            citizen.birthCooldown = 600 + static_cast<int>(rng_() % 301);
            other.birthCooldown = std::max(other.birthCooldown, 150);
            citizen.social = unit(citizen.social + .15f);
            emit("birth", "Citizen " + std::to_string(child.id) + " is born into generation " + std::to_string(child.generation), .7f, child.x, child.y);
            citizens_.push_back(std::move(child)); // Reserved to PopulationLimit; newborns act next tick.
            ++stats_.births; ++stats_.population;
            return .8f;
        }
        const float stolen = std::min(.6f, other.food);
        citizen.food = std::min(10.0f, citizen.food + stolen); other.food -= stolen;
        const float injury = .015f + citizen.aggression * .06f;
        other.health = unit(other.health - injury);
        citizen.health = unit(citizen.health - (.008f + other.aggression * .025f));
        citizen.social = unit(citizen.social - .035f); other.social = unit(other.social - .07f);
        emit("conflict", "Citizen " + std::to_string(citizen.id) + " raids " + std::to_string(other.id), .45f + .4f * citizen.aggression, citizen.x, citizen.y);
        if (other.health <= 0) {
            const Observation finalState = observe(other);
            die(other, "conflict");
            other.brain.learn(compose(finalState, advice_), other.action, -1.5f,
                              compose(finalState, advice_), legalActions(other), true);
            ++stats_.learningUpdates;
        }
        return stolen * citizen.hunger * citizen.aggression * .15f - .22f * (1 - citizen.aggression);
    }
    }
    return 0;
}

void Simulation::die(Citizen& citizen, const std::string& reason) {
    if (!citizen.alive) return;
    citizen.alive = false;
    citizen.health = 0;
    Tile& ground = tiles_[indexOf(citizen.x, citizen.y)];
    ground.food = std::min(ground.structure == Structure::Farm ? 8.0f : 3.0f, ground.food + citizen.food);
    if (ground.terrain == Terrain::Forest) ground.wood = std::min(5.0f, ground.wood + citizen.wood);
    emit("death", "Citizen " + std::to_string(citizen.id) + " dies: " + reason, .9f, citizen.x, citizen.y);
    ++stats_.deaths;
    --stats_.population;
}

std::string Simulation::seasonName() const {
    static const char* names[] = {"SPRING", "SUMMER", "AUTUMN", "WINTER"};
    return names[(tick_ / (TicksPerDay * 4)) % 4];
}
float Simulation::seasonYield() const {
    static constexpr float yield[] = {1.0f, .8f, .6f, .3f};
    return yield[(tick_ / (TicksPerDay * 4)) % 4];
}

void Simulation::environment() {
    if (tick_ % (TicksPerDay * 4) == 0) emit("season", seasonName() + " changes the land's yield", .35f);
    if (tick_ % TicksPerDay == 0) emit("day", "Day " + std::to_string(tick_ / TicksPerDay) + " begins", .05f);
    const float yield = seasonYield();
    std::vector<int> ignitions;
    for (int i = 0; i < Area; ++i) {
        Tile& ground = tiles_[i];
        ground.traffic *= .999f;
        if (ground.terrain == Terrain::Water || ground.terrain == Terrain::Rock) continue;
        if (ground.fire > 0) {
            ground.fire = std::max(0.0f, ground.fire - .012f);
            ground.food = std::max(0.0f, ground.food - .08f);
            ground.wood = std::max(0.0f, ground.wood - .06f);
            if (ground.structure != Structure::None && ground.fire > .45f && randomUnit(rng_) < .015f) {
                const bool home = ground.structure == Structure::Home;
                ground.structure = Structure::None; ground.owner = -1;
                emit("destruction", home ? "A home burns down" : "Fire consumes a field", .72f, i % WorldWidth, i / WorldWidth);
            }
            if (ground.fire > .4f && randomUnit(rng_) < .04f * config_.hazards) {
                const auto& direction = Directions[rng_() % 4];
                const int nx = i % WorldWidth + direction[0], ny = i / WorldWidth + direction[1];
                if (walkable(nx, ny) && tiles_[indexOf(nx, ny)].fire <= 0 && tiles_[indexOf(nx, ny)].wood > .5f)
                    ignitions.push_back(indexOf(nx, ny));
            }
            continue;
        }
        if (ground.structure == Structure::Home) continue;
        if (ground.structure == Structure::Farm) ground.food = std::min(8.0f, ground.food + .013f * ground.fertility * yield);
        else ground.food = std::min(3.0f, ground.food + .0018f * ground.fertility * yield);
        if (ground.terrain == Terrain::Forest) ground.wood = std::min(5.0f, ground.wood + .0025f * yield);
    }
    for (int cell : ignitions) tiles_[cell].fire = .7f;
    if (tick_ % 150 == 0 && randomUnit(rng_) < config_.hazards * .22f) {
        const int cell = static_cast<int>(rng_() % Area);
        Tile& ground = tiles_[cell];
        if (ground.terrain == Terrain::Forest && ground.wood > 1) {
            ground.fire = 1;
            emit("fire", "Lightning ignites the forest", .78f, cell % WorldWidth, cell / WorldWidth);
        }
    }
    if (tick_ % 450 == 0 && randomUnit(rng_) < config_.hazards * .5f) {
        const int center = static_cast<int>(rng_() % Area);
        const int cx = center % WorldWidth, cy = center / WorldWidth;
        for (int y = std::max(0, cy - 7); y <= std::min(WorldHeight - 1, cy + 7); ++y) {
            for (int x = std::max(0, cx - 7); x <= std::min(WorldWidth - 1, cx + 7); ++x) {
                Tile& ground = tiles_[indexOf(x, y)];
                ground.fire = 0;
                if (ground.terrain == Terrain::Grass || ground.terrain == Terrain::Forest) ground.fertility = unit(ground.fertility + .025f);
            }
        }
        for (Citizen& citizen : citizens_) {
            if (citizen.alive && distance(citizen.x, citizen.y, cx, cy) <= 7 &&
                tiles_[indexOf(citizen.x, citizen.y)].structure != Structure::Home)
                citizen.energy = unit(citizen.energy - .08f);
        }
        emit("storm", "A rainstorm drenches the region and quenches fires", .58f, cx, cy);
    }
    // Resource maps remain accurate enough for one second of real time. They
    // are intentionally cached so a populous world can stay responsive at the
    // fixed five-ticks-per-second rate instead of spending each tick rebuilding
    // the entire map for every citizen's changing inventory.
    if (tick_ % TicksPerSecond == 0) rebuildDestinations();
}

void Simulation::step() {
    ++tick_;
    citizens_.erase(std::remove_if(citizens_.begin(), citizens_.end(), [](const Citizen& citizen) { return !citizen.alive; }), citizens_.end());
    environment();
    // This tick's shared advisory context, derived from the current population.
    advice_ = currentAdvice();
    const std::size_t actors = citizens_.size();
    for (std::size_t i = 0; i < actors; ++i) {
        Citizen& citizen = citizens_[i];
        if (!citizen.alive) continue;
        const BrainInput before = compose(observe(citizen), advice_);
        const ActionMask allowed = legalActions(citizen);
        const float previousComfort = comfort(citizen);
        const float exploration = .025f + .035f * (1 - citizen.cooperation);
        citizen.action = citizen.brain.choose(before, allowed, exploration, rng_);
        ++stats_.decisions;
        ++stats_.actions[static_cast<int>(citizen.action)];
        float reward = act(citizen, citizen.action);
        ++citizen.age;
        if (citizen.birthCooldown > 0) --citizen.birthCooldown;
        citizen.hunger = unit(citizen.hunger + .0018f);
        citizen.thirst = unit(citizen.thirst + .0030f);
        citizen.energy = unit(citizen.energy - .0022f);
        citizen.social = unit(citizen.social - .0012f);
        const Tile& ground = tiles_[indexOf(citizen.x, citizen.y)];
        float damage = std::max(0.0f, citizen.hunger - .86f) * .022f +
                       std::max(0.0f, citizen.thirst - .85f) * .035f +
                       std::max(0.0f, .06f - citizen.energy) * .012f + ground.fire * .025f;
        if (citizen.age > 7200) damage += .0008f + (citizen.age - 7200) * .000001f;
        if (damage > 0) citizen.health = unit(citizen.health - damage);
        else if (citizen.hunger < .7f && citizen.thirst < .7f && citizen.energy > .15f) citizen.health = unit(citizen.health + .0015f);
        reward += (comfort(citizen) - previousComfort) * 1.5f -
                  .035f * (citizen.hunger * citizen.hunger + citizen.thirst * citizen.thirst) - damage * 5;
        if (citizen.health <= 0) {
            const std::string cause = citizen.age > 7200 ? "old age" : ground.fire > .1f ? "fire" :
                                      citizen.thirst >= .95f ? "dehydration" : citizen.hunger >= .95f ? "starvation" : "injury";
            die(citizen, cause);
            reward -= 1.5f;
        }
        citizen.lastReward = std::clamp(reward, -2.0f, 2.0f);
        citizen.brain.learn(before, citizen.action, citizen.lastReward, compose(observe(citizen), advice_), legalActions(citizen), !citizen.alive);
        ++stats_.learningUpdates;
    }
    refreshStatistics();
    if (tick_ % 25 == 0) {
        history_.push_back({tick_, stats_.population, stats_.wellbeing});
        while (history_.size() > HistoryLimit) history_.pop_front();
    }
}

void Simulation::refreshStatistics() {
    stats_.population = 0; stats_.homes = 0; stats_.farms = 0;
    stats_.generation = 0;
    stats_.wellbeing = 0; stats_.food = 0; stats_.cooperation = 0;
    for (const Citizen& citizen : citizens_) {
        if (!citizen.alive) continue;
        ++stats_.population;
        stats_.generation = std::max(stats_.generation, citizen.generation);
        stats_.wellbeing += comfort(citizen);
        stats_.food += citizen.food;
        stats_.cooperation += citizen.cooperation;
    }
    if (stats_.population > 0) {
        stats_.wellbeing /= stats_.population;
        stats_.cooperation /= stats_.population;
    }
    for (const Tile& ground : tiles_) {
        if (ground.structure == Structure::Home) ++stats_.homes;
        if (ground.structure == Structure::Farm) ++stats_.farms;
    }
}

std::uint64_t Simulation::digest() const {
    std::uint64_t hash = 1469598103934665603ull;
    auto bytes = [&](const void* value, std::size_t size) {
        const auto* data = static_cast<const unsigned char*>(value);
        for (std::size_t i = 0; i < size; ++i) { hash ^= data[i]; hash *= 1099511628211ull; }
    };
    auto add = [&](auto value) { bytes(&value, sizeof(value)); };
    auto string = [&](const std::string& value) { add(value.size()); bytes(value.data(), value.size()); };
    add(config_.seed); add(config_.founders); add(config_.fertility); add(config_.cooperation); add(config_.hazards);
    add(tick_); add(nextId_);
    add(core_ ? core_->digest() : 0ull);
    for (float value : advice_) add(value);
    auto rngCopy = rng_;
    for (std::size_t i = 0; i < std::mt19937::state_size; ++i) add(rngCopy());
    for (const Tile& ground : tiles_) {
        add(ground.terrain); add(ground.structure); add(ground.food); add(ground.wood); add(ground.fertility);
        add(ground.fire); add(ground.traffic); add(ground.owner);
    }
    add(citizens_.size());
    for (const Citizen& citizen : citizens_) {
        add(citizen.id); add(citizen.x); add(citizen.y); add(citizen.clan); add(citizen.generation);
        add(citizen.age); add(citizen.birthCooldown); add(citizen.alive);
        add(citizen.health); add(citizen.hunger); add(citizen.thirst); add(citizen.energy); add(citizen.social);
        add(citizen.food); add(citizen.wood); add(citizen.cooperation); add(citizen.aggression);
        add(citizen.action); add(citizen.lastReward); add(citizen.brain.updates()); add(citizen.brain.digest());
    }
    for (const Event& event : events_) {
        add(event.tick); add(event.score); string(event.kind); string(event.text); add(event.x); add(event.y);
    }
    for (const HistoryPoint& point : history_) { add(point.tick); add(point.population); add(point.wellbeing); }
    add(stats_.population); add(stats_.births); add(stats_.deaths); add(stats_.homes); add(stats_.farms);
    add(stats_.generation); add(stats_.wellbeing); add(stats_.food); add(stats_.cooperation);
    add(stats_.decisions); add(stats_.learningUpdates); add(stats_.eventCount);
    for (auto count : stats_.actions) add(count);
    return hash;
}
}
