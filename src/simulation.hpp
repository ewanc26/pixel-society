#pragma once
#include "neural.hpp"
#include <array>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace pixels {
// Classic default dimensions; the chosen world preset may override them.
inline constexpr int DefaultWorldWidth = 96;
inline constexpr int DefaultWorldHeight = 64;
// Source-compatible names for callers that mean the classic preset.
inline constexpr int WorldWidth = DefaultWorldWidth;
inline constexpr int WorldHeight = DefaultWorldHeight;
inline constexpr int TicksPerSecond = 5;
inline constexpr int TicksPerDay = 300;
inline constexpr int PopulationLimit = 256;

// Terrain-generation presets for the WORLD SHAPE setup option. Every shape is
// deterministic from the world seed; they change layout, not the rules.
enum class WorldShape { Island, Archipelago, InlandSea, Highlands, Riverlands };
// Discrete world-size presets for the WORLD SIZE setup option.
enum class WorldSize { Tiny, Small, Classic, Large, Huge };
inline int worldWidth(WorldSize size) {
    switch (size) {
    case WorldSize::Tiny: return 48;
    case WorldSize::Small: return 64;
    case WorldSize::Large: return 128;
    case WorldSize::Huge: return 192;
    default: return DefaultWorldWidth;
    }
}
inline int worldHeight(WorldSize size) {
    switch (size) {
    case WorldSize::Tiny: return 32;
    case WorldSize::Small: return 40;
    case WorldSize::Large: return 80;
    case WorldSize::Huge: return 128;
    default: return DefaultWorldHeight;
    }
}
inline const char* nameOf(WorldShape shape) {
    switch (shape) {
    case WorldShape::Archipelago: return "ARCHIPELAGO";
    case WorldShape::InlandSea: return "INLAND SEA";
    case WorldShape::Highlands: return "HIGHLANDS";
    case WorldShape::Riverlands: return "RIVERLANDS";
    default: return "ISLAND";
    }
}
inline const char* nameOf(WorldSize size) {
    switch (size) {
    case WorldSize::Tiny: return "TINY";
    case WorldSize::Small: return "SMALL";
    case WorldSize::Large: return "LARGE";
    case WorldSize::Huge: return "HUGE";
    default: return "CLASSIC";
    }
}

struct Config {
    std::uint32_t seed = 2026;
    int founders = 48;
    float fertility = 0.65f;
    float cooperation = 0.7f;
    float hazards = 0.35f;
    WorldShape shape = WorldShape::Island;
    WorldSize worldSize = WorldSize::Classic;
    // Worker threads for the deterministic pool (0 = all available cores).
    // The simulation's trace is independent of this value.
    int threads = 0;
};
enum class Terrain { Water, Sand, Grass, Forest, Rock };
enum class Structure { None, Home, Farm };
struct Tile {
    Terrain terrain = Terrain::Grass;
    Structure structure = Structure::None;
    float food = 0;
    float wood = 0;
    float fertility = 0;
    float fire = 0;
    float traffic = 0;
    int owner = -1;
    bool burned = false;
};
struct Citizen {
    int id = 0, x = 0, y = 0, clan = 0, generation = 0;
    int age = 0, birthCooldown = 0;
    int sick = 0, immune = 0;
    bool alive = true;
    float health = 1, hunger = 0.2f, thirst = 0.2f, energy = 0.85f, social = 0.5f;
    float food = 4, wood = 0, cooperation = 0.7f, aggression = 0.15f;
    Action action = Action::Wander;
    float lastReward = 0;
    Brain brain;
};
struct Event {
    std::uint64_t tick = 0;
    int score = 0;
    std::string kind;
    std::string text;
    int x = -1, y = -1;
};
struct Statistics {
    int population = 0, births = 0, deaths = 0, homes = 0, farms = 0;
    int generation = 0;
    float wellbeing = 0, food = 0, cooperation = 0;
    std::uint64_t decisions = 0, learningUpdates = 0, eventCount = 0;
    std::array<std::uint64_t, ActionCount> actions{};
};
struct HistoryPoint { std::uint64_t tick; int population; float wellbeing; };

// Neural observation contract. Ranges [0,19] preserve the original survival
// inputs: hunger, thirst, fatigue, social need, health deficit, food/10,
// wood/10, cooperation, aggression, age/6000, forage/water/forest/home/farm/
// neighbour proximity, local fire, seasonal yield, birth readiness and local
// fertility. [20,27] add recent reward and society-wide capacity/wellbeing;
// [28,31] are a season one-hot; [32,40] describe the current tile; [41,51]
// and [52,62] summarize radius-two and radius-five square neighborhoods;
// [63,69] describe nearest social contacts; and [70,81] encode the most
// recently selected action. All components are finite and normalized to [0,1].
class Simulation {
public:
    explicit Simulation(Config config = {});
    void step();
    const Config& config() const { return config_; }
    // Dimensions of the selected world preset.
    int width() const { return width_; }
    int height() const { return height_; }
    std::uint64_t tick() const { return tick_; }
    const std::vector<Tile>& tiles() const { return tiles_; }
    const std::vector<Citizen>& citizens() const { return citizens_; }
    const std::deque<Event>& events() const { return events_; }
    const std::deque<HistoryPoint>& history() const { return history_; }
    const Statistics& stats() const { return stats_; }
    const Tile& tile(int x, int y) const;
    // Deterministic clan territory labels (one per tile): the civilisation that
    // claims a tile, or -1 where no civilisation has claimed the land yet.
    // Drives the observer's border overlay and is covered by the digest.
    int territory(int x, int y) const;
    const std::vector<int>& territory() const { return territory_; }
    Observation observe(const Citizen& citizen) const;
    ActionMask legalActions(const Citizen& citizen) const;
    std::string seasonName() const;
    float seasonYield() const;
    // This civilization's society core and the advisory values derived from it
    // this tick. The core is trained from the world seed, so different seeds
    // give different civilizations distinct collective instincts.
    const SocietyCore* core() const { return core_.get(); }
    const Values& advice() const { return advice_; }
    // Reproducibility digest includes the world, residents, event history and learned policy state.
    std::uint64_t digest() const;
private:
    std::shared_ptr<SocietyCore> core_;
    Values advice_ = {};
    Values currentAdvice() const;
    Config config_;
    std::mt19937 rng_;
    std::uint64_t tick_ = 0;
    int nextId_ = 0;
    // World dimensions from the selected size preset (Classic = 96 x 64).
    int width_ = DefaultWorldWidth;
    int height_ = DefaultWorldHeight;
    std::vector<Tile> tiles_;
    std::vector<Citizen> citizens_;
    std::deque<Event> events_;
    std::deque<HistoryPoint> history_;
    Statistics stats_;
    // Epidemic state: a nonzero counter is an active plague wave in progress.
    int epidemic_ = 0;
    // Multi-source breadth-first fields provide reachable resources, not straight-line guesses.
    std::array<std::vector<int>, 5> destinations_;
    std::array<std::vector<int>, 5> distances_;
    std::vector<int> landComponents_;
    // Per-tile clan territory labels used by the observer's border overlay.
    std::vector<int> territory_;
    void rebuildDestinations();
    void computeTerritory();
    void generate();
    void classifyLand(int x, int y, float n, float shore);
    void generateIsland(float phase);
    void generateArchipelago(float phase);
    void generateInlandSea(float phase);
    void generateHighlands(float phase);
    void generateRiverlands(float phase);
    void environment();
    void epidemiology();
    float act(Citizen& citizen, Action action);
    void emit(std::string kind, std::string text, float impact, int x = -1, int y = -1);
    void refreshStatistics();
    int area() const { return width_ * height_; }
    int indexOf(int x, int y) const { return y * width_ + x; }
    bool inBounds(int x, int y) const { return x >= 0 && y >= 0 && x < width_ && y < height_; }
    bool walkable(int x, int y) const;
    int nearest(int x, int y, int kind, int radius = 24, int exclude = -1) const;
    bool moveToward(Citizen& citizen, int target);
    bool moveAlongField(Citizen& citizen, int kind);
    void die(Citizen& citizen, const std::string& reason);
};
}
