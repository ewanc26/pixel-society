#pragma once
#include "neural.hpp"
#include <array>
#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

namespace pixels {
inline constexpr int WorldWidth = 96;
inline constexpr int WorldHeight = 64;
inline constexpr int TicksPerSecond = 5;
inline constexpr int TicksPerDay = 300;
inline constexpr int PopulationLimit = 256;

struct Config {
    std::uint32_t seed = 2026;
    int founders = 48;
    float fertility = 0.65f;
    float cooperation = 0.7f;
    float hazards = 0.35f;
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
};
struct Citizen {
    int id = 0, x = 0, y = 0, clan = 0, generation = 0;
    int age = 0, birthCooldown = 0;
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
    std::uint64_t tick() const { return tick_; }
    const std::vector<Tile>& tiles() const { return tiles_; }
    const std::vector<Citizen>& citizens() const { return citizens_; }
    const std::deque<Event>& events() const { return events_; }
    const std::deque<HistoryPoint>& history() const { return history_; }
    const Statistics& stats() const { return stats_; }
    const Tile& tile(int x, int y) const;
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
    std::vector<Tile> tiles_;
    std::vector<Citizen> citizens_;
    std::deque<Event> events_;
    std::deque<HistoryPoint> history_;
    Statistics stats_;
    // Multi-source breadth-first fields provide reachable resources, not straight-line guesses.
    std::array<std::vector<int>, 5> destinations_;
    std::array<std::vector<int>, 5> distances_;
    std::vector<int> landComponents_;
    void rebuildDestinations();
    void generate();
    void environment();
    float act(Citizen& citizen, Action action);
    void emit(std::string kind, std::string text, float impact, int x = -1, int y = -1);
    void refreshStatistics();
    bool walkable(int x, int y) const;
    int nearest(int x, int y, int kind, int radius = 24, int exclude = -1) const;
    bool moveToward(Citizen& citizen, int target);
    bool moveAlongField(Citizen& citizen, int kind);
    void die(Citizen& citizen, const std::string& reason);
};
}
