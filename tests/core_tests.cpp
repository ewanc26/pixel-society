#include "simulation.hpp"
#include "ticker.hpp"
#include "parallel.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

using namespace pixels;
namespace {
#if defined(PIXEL_SOCIETY_SANITIZE_BUILD)
constexpr int AutonomousRunTicks = 1200;
constexpr int HarshRunTicks = 500;
#else
constexpr int AutonomousRunTicks = 6000;
constexpr int HarshRunTicks = 1500;
#endif
// The emergent-mechanics suite runs a fixed tick window in every build so the
// disease, fire and transmission assertions are consistent under sanitizers.
constexpr int MechanicsRunTicks = 800;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F&& operation, const std::string& message) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

struct ObservationSurvey {
    std::array<float, InputCount> minimum{};
    std::array<float, InputCount> maximum{};
    int samples = 0;

    void add(const Observation& observation) {
        for (int i = 0; i < InputCount; ++i) {
            const float value = observation[static_cast<std::size_t>(i)];
            require(std::isfinite(value) && value >= 0.0f && value <= 1.0f,
                    "every rich observation feature must be finite and normalized");
            if (samples == 0) minimum[static_cast<std::size_t>(i)] = maximum[static_cast<std::size_t>(i)] = value;
            else {
                minimum[static_cast<std::size_t>(i)] = std::min(minimum[static_cast<std::size_t>(i)], value);
                maximum[static_cast<std::size_t>(i)] = std::max(maximum[static_cast<std::size_t>(i)], value);
            }
        }
        ++samples;
    }

    int nonzero(int first = 0, int last = InputCount) const {
        int count = 0;
        for (int i = first; i < last; ++i)
            if (maximum[static_cast<std::size_t>(i)] > 0.001f) ++count;
        return count;
    }

    int varying(int first = 0, int last = InputCount) const {
        int count = 0;
        for (int i = first; i < last; ++i)
            if (maximum[static_cast<std::size_t>(i)] - minimum[static_cast<std::size_t>(i)] > 0.001f) ++count;
        return count;
    }
};

void collectObservations(const Simulation& sim, ObservationSurvey& survey) {
    int living = 0;
    for (const auto& citizen : sim.citizens()) {
        if (!citizen.alive) continue;
        survey.add(sim.observe(citizen));
        ++living;
    }
    require(living > 0, "a normal seeded world must expose citizens to observe");
}

void neural() {
    require(InputCount == 82, "the policy must expose 82 input features");
    require(HiddenOneCount == 56 && HiddenTwoCount == 28,
            "the policy must retain its 56 then 28 hidden-neuron architecture");
    require(ActionCount == 12, "the policy must retain twelve ranked intentions");
    require(BrainInputCount == InputCount + ActionCount,
            "an individual policy joins its observation with one advisory value per action");

    // The society core is the shared wide neural AI behind action selection.
    require(SocietyCore::parameterCount() >= 10'000'000,
            "the society core neural AI must contain at least ten million parameters");
    require(Brain::parameterCount() < SocietyCore::parameterCount(),
            "the shared society core is the largest network in the system");
    auto coreSeven = std::make_unique<SocietyCore>(7);
    auto coreSevenAgain = std::make_unique<SocietyCore>(7);
    auto coreEight = std::make_unique<SocietyCore>(8);
    CoreInput society{};
    society[0] = 0.8f; society[21] = 0.5f; society[39] = 1.0f;
    const Values coreAdvice = coreSeven->advise(society);
    require(coreSeven->advise(society) == coreSevenAgain->advise(society),
            "seeded society core inference must reproduce");
    require(coreSeven->advise(society) != coreEight->advise(society),
            "different seeds must vary the society core");
    for (float value : coreAdvice) {
        require(std::isfinite(value) && value >= 0.0f && value <= 1.0f,
                "core advisory values are normalized for the individual networks");
    }
    const auto civAlpha = makeSocietyCore(7);
    require(civAlpha->digest() == makeSocietyCore(7)->digest(),
            "a civilization's society core is deterministic for its seed");
    const auto civBeta = makeSocietyCore(8);
    require(civAlpha->digest() != civBeta->digest(),
            "different civilization seeds must train different society cores");
    require(civAlpha->advise(society) != civBeta->advise(society),
            "different civilizations must advise their citizens differently");

    Brain brain(7), same(7), other(8);
    Observation obs{}; obs[0] = 0.8f; obs[5] = 0.6f; obs[7] = 0.7f;
    Values advice{}; advice.fill(0.5f);
    const BrainInput input = compose(obs, advice);
    require(brain.predict(input) == same.predict(input), "seeded inference must reproduce");
    require(brain.predict(input) != other.predict(input), "different seeds must vary networks");
    const auto before = brain.predict(input);
    for (int i = 0; i < 300; ++i) brain.train(input, Action::Eat, 1.5f);
    require(std::abs(brain.predict(input)[2] - 1.5f) < std::abs(before[2] - 1.5f) * 0.1f,
            "backpropagation must learn the target, not only count updates");
    require(brain.updates() == 300, "training updates must be recorded");
    ActionMask legal{}; legal.fill(true);
    std::mt19937 rng(42);
    Brain preferGather(12), preferDrink(12);
    for (int n = 0; n < 200; ++n) {
        for (int a = 0; a < ActionCount; ++a) {
            preferGather.train(input, static_cast<Action>(a), a == 1 ? 2.f : -1.f);
            preferDrink.train(input, static_cast<Action>(a), a == 3 ? 2.f : -1.f);
        }
    }
    require(preferGather.choose(input, legal, 0, rng) == Action::Gather, "learned weights must control decisions");
    require(preferDrink.choose(input, legal, 0, rng) == Action::Drink, "changing learned weights must change intention");
    legal.fill(false); legal[4] = true;
    for (int n = 0; n < 100; ++n)
        require(brain.choose(input, legal, n % 2 ? 1 : 0, rng) == Action::Rest, "exploration and inference must both respect legality");
    legal.fill(false);
    rejects([&] { brain.choose(input, legal, 0, rng); }, "no legal action must fail explicitly");
    auto inherited = brain;
    require(inherited.fingerprint() == brain.fingerprint(), "children can inherit exact trained weights");
    inherited.mutate(rng);
    require(inherited.fingerprint() != brain.fingerprint(), "mutation must actually change policy");
    legal.fill(true);
    auto previous = brain.fingerprint();
    brain.learn(input, Action::Eat, -0.7f, input, legal);
    require(brain.fingerprint() != previous, "online temporal difference update must affect weights");
    for (int i = 0; i < 2000; ++i) brain.learn(input, Action::Eat, i % 2 ? 1.f : -1.f, input, legal);
    for (float value : brain.predict(input)) require(std::isfinite(value), "online learning must stay finite");

    // The society core's advisory channel must genuinely reach the individual
    // policy's learned action values, not sit as a disconnected input tail.
    {
        const int farm = static_cast<int>(Action::Farm);
        Brain adviceSensitive(606);
        Values lowAdvice = advice; lowAdvice[farm] = 0.05f;
        Values highAdvice = advice; highAdvice[farm] = 0.95f;
        for (int i = 0; i < 400; ++i) {
            adviceSensitive.train(compose(obs, lowAdvice), Action::Farm, -1.25f, 0.02f);
            adviceSensitive.train(compose(obs, highAdvice), Action::Farm, 1.25f, 0.02f);
        }
        const float afterLow = adviceSensitive.predict(compose(obs, lowAdvice))[farm];
        const float afterHigh = adviceSensitive.predict(compose(obs, highAdvice))[farm];
        require(afterHigh - afterLow > 0.35f,
                "training must make action values discriminate the society core advisory input");
    }

    // The final observation sensor (a previous-action bit) is deliberately
    // isolated while the advice channels are fixed: this catches networks that
    // allocate the advertised high-dimensional observation but never connect
    // its deep tail to the learned action values.
    {
        Observation tailLow{};
        for (int i = 0; i < InputCount - 1; ++i)
            tailLow[static_cast<std::size_t>(i)] = static_cast<float>((i * 17) % 31) / 30.0f;
        Observation tailHigh = tailLow;
        tailLow.back() = 0.05f;
        tailHigh.back() = 0.95f;
        const BrainInput low = compose(tailLow, advice);
        const BrainInput high = compose(tailHigh, advice);
        Brain tailSensitive(909);
        const float beforeLow = tailSensitive.predict(low)[static_cast<int>(Action::Farm)];
        const float beforeHigh = tailSensitive.predict(high)[static_cast<int>(Action::Farm)];
        for (int i = 0; i < 650; ++i) {
            tailSensitive.train(low, Action::Farm, -1.25f, 0.02f);
            tailSensitive.train(high, Action::Farm, 1.25f, 0.02f);
        }
        const float afterLow = tailSensitive.predict(low)[static_cast<int>(Action::Farm)];
        const float afterHigh = tailSensitive.predict(high)[static_cast<int>(Action::Farm)];
        require(afterHigh - afterLow > 0.35f,
                "training must make an action value discriminate observations that differ only at input 81");
        require(std::abs(afterLow - beforeLow) > 0.05f || std::abs(afterHigh - beforeHigh) > 0.05f,
                "deep-tail training must materially change predicted action values");
    }
}

void culture() {
    // Social transmission: imitation must move a policy toward a donor, stop
    // at the donor for rate one, and leave the donor and its training history
    // untouched.
    const auto core = makeSocietyCore(9182);
    Brain teacher = makeFounderBrain(11, *core);
    Brain student = makeFounderBrain(22, *core);
    require(teacher.fingerprint() != student.fingerprint(), "distinct founder seeds must train distinct policies");
    const std::uint64_t donorUpdates = teacher.updates();
    student.imitate(teacher, 1.0f);
    require(student.fingerprint() == teacher.fingerprint(), "full imitation must reproduce the donor policy exactly");
    require(teacher.updates() == donorUpdates, "imitation must not mutate the donor");
    Brain partial = makeFounderBrain(33, *core);
    const double frozen = partial.fingerprint();
    partial.imitate(teacher, 0.0f);
    require(partial.fingerprint() == frozen, "zero-rate imitation must be a no-op");
    partial.imitate(teacher, 1.0f);
    require(partial.fingerprint() == teacher.fingerprint(), "rate one must converge to the donor");
    std::mt19937 fuzz(7);
    partial.mutate(fuzz);
    const double fuzzed = partial.fingerprint();
    partial.imitate(teacher, 0.5f);
    require(std::isfinite(partial.fingerprint()) && partial.fingerprint() != fuzzed,
            "half-rate imitation must move a mutated policy partway toward the donor");
}
void observations() {
    Config config;
    config.seed = 9182;
    Simulation sim(config);
    ObservationSurvey initial;
    collectObservations(sim, initial);
    require(initial.samples >= 2, "the normal world must contain multiple individual perspectives");
    // Empty infrastructure and the absence of an active disaster are valid
    // initial conditions, so this asks for broad live data without demanding
    // every optional world signal at tick zero.
    require(initial.nonzero() >= InputCount * 11 / 20,
            "a new world must populate a substantial portion of its observation data");
    require(initial.nonzero(InputCount / 2) >= InputCount / 5,
            "a new world must expose substantial high-index sensor data");
    require(initial.varying() >= 8,
            "individual citizens must see meaningfully different surroundings and states");

    ObservationSurvey evolving = initial;
    for (int tick = 0; tick < 240; ++tick) {
        sim.step();
        if (tick % 12 == 11) collectObservations(sim, evolving);
    }
    require(evolving.samples > initial.samples, "an autonomous world must keep producing observation samples");
    require(evolving.nonzero() >= InputCount * 2 / 3,
            "individual and environmental data must activate most rich observation features over time");
    require(evolving.nonzero(InputCount / 2) >= InputCount / 4,
            "the high-index half must carry substantial live world data over time");
    require(evolving.varying() >= InputCount / 3,
            "the observation vector must evolve across changing citizens and environment");
    require(evolving.varying(InputCount / 2) >= InputCount / 8,
            "high-index sensors must respond as the autonomous world evolves");
}
void timing() {
    using namespace std::chrono_literals;
    FixedTicker ticker;
    int ticks = 0;
    for (int i = 0; i < 1000; ++i) ticker.advance(1ms, [&] { ++ticks; });
    require(ticks == 5 && ticker.debt() == 0ns, "one second must produce exactly five game ticks");
    ticker.advance(199ms, [&] { ++ticks; });
    require(ticks == 5, "no early tick");
    ticker.advance(1ms, [&] { ++ticks; });
    require(ticks == 6, "tick at exact 200ms boundary");
    ticker.advance(12s, [&] { ++ticks; });
    require(ticks == 31 && ticker.debt() == 7s, "slow frames must retain all time debt");
    ticker.advance(0ns, [&] { ++ticks; });
    ticker.advance(0ns, [&] { ++ticks; });
    require(ticks == 66 && ticker.debt() == 0ns, "catchup must not discard simulation ticks");
    ticker.advance(-1s, [&] { ++ticks; });
    require(ticks == 66, "clock anomalies must not reverse time");
}
void invariants(const Simulation& sim) {
    const auto& s = sim.stats();
    require(s.population >= 0 && s.population <= PopulationLimit, "population bounded");
    require(sim.citizens().size() <= PopulationLimit, "citizen storage bounded across generations");
    require(sim.tiles().size() == WorldWidth * WorldHeight, "complete world");
    require(std::isfinite(s.wellbeing) && s.wellbeing >= 0 && s.wellbeing <= 1, "wellbeing normalized");
    std::set<int> ids;
    int alive = 0;
    for (const auto& c : sim.citizens()) {
        if (!c.alive) continue;
        ++alive;
        require(ids.insert(c.id).second, "living citizen ids unique");
        require(c.x >= 0 && c.x < WorldWidth && c.y >= 0 && c.y < WorldHeight, "citizens stay on map");
        require(sim.tile(c.x, c.y).terrain != Terrain::Water, "citizens cannot walk on water");
        require(std::isfinite(c.health) && c.health > 0 && c.health <= 1, "living health valid");
        require(std::isfinite(c.food) && c.food >= 0 && std::isfinite(c.wood) && c.wood >= 0, "inventories finite nonnegative");
        for (float input : sim.observe(c)) require(std::isfinite(input) && input >= 0 && input <= 1, "features normalized");
        for (float output : c.brain.predict(compose(sim.observe(c), sim.advice())))
            require(std::isfinite(output), "network values finite");
    }
    require(alive == s.population, "population statistics match actual citizens");
    require(s.population == sim.config().founders + s.births - s.deaths, "births and deaths conserve population");
    for (const auto& t : sim.tiles()) {
        require(std::isfinite(t.food) && t.food >= 0 && std::isfinite(t.wood) && t.wood >= 0, "world resources finite and nonnegative");
        require(std::isfinite(t.fire) && t.fire >= 0, "fire finite and nonnegative");
        if (t.structure != Structure::None) require(t.terrain != Terrain::Water, "structures stay on land");
    }
    for (const auto& event : sim.events()) {
        require(event.score >= 0 && event.score <= 100, "every event score in 0..100");
        require(event.tick <= sim.tick() && !event.kind.empty() && !event.text.empty(), "events have timestamp and content");
    }
}
void society() {
    Config config; config.seed = 42;
    Simulation a(config), b(config);
    require(a.digest() == b.digest(), "seeded worlds reproduce exactly");
    config.seed = 43;
    Simulation different(config);
    require(a.digest() != different.digest(), "world seed changes simulation");
    const auto initial = a.digest();
    for (int i = 0; i < 500; ++i) { a.step(); b.step(); }
    require(a.digest() == b.digest(), "randomness and learned behavior reproduce after500 ticks");
    require(a.digest() != initial, "world develops without player interaction");
    invariants(a);
    require(a.stats().decisions > 0 && a.stats().learningUpdates >= a.stats().decisions,
            "citizens perform inference and learn from their autonomous decisions");
    std::uint64_t decisions = 0;
    for (auto count : a.stats().actions) decisions += count;
    require(decisions == a.stats().decisions, "action counters account for every decision");
    for (int i = 500; i < AutonomousRunTicks; ++i) {
        a.step();
        if (i % 500 == 0) invariants(a);
    }
    invariants(a);
    const auto& s = a.stats();
    std::cout << "seed42 at" << AutonomousRunTicks << " ticks: population=" << s.population << " births=" << s.births
              << " homes=" << s.homes << " farms=" << s.farms << " updates=" << s.learningUpdates << '\n';
    require(s.births > 0, "neural citizens must autonomously produce new generations");
    require(s.homes > 0 && s.farms > 0, "neural citizens must autonomously construct society");
    require(s.population > 0, "ordinary starting conditions must support sustained society");
    require(s.eventCount > 0, "simulation produces scored events");
}
void extremes() {
    Config config;
    config.founders = 0;
    rejects([&] { Simulation sim(config); }, "zero founders rejected");
    config.founders = PopulationLimit + 1;
    rejects([&] { Simulation sim(config); }, "oversized founders rejected");
    config.founders = 2; config.fertility = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { Simulation sim(config); }, "NaN config rejected");
    config.fertility = 0; config.hazards = 1; config.cooperation = 0;
    Simulation harsh(config);
    for (int i = 0; i < HarshRunTicks; ++i) harsh.step();
    invariants(harsh);
    config.founders = PopulationLimit; config.fertility = 1; config.hazards = 0; config.cooperation = 1;
    Simulation crowded(config);
    for (int i = 0; i < 100; ++i) crowded.step();
    invariants(crowded);
}
void mechanics() {
    // A dense, hazard-ridden world is the research-grounded setting for the
    // emergent disease, fire-succession and cultural-transmission mechanics.
    // The seed is selected deterministically so the suite never races the RNG.
    Config config;
    config.seed = 43;
    config.founders = 256;
    config.fertility = 1;
    config.hazards = 1;
    config.cooperation = 0.9f;
    Simulation dense(config);
    const auto initialForest = std::count_if(dense.tiles().begin(), dense.tiles().end(),
                                             [](const Tile& tile) { return tile.terrain == Terrain::Forest; });
    bool plague = false, recovery = false, diseaseDeath = false, clearing = false;
    for (int i = 0; i < MechanicsRunTicks; ++i) {
        dense.step();
        if (!plague && std::any_of(dense.events().begin(), dense.events().end(),
                                   [](const Event& event) { return event.kind == "plague"; })) plague = true;
        if (!recovery && std::any_of(dense.events().begin(), dense.events().end(),
                                     [](const Event& event) { return event.kind == "recovery"; })) recovery = true;
        if (!diseaseDeath && std::any_of(dense.events().begin(), dense.events().end(), [](const Event& event) {
                return event.kind == "death" && event.text.find("disease") != std::string::npos;
            })) diseaseDeath = true;
        if (!clearing && std::any_of(dense.tiles().begin(), dense.tiles().end(),
                                     [](const Tile& tile) { return tile.burned; })) clearing = true;
        if (i % 400 == 0) invariants(dense);
    }
    invariants(dense);
    const auto finalForest = std::count_if(dense.tiles().begin(), dense.tiles().end(),
                                           [](const Tile& tile) { return tile.terrain == Terrain::Forest; });
    std::cout << "mechanics seed" << config.seed << ": plague=" << plague << " recovery=" << recovery
              << " disease_death=" << diseaseDeath << " burned_clearing=" << clearing
              << " forest=" << finalForest << "/" << initialForest
              << " population=" << dense.stats().population << '\n';
    require(plague && recovery, "disease must ignite and infected citizens must recover with immunity");
    require(diseaseDeath, "illness must weaken citizens severely enough to cause deaths");
    require(clearing, "burning forest must leave charred succession clears");
    require(finalForest < initialForest, "stand-replacing fires must carve grass clearings out of the woodland");
    require(finalForest > 0, "fires must not erase the woodland entirely");
    require(dense.stats().population > 0, "the epidemic must not extinguish the civilization");
    require(dense.stats().births > 0, "a dense society must still reproduce under disease pressure");
    // Fire ecology without hazard pressure: same seed, no ignition source, so
    // there must be no charred clearing anywhere. This isolates fire as the
    // disturbance driver behind succession.
    Config calm = config;
    calm.hazards = 0;
    Simulation serene(calm);
    int calmBurned = 0;
    for (int i = 0; i < MechanicsRunTicks; ++i) {
        serene.step();
        if (i % 400 == 0) invariants(serene);
    }
    invariants(serene);
    for (const Tile& ground : serene.tiles()) if (ground.burned) ++calmBurned;
    std::cout << "mechanics calm: burned=" << calmBurned << " population=" << serene.stats().population << '\n';
    require(calmBurned == 0, "without fires, no tile may enter the burned succession state");
    std::uint64_t serenefires = 0;
    for (const Event& event : serene.events()) if (event.kind == "fire" || event.kind == "destruction") ++serenefires;
    require(serenefires == 0, "without hazard pressure, lightning must never ignite the forest");
}
bool territoryBoundary(const Simulation& sim) {
    const auto& tiles = sim.tiles();
    for (int y = 0; y < WorldHeight; ++y) {
        for (int x = 0; x < WorldWidth; ++x) {
            const int here = sim.territory(x, y);
            const int cell = y * WorldWidth + x;
            if (x + 1 < WorldWidth &&
                tiles[static_cast<std::size_t>(cell)].terrain != Terrain::Water &&
                tiles[static_cast<std::size_t>(cell + 1)].terrain != Terrain::Water &&
                here != sim.territory(x + 1, y)) return true;
            if (y + 1 < WorldHeight &&
                tiles[static_cast<std::size_t>(cell)].terrain != Terrain::Rock &&
                tiles[static_cast<std::size_t>(cell + WorldWidth)].terrain != Terrain::Rock &&
                here != sim.territory(x, y + 1)) return true;
        }
    }
    return false;
}
void borders() {
    Config config;
    config.seed = 42;
    Simulation a(config), b(config);
    require(a.territory().size() == static_cast<std::size_t>(WorldWidth * WorldHeight),
            "territory labels must cover every tile");
    require(a.territory() == b.territory(), "territory must derive deterministically from the seed");
    std::set<int> clans;
    for (int label : a.territory()) if (label >= 0) clans.insert(label);
    require(clans.size() >= 2, "distinct civilisations must claim distinct land immediately");
    require(territoryBoundary(a), "claimed land must meet at a visible border from the start");
    for (int i = 0; i < HarshRunTicks; ++i) {
        a.step();
        if (i % 500 == 0) require(territoryBoundary(a), "borders must persist as the world develops");
    }
    require(territoryBoundary(a), "borders must survive sustained settlement");
    int claimed = 0;
    for (int label : a.territory()) if (label >= 0) ++claimed;
    std::cout << "seed42 borders: clans=" << clans.size() << " claimed=" << claimed << '\n';
}
void threadDeterminism() {
    Config config;
    config.seed = 42;
    config.threads = 1;
    Simulation serial(config);
    config.threads = 4;
    Simulation parallel(config);
    const int ticks = 250;
    for (int i = 0; i < ticks; ++i) { serial.step(); parallel.step(); }
    require(serial.digest() == parallel.digest(),
            "digest is independent of the configured worker count");
    parallel::setWorkerCount(0);
}
}
int main() {
    try {
        timing(); std::cout << "PASS fixed5Hz timing and catchup\n";
        neural(); std::cout << "PASS deep neural inference, learning, masks, inheritance\n";
        culture(); std::cout << "PASS social imitation and cultural transmission\n";
        observations(); std::cout << "PASS 82-feature individual and environmental observations\n";
        society(); std::cout << "PASS autonomous society, event ranks, determinism\n";
        borders(); std::cout << "PASS deterministic civilisation territory borders\n";
        mechanics(); std::cout << "PASS disease, fire succession and cultural mechanics\n";
        threadDeterminism(); std::cout << "PASS deterministic multithreaded simulation\n";
        extremes(); std::cout << "PASS invalid and extreme configurations\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
