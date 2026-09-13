#include "simulation.hpp"
#include "ticker.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

using namespace pixels;
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F&& operation, const std::string& message) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}
void neural() {
    Brain brain(7), same(7), other(8);
    Observation obs{}; obs[0] = 0.8f; obs[5] = 0.6f; obs[7] = 0.7f;
    require(brain.predict(obs) == same.predict(obs), "seeded inference must reproduce");
    require(brain.predict(obs) != other.predict(obs), "different seeds must vary networks");
    const auto before = brain.predict(obs);
    for (int i = 0; i < 300; ++i) brain.train(obs, Action::Eat, 1.5f);
    require(std::abs(brain.predict(obs)[2] - 1.5f) < std::abs(before[2] - 1.5f) * 0.1f,
            "backpropagation must learn the target, not only count updates");
    require(brain.updates() == 300, "training updates must be recorded");
    ActionMask legal{}; legal.fill(true);
    std::mt19937 rng(42);
    Brain preferGather(12), preferDrink(12);
    for (int n = 0; n < 200; ++n) {
        for (int a = 0; a < ActionCount; ++a) {
            preferGather.train(obs, static_cast<Action>(a), a == 1 ? 2.f : -1.f);
            preferDrink.train(obs, static_cast<Action>(a), a == 3 ? 2.f : -1.f);
        }
    }
    require(preferGather.choose(obs, legal, 0, rng) == Action::Gather, "learned weights must control decisions");
    require(preferDrink.choose(obs, legal, 0, rng) == Action::Drink, "changing learned weights must change intention");
    legal.fill(false); legal[4] = true;
    for (int n = 0; n < 100; ++n)
        require(brain.choose(obs, legal, n % 2 ? 1 : 0, rng) == Action::Rest, "exploration and inference must both respect legality");
    legal.fill(false);
    rejects([&] { brain.choose(obs, legal, 0, rng); }, "no legal action must fail explicitly");
    auto inherited = brain;
    require(inherited.fingerprint() == brain.fingerprint(), "children can inherit exact trained weights");
    inherited.mutate(rng);
    require(inherited.fingerprint() != brain.fingerprint(), "mutation must actually change policy");
    legal.fill(true);
    auto previous = brain.fingerprint();
    brain.learn(obs, Action::Eat, -0.7f, obs, legal);
    require(brain.fingerprint() != previous, "online temporal difference update must affect weights");
    for (int i = 0; i < 2000; ++i) brain.learn(obs, Action::Eat, i % 2 ? 1.f : -1.f, obs, legal);
    for (float value : brain.predict(obs)) require(std::isfinite(value), "online learning must stay finite");
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
        for (float output : c.brain.predict(sim.observe(c))) require(std::isfinite(output), "network values finite");
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
    for (int i = 500; i < 6000; ++i) {
        a.step();
        if (i % 500 == 0) invariants(a);
    }
    invariants(a);
    const auto& s = a.stats();
    std::cout << "seed42 at6000 ticks: population=" << s.population << " births=" << s.births
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
    for (int i = 0; i < 1500; ++i) harsh.step();
    invariants(harsh);
    config.founders = PopulationLimit; config.fertility = 1; config.hazards = 0; config.cooperation = 1;
    Simulation crowded(config);
    for (int i = 0; i < 100; ++i) crowded.step();
    invariants(crowded);
}
}
int main() {
    try {
        timing(); std::cout << "PASS fixed5Hz timing and catchup\n";
        neural(); std::cout << "PASS neural inference, learning, masks, inheritance\n";
        society(); std::cout << "PASS autonomous society, event ranks, determinism\n";
        extremes(); std::cout << "PASS invalid and extreme configurations\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
