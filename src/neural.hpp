#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <random>

namespace pixels {
// The observation contract is deliberately explicit: every citizen sees the
// same 82 normalized environmental, social and historical measurements.
inline constexpr int InputCount = 82;
inline constexpr int HiddenOneCount = 56;
inline constexpr int HiddenTwoCount = 28;
// Kept as a source-compatible name for callers that used the original single
// hidden-layer constant. New code should name the layer it means.
inline constexpr int HiddenCount = HiddenOneCount;
inline constexpr int ActionCount = 12;

// The society core is a per-civilization neural network with over ten million
// trainable parameters. Every simulation trains its own core from its world
// seed, then every living citizen reads that civilization's advice on top of
// its own 82 live sensors. One advisory channel is produced per action.
inline constexpr int CoreHiddenOneCount = 2048;
inline constexpr int CoreHiddenTwoCount = 2560;
inline constexpr int CoreHiddenThreeCount = 2560;
inline constexpr int CoreAdviceCount = ActionCount;

using Observation = std::array<float, InputCount>;
using Values = std::array<float, ActionCount>;
using ActionMask = std::array<bool, ActionCount>;
using CoreInput = std::array<float, InputCount>;
// A citizen policy input is its own 82-feature observation plus the society
// core's 12 normalized advisory values, so individual needs and shared
// society context always reach the same network.
inline constexpr int BrainInputCount = InputCount + CoreAdviceCount;
using BrainInput = std::array<float, BrainInputCount>;

BrainInput compose(const Observation& observation, const Values& advice);

// Order is part of the policy's observation/action contract.
enum class Action { Wander, Gather, Eat, Drink, Rest, Chop, Build, Farm, Share, Socialize, Reproduce, Attack };
const char* actionName(Action action);

// Shared society-wide core network. One instance exists per simulation and is
// shared by every citizen of that civilization; it summarizes the population's
// mean observation into normalized per-action advice that individual policies
// combine with their own state. Its trained parameters are distinct for each
// world seed. It holds 12,002,316 trainable scalar parameters.
class SocietyCore {
public:
    explicit SocietyCore(std::uint32_t seed = 0xC03E5EEDu);
    // Advisor values in [0,1], one per action, for one society aggregate
    // observation (normally the population mean of all living citizens).
    Values advise(const CoreInput& society) const;
    // Fit all twelve advisory outputs toward bounded targets (mean squared
    // error). Returns the mean squared error over all twelve outputs.
    float train(const CoreInput& society, const Values& targets, float rate = 0.05f);
    static int parameterCount();
    std::uint64_t digest() const;

private:
    std::array<std::array<float, InputCount>, CoreHiddenOneCount> c1_{};
    std::array<float, CoreHiddenOneCount> cb1_{};
    std::array<std::array<float, CoreHiddenOneCount>, CoreHiddenTwoCount> c2_{};
    std::array<float, CoreHiddenTwoCount> cb2_{};
    std::array<std::array<float, CoreHiddenTwoCount>, CoreHiddenThreeCount> c3_{};
    std::array<float, CoreHiddenThreeCount> cb3_{};
    std::array<std::array<float, CoreHiddenThreeCount>, ActionCount> c4_{};
    std::array<float, ActionCount> cb4_{};
    // A core is only trained while it is being prepared.  Thereafter it is
    // shared read-only by every resident, so cache its very expensive
    // parameter digest instead of walking twelve million floats for every
    // diagnostic world digest.
    mutable std::uint64_t cachedDigest_ = 0;
    mutable bool digestValid_ = false;

    struct ForwardPass {
        CoreInput input{};
        std::array<float, CoreHiddenOneCount> first{};
        std::array<float, CoreHiddenTwoCount> second{};
        std::array<float, CoreHiddenThreeCount> third{};
        Values raw{};
    };
    ForwardPass forward(const CoreInput& society) const;
};

// Trains a civilization's society core from its world seed: a distinct fitted
// prior for every seeded world, using the same synthetic curriculum as the
// founders. Cached per seed so an identical seed (a restarted experiment, or
// the observer's reference copy) reuses the exact same trained core at no cost.
std::shared_ptr<SocietyCore> makeSocietyCore(std::uint32_t seed);

class Brain {
public:
    explicit Brain(std::uint32_t seed = 1);
    Values predict(const BrainInput& input) const;
    Action choose(const BrainInput& input, const ActionMask& legal, float exploration, std::mt19937& rng) const;
    float train(const BrainInput& input, Action action, float target, float rate = 0.015f);
    void learn(const BrainInput& before, Action action, float reward, const BrainInput& after,
               const ActionMask& legal, bool terminal = false);
    void mutate(std::mt19937& rng, float amount = 0.025f);
    // Social/cultural transmission: blend every parameter toward a donor's
    // policy. The donor keeps its weights untouched; only the learner moves.
    void imitate(const Brain& donor, float rate);
    static int parameterCount();
    std::uint64_t updates() const { return updates_; }
    double fingerprint() const;
    std::uint64_t digest() const;
private:
    friend Brain makeFounderBrain(std::uint32_t seed, const SocietyCore& core);
    std::array<std::array<float, BrainInputCount>, HiddenOneCount> w1_{};
    std::array<float, HiddenOneCount> b1_{};
    std::array<std::array<float, HiddenOneCount>, HiddenTwoCount> w2_{};
    std::array<float, HiddenTwoCount> b2_{};
    std::array<std::array<float, HiddenTwoCount>, ActionCount> w3_{};
    std::array<float, ActionCount> b3_{};
    std::uint64_t updates_ = 0;

    struct ForwardPass {
        BrainInput input{};
        std::array<float, HiddenOneCount> first{};
        std::array<float, HiddenTwoCount> second{};
        Values values{};
    };
    ForwardPass forward(const BrainInput& input) const;

    // Founder preparation can fit every action target from one shared forward
    // pass. It remains private so runtime learning continues to update only
    // the action the citizen actually selected.
    float trainTargets(const BrainInput& input, const Values& targets, float rate);
};

// One reproducible bootstrap policy shared by founders; subsequent learning is individual.
Brain makeFounderBrain(std::uint32_t seed, const SocietyCore& core);
}
