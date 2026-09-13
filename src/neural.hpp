#pragma once
#include <array>
#include <cstdint>
#include <random>

namespace pixels {
inline constexpr int InputCount = 20;
inline constexpr int HiddenCount = 24;
inline constexpr int ActionCount = 12;
using Observation = std::array<float, InputCount>;
using Values = std::array<float, ActionCount>;
using ActionMask = std::array<bool, ActionCount>;

// Order is part of the policy's observation/action contract.
enum class Action { Wander, Gather, Eat, Drink, Rest, Chop, Build, Farm, Share, Socialize, Reproduce, Attack };
const char* actionName(Action action);

class Brain {
public:
    explicit Brain(std::uint32_t seed = 1);
    Values predict(const Observation& input) const;
    Action choose(const Observation& input, const ActionMask& legal, float exploration, std::mt19937& rng) const;
    float train(const Observation& input, Action action, float target, float rate = 0.015f);
    void learn(const Observation& before, Action action, float reward, const Observation& after,
               const ActionMask& legal, bool terminal = false);
    void mutate(std::mt19937& rng, float amount = 0.025f);
    std::uint64_t updates() const { return updates_; }
    double fingerprint() const;
    std::uint64_t digest() const;
private:
    std::array<std::array<float, InputCount>, HiddenCount> w1_{};
    std::array<float, HiddenCount> b1_{};
    std::array<std::array<float, HiddenCount>, ActionCount> w2_{};
    std::array<float, ActionCount> b2_{};
    std::uint64_t updates_ = 0;
};

// One reproducible bootstrap policy shared by founders; subsequent learning is individual.
Brain makeFounderBrain(std::uint32_t seed);
}
