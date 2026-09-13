#include "neural.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace pixels {
namespace {
float unit(float value) {
    return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f;
}

Observation normalized(const Observation& input) {
    Observation result{};
    std::transform(input.begin(), input.end(), result.begin(), unit);
    return result;
}

// These targets are only an offline curriculum. Runtime choices always use the
// learned network; neither this function nor an urgency rule runs in choose().
Values curriculum(const Observation& x) {
    const float hunger = x[0], thirst = x[1], fatigue = x[2];
    const float food = x[5], wood = x[6], cooperative = x[7];
    const float needs = std::max({hunger, thirst, fatigue});
    const float scarcity = 1.0f - food;
    Values q{};
    q[static_cast<int>(Action::Wander)] = -0.15f + 3.4f * x[16] + 0.08f * (1 - x[13]);
    q[static_cast<int>(Action::Gather)] = -0.25f + 1.5f * scarcity + 0.55f * hunger * scarcity
        - 0.35f * std::max(thirst, fatigue) + 0.06f * x[10];
    q[static_cast<int>(Action::Eat)] = -0.50f + 3.2f * hunger;
    q[static_cast<int>(Action::Drink)] = -0.55f + 3.4f * thirst + 0.05f * x[11];
    q[static_cast<int>(Action::Rest)] = -0.45f + 2.8f * fatigue + 0.10f * x[13];
    q[static_cast<int>(Action::Chop)] = 0.85f - 1.2f * wood - 0.35f * needs
        - 0.35f * scarcity + 0.25f * (1 - x[13]) + 0.06f * x[12];
    q[static_cast<int>(Action::Build)] = 0.20f + 0.45f * wood + 0.28f * (1 - x[13])
        - 0.50f * needs;
    q[static_cast<int>(Action::Farm)] = 0.12f + 0.20f * wood + 0.35f * x[19]
        + 0.15f * (1 - x[14]) - 0.40f * needs;
    q[static_cast<int>(Action::Share)] = -0.35f + 1.2f * food * cooperative
        + 0.12f * x[15] - 0.30f * needs;
    q[static_cast<int>(Action::Socialize)] = -0.30f + 1.4f * x[3]
        + 0.15f * cooperative + 0.12f * x[15] - 0.25f * needs;
    q[static_cast<int>(Action::Reproduce)] = -0.20f + 1.20f * x[18] + 0.15f * food
        + 0.15f * x[15] - 0.60f * needs;
    q[static_cast<int>(Action::Attack)] = -0.55f + 1.25f * x[8] - 0.60f * cooperative
        + 0.16f * x[15] + 0.20f * hunger * scarcity;
    return q;
}
} // namespace

const char* actionName(Action action) {
    static constexpr const char* names[ActionCount] = {
        "WANDER", "GATHER", "EAT", "DRINK", "REST", "CHOP",
        "BUILD", "FARM", "SHARE", "SOCIALIZE", "REPRODUCE", "ATTACK"
    };
    const int index = static_cast<int>(action);
    return index >= 0 && index < ActionCount ? names[index] : "UNKNOWN";
}

Brain::Brain(std::uint32_t seed) {
    std::mt19937 rng(seed);
    // Xavier initialization keeps tanh units away from saturation.
    std::uniform_real_distribution<float> inputWeight(-std::sqrt(6.0f / (InputCount + HiddenCount)),
                                                     std::sqrt(6.0f / (InputCount + HiddenCount)));
    std::uniform_real_distribution<float> outputWeight(-std::sqrt(6.0f / (HiddenCount + ActionCount)),
                                                      std::sqrt(6.0f / (HiddenCount + ActionCount)));
    for (auto& row : w1_) for (float& value : row) value = inputWeight(rng);
    for (auto& row : w2_) for (float& value : row) value = outputWeight(rng);
}

Values Brain::predict(const Observation& input) const {
    const Observation x = normalized(input);
    std::array<float, HiddenCount> hidden{};
    for (int h = 0; h < HiddenCount; ++h) {
        float sum = b1_[h];
        for (int i = 0; i < InputCount; ++i) sum += w1_[h][i] * x[i];
        hidden[h] = std::tanh(sum);
    }
    Values values{};
    for (int a = 0; a < ActionCount; ++a) {
        values[a] = b2_[a];
        for (int h = 0; h < HiddenCount; ++h) values[a] += w2_[a][h] * hidden[h];
    }
    return values;
}

Action Brain::choose(const Observation& input, const ActionMask& legal,
                     float exploration, std::mt19937& rng) const {
    std::array<int, ActionCount> candidates{};
    int count = 0;
    for (int a = 0; a < ActionCount; ++a) if (legal[a]) candidates[count++] = a;
    if (count == 0) throw std::invalid_argument("Brain::choose requires at least one legal action");

    std::uniform_real_distribution<float> chance(0.0f, 1.0f);
    if (chance(rng) < unit(exploration)) {
        std::uniform_int_distribution<int> pick(0, count - 1);
        return static_cast<Action>(candidates[pick(rng)]);
    }
    const Values values = predict(input);
    int best = candidates[0];
    for (int i = 1; i < count; ++i) {
        if (values[candidates[i]] > values[best]) best = candidates[i];
    }
    return static_cast<Action>(best);
}

float Brain::train(const Observation& input, Action action, float target, float rate) {
    const int a = static_cast<int>(action);
    if (a < 0 || a >= ActionCount) throw std::invalid_argument("Unknown training action");
    const Observation x = normalized(input);
    target = std::isfinite(target) ? std::clamp(target, -4.0f, 4.0f) : 0.0f;
    rate = std::isfinite(rate) ? std::clamp(rate, 0.0f, 0.1f) : 0.0f;
    std::array<float, HiddenCount> hidden{};
    for (int h = 0; h < HiddenCount; ++h) {
        float sum = b1_[h];
        for (int i = 0; i < InputCount; ++i) sum += w1_[h][i] * x[i];
        hidden[h] = std::tanh(sum);
    }
    float value = b2_[a];
    for (int h = 0; h < HiddenCount; ++h) value += w2_[a][h] * hidden[h];
    const float error = value - target;
    const float gradient = std::clamp(error, -2.0f, 2.0f);
    // Keep the pre-update output weights: the hidden gradient must use the
    // weights from the same forward pass, not the already modified weights.
    const auto outputWeights = w2_[a];
    auto update = [rate](float& parameter, float derivative) {
        parameter = std::clamp(parameter - rate * derivative, -6.0f, 6.0f);
    };
    update(b2_[a], gradient);
    for (int h = 0; h < HiddenCount; ++h) {
        update(w2_[a][h], gradient * hidden[h]);
        const float delta = gradient * outputWeights[h] * (1.0f - hidden[h] * hidden[h]);
        update(b1_[h], delta);
        for (int i = 0; i < InputCount; ++i) update(w1_[h][i], delta * x[i]);
    }
    ++updates_;
    return error * error;
}

void Brain::learn(const Observation& before, Action action, float reward,
                  const Observation& after, const ActionMask& legal, bool terminal) {
    float future = 0.0f;
    if (!terminal) {
        const Values values = predict(after);
        bool hasFuture = false;
        for (int a = 0; a < ActionCount; ++a) if (legal[a]) {
            if (!hasFuture || values[a] > future) future = values[a];
            hasFuture = true;
        }
    }
    reward = std::isfinite(reward) ? std::clamp(reward, -2.0f, 2.0f) : 0.0f;
    constexpr float discount = 0.85f;
    train(before, action, reward + discount * future, 0.008f);
}

void Brain::mutate(std::mt19937& rng, float amount) {
    amount = std::isfinite(amount) ? std::clamp(amount, 0.0f, 1.0f) : 0.0f;
    if (amount == 0.0f) return;
    std::normal_distribution<float> noise(0.0f, amount);
    auto perturb = [&](float& weight) { weight = std::clamp(weight + noise(rng), -6.0f, 6.0f); };
    for (auto& row : w1_) for (float& weight : row) perturb(weight);
    for (auto& weight : b1_) perturb(weight);
    for (auto& row : w2_) for (float& weight : row) perturb(weight);
    for (auto& weight : b2_) perturb(weight);
}

double Brain::fingerprint() const {
    double sum = 0;
    std::uint64_t index = 1;
    auto include = [&](float parameter) {
        sum += static_cast<double>(parameter) * static_cast<double>(index++);
    };
    for (const auto& row : w1_) for (float weight : row) include(weight);
    for (float weight : b1_) include(weight);
    for (const auto& row : w2_) for (float weight : row) include(weight);
    for (float weight : b2_) include(weight);
    return sum;
}

std::uint64_t Brain::digest() const {
    std::uint64_t hash = 14695981039346656037ull;
    auto includeByte = [&](std::uint8_t byte) {
        hash ^= byte;
        hash *= 1099511628211ull;
    };
    auto include = [&](float parameter) {
        static_assert(sizeof(float) == sizeof(std::uint32_t), "32-bit float required");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &parameter, sizeof(bits));
        for (int shift = 0; shift < 32; shift += 8) includeByte(static_cast<std::uint8_t>(bits >> shift));
    };
    for (const auto& row : w1_) for (float weight : row) include(weight);
    for (float weight : b1_) include(weight);
    for (const auto& row : w2_) for (float weight : row) include(weight);
    for (float weight : b2_) include(weight);
    for (int shift = 0; shift < 64; shift += 8) includeByte(static_cast<std::uint8_t>(updates_ >> shift));
    return hash;
}

Brain makeFounderBrain(std::uint32_t seed) {
    Brain brain(seed);
    std::mt19937 rng(seed ^ 0xA17E5EEDu);
    std::uniform_real_distribution<float> unitSample(0.0f, 1.0f);
    // Broad synthetic coverage includes quiet daily life, emergencies, scarce
    // resources and different temperaments. This is supervised initialization,
    // followed by individual online temporal-difference learning in the world.
    constexpr int experiences = 8000;
    for (int sample = 0; sample < experiences; ++sample) {
        Observation input{};
        for (float& value : input) value = unitSample(rng);
        // Most time is spent in non-emergency states and near no fire.
        if (sample % 3 != 0) {
            for (int i = 0; i < 5; ++i) input[i] *= 0.55f;
            input[16] = 0.0f;
        }
        for (int i = 10; i <= 15; ++i) {
            const float distance = std::floor(unitSample(rng) * 20.0f);
            input[i] = 1.0f / (1.0f + distance);
        }
        input[18] = unitSample(rng) < 0.35f ? 1.0f : 0.0f;
        const Values targets = curriculum(input);
        const int first = sample % ActionCount;
        const float rate = sample < 6000 ? 0.012f : 0.005f;
        for (int offset = 0; offset < ActionCount; ++offset) {
            const int a = (first + offset) % ActionCount;
            brain.train(input, static_cast<Action>(a), targets[a], rate);
        }
    }
    return brain;
}
} // namespace pixels
