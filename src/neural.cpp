#include "neural.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pixels {
namespace {
constexpr float ParameterLimit = 6.0f;
constexpr float GradientLimit = 2.0f;

float unit(float value) {
    return std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f;
}

float sampleUnit(std::mt19937& rng) {
    // Use a fixed set of engine bits instead of implementation-defined real
    // distributions for reproducible founder examples and initialization.
    return static_cast<float>(rng() >> 8) / 16777216.0f;
}

template<std::size_t N>
std::array<float, N> normalized(const std::array<float, N>& values) {
    std::array<float, N> result{};
    std::transform(values.begin(), values.end(), result.begin(), unit);
    return result;
}

void updateParameter(float& parameter, float rate, float derivative) {
    parameter = std::clamp(parameter - rate * derivative, -ParameterLimit, ParameterLimit);
}

float prior(const Observation& x, Action action) {
    return x[70 + static_cast<int>(action)];
}

// These targets are an offline founder curriculum only. Runtime decisions use
// neural action values exclusively; no branch in choose() reads these rules.
Values curriculum(const Observation& x) {
    const float hunger = x[0];
    const float thirst = x[1];
    const float fatigue = x[2];
    const float socialNeed = x[3];
    const float healthDeficit = x[4];
    const float food = x[5];
    const float wood = x[6];
    const float cooperation = x[7];
    const float aggression = x[8];
    const float age = x[9];
    const float nearForage = x[10];
    const float nearWater = x[11];
    const float nearForest = x[12];
    const float nearHome = x[13];
    const float nearFarm = x[14];
    const float nearCitizen = x[15];
    const float localFire = x[16];
    const float seasonalYield = x[17];
    const float birthReady = x[18];
    const float localFertility = x[19];

    const float recentReward = x[20];
    const float population = x[21];
    const float homeCoverage = x[22];
    const float farmCoverage = x[23];
    const float foodSecurity = x[24];
    const float wellbeing = x[25];
    const float communityCooperation = x[26];
    const float generation = x[27];
    const float seasonalGrowth = x[28] + .80f * x[29] + .60f * x[30] + .30f * x[31];

    const float tileFood = x[32];
    const float tileWood = x[33];
    const float traffic = x[34];
    const float onHome = x[35];
    const float onFarm = x[36];
    const float onSand = x[37];
    const float onGrass = x[38];
    const float onForest = x[39];
    const float onRock = x[40];

    const float nearFood2 = x[41];
    const float nearWood2 = x[42];
    const float fertility2 = x[43];
    const float fire2 = x[44];
    const float traffic2 = x[45];
    const float homes2 = x[46];
    const float farms2 = x[47];
    const float people2 = x[48];
    const float cooperation2 = x[49];
    const float aggression2 = x[50];
    const float diversity2 = x[51];
    const float nearFood5 = x[52];
    const float nearWood5 = x[53];
    const float fertility5 = x[54];
    const float fire5 = x[55];
    const float traffic5 = x[56];
    const float homes5 = x[57];
    const float farms5 = x[58];
    const float people5 = x[59];
    const float cooperation5 = x[60];
    const float aggression5 = x[61];
    const float diversity5 = x[62];

    const float neighbourHunger = x[63];
    const float neighbourFood = x[64];
    const float neighbourCooperation = x[65];
    const float neighbourAggression = x[66];
    const float neighbourSocialNeed = x[67];
    const float sameClan = x[68];
    const float differentClan = x[69];
    const float reserveFood = x[83];
    const float reserveWood = x[84];
    const float storeCoverage = x[85];
    const float storeNear = x[86];
    const float onStore = x[87];
    const float foodStoreSpace = x[88];
    const float woodStoreSpace = x[89];
    const float logisticsSkill = x[90];
    const float nearbyNeed = x[91];

    const float needs = std::max({hunger, thirst, fatigue, healthDeficit * .85f});
    const float scarcity = 1.0f - foodSecurity;
    const float foodNearby = std::max({nearForage, tileFood, nearFood2, nearFood5});
    const float woodNearby = std::max({nearForest, tileWood, nearWood2, nearWood5});
    const float fertileNearby = std::max({localFertility, fertility2, fertility5});
    const float fireRisk = std::max({localFire, fire2, fire5});
    const float housingNearby = std::max({nearHome, onHome, homes2, homes5});
    const float farmingNearby = std::max({nearFarm, onFarm, farms2, farms5});
    const float company = std::max({nearCitizen, people2, people5});
    const float localCooperation = (cooperation2 + cooperation5 + neighbourCooperation) / 3.0f;
    const float localAggression = (aggression2 + aggression5 + neighbourAggression) / 3.0f;
    const float diversity = (diversity2 + diversity5 + differentClan) / 3.0f;
    const float infrastructureGap = 1.0f - std::max(homeCoverage, housingNearby);
    const float agricultureGap = 1.0f - std::max(farmCoverage, farmingNearby);
    const float socialStrain = std::max(socialNeed, neighbourSocialNeed);
    const float crowding = population * company;
    const float terrainForage = .72f * onGrass + .28f * onSand + .15f * onForest;
    const float terrainWood = .70f * onForest + .10f * onGrass;
    const float settlementFlow = .5f * traffic + .25f * traffic2 + .25f * traffic5;
    const float hostileContext = unit(.45f * aggression + .25f * localAggression +
                                      .20f * differentClan + .10f * diversity);
    const float continuity = .025f + .055f * recentReward;

    Values q{};
    q[static_cast<int>(Action::Wander)] = -.22f + 3.25f * fireRisk + .32f * (1.0f - foodNearby) * scarcity +
        .15f * (1.0f - settlementFlow) + .10f * onRock + .06f * prior(x, Action::Wander);
    q[static_cast<int>(Action::Gather)] = -.30f + 1.28f * scarcity + .62f * hunger * scarcity +
        .56f * foodNearby + .22f * terrainForage + .18f * seasonalGrowth + .12f * (1.0f - farmCoverage) -
        .36f * std::max(thirst, fatigue) - .55f * fireRisk + continuity * prior(x, Action::Gather);
    q[static_cast<int>(Action::Eat)] = -.55f + 3.30f * hunger + .38f * food - .28f * fireRisk +
        .08f * prior(x, Action::Eat);
    q[static_cast<int>(Action::Drink)] = -.58f + 3.45f * thirst + .24f * nearWater - .20f * fireRisk +
        .08f * prior(x, Action::Drink);
    q[static_cast<int>(Action::Rest)] = -.48f + 2.82f * fatigue + .28f * housingNearby + .16f * homes2 +
        .12f * wellbeing - .36f * fireRisk + .07f * prior(x, Action::Rest);
    q[static_cast<int>(Action::Chop)] = .48f + .83f * woodNearby + .32f * terrainWood + .34f * infrastructureGap +
        .29f * agricultureGap + .12f * settlementFlow - 1.12f * wood - .40f * needs - .33f * scarcity -
        .42f * fireRisk + .06f * prior(x, Action::Chop);
    q[static_cast<int>(Action::Build)] = -.22f + 1.22f * infrastructureGap * (.25f + wood) + .42f * wood +
        .30f * company + .20f * (communityCooperation + localCooperation) + .10f * generation - .56f * needs -
        .48f * fireRisk + .06f * prior(x, Action::Build);
    q[static_cast<int>(Action::Farm)] = -.25f + 1.16f * agricultureGap * (.28f + wood) + .42f * fertileNearby +
        .31f * seasonalYield + .25f * seasonalGrowth + .25f * scarcity + .12f * onFarm - .52f * needs -
        .44f * fireRisk + .06f * prior(x, Action::Farm);
    q[static_cast<int>(Action::Share)] = -.50f + 1.08f * food * cooperation + .54f * neighbourHunger * company +
        .25f * socialStrain + .24f * (communityCooperation + localCooperation) + .14f * sameClan -
        .33f * needs - .28f * hostileContext + .06f * prior(x, Action::Share);
    q[static_cast<int>(Action::Socialize)] = -.38f + 1.40f * socialNeed + .29f * company +
        .20f * (cooperation + neighbourCooperation) + .17f * sameClan + .12f * (1.0f - neighbourFood) -
        .30f * needs - .30f * hostileContext + .06f * prior(x, Action::Socialize);
    q[static_cast<int>(Action::Reproduce)] = -.34f + 1.38f * birthReady + .25f * foodSecurity + .22f * wellbeing +
        .26f * sameClan + .18f * (1.0f - crowding) + .10f * age * (1.0f - age) - .60f * needs -
        .35f * fireRisk + .05f * prior(x, Action::Reproduce);
    q[static_cast<int>(Action::Attack)] = -.82f + 1.42f * aggression + .48f * hostileContext +
        .31f * differentClan * (1.0f - neighbourCooperation) + .22f * neighbourFood * scarcity +
        .16f * localAggression * company - .78f * cooperation - .24f * communityCooperation -
        .18f * wellbeing + .05f * prior(x, Action::Attack);
    const float storedProvision = .55f * reserveFood + .45f * reserveWood;
    const float carryingSurplus = unit(.62f * food + .38f * wood);
    const float storeSpace = std::max(foodStoreSpace, woodStoreSpace);
    q[static_cast<int>(Action::Haul)] = -.72f + .82f * storeNear + .52f * onStore +
        .68f * carryingSurplus * storeSpace + .46f * nearbyNeed * reserveFood +
        .20f * storeCoverage + .17f * logisticsSkill + .12f * storedProvision -
        .46f * needs - .38f * fireRisk + .06f * prior(x, Action::Haul);
    for (float& value : q) value = std::clamp(value, -3.5f, 3.5f);
    return q;
}

Observation founderObservation(std::mt19937& rng, int sample) {
    Observation x{};
    for (float& value : x) value = sampleUnit(rng);

    // Season, terrain and previous intention are categorical observations,
    // rather than arbitrary fractional bits in the founder dataset.
    x[28] = x[29] = x[30] = x[31] = 0.0f;
    const int season = static_cast<int>(rng() % 4);
    x[28 + season] = 1.0f;
    static constexpr float yields[] = {1.0f, .80f, .60f, .30f};
    x[17] = yields[season];
    x[37] = x[38] = x[39] = x[40] = 0.0f;
    const int terrain = static_cast<int>(rng() % 4);
    x[37 + terrain] = 1.0f;
    for (int action = 0; action < ActionCount; ++action) x[70 + action] = 0.0f;
    x[70 + static_cast<int>(rng() % ActionCount)] = 1.0f;

    // Couple broad measurements to nearby measurements so the network sees
    // plausible relationships as well as independent edge cases.
    x[24] = unit(.65f * x[5] + .20f * x[23] + .15f * sampleUnit(rng));
    x[25] = unit(.30f * (1.0f - x[0]) + .25f * (1.0f - x[1]) + .20f * (1.0f - x[2]) +
                 .15f * (1.0f - x[4]) + .10f * (1.0f - x[3]));
    x[26] = unit(.55f * x[7] + .45f * sampleUnit(rng));
    x[32] = unit(.60f * x[10] + .40f * sampleUnit(rng));
    x[33] = unit(.60f * x[12] + .40f * sampleUnit(rng));
    x[34] = unit(.35f * x[15] + .25f * x[22] + .25f * x[23] + .15f * sampleUnit(rng));
    x[35] = sampleUnit(rng) < x[22] ? 1.0f : 0.0f;
    x[36] = sampleUnit(rng) < x[23] ? 1.0f : 0.0f;
    for (int offset : {0, 11}) {
        const int base = 41 + offset;
        x[base + 0] = unit(.58f * x[32] + .42f * sampleUnit(rng));
        x[base + 1] = unit(.58f * x[33] + .42f * sampleUnit(rng));
        x[base + 2] = unit(.62f * x[19] + .38f * sampleUnit(rng));
        x[base + 3] = unit(.72f * x[16] + .28f * sampleUnit(rng));
        x[base + 4] = unit(.70f * x[34] + .30f * sampleUnit(rng));
        x[base + 5] = unit(.65f * x[22] + .35f * sampleUnit(rng));
        x[base + 6] = unit(.65f * x[23] + .35f * sampleUnit(rng));
        x[base + 7] = unit(.50f * x[15] + .50f * sampleUnit(rng));
        x[base + 8] = unit(.60f * x[26] + .40f * sampleUnit(rng));
        x[base + 9] = unit(.45f * x[8] + .55f * sampleUnit(rng));
        x[base + 10] = sampleUnit(rng);
    }
    const float neighbourChance = std::max(x[48], x[59]);
    for (int i = 63; i <= 67; ++i) x[i] = sampleUnit(rng) < neighbourChance ? sampleUnit(rng) : 0.0f;
    x[68] = sampleUnit(rng) < neighbourChance ? sampleUnit(rng) : 0.0f;
    x[69] = sampleUnit(rng) < neighbourChance ? sampleUnit(rng) : 0.0f;
    x[83] = unit(.60f * x[24] + .40f * sampleUnit(rng));
    x[84] = unit(.42f * x[6] + .58f * sampleUnit(rng));
    x[85] = unit(.55f * x[22] + .45f * sampleUnit(rng));
    x[86] = sampleUnit(rng) < x[85] ? sampleUnit(rng) : 0.0f;
    x[87] = sampleUnit(rng) < x[86] ? 1.0f : 0.0f;
    x[88] = 1.0f - x[83];
    x[89] = 1.0f - x[84];
    x[90] = sampleUnit(rng);
    x[91] = unit(.55f * x[0] + .25f * x[3] + .20f * sampleUnit(rng));

    // Repeated archetypes make rare but important choices observable during a
    // small bootstrap without replacing the broad background data above.
    const int scenario = sample % ActionCount;
    for (int action = 0; action < ActionCount; ++action) x[70 + action] = 0.0f;
    x[70 + scenario] = 1.0f;
    switch (scenario) {
    case static_cast<int>(Action::Wander):
        x[16] = x[44] = .85f; x[55] = .55f; x[32] = x[41] = .05f; break;
    case static_cast<int>(Action::Gather):
        x[0] = .72f; x[5] = .08f; x[10] = x[32] = x[41] = .90f; x[24] = .10f; break;
    case static_cast<int>(Action::Eat):
        x[0] = .92f; x[5] = .82f; x[24] = .40f; break;
    case static_cast<int>(Action::Drink):
        x[1] = .94f; x[11] = .92f; break;
    case static_cast<int>(Action::Rest):
        x[2] = .90f; x[13] = x[35] = x[46] = .75f; break;
    case static_cast<int>(Action::Chop):
        x[6] = .06f; x[12] = x[33] = x[42] = .90f; x[39] = 1.0f; x[37] = x[38] = x[40] = 0.0f; break;
    case static_cast<int>(Action::Build):
        x[6] = .78f; x[22] = x[46] = .08f; x[15] = x[48] = .72f; break;
    case static_cast<int>(Action::Farm):
        x[6] = .42f; x[19] = x[43] = x[54] = .92f; x[23] = x[47] = .06f; x[24] = .20f; break;
    case static_cast<int>(Action::Share):
        x[5] = .88f; x[7] = x[26] = .90f; x[15] = x[48] = .90f; x[63] = .90f; x[68] = .85f; break;
    case static_cast<int>(Action::Socialize):
        x[3] = .92f; x[15] = x[48] = .88f; x[65] = .85f; x[67] = .80f; x[68] = .80f; break;
    case static_cast<int>(Action::Reproduce):
        x[18] = .98f; x[5] = .72f; x[24] = x[25] = .80f; x[68] = .90f; x[21] = .22f; break;
    case static_cast<int>(Action::Attack):
        x[8] = x[50] = x[61] = .90f; x[69] = x[62] = .85f; x[65] = .12f; x[66] = .84f; break;
    case static_cast<int>(Action::Haul):
        x[5] = .82f; x[6] = .66f; x[83] = .28f; x[84] = .18f; x[85] = .55f;
        x[86] = x[87] = .92f; x[88] = .72f; x[89] = .82f; x[90] = .45f; x[91] = .62f; break;
    }
    return x;
}
} // namespace

BrainInput compose(const Observation& observation, const Values& advice) {
    BrainInput input{};
    for (int i = 0; i < InputCount; ++i)
        input[static_cast<std::size_t>(i)] = observation[static_cast<std::size_t>(i)];
    for (int a = 0; a < ActionCount; ++a)
        input[static_cast<std::size_t>(InputCount + a)] = advice[static_cast<std::size_t>(a)];
    return input;
}

const char* actionName(Action action) {
    static constexpr const char* names[ActionCount] = {
        "WANDER", "GATHER", "EAT", "DRINK", "REST", "CHOP",
        "BUILD", "FARM", "SHARE", "SOCIALIZE", "REPRODUCE", "ATTACK", "HAUL"
    };
    const int index = static_cast<int>(action);
    return index >= 0 && index < ActionCount ? names[index] : "UNKNOWN";
}

// ---- Society core ----

SocietyCore::SocietyCore(std::uint32_t seed) {
    std::mt19937 rng(seed);
    const float firstLimit = std::sqrt(6.0f / static_cast<float>(InputCount + CoreHiddenOneCount));
    const float secondLimit = std::sqrt(6.0f / static_cast<float>(CoreHiddenOneCount + CoreHiddenTwoCount));
    const float thirdLimit = std::sqrt(6.0f / static_cast<float>(CoreHiddenTwoCount + CoreHiddenThreeCount));
    const float outputLimit = std::sqrt(6.0f / static_cast<float>(CoreHiddenThreeCount + ActionCount));
    auto fill = [&](auto& matrix, float limit) {
        for (auto& row : matrix) for (float& value : row) value = (sampleUnit(rng) * 2.0f - 1.0f) * limit;
    };
    fill(c1_, firstLimit);
    fill(c2_, secondLimit);
    fill(c3_, thirdLimit);
    fill(c4_, outputLimit);
}

SocietyCore::ForwardPass SocietyCore::forward(const CoreInput& society) const {
    ForwardPass pass;
    pass.input = normalized(society);
    // The twelve-million-plus-parameter core is by far the widest network in the
    // simulation, so its hidden layers are worth splitting across the worker
    // pool. Every output unit is an independent dot product that reads only
    // the previous activations and its own weight row, so the row partition is
    // bit-for-bit identical to running the loops serially. runRanges waits for
    // each layer to finish before the next one starts.
    constexpr std::size_t rowChunk = 256;
    parallel::runRanges(static_cast<std::size_t>(CoreHiddenOneCount), rowChunk, [&](std::size_t begin, std::size_t end) {
        for (int h = static_cast<int>(begin); h < static_cast<int>(end); ++h) {
            float sum = cb1_[h];
            for (int i = 0; i < InputCount; ++i) sum += c1_[h][i] * pass.input[i];
            pass.first[h] = std::tanh(sum);
        }
    });
    parallel::runRanges(static_cast<std::size_t>(CoreHiddenTwoCount), rowChunk, [&](std::size_t begin, std::size_t end) {
        for (int h = static_cast<int>(begin); h < static_cast<int>(end); ++h) {
            float sum = cb2_[h];
            for (int i = 0; i < CoreHiddenOneCount; ++i) sum += c2_[h][i] * pass.first[i];
            pass.second[h] = std::tanh(sum);
        }
    });
    parallel::runRanges(static_cast<std::size_t>(CoreHiddenThreeCount), rowChunk, [&](std::size_t begin, std::size_t end) {
        for (int h = static_cast<int>(begin); h < static_cast<int>(end); ++h) {
            float sum = cb3_[h];
            for (int i = 0; i < CoreHiddenTwoCount; ++i) sum += c3_[h][i] * pass.second[i];
            pass.third[h] = std::tanh(sum);
        }
    });
    for (int a = 0; a < ActionCount; ++a) {
        float value = cb4_[a];
        for (int h = 0; h < CoreHiddenThreeCount; ++h) value += c4_[a][h] * pass.third[h];
        pass.raw[a] = value;
    }
    return pass;
}

Values SocietyCore::advise(const CoreInput& society) const {
    const ForwardPass pass = forward(society);
    Values advice{};
    for (int a = 0; a < ActionCount; ++a) {
        advice[a] = unit((std::clamp(pass.raw[a], -3.5f, 3.5f) + 3.5f) * (1.0f / 7.0f));
    }
    return advice;
}

float SocietyCore::train(const CoreInput& society, const Values& targets, float rate) {
    rate = std::isfinite(rate) ? std::clamp(rate, 0.0f, 0.1f) : 0.0f;
    const ForwardPass pass = forward(society);
    std::array<float, ActionCount> outputDelta{};
    float loss = 0.0f;
    for (int a = 0; a < ActionCount; ++a) {
        const float target = std::isfinite(targets[a]) ? std::clamp(targets[a], -4.0f, 4.0f) : 0.0f;
        const float error = pass.raw[a] - target;
        loss += error * error;
        // Mean squared error across all thirteen advisory outputs keeps the
        // shared hidden gradients comparable to one citizen TD update.
        outputDelta[a] = std::clamp(error, -GradientLimit, GradientLimit) /
            static_cast<float>(ActionCount);
    }
    std::array<float, CoreHiddenThreeCount> thirdDelta{};
    for (int h = 0; h < CoreHiddenThreeCount; ++h) {
        float sum = 0.0f;
        for (int a = 0; a < ActionCount; ++a) sum += outputDelta[a] * c4_[a][h];
        thirdDelta[h] = std::clamp(sum, -GradientLimit, GradientLimit) *
            (1.0f - pass.third[h] * pass.third[h]);
    }
    std::array<float, CoreHiddenTwoCount> secondDelta{};
    for (int h = 0; h < CoreHiddenTwoCount; ++h) {
        float sum = 0.0f;
        for (int next = 0; next < CoreHiddenThreeCount; ++next) sum += thirdDelta[next] * c3_[next][h];
        secondDelta[h] = std::clamp(sum, -GradientLimit, GradientLimit) *
            (1.0f - pass.second[h] * pass.second[h]);
    }
    std::array<float, CoreHiddenOneCount> firstDelta{};
    for (int h = 0; h < CoreHiddenOneCount; ++h) {
        float sum = 0.0f;
        for (int next = 0; next < CoreHiddenTwoCount; ++next) sum += secondDelta[next] * c2_[next][h];
        firstDelta[h] = std::clamp(sum, -GradientLimit, GradientLimit) *
            (1.0f - pass.first[h] * pass.first[h]);
    }

    for (int a = 0; a < ActionCount; ++a) {
        updateParameter(cb4_[a], rate, outputDelta[a]);
        for (int h = 0; h < CoreHiddenThreeCount; ++h)
            updateParameter(c4_[a][h], rate, outputDelta[a] * pass.third[h]);
    }
    for (int h = 0; h < CoreHiddenThreeCount; ++h) {
        updateParameter(cb3_[h], rate, thirdDelta[h]);
        for (int previous = 0; previous < CoreHiddenTwoCount; ++previous)
            updateParameter(c3_[h][previous], rate, thirdDelta[h] * pass.second[previous]);
    }
    for (int h = 0; h < CoreHiddenTwoCount; ++h) {
        updateParameter(cb2_[h], rate, secondDelta[h]);
        for (int previous = 0; previous < CoreHiddenOneCount; ++previous)
            updateParameter(c2_[h][previous], rate, secondDelta[h] * pass.first[previous]);
    }
    for (int h = 0; h < CoreHiddenOneCount; ++h) {
        updateParameter(cb1_[h], rate, firstDelta[h]);
        for (int i = 0; i < InputCount; ++i)
            updateParameter(c1_[h][i], rate, firstDelta[h] * pass.input[i]);
    }
    digestValid_ = false;
    return loss / static_cast<float>(ActionCount);
}

int SocietyCore::parameterCount() {
    return CoreHiddenOneCount * (InputCount + 1) +
           CoreHiddenTwoCount * (CoreHiddenOneCount + 1) +
           CoreHiddenThreeCount * (CoreHiddenTwoCount + 1) +
           ActionCount * (CoreHiddenThreeCount + 1);
}

std::uint64_t SocietyCore::digest() const {
    if (digestValid_) return cachedDigest_;
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
    for (const auto& row : c1_) for (float weight : row) include(weight);
    for (float weight : cb1_) include(weight);
    for (const auto& row : c2_) for (float weight : row) include(weight);
    for (float weight : cb2_) include(weight);
    for (const auto& row : c3_) for (float weight : row) include(weight);
    for (float weight : cb3_) include(weight);
    for (const auto& row : c4_) for (float weight : row) include(weight);
    for (float weight : cb4_) include(weight);
    cachedDigest_ = hash;
    digestValid_ = true;
    return cachedDigest_;
}

std::shared_ptr<SocietyCore> makeSocietyCore(std::uint32_t seed) {
    // A civilization's instinct is a world property, not a process constant:
    // every seeded world trains its own core. Caching per seed keeps a
    // restarted experiment or the observer's reference copy identical and free,
    // while a different world seed always yields a different society core.
    struct Entry { std::uint32_t seed; std::shared_ptr<SocietyCore> core; };
    static std::vector<Entry> cache;
    for (const Entry& entry : cache)
        if (entry.seed == seed) return entry.core;
    auto prepared = std::make_shared<SocietyCore>(seed);
    std::mt19937 rng(seed ^ 0xA17E5EEDu);
    // Compact, stratified curriculum for a very wide network. Every epoch
    // aggregates a small tribe of founder observations into one
    // society-mean observation and one society-mean target vector, so the
    // civilization core learns a collective instinct rather than a personal one.
    constexpr int Epochs = 64;
    constexpr int Tribe = 48;
    for (int pass = 0; pass < 2; ++pass) {
        const float rate = pass == 0 ? .05f : .025f;
        for (int epoch = 0; epoch < Epochs; ++epoch) {
            CoreInput mean{};
            Values target{};
            for (int i = 0; i < Tribe; ++i) {
                const Observation observation = founderObservation(rng, epoch * Tribe + i);
                const Values values = curriculum(observation);
                for (int k = 0; k < InputCount; ++k) mean[k] += observation[k];
                for (int a = 0; a < ActionCount; ++a) target[a] += values[a];
            }
            for (int k = 0; k < InputCount; ++k) mean[k] /= static_cast<float>(Tribe);
            for (int a = 0; a < ActionCount; ++a) target[a] /= static_cast<float>(Tribe);
            prepared->train(mean, target, rate);
        }
    }
    // Bound the cache so a long-lived process that visits many seeds keeps a
    // stable memory footprint; the desktop holds one live civilization anyway.
    constexpr std::size_t MaxCachedCores = 8;
    cache.push_back({seed, prepared});
    if (cache.size() > MaxCachedCores) cache.erase(cache.begin(), cache.end() - MaxCachedCores);
    return prepared;
}

// ---- Individual policy ----

Brain::Brain(std::uint32_t seed) {
    std::mt19937 rng(seed);
    const float inputLimit = std::sqrt(6.0f / static_cast<float>(BrainInputCount + HiddenOneCount));
    const float middleLimit = std::sqrt(6.0f / static_cast<float>(HiddenOneCount + HiddenTwoCount));
    const float outputLimit = std::sqrt(6.0f / static_cast<float>(HiddenTwoCount + ActionCount));
    auto fill = [&](auto& matrix, float limit) {
        for (auto& row : matrix) for (float& value : row) value = (sampleUnit(rng) * 2.0f - 1.0f) * limit;
    };
    fill(w1_, inputLimit);
    fill(w2_, middleLimit);
    fill(w3_, outputLimit);
}

Brain::ForwardPass Brain::forward(const BrainInput& input) const {
    ForwardPass pass;
    pass.input = normalized(input);
    for (int h = 0; h < HiddenOneCount; ++h) {
        float sum = b1_[h];
        for (int i = 0; i < BrainInputCount; ++i) sum += w1_[h][i] * pass.input[i];
        pass.first[h] = std::tanh(sum);
    }
    for (int h = 0; h < HiddenTwoCount; ++h) {
        float sum = b2_[h];
        for (int i = 0; i < HiddenOneCount; ++i) sum += w2_[h][i] * pass.first[i];
        pass.second[h] = std::tanh(sum);
    }
    for (int a = 0; a < ActionCount; ++a) {
        float value = b3_[a];
        for (int h = 0; h < HiddenTwoCount; ++h) value += w3_[a][h] * pass.second[h];
        pass.values[a] = value;
    }
    return pass;
}

Values Brain::predict(const BrainInput& input) const {
    return forward(input).values;
}

int Brain::parameterCount() {
    return HiddenOneCount * (BrainInputCount + 1) +
           HiddenTwoCount * (HiddenOneCount + 1) +
           ActionCount * (HiddenTwoCount + 1);
}

Action Brain::choose(const BrainInput& input, const ActionMask& legal,
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

float Brain::train(const BrainInput& input, Action action, float target, float rate) {
    const int actionIndex = static_cast<int>(action);
    if (actionIndex < 0 || actionIndex >= ActionCount) throw std::invalid_argument("Unknown training action");
    target = std::isfinite(target) ? std::clamp(target, -4.0f, 4.0f) : 0.0f;
    rate = std::isfinite(rate) ? std::clamp(rate, 0.0f, 0.1f) : 0.0f;
    const ForwardPass pass = forward(input);
    const float error = pass.values[actionIndex] - target;
    const float outputGradient = std::clamp(error, -GradientLimit, GradientLimit);

    // All hidden deltas are computed before any parameter changes. This is the
    // chain rule for the two tanh layers using weights from this same pass.
    std::array<float, HiddenTwoCount> secondDelta{};
    for (int h = 0; h < HiddenTwoCount; ++h) {
        secondDelta[h] = outputGradient * w3_[actionIndex][h] *
            (1.0f - pass.second[h] * pass.second[h]);
    }
    std::array<float, HiddenOneCount> firstDelta{};
    for (int h = 0; h < HiddenOneCount; ++h) {
        float sum = 0.0f;
        for (int next = 0; next < HiddenTwoCount; ++next) sum += secondDelta[next] * w2_[next][h];
        firstDelta[h] = std::clamp(sum, -GradientLimit, GradientLimit) *
            (1.0f - pass.first[h] * pass.first[h]);
    }

    updateParameter(b3_[actionIndex], rate, outputGradient);
    for (int h = 0; h < HiddenTwoCount; ++h)
        updateParameter(w3_[actionIndex][h], rate, outputGradient * pass.second[h]);
    for (int h = 0; h < HiddenTwoCount; ++h) {
        updateParameter(b2_[h], rate, secondDelta[h]);
        for (int previous = 0; previous < HiddenOneCount; ++previous)
            updateParameter(w2_[h][previous], rate, secondDelta[h] * pass.first[previous]);
    }
    for (int h = 0; h < HiddenOneCount; ++h) {
        updateParameter(b1_[h], rate, firstDelta[h]);
        for (int i = 0; i < BrainInputCount; ++i)
            updateParameter(w1_[h][i], rate, firstDelta[h] * pass.input[i]);
    }
    ++updates_;
    return error * error;
}

float Brain::trainTargets(const BrainInput& input, const Values& targets, float rate) {
    rate = std::isfinite(rate) ? std::clamp(rate, 0.0f, 0.1f) : 0.0f;
    const ForwardPass pass = forward(input);
    std::array<float, ActionCount> outputDelta{};
    float loss = 0.0f;
    for (int a = 0; a < ActionCount; ++a) {
        const float target = std::isfinite(targets[a]) ? std::clamp(targets[a], -4.0f, 4.0f) : 0.0f;
        const float error = pass.values[a] - target;
        loss += error * error;
        // This is mean-squared error across all action values. Averaging keeps
        // a shared hidden gradient comparable to one online TD update.
        outputDelta[a] = std::clamp(error, -GradientLimit, GradientLimit) /
            static_cast<float>(ActionCount);
    }
    std::array<float, HiddenTwoCount> secondDelta{};
    for (int h = 0; h < HiddenTwoCount; ++h) {
        float sum = 0.0f;
        for (int a = 0; a < ActionCount; ++a) sum += outputDelta[a] * w3_[a][h];
        secondDelta[h] = std::clamp(sum, -GradientLimit, GradientLimit) *
            (1.0f - pass.second[h] * pass.second[h]);
    }
    std::array<float, HiddenOneCount> firstDelta{};
    for (int h = 0; h < HiddenOneCount; ++h) {
        float sum = 0.0f;
        for (int next = 0; next < HiddenTwoCount; ++next) sum += secondDelta[next] * w2_[next][h];
        firstDelta[h] = std::clamp(sum, -GradientLimit, GradientLimit) *
            (1.0f - pass.first[h] * pass.first[h]);
    }

    for (int a = 0; a < ActionCount; ++a) {
        updateParameter(b3_[a], rate, outputDelta[a]);
        for (int h = 0; h < HiddenTwoCount; ++h)
            updateParameter(w3_[a][h], rate, outputDelta[a] * pass.second[h]);
    }
    for (int h = 0; h < HiddenTwoCount; ++h) {
        updateParameter(b2_[h], rate, secondDelta[h]);
        for (int previous = 0; previous < HiddenOneCount; ++previous)
            updateParameter(w2_[h][previous], rate, secondDelta[h] * pass.first[previous]);
    }
    for (int h = 0; h < HiddenOneCount; ++h) {
        updateParameter(b1_[h], rate, firstDelta[h]);
        for (int i = 0; i < BrainInputCount; ++i)
            updateParameter(w1_[h][i], rate, firstDelta[h] * pass.input[i]);
    }
    ++updates_;
    return loss / static_cast<float>(ActionCount);
}

void Brain::learn(const BrainInput& before, Action action, float reward,
                  const BrainInput& after, const ActionMask& legal, bool terminal) {
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
    auto perturb = [&](float& weight) { weight = std::clamp(weight + noise(rng), -ParameterLimit, ParameterLimit); };
    for (auto& row : w1_) for (float& weight : row) perturb(weight);
    for (float& weight : b1_) perturb(weight);
    for (auto& row : w2_) for (float& weight : row) perturb(weight);
    for (float& weight : b2_) perturb(weight);
    for (auto& row : w3_) for (float& weight : row) perturb(weight);
    for (float& weight : b3_) perturb(weight);
}

void Brain::imitate(const Brain& donor, float rate) {
    rate = std::isfinite(rate) ? std::clamp(rate, 0.0f, 1.0f) : 0.0f;
    if (rate <= 0.0f || this == &donor) return;
    auto blend = [&](float& weight, float skill) {
        if (rate >= 1.0f) { weight = skill; return; }
        weight = std::clamp(weight + rate * (skill - weight), -ParameterLimit, ParameterLimit);
    };
    for (int h = 0; h < HiddenOneCount; ++h) {
        for (int i = 0; i < BrainInputCount; ++i) blend(w1_[h][i], donor.w1_[h][i]);
        blend(b1_[h], donor.b1_[h]);
    }
    for (int h = 0; h < HiddenTwoCount; ++h) {
        for (int i = 0; i < HiddenOneCount; ++i) blend(w2_[h][i], donor.w2_[h][i]);
        blend(b2_[h], donor.b2_[h]);
    }
    for (int a = 0; a < ActionCount; ++a) {
        for (int i = 0; i < HiddenTwoCount; ++i) blend(w3_[a][i], donor.w3_[a][i]);
        blend(b3_[a], donor.b3_[a]);
    }
}

double Brain::fingerprint() const {
    double sum = 0.0;
    std::uint64_t index = 1;
    auto include = [&](float parameter) {
        sum += static_cast<double>(parameter) * static_cast<double>(index++);
    };
    for (const auto& row : w1_) for (float weight : row) include(weight);
    for (float weight : b1_) include(weight);
    for (const auto& row : w2_) for (float weight : row) include(weight);
    for (float weight : b2_) include(weight);
    for (const auto& row : w3_) for (float weight : row) include(weight);
    for (float weight : b3_) include(weight);
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
    for (const auto& row : w3_) for (float weight : row) include(weight);
    for (float weight : b3_) include(weight);
    for (int shift = 0; shift < 64; shift += 8) includeByte(static_cast<std::uint8_t>(updates_ >> shift));
    return hash;
}

Brain makeFounderBrain(std::uint32_t seed, const SocietyCore& core) {
    Brain brain(seed);
    std::mt19937 rng(seed ^ 0xA17E5EEDu);
    // A small palette of society-mean advisories, derived from the shared core,
    // gives each individual policy exposure to a range of collective contexts
    // during bootstrap. Runtime citizens instead read that tick's live advice.
    std::array<Values, ActionCount> adviceSet{};
    for (int palette = 0; palette < ActionCount; ++palette) {
        CoreInput mean{};
        for (int i = 0; i < 48; ++i) {
            const Observation observation = founderObservation(rng, palette * 48 + i);
            for (int k = 0; k < InputCount; ++k) mean[k] += observation[k];
        }
        for (int k = 0; k < InputCount; ++k) mean[k] /= 48.0f;
        adviceSet[palette] = core.advise(mean);
    }
    // 1,536 compact, stratified samples are trained twice. Each optimizer step
    // fits all 13 values from a single forward pass, avoiding the former
    // 96,000 separate forward/backward passes while retaining broad coverage.
    constexpr int SamplesPerPass = 1536;
    for (int pass = 0; pass < 2; ++pass) {
        const float rate = pass == 0 ? .048f : .024f;
        for (int sample = 0; sample < SamplesPerPass; ++sample) {
            const Observation observation = founderObservation(rng, sample);
            const BrainInput input = compose(observation, adviceSet[sample % ActionCount]);
            brain.trainTargets(input, curriculum(observation), rate);
        }
    }
    return brain;
}
} // namespace pixels
