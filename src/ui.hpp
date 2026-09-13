#pragma once
#include "simulation.hpp"
#include <string>
namespace pixels {
struct UiOptions {
    Config config;
    bool smokeTest = false;
    std::string screenshotPath;
};
int runUi(const UiOptions& options);
}
