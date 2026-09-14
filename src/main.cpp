#include "simulation.hpp"
#include "ticker.hpp"
#include "parallel.hpp"
#ifdef PIXEL_SOCIETY_GUI
#include "ui.hpp"
#endif
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
template<class T> T integer(const std::string& value, const char* option) {
    T result{};
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size())
        throw std::invalid_argument(std::string(option) + " needs a valid integer");
    return result;
}
float fraction(const std::string& value, const char* option) {
    std::size_t end = 0;
    float result = std::stof(value, &end);
    if (end != value.size() || !std::isfinite(result) || result < 0 || result > 1)
        throw std::invalid_argument(std::string(option) + " must be between 0 and 1");
    return result;
}
pixels::WorldShape shape(const std::string& value) {
    if (value == "island") return pixels::WorldShape::Island;
    if (value == "archipelago") return pixels::WorldShape::Archipelago;
    if (value == "inland-sea" || value == "inlandsea") return pixels::WorldShape::InlandSea;
    if (value == "highlands") return pixels::WorldShape::Highlands;
    if (value == "riverlands") return pixels::WorldShape::Riverlands;
    throw std::invalid_argument("--shape must be island, archipelago, inland-sea, highlands or riverlands");
}
pixels::WorldSize worldSize(const std::string& value) {
    if (value == "tiny") return pixels::WorldSize::Tiny;
    if (value == "small") return pixels::WorldSize::Small;
    if (value == "classic") return pixels::WorldSize::Classic;
    if (value == "large") return pixels::WorldSize::Large;
    if (value == "huge") return pixels::WorldSize::Huge;
    throw std::invalid_argument("--size must be tiny, small, classic, large or huge");
}
std::string jsonString(const std::string& value) {
    std::string out = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else if (c < 0x20) {
            const char* digits = "0123456789abcdef";
            out += "\\u00"; out += digits[c >> 4]; out += digits[c & 15];
        } else out += static_cast<char>(c);
    }
    return out + '"';
}
void help() {
    std::cout << R"(Pixel Society — seed a world, then let its citizens decide.

Usage: pixel-society [options]
  --seed N             Reproducible world seed (default 2026)
  --founders N         Initial population, 2..256 (default 48)
  --fertility F        Resource richness, 0..1 (default 0.65)
  --cooperation F      Initial social disposition, 0..1 (default 0.7)
  --hazards F          Environmental hazard intensity, 0..1 (default 0.35)
  --shape NAME         Terrain: island, archipelago, inland-sea, highlands, riverlands
  --size NAME          World: tiny 48x32, small 64x40, classic 96x64, large 128x80, huge 192x128
  --advisor-every N    Refresh shared society advice every N ticks, 1..300 (default 1)
  --headless           Run without graphics for experiments
  --ticks N            Headless ticks (default 3000); each represents 0.2 seconds
  --threads N          Worker threads for the simulation pool (default: all cores)
  --realtime           Pace headless mode at five ticks per second
  --events PATH        Write all headless events as scored JSONL
  --smoke-test         Exercise setup and observer UI automatically, then exit
  --screenshot PATH    Save UI test screenshot as BMP (use with --smoke-test)
  --help               Show this help

The desktop simulation always advances at 5 ticks/second. The headless runner
executes those same fixed ticks as fast as possible unless --realtime is set.
After Begin, all controls are observation only. There are no external AI services.
)";
}
}

int main(int argc, char** argv) {
    try {
        pixels::Config config;
        bool headless = false, realtime = false, smoke = false;
        std::uint64_t ticks = 3000;
        std::string eventsPath, screenshot;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            const auto next = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("Missing value for " + option);
                return argv[i];
            };
            if (option == "--help" || option == "-h") { help(); return 0; }
            if (option == "--seed") config.seed = integer<std::uint32_t>(next(), "--seed");
            else if (option == "--founders") config.founders = integer<int>(next(), "--founders");
            else if (option == "--fertility") config.fertility = fraction(next(), "--fertility");
            else if (option == "--cooperation") config.cooperation = fraction(next(), "--cooperation");
            else if (option == "--hazards") config.hazards = fraction(next(), "--hazards");
            else if (option == "--shape") config.shape = shape(next());
            else if (option == "--size") config.worldSize = worldSize(next());
            else if (option == "--advisor-every") config.advisorEvery = integer<int>(next(), "--advisor-every");
            else if (option == "--ticks") ticks = integer<std::uint64_t>(next(), "--ticks");
            else if (option == "--headless") headless = true;
            else if (option == "--threads") config.threads = integer<int>(next(), "--threads");
            else if (option == "--realtime") realtime = true;
            else if (option == "--events") eventsPath = next();
            else if (option == "--smoke-test") smoke = true;
            else if (option == "--screenshot") screenshot = next();
            else throw std::invalid_argument("Unknown option: " + option);
        }
        if (config.founders < 2 || config.founders > pixels::PopulationLimit)
            throw std::invalid_argument("--founders must be between 2 and 256");
        if (config.threads < 0 || config.threads > 256)
            throw std::invalid_argument("--threads must be between 0 and 256");
        if (config.advisorEvery < 1 || config.advisorEvery > pixels::TicksPerDay)
            throw std::invalid_argument("--advisor-every must be between 1 and 300");
        if (!headless && (realtime || !eventsPath.empty()))
            throw std::invalid_argument("--realtime and --events require --headless");
        if ((!smoke && !screenshot.empty()) || (headless && smoke))
            throw std::invalid_argument("--screenshot requires --smoke-test; UI tests cannot be headless");
        if (!headless) {
#ifdef PIXEL_SOCIETY_GUI
            return pixels::runUi({config, smoke, screenshot});
#else
            throw std::runtime_error("This build has no desktop UI. Use --headless or build with PIXEL_SOCIETY_GUI=ON.");
#endif
        }

        std::ofstream log;
        if (!eventsPath.empty()) {
            log.open(eventsPath, std::ios::trunc);
            if (!log) throw std::runtime_error("Cannot open event log: " + eventsPath);
        }
        pixels::Simulation simulation(config);
        std::uint64_t logged = 0;
        const auto flushEvents = [&] {
            if (eventsPath.empty()) return;
            const auto count = simulation.stats().eventCount;
            const auto& events = simulation.events();
            const auto first = count - events.size();
            if (logged < first) throw std::runtime_error("Event log could not keep up with simulation");
            for (std::uint64_t id = logged; id < count; ++id) {
                const auto& e = events[static_cast<std::size_t>(id - first)];
                log << "{\"id\":" << id << ",\"tick\":" << e.tick << ",\"score\":" << e.score
                    << ",\"kind\":" << jsonString(e.kind) << ",\"text\":" << jsonString(e.text)
                    << ",\"x\":" << e.x << ",\"y\":" << e.y << "}\n";
            }
            logged = count;
        };
        flushEvents();
        const auto start = std::chrono::steady_clock::now();
        for (std::uint64_t t = 0; t < ticks; ++t) {
            if (realtime)
                std::this_thread::sleep_until(start + std::chrono::milliseconds(200) * (t + 1));
            simulation.step();
            flushEvents();
        }
        if (!eventsPath.empty()) { log.flush(); if (!log) throw std::runtime_error("Writing event log failed"); }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto& s = simulation.stats();
        std::cout << "{\"seed\":" << config.seed << ",\"ticks\":" << simulation.tick()
                  << ",\"simulated_seconds\":" << static_cast<double>(simulation.tick()) / pixels::TicksPerSecond
                  << ",\"wall_seconds\":" << elapsed << ",\"threads\":" << pixels::parallel::workerCount()
                  << ",\"advisor_every\":" << config.advisorEvery
                  << ",\"shape\":" << jsonString(pixels::nameOf(config.shape))
                  << ",\"size\":" << jsonString(pixels::nameOf(config.worldSize))
                  << ",\"width\":" << simulation.width() << ",\"height\":" << simulation.height()
                  << ",\"population\":" << s.population
                  << ",\"births\":" << s.births << ",\"deaths\":" << s.deaths
                  << ",\"homes\":" << s.homes << ",\"farms\":" << s.farms
                  << ",\"generation\":" << s.generation << ",\"wellbeing\":" << s.wellbeing
                  << ",\"decisions\":" << s.decisions << ",\"learning_updates\":" << s.learningUpdates
                  << ",\"events\":" << s.eventCount << ",\"core_parameters\":" << pixels::SocietyCore::parameterCount()
                  << ",\"citizen_parameters\":" << pixels::Brain::parameterCount()
                  << ",\"digest\":" << simulation.digest()
                  << ",\"actions\":{";
        for (int a = 0; a < pixels::ActionCount; ++a) {
            if (a) std::cout << ',';
            std::cout << jsonString(pixels::actionName(static_cast<pixels::Action>(a))) << ':' << s.actions[a];
        }
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Pixel Society: " << e.what() << '\n';
        return 1;
    }
}
