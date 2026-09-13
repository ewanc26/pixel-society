#pragma once
#include "simulation.hpp"
#include <chrono>
#include <cstdint>
namespace pixels {
// Render independently; retain the entire elapsed-time debt instead of dropping ticks.
class FixedTicker {
public:
    using Duration = std::chrono::nanoseconds;
    static constexpr auto interval = std::chrono::milliseconds(1000 / TicksPerSecond);
    template<class F> int advance(Duration elapsed, F&& step, int batchLimit = 25) {
        if (elapsed.count() > 0) debt_ += elapsed;
        int count = 0;
        while (debt_ >= interval && count < batchLimit) {
            step(); debt_ -= interval; ++count;
        }
        return count;
    }
    Duration debt() const { return debt_; }
private:
    Duration debt_{};
};
}
