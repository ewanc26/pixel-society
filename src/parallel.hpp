#pragma once
#include <cstddef>
#include <functional>

namespace pixels {
namespace parallel {

// Fixed-size worker pool for deterministic, thread-count-independent
// parallelization. Chunk partitions are always derived from chunk counts and
// block sizes chosen by the caller, never from how many workers happen to run,
// so a run with one worker and a run with many produce identical results.
void setWorkerCount(int count);
int workerCount();
int hardwareCount();

// Runs fn(chunks) for chunk in [0, chunks). fn(chunk) may run concurrently for
// different chunks; it must write only chunk-indexed outputs so the combined
// result does not depend on completion order. Returns after every chunk done.
void runChunks(int chunks, const std::function<void(int)>& fn);

// Splits [0, total) into fixed-size chunks and runs body(begin, end) for each.
// The chunk boundaries depend only on total and chunkSize, which keeps results
// reproducible regardless of the worker count configured for the process.
inline void runRanges(std::size_t total, std::size_t chunkSize,
                      const std::function<void(std::size_t, std::size_t)>& body) {
    if (total == 0) return;
    if (chunkSize == 0) chunkSize = 1;
    const std::size_t chunks = (total + chunkSize - 1) / chunkSize;
    runChunks(static_cast<int>(chunks), [&](int chunk) {
        const std::size_t begin = static_cast<std::size_t>(chunk) * chunkSize;
        body(begin, begin + chunkSize < total ? begin + chunkSize : total);
    });
}
} // namespace parallel
} // namespace pixels