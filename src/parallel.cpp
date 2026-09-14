#include "parallel.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace pixels {
namespace parallel {
namespace {

class Pool {
public:
    explicit Pool(int workers) : workers_(workers > 0 ? workers : 1) {
        for (int i = 0; i < workers_; ++i) threads_.emplace_back([this] { workerLoop(); });
    }
    ~Pool() { shutdown(); }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    int workers() const { return workers_; }

    void run(int chunks, const std::function<void(int)>& fn) {
        if (chunks <= 1 || workers_ <= 1) {
            for (int chunk = 0; chunk < chunks; ++chunk) fn(chunk);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pending_ = chunks;
            for (int chunk = 0; chunk < chunks; ++chunk) {
                tasks_.emplace_back([this, chunk, &fn] { fn(chunk); finished(); });
            }
        }
        work_.notify_all();
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock, [this] { return pending_ == 0; });
    }

private:
    void finished() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--pending_ == 0) idle_.notify_one();
    }

    void workerLoop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        work_.notify_all();
        for (std::thread& thread : threads_) thread.join();
        threads_.clear();
    }

    int workers_ = 1;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable work_;
    std::condition_variable idle_;
    std::deque<std::function<void()>> tasks_;
    int pending_ = 0;
    bool stop_ = false;
};

int hardwareThreads() {
    const unsigned int detected = std::thread::hardware_concurrency();
    return detected > 0 ? static_cast<int>(detected) : 1;
}

std::mutex guard;
Pool* pool = nullptr;
int configured = 0;

struct Holder {
    ~Holder() { delete pool; }
};
Holder holder;

Pool* obtain() {
    if (pool == nullptr) pool = new Pool(configured > 0 ? configured : hardwareThreads());
    return pool;
}
} // namespace

void setWorkerCount(int count) {
    const int target = count > 0 ? count : hardwareThreads();
    std::lock_guard<std::mutex> lock(guard);
    if (pool != nullptr && pool->workers() == target) {
        configured = target;
        return;
    }
    // Replace the pool only while it is guaranteed idle (callers configure the
    // worker count between runs). The destructor joins the previous workers.
    Pool* replacement = new Pool(target);
    delete pool;
    pool = replacement;
    configured = target;
}

int workerCount() {
    std::lock_guard<std::mutex> lock(guard);
    return obtain()->workers();
}

int hardwareCount() { return hardwareThreads(); }

void runChunks(int chunks, const std::function<void(int)>& fn) {
    if (chunks <= 0) return;
    // The guard stays held for the whole batch so setWorkerCount cannot free
    // the pool underneath an in-flight run.
    std::lock_guard<std::mutex> lock(guard);
    obtain()->run(chunks, fn);
}
} // namespace parallel
} // namespace pixels