#include <lr/scene/ScriptHost.h>

#include <co2/manual_executor.hpp>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace lr::scene {

namespace detail {

void TimedAwaiter::await_suspend(co2::coroutine_handle<> const continuation) {
    auto &context = *wait.script->context;
    awaitedAt = context.now();
    wait.deadline = awaitedAt + std::max(delay, SimDuration{0});
    wait.continuation = continuation;
    context.awaitTime(wait);
}

SimDuration TimedAwaiter::await_resume() const {
    if (wait.script->stopSource.stop_requested()) {
        throw lexec::coro::stopped_error{};
    }
    return wait.script->context->now() - awaitedAt;
}

} // namespace detail

namespace {

using detail::OperationGate;
using detail::ScriptPromise;
using detail::TimedWait;

struct WaitEntry {
    SimDuration deadline;
    std::uint64_t sequence;
    ScriptId script;
};

// For a min-heap: the soonest deadline, then the earliest await, on top.
bool later(WaitEntry const &a, WaitEntry const &b) noexcept {
    return std::tie(a.deadline, a.sequence) > std::tie(b.deadline, b.sequence);
}

struct Resumption {
    std::uint64_t sequence;
    co2::coroutine_handle<> continuation;
};

struct ScriptRecord {
    co2::coroutine_handle<ScriptPromise> handle;
    TimedWait *wait = nullptr; // the wait in the heap it is suspended in
    bool ended = false;
};

bool isStop(std::exception_ptr const &error) noexcept {
    try {
        std::rethrow_exception(error);
    } catch (lexec::coro::stopped_error const &) {
        return true;
    } catch (...) {
        return false;
    }
}

struct ScriptHostImpl final : ScriptHost, detail::ScriptContext {
    explicit ScriptHostImpl(ScriptHostConfig const &config_) : config(config_) {}

    ScriptHostImpl(ScriptHostImpl const &) = delete;
    ScriptHostImpl &operator=(ScriptHostImpl const &) = delete;

    ~ScriptHostImpl() override { shutdown(); }

    ScriptId start(ScriptTask script) override {
        auto const handle = detail::ScriptTaskAccess::handleOf(script);
        if (not handle) {
            throw std::invalid_argument{"lr::scene: the script was moved from"};
        }
        auto const id = ScriptId{nextScript++};
        if (shuttingDown) {
            return id; // `script` destroys the coroutine, which never ran
        }
        scripts.emplace(id, ScriptRecord{handle});
        detail::ScriptTaskAccess::release(script);
        auto &promise = handle.promise();
        promise.context = this;
        promise.id = id;
        executor.schedule(handle);
        return id;
    }

    void requestStop(ScriptId const script) override {
        auto const found = scripts.find(script);
        if (found == scripts.end() or found->second.ended) {
            return;
        }
        auto &record = found->second;
        if (record.wait != nullptr) {
            stoppedWaits.push_back(std::exchange(record.wait, nullptr));
        }
        // Stop callbacks may complete the awaited sender right here, which takes the lock.
        record.handle.promise().stopSource.request_stop();
    }

    void tick(SimDuration const elapsed) override {
        if (ticking) {
            throw std::logic_error{"lr::scene: ScriptHost::tick() was called from a script"};
        }
        if (elapsed < SimDuration{0}) {
            throw std::invalid_argument{"lr::scene: simulation time cannot go back"};
        }
        ticking = true;
        auto const done = Ticking{ticking};
        ++ticks;
        time += elapsed;
        due.clear();
        releaseWaits();
        releaseOperations();
        std::sort(due.begin(), due.end(),
                  [](Resumption const &a, Resumption const &b) { return a.sequence < b.sequence; });
        for (auto const &resumption : due) {
            executor.schedule(resumption.continuation);
        }
        executor.run();
        destroyEnded();
    }

    bool isRunning(ScriptId const script) const noexcept override {
        auto const found = scripts.find(script);
        return found != scripts.end() and not found->second.ended;
    }

    bool isIdle() const noexcept override { return scripts.empty(); }
    SimDuration now() const noexcept override { return time; }
    std::uint64_t tickCount() const noexcept override { return ticks; }

    std::vector<ScriptFailure> takeFailures() override { return std::exchange(failures, {}); }

    void shutdown() override {
        if (ticking) {
            throw std::logic_error{"lr::scene: ScriptHost::shutdown() was called from a script"};
        }
        shuttingDown = true;
        auto running = std::vector<ScriptId>{};
        for (auto const &entry : scripts) {
            running.push_back(entry.first);
        }
        for (auto const script : running) {
            requestStop(script);
        }
        while (not scripts.empty()) {
            if (config.release == ReleaseMode::AsSoonAsReady and stoppedWaits.empty() and executor.pending() == 0) {
                waitForAnyOperation();
            }
            tick(SimDuration{0});
        }
    }

    void awaitTime(TimedWait &wait) override {
        wait.sequence = nextSequence++;
        auto &promise = *wait.script;
        if (promise.stopSource.stop_requested()) {
            stoppedWaits.push_back(&wait);
            return;
        }
        waits.push_back(WaitEntry{wait.deadline, wait.sequence, promise.id});
        std::push_heap(waits.begin(), waits.end(), later);
        scripts.at(promise.id).wait = &wait;
    }

    void awaitOperation(OperationGate &gate) override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        gate.sequence = nextSequence++;
        gate.awaitedTick = ticks;
        operations.push_back(&gate);
    }

    void abandonOperation(OperationGate &gate) noexcept override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        operations.erase(std::find(operations.begin(), operations.end(), &gate));
    }

    void operationCompleted(OperationGate &gate, co2::coroutine_handle<> const continuation) noexcept override {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            gate.continuation = continuation;
        }
        completed.notify_all();
    }

    void scriptEnded(ScriptPromise &script) noexcept override {
        scripts.at(script.id).ended = true;
        ended.push_back(script.id);
        if (script.error != nullptr and not isStop(script.error)) {
            failures.push_back(ScriptFailure{script.id, script.error});
        }
    }

private:
    struct Ticking {
        explicit Ticking(bool &flag_) noexcept : flag(flag_) {}
        Ticking(Ticking const &) = delete;
        Ticking &operator=(Ticking const &) = delete;
        ~Ticking() { flag = false; }

        bool &flag;
    };

    void releaseWaits() {
        for (auto *const wait : stoppedWaits) {
            due.push_back(Resumption{wait->sequence, wait->continuation});
        }
        stoppedWaits.clear();
        while (not waits.empty() and waits.front().deadline <= time) {
            std::pop_heap(waits.begin(), waits.end(), later);
            auto const entry = waits.back();
            waits.pop_back();
            // Entries of waits that ended early, by a stop request, stay in the heap.
            auto const found = scripts.find(entry.script);
            if (found == scripts.end() or found->second.wait == nullptr
                or found->second.wait->sequence != entry.sequence) {
                continue;
            }
            due.push_back(Resumption{entry.sequence, found->second.wait->continuation});
            found->second.wait = nullptr;
        }
    }

    void releaseOperations() {
        auto lock = std::unique_lock<std::mutex>{mutex};
        if (config.release == ReleaseMode::Deterministic) {
            while (not operations.empty() and operations.front()->awaitedTick + config.deterministicDelay <= ticks) {
                auto *const gate = operations.front();
                completed.wait(lock, [gate] { return static_cast<bool>(gate->continuation); });
                due.push_back(Resumption{gate->sequence, gate->continuation});
                operations.pop_front();
            }
            return;
        }
        auto const ready = std::stable_partition(operations.begin(), operations.end(),
                                                 [](OperationGate const *const gate) { return not gate->continuation; });
        for (auto gate = ready; gate != operations.end(); ++gate) {
            due.push_back(Resumption{(*gate)->sequence, (*gate)->continuation});
        }
        operations.erase(ready, operations.end());
    }

    void waitForAnyOperation() {
        auto lock = std::unique_lock<std::mutex>{mutex};
        completed.wait(lock, [this] {
            return operations.empty() or std::any_of(operations.begin(), operations.end(), [](OperationGate const *gate) {
                       return static_cast<bool>(gate->continuation);
                   });
        });
    }

    void destroyEnded() noexcept {
        for (auto const script : ended) {
            auto const found = scripts.find(script);
            found->second.handle.destroy();
            scripts.erase(found);
        }
        ended.clear();
    }

    ScriptHostConfig config;
    co2::ManualExecutor executor;
    std::unordered_map<ScriptId, ScriptRecord> scripts;
    std::uint32_t nextScript = 0;
    std::uint64_t nextSequence = 0;
    std::uint64_t ticks = 0;
    SimDuration time{0};
    std::vector<WaitEntry> waits; // a heap
    std::vector<TimedWait *> stoppedWaits; // of scripts asked to stop: released in the next tick
    std::vector<ScriptId> ended;
    std::vector<ScriptFailure> failures;
    std::vector<Resumption> due; // tick()'s
    bool ticking = false;
    bool shuttingDown = false;

    std::mutex mutex; // guards `operations` and their gates, which other threads complete
    std::condition_variable completed;
    std::deque<OperationGate *> operations; // in the order they were awaited
};

} // namespace

std::unique_ptr<ScriptHost> makeScriptHost(ScriptHostConfig const &config) {
    if (config.deterministicDelay == 0) {
        throw std::invalid_argument{"lr::scene: the deterministic delay is at least one tick"};
    }
    return std::make_unique<ScriptHostImpl>(config);
}

} // namespace lr::scene
