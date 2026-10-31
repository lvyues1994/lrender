#pragma once

#include <lr/scene/ScriptTask.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

namespace lr::scene {

enum class ReleaseMode : std::uint8_t {
    // An awaited sender that has completed resumes its script in the next tick.
    AsSoonAsReady,
    // A sender awaited in tick t resumes its script in tick t + deterministicDelay, which
    // waits for it to complete if it has not: how long operations take does not change
    // when scripts resume. A sender that never completes blocks that tick for good.
    Deterministic,
};

struct ScriptHostConfig {
    ReleaseMode release = ReleaseMode::AsSoonAsReady;
    std::uint32_t deterministicDelay = 1; // ticks, at least 1
};

struct ScriptFailure {
    ScriptId script;
    std::exception_ptr error;
};

// Runs scripts on the thread that calls tick(), the host thread, which makes every other
// call too. A tick runs the scripts started since the previous one, then resumes, in the
// order they were awaited, the waits and operations due in it.
//
// Stopping is cooperative: requestStop() requests the script's stop token, which awaited
// senders see; the script gets lexec::coro::stopped_error where it resumes next and ends,
// at the next tick for nextFrame() and waitFor(), when the sender completes otherwise.
// A script suspended on a sender cannot be destroyed, so the host's destructor stops every
// script and ticks until they have ended, waiting for senders that ignore stop requests.
struct ScriptHost {
    virtual ~ScriptHost() = default;

    // The script runs from the next tick, or later in this one when a script starts it.
    virtual ScriptId start(ScriptTask script) = 0;
    // Nothing for scripts that have ended. A script that has not run yet still runs up to
    // its first await.
    virtual void requestStop(ScriptId script) = 0;
    // Advances simulation time by `elapsed` (not negative) and runs what is due.
    virtual void tick(SimDuration elapsed) = 0;

    virtual bool isRunning(ScriptId script) const noexcept = 0;
    // No script is running.
    virtual bool isIdle() const noexcept = 0;
    virtual SimDuration now() const noexcept = 0;
    virtual std::uint64_t tickCount() const noexcept = 0;
    // The scripts that ended with an exception other than stopped_error since the last call.
    virtual std::vector<ScriptFailure> takeFailures() = 0;
    // Stops every script and ticks with no time passing until all have ended; scripts
    // started afterwards never run.
    virtual void shutdown() = 0;
};

// Throws std::invalid_argument for a deterministic delay of 0.
std::unique_ptr<ScriptHost> makeScriptHost(ScriptHostConfig const &config);

} // namespace lr::scene
