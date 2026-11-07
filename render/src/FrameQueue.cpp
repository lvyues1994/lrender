#include "FrameQueue.h"

#include "kernels/PathTrace.h"
#include "kernels/Textures.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <unistd.h>

namespace lr::render {

namespace {

// The frame queue's own dispatches, recorded before the views (pool growth, acceleration
// structures) or after them (mip levels of textures that arrived).
bool ownProgram(std::string const &program) {
    using namespace kernels;
    return program == CopyTextureLayer::fileLocation or program == DownsampleTexture::fileLocation
        or program == CopyPositions::fileLocation or program == WriteTopLevelInstances::fileLocation;
}

// Where the runs' dispatches start among the frame's timings. The device lists them in
// execution order, the runs in recording order, and between the frame queue's own
// dispatches the two agree unless a pass dispatched behind PassContext's back; nullopt
// then.
std::optional<std::size_t> firstViewTiming(std::vector<GraphRun> const &runs,
                                           std::vector<tc::KernelTiming> const &timings) {
    auto first = std::size_t{0};
    while (first < timings.size() and ownProgram(timings[first].program)) {
        ++first;
    }
    auto next = first;
    for (auto const &run : runs) {
        for (auto const &dispatch : run.dispatches()) {
            if (next == timings.size() or timings[next].program != dispatch.program) {
                return std::nullopt;
            }
            ++next;
        }
    }
    auto const rest = std::all_of(timings.begin() + static_cast<std::ptrdiff_t>(next), timings.end(),
                                  [](tc::KernelTiming const &timing) { return ownProgram(timing.program); });
    return rest ? std::optional<std::size_t>{first} : std::nullopt;
}

std::exception_ptr waitStopped() noexcept {
    return std::make_exception_ptr(std::runtime_error{"lr::render: the wait for the device was stopped"});
}

} // namespace

void FrameQueue::accept(gpu::GpuContext &context, PreparedFrame frame) noexcept {
    waiting.push_back(std::move(frame));
    advance(context);
}

void FrameQueue::clear() noexcept {
    waiting.clear();
    inFlight.clear();
    pool.clear();
    historyImages.clear();
    sharedOutputs.clear();
    deviceScene = DeviceScene{};
    accelerations = AccelerationScene{};
    textures.clear();
    lastSubmission = tc::Submission{};
}

void FrameQueue::advance(gpu::GpuContext &context) noexcept {
    for (;;) {
        finishCompleted(context);
        if (waiting.empty() or inFlight.size() >= config.framesInFlight) {
            break;
        }
        auto frame = std::move(waiting.front());
        waiting.pop_front();
        start(context, std::move(frame));
    }
    deliver();
}

void FrameQueue::start(gpu::GpuContext &context, PreparedFrame prepared) {
    auto &frame = inFlight.emplace_back();
    frame.index = prepared.index;
    frame.submitted = prepared.submitted;
    frame.completion = prepared.completion;
    frame.result.frameIndex = prepared.index;
    frame.result.prepareTime = prepared.prepareTime;
    frame.result.lostTextures = std::move(prepared.lostTextures);
    for (auto const &view : prepared.views) {
        auto result = ViewResult{};
        result.view = view.view;
        result.target = view.target;
        frame.result.views.push_back(std::move(result));
    }
    if (prepared.failure or prepared.deviceGeneration != config.deviceGeneration->load(std::memory_order_relaxed)) {
        frame.failure = prepared.failure ? prepared.failure : lostWith;
        frame.deviceDone = true;
        return;
    }
    try {
        if (context.device == nullptr) {
            recreateDevice(context);
        }
        record(context, prepared, frame);
    } catch (tc::DeviceError const &) {
        loseDevice(context, std::current_exception());
    }
}

// A view that failed fails the frame after the scene update is applied, so both copies of
// the scene stay in step; the work recorded before a failure is still submitted, and the
// frame keeps its resources until the device is done with it.
void FrameQueue::record(gpu::GpuContext &context, PreparedFrame &prepared, InFlight &frame) {
    auto const start = std::chrono::steady_clock::now();
    auto &device = *context.device;
    pool.endFrame();
    sharedOutputs.endFrame();
    textures.apply(device, prepared.scene, frame.retired);
    textures.send(device, lastSubmission, frame.retired);
    deviceScene.apply(device, prepared.scene, textures, frame.retired);
    accelerations.apply(prepared.scene);
    if (prepared.tracesWithRayQueries) {
        accelerations.build(device, deviceScene.tables(), frame.retired);
    }
    historyImages.release(prepared.releasedHistories, frame.retired);
    try {
        recordViews(device, prepared, frame);
    } catch (tc::DeviceError const &) {
        throw;
    } catch (...) {
        frame.failure = std::current_exception();
    }
    textures.arrive(device);
    auto submitted = config.gpu->submit(context);
    frame.submission = submitted.submission;
    lastSubmission = submitted.submission;
    handOver(frame, submitted.fence);
    watch(frame.index, std::move(submitted));
    frame.result.recordTime = std::chrono::steady_clock::now() - start;
}

void FrameQueue::recordViews(tc::Device &device, PreparedFrame &prepared, InFlight &frame) {
    for (auto const &view : prepared.views) {
        if (view.failure) {
            std::rethrow_exception(view.failure);
        }
    }
    auto tables = deviceScene.tables();
    tables.topLevel = accelerations.topLevel();
    tables.textures = textures.textures();
    frame.runs.reserve(prepared.views.size());
    for (auto &view : prepared.views) {
        auto externals = ExternalResources{};
        if (view.sharedOutput) {
            externals.emplace_back(view.sharedOutput->resource, sharedOutputs.acquire(device, view, frame.sharedSlots));
        }
        frame.runs.emplace_back(std::move(*view.graph), pool, historyImages, device, config.aliasGraphResources,
                                externals);
        frame.runs.back().record(device, tables);
    }
}

void FrameQueue::handOver(InFlight &frame, tc::SyncFence const &fence) {
    for (auto &held : frame.sharedSlots) {
        auto const copy = fence.empty() ? -1 : ::dup(fence.get());
        if (frame.failure or (not fence.empty() and copy < 0)) {
            held.presenter->discardShared(held.slot);
            continue;
        }
        held.presenter->presentShared(
            SharedBufferFrame{frame.index, held.view, held.target, held.slot, tc::SyncFence{copy}});
    }
    frame.sharedSlots.clear();
}

void FrameQueue::watch(std::uint64_t const index, tc::FencedSubmission submitted) {
    auto &queue = *config.gpu;
    auto backOnGpuThread = [this, &queue, index](std::exception_ptr &failure) {
        return queue.record(
            [this, index, failure](gpu::GpuContext &context) noexcept { deviceDone(context, index, failure); });
    };
    // Scheduling onto the GPU thread fails only when its queue's mutex cannot be locked,
    // and the frames cannot go on without it.
    lexec::spawn(queue.whenComplete(std::move(submitted))
                     | lexec::then([]() noexcept { return std::exception_ptr{}; })
                     | lexec::upon_error([](std::exception_ptr failure) noexcept { return failure; })
                     | lexec::upon_stopped([]() noexcept { return waitStopped(); })
                     | lexec::let_value(std::move(backOnGpuThread))
                     | lexec::upon_error([](std::exception_ptr) noexcept { std::terminate(); }),
                 config.scope->get_token());
}

// A failed wait leaves it unknown whether the device still uses the frame's resources,
// so it is handled as a lost device.
void FrameQueue::deviceDone(gpu::GpuContext &context, std::uint64_t const index,
                            std::exception_ptr const &failure) noexcept {
    auto const found = std::find_if(inFlight.begin(), inFlight.end(),
                                    [index](InFlight const &frame) { return frame.index == index; });
    if (found == inFlight.end()) {
        return; // failed with a lost device already
    }
    if (failure) {
        loseDevice(context, failure);
    } else {
        found->deviceDone = true;
    }
    advance(context);
}

void FrameQueue::finishCompleted(gpu::GpuContext &context) {
    while (not inFlight.empty() and inFlight.front().deviceDone) {
        auto lost = std::exception_ptr{};
        {
            auto frame = std::move(inFlight.front());
            inFlight.pop_front();
            auto lostTextures = std::move(frame.result.lostTextures);
            try {
                auto result = finish(context, frame);
                result.lostTextures = std::move(unreportedLostTextures);
                unreportedLostTextures.clear();
                result.lostTextures.insert(result.lostTextures.end(), lostTextures.begin(), lostTextures.end());
                outcomes.push_back(Outcome{frame.completion, std::move(result), nullptr});
            } catch (tc::DeviceError const &) {
                lost = std::current_exception();
                outcomes.push_back(Outcome{frame.completion, {}, lost});
            } catch (...) {
                outcomes.push_back(Outcome{frame.completion, {}, std::current_exception()});
            }
            if (not outcomes.back().failure) {
                lostTextures.clear();
            }
            unreportedLostTextures.insert(unreportedLostTextures.end(), lostTextures.begin(), lostTextures.end());
        }
        // After the frame's resources went back to the pool, which this empties.
        if (lost) {
            loseDevice(context, lost);
        }
    }
}

FrameResult FrameQueue::finish(gpu::GpuContext &context, InFlight &frame) {
    auto const start = std::chrono::steady_clock::now();
    auto const recorded = frame.submission.sequence() != 0;
    auto const timings = config.timers and recorded ? context.device->takeTimings(frame.submission)
                                                    : std::vector<tc::KernelTiming>{};
    if (frame.failure) {
        std::rethrow_exception(frame.failure);
    }
    auto const first = config.timers ? firstViewTiming(frame.runs, timings) : std::nullopt;
    auto firstTiming = first.value_or(0);
    for (auto index = std::size_t{0}; index < frame.runs.size(); ++index) {
        auto &view = frame.result.views[index];
        if (first) {
            view.passTimings = frame.runs[index].passTimings(timings, firstTiming);
            firstTiming += frame.runs[index].dispatches().size();
        }
        frame.runs[index].finish(view);
    }
    auto const finished = std::chrono::steady_clock::now();
    frame.result.finishTime = finished - start;
    frame.result.wallTime = finished - frame.submitted;
    return std::move(frame.result);
}

// Everything made on the old device goes before it does.
void FrameQueue::loseDevice(gpu::GpuContext &context, std::exception_ptr const &failure) {
    for (auto &frame : inFlight) {
        outcomes.push_back(Outcome{frame.completion, {}, failure});
        auto const &lostTextures = frame.result.lostTextures;
        unreportedLostTextures.insert(unreportedLostTextures.end(), lostTextures.begin(), lostTextures.end());
        for (auto const &held : frame.sharedSlots) {
            held.presenter->discardShared(held.slot);
        }
    }
    inFlight.clear();
    pool.clear();
    historyImages.clear();
    sharedOutputs.clear();
    deviceScene = DeviceScene{};
    accelerations = AccelerationScene{};
    textures.clear();
    lastSubmission = tc::Submission{};
    lostWith = failure;
    try {
        recreateDevice(context);
    } catch (tc::DeviceError const &) {
        // The next frame tries again, and fails with the reason if it cannot.
    }
    config.deviceGeneration->fetch_add(1, std::memory_order_release);
}

void FrameQueue::recreateDevice(gpu::GpuContext &context) {
    config.gpu->recreateDevice(context);
    if (config.checkDevice) {
        config.checkDevice(*context.device);
    }
    if (config.timers) {
        context.device->enableTimers(true);
    }
}

void FrameQueue::deliver() noexcept {
    auto ready = std::move(outcomes);
    outcomes.clear();
    for (auto &outcome : ready) {
        if (outcome.failure) {
            outcome.completion.setException(outcome.failure);
        } else {
            outcome.completion.set(std::move(outcome.result));
        }
    }
}

} // namespace lr::render
