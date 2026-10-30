#pragma once

#include "DeviceScene.h"
#include "FrameGraphImpl.h"
#include "PreparedFrame.h"
#include "RetireList.h"

#include <lr/gpu/GpuQueue.h>
#include <lr/render/Renderer.h>

#include <lexec/scopes/counting_scope.hpp>
#include <stackfull/sync/Completion.h>
#include <tc/device.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <exception>
#include <vector>

namespace lr::render {

struct FrameQueueConfig {
    gpu::GpuQueue *gpu = nullptr;
    // Holds the waits for the device; joined before the queue goes away.
    lexec::counting_scope *scope = nullptr;
    // Bumped once a lost device has been replaced and everything made on it dropped;
    // frames prepared for an earlier generation fail instead of being recorded.
    std::atomic<std::uint64_t> *deviceGeneration = nullptr;
    std::uint32_t framesInFlight = 2;
    bool timers = false;
};

// The GPU thread's frames. Prepared frames are recorded in order, at most framesInFlight
// of them ahead of the device; the rest wait. Each is finished once the device has
// finished it and every frame before it (readbacks, timings, presenting, releasing what
// it held), and its sender completes then, so senders complete in frame order. Runs only
// on the GPU thread, so it needs no locks.
struct FrameQueue {
    explicit FrameQueue(FrameQueueConfig const &config_) noexcept : config(config_) {}

    void accept(gpu::GpuContext &context, PreparedFrame frame) noexcept;
    // Drops every resource the GPU thread holds; at shutdown, once no frame is left.
    void clear() noexcept;

private:
    struct InFlight {
        std::uint64_t index = 0;
        std::chrono::steady_clock::time_point submitted;
        FrameResult result; // the views' ids until finish() fills in the rest
        std::vector<GraphRun> runs;
        tc::Submission submission;
        RetireList retired;
        // Reported once the device is done with what the frame recorded before it.
        std::exception_ptr failure;
        bool deviceDone = false;
        stackfull::sync::Completion<FrameResult> completion;
    };

    struct Outcome {
        stackfull::sync::Completion<FrameResult> completion;
        FrameResult result;
        std::exception_ptr failure;
    };

    void advance(gpu::GpuContext &context) noexcept;
    void start(gpu::GpuContext &context, PreparedFrame prepared);
    void record(gpu::GpuContext &context, PreparedFrame &prepared, InFlight &frame);
    void recordViews(tc::Device &device, PreparedFrame &prepared, InFlight &frame);
    void watch(std::uint64_t index, tc::FencedSubmission submitted);
    void deviceDone(gpu::GpuContext &context, std::uint64_t index, std::exception_ptr const &failure) noexcept;
    void finishCompleted(gpu::GpuContext &context);
    FrameResult finish(gpu::GpuContext &context, InFlight &frame);
    void loseDevice(gpu::GpuContext &context, std::exception_ptr const &failure);
    void recreateDevice(gpu::GpuContext &context);
    // Completes the senders of the frames finished so far, last, so that what a sender's
    // receiver runs on this thread finds the queue in order.
    void deliver() noexcept;

    FrameQueueConfig config;
    DeviceScene deviceScene;
    ResourcePool pool;
    HistoryResources historyImages;
    std::deque<PreparedFrame> waiting;
    std::deque<InFlight> inFlight;
    std::vector<Outcome> outcomes;
    std::exception_ptr lostWith; // what the frames prepared for a lost device fail with
};

} // namespace lr::render
