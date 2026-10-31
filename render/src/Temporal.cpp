#include "Temporal.h"

namespace lr::render {

namespace {

double radicalInverse(std::uint32_t index, std::uint32_t const base) {
    auto result = 0.0;
    auto scale = 1.0 / base;
    while (index > 0) {
        result += (index % base) * scale;
        index /= base;
        scale /= base;
    }
    return result;
}

} // namespace

core::vec2 temporalJitter(std::uint32_t const sample) {
    auto const index = 1 + sample % kTemporalJitters;
    return core::vec2{static_cast<float>(radicalInverse(index, 2) - 0.5),
                      static_cast<float>(radicalInverse(index, 3) - 0.5)};
}

core::mat4 jitteredViewProjection(core::mat4 const &viewProjection, core::vec2 const jitter,
                                  core::ivec2 const extent) {
    auto shift = core::mat4{1.0f};
    shift[3][0] = 2.0f * jitter.x / static_cast<float>(extent.x);
    shift[3][1] = -2.0f * jitter.y / static_cast<float>(extent.y);
    return shift * viewProjection;
}

TemporalView TemporalViews::advance(TemporalRequest const &request) {
    auto const found = records.find(request.view);
    auto const continues = found != records.end() and found->second.frame + 1 == request.frame
                       and found->second.extent == request.extent and not request.cameraCut;
    auto view = TemporalView{};
    view.continues = continues;
    view.sample = continues ? found->second.sample + 1 : 0;
    view.jitter = temporalJitter(view.sample);
    view.viewProjection = request.viewProjection;
    view.previousViewProjection = continues ? found->second.viewProjection : request.viewProjection;
    records[request.view] = Record{request.frame, request.extent, request.viewProjection, view.sample};
    return view;
}

void TemporalViews::expire(std::uint64_t const frame) {
    for (auto record = records.begin(); record != records.end();) {
        if (record->second.frame + keepFrames < frame) {
            record = records.erase(record);
        } else {
            ++record;
        }
    }
}

} // namespace lr::render
