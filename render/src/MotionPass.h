#pragma once

#include <lr/render/Features.h>

namespace lr::render {

// Where the surface each pixel of the visibility buffer shows was in the previous frame
// (TemporalOutputs): the pass that finds it, added by the first feature that asks, and
// put on the blackboard. Needs ViewContext::temporal.
BufferRef<tc::vec2> motionOf(FrameGraphBuilder &graph, ViewContext const &view);

} // namespace lr::render
