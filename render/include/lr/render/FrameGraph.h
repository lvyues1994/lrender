#pragma once

// How a view's frame is built: features declare passes and the resources they read and
// write on the render thread; the passes are recorded on the GPU thread.

#include <lr/core/Math.h>
#include <lr/core/Scene.h>
#include <lr/render/Capabilities.h>
#include <lr/render/GpuTypes.h>
#include <lr/render/View.h>

#include <tc/device.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lr::render {

// A buffer of T or a 2D image of GPU format G in one view's frame graph.
template <class T>
struct BufferRef {
    std::uint32_t index = 0xFFFFFFFFu;
};

template <class G>
struct ImageRef {
    std::uint32_t index = 0xFFFFFFFFu;
};

template <class G>
using GraphImage = tc::DeviceImage<G, tc::Dim::D2>;

// An image a view keeps from frame to frame: two images that trade places every frame
// the view uses them, so passes read what the view's previous such frame wrote into
// `current` from `previous`, and write this frame's into `current`. Both count as
// written when the frame starts, and whatever writes `current` runs.
template <class G>
struct HistoryImage {
    ImageRef<G> previous;
    ImageRef<G> current;
    // False when `previous` holds nothing yet: the images are new, because the view had
    // not used them, had not for a while, or asked for another extent.
    bool valid = false;
};

// What FrameGraphBuilder implementations get for each resource.
struct ResourceDesc {
    std::string name;
    std::type_index type;  // the tc::BufferResource<T> or GraphImage<G>
    std::uint64_t size;    // elements, or width << 32 | height
    bool hostVisible;      // read by the host once the frame is done
    std::function<std::shared_ptr<void>(tc::Device &)> create;
    std::function<void const *(void *)> hostData; // hostVisible buffers
};

// A pass's view of the GPU thread while it records.
struct PassContext {
    virtual ~PassContext() = default;

    // Dispatch through dispatch() and dispatchIndirect(), which attribute device time to
    // the pass (RendererConfig::gpuTimers).
    virtual tc::Device &device() noexcept = 0;
    virtual SceneTables const &scene() const noexcept = 0;

    template <class T>
    tc::BufferResource<T> &buffer(BufferRef<T> const ref) {
        return *static_cast<tc::BufferResource<T> *>(resource(ref.index, typeid(tc::BufferResource<T>)));
    }

    template <class G>
    GraphImage<G> &image(ImageRef<G> const ref) {
        return *static_cast<GraphImage<G> *>(resource(ref.index, typeid(GraphImage<G>)));
    }

    // device().execute() and device().executeIndirect(), as part of this pass.
    template <class K>
    void dispatch(K &kernel, tc::uvec3 const workGroups) {
        device().execute(kernel, workGroups);
        dispatched(K::fileLocation);
    }
    // With the workgroup counts arguments[index] holds when the device gets there.
    template <class K>
    void dispatchIndirect(K &kernel, BufferRef<tc::DispatchIndirectCommand> const arguments,
                          std::size_t const index = 0) {
        device().executeIndirect(kernel, buffer(arguments), index);
        dispatched(K::fileLocation);
    }

protected:
    // Throws std::logic_error when `index` is not a resource of `type`.
    virtual void *resource(std::uint32_t index, std::type_info const &type) = 0;
    virtual void dispatched(std::string_view program) = 0;
};

using PassRecorder = std::function<void(PassContext &)>;

struct PassBuilder {
    virtual ~PassBuilder() = default;

    template <class T>
    BufferRef<T> read(BufferRef<T> const ref) {
        declare(ref.index, false);
        return ref;
    }
    template <class T>
    BufferRef<T> write(BufferRef<T> const ref) {
        declare(ref.index, true);
        return ref;
    }
    template <class G>
    ImageRef<G> read(ImageRef<G> const ref) {
        declare(ref.index, false);
        return ref;
    }
    template <class G>
    ImageRef<G> write(ImageRef<G> const ref) {
        declare(ref.index, true);
        return ref;
    }

    // Runs on the GPU thread when the frame is recorded, after the passes declared before it.
    virtual void record(PassRecorder recorder) = 0;

protected:
    virtual void declare(std::uint32_t resource, bool writes) = 0;
};

// What the features of one view hand to the features after them: one value per type.
struct Blackboard {
    template <class T>
    void put(T value) {
        entries[std::type_index{typeid(T)}] = std::make_shared<T>(std::move(value));
    }

    template <class T>
    bool contains() const {
        return entries.count(std::type_index{typeid(T)}) != 0;
    }

    // Throws std::logic_error when no feature before put a T.
    template <class T>
    T const &get() const {
        auto const found = entries.find(std::type_index{typeid(T)});
        if (found == entries.end()) {
            throw std::logic_error{std::string{"lr::render::Blackboard: nothing put for "} + typeid(T).name()};
        }
        return *static_cast<T const *>(found->second.get());
    }

private:
    std::unordered_map<std::type_index, std::shared_ptr<void>> entries;
};

using ResultHandler = std::function<void(ViewResult &)>;

struct FrameGraphBuilder {
    virtual ~FrameGraphBuilder() = default;

    // Device memory, contents undefined at the start of the frame.
    template <class T>
    BufferRef<T> createBuffer(std::string name, std::size_t const count) {
        return BufferRef<T>{declareResource(bufferDesc<T>(std::move(name), count, false))};
    }

    // Memory the host reads with readback() once the frame is done.
    template <class T>
    BufferRef<T> createReadbackBuffer(std::string name, std::size_t const count) {
        return BufferRef<T>{declareResource(bufferDesc<T>(std::move(name), count, true))};
    }

    template <class G>
    ImageRef<G> createImage(std::string name, core::ivec2 const extent) {
        return ImageRef<G>{declareResource(imageDesc<G>(std::move(name), extent))};
    }

    // The view's history images called `name`; the same pair, swapped, in every frame of
    // the view that asks for them with the same format and extent. Throws
    // std::logic_error for a graph that belongs to no view.
    template <class G>
    HistoryImage<G> history(std::string name, core::ivec2 const extent) {
        auto const declared = declareHistory(imageDesc<G>(std::move(name), extent));
        return HistoryImage<G>{ImageRef<G>{declared.previous}, ImageRef<G>{declared.current}, declared.valid};
    }

    virtual PassBuilder &addPass(std::string name) = 0;

    // Calls `handler` with the buffer's contents on the GPU thread once the frame is done.
    // The buffer must come from createReadbackBuffer(); this makes it an output of the
    // graph, so the passes that write it run.
    template <class T>
    void readback(BufferRef<T> const ref, std::function<void(T const *data, std::size_t count, ViewResult &)> handler) {
        addReadback(ref.index, [handler = std::move(handler)](void const *data, std::size_t const count,
                                                              ViewResult &result) {
            handler(static_cast<T const *>(data), count, result);
        });
    }

    // Calls `handler` on the GPU thread once the frame is done, after the readbacks.
    virtual void onFinish(ResultHandler handler) = 0;

    virtual Blackboard &blackboard() noexcept = 0;

protected:
    using ReadbackHandler = std::function<void(void const *data, std::size_t count, ViewResult &)>;

    struct DeclaredHistory {
        std::uint32_t previous;
        std::uint32_t current;
        bool valid;
    };

    virtual std::uint32_t declareResource(ResourceDesc desc) = 0;
    virtual DeclaredHistory declareHistory(ResourceDesc desc) = 0;
    virtual void addReadback(std::uint32_t resource, ReadbackHandler handler) = 0;

private:
    template <class T>
    static ResourceDesc bufferDesc(std::string name, std::size_t const count, bool const hostVisible) {
        if (count == 0 or count > 0x7FFFFFFFu) {
            throw std::invalid_argument{"lr::render: graph buffer '" + name + "' needs 1 to 2^31 - 1 elements"};
        }
        auto const elements = static_cast<std::int32_t>(count);
        auto create = [elements, hostVisible](tc::Device &device) -> std::shared_ptr<void> {
            if (hostVisible) {
                return std::make_shared<tc::BufferResource<T>>(device.createBuffer<T>(elements));
            }
            auto buffer = std::make_shared<tc::BufferResource<T>>(elements);
            device.uploadBuffer(*buffer);
            return buffer;
        };
        auto hostData = [](void *const resource) -> void const * {
            return static_cast<tc::BufferResource<T> const *>(resource)->data();
        };
        return ResourceDesc{std::move(name), std::type_index{typeid(tc::BufferResource<T>)}, count, hostVisible,
                            std::move(create), std::move(hostData)};
    }

    template <class G>
    static ResourceDesc imageDesc(std::string name, core::ivec2 const extent) {
        auto const size = static_cast<std::uint64_t>(extent.x) << 32 | static_cast<std::uint32_t>(extent.y);
        auto create = [extent](tc::Device &device) -> std::shared_ptr<void> {
            return std::make_shared<GraphImage<G>>(device.createImage<G, tc::Dim::D2>(extent, 1));
        };
        return ResourceDesc{std::move(name), std::type_index{typeid(GraphImage<G>)}, size, false, std::move(create), {}};
    }
};

// The directional light the visibility pipeline gives shadows to.
struct ShadowLight {
    std::uint32_t index = 0;   // in the light table
    core::vec3 direction{0.0f}; // that the light travels, unit length
};

// A view whose pipeline jitters (FeatureRequirements::temporalJitter) and whose settings
// ask for temporal anti-aliasing: ViewContext::viewProjection moves where points land by
// `jitter` pixels this frame.
struct TemporalView {
    core::vec2 jitter{0.0f};   // pixels, right and down; within half a pixel
    std::uint32_t sample = 0;  // frames since the view's history started over
    core::mat4 viewProjection{1.0f};         // this frame's, without the jitter
    core::mat4 previousViewProjection{1.0f}; // the previous frame's, without its jitter
    // The previous frame rendered the view, jittered, at this extent and the camera did not
    // cut: its history still shows the scene.
    bool continues = false;
};

struct MaterialModel;

// What a feature knows about the view it sets up.
struct ViewContext {
    ViewId view;
    core::ivec2 extent{0};
    core::CameraDesc camera;
    ViewSettings settings;
    core::EnvironmentDesc environment;
    core::mat4 viewMatrix{1.0f};
    core::mat4 projection{1.0f};
    // Jittered where `temporal` is set.
    core::mat4 viewProjection{1.0f};
    core::mat4 inverseViewProjection{1.0f};
    std::optional<TemporalView> temporal;
    RendererCapabilities capabilities;
    // The renderer's material models when the frame was prepared, by MaterialModelId.
    std::vector<std::shared_ptr<MaterialModel>> materialModels;
    std::uint32_t instanceCount = 0;
    std::uint32_t lightCount = 0;
    // Instances whose material blends.
    std::uint32_t transparentInstanceCount = 0;
    // Of every instance; min > max when there are none.
    core::Aabb sceneBounds;
    std::optional<ShadowLight> shadowLight;
    // Changes whenever a frame's delta changes meshes, textures, materials, instances,
    // lights or the environment.
    std::uint64_t sceneVersion = 0;
    // Shading writes HDR radiance here; post-processing reads and rewrites it.
    ImageRef<HdrFormat> sceneColor;
    // The final pass writes the view's packed RGBA8 pixels here: a readback buffer the
    // renderer hands to the view's target once the frame is done.
    BufferRef<tc::uint> output;
};

enum class FeatureStage : std::uint8_t {
    Visibility,
    Shadow,
    Shading,
    Transparent,
    PostProcess,
    Output,
};

// What a feature needs of the scene and the view beyond what every feature gets.
struct FeatureRequirements {
    // The hierarchies of SceneTables. Built from the first frame with a view that needs
    // them, then kept up to date.
    bool rayTracing = false;
    // A projection jittered anew every frame (ViewContext::temporal) for views whose
    // settings ask for temporal anti-aliasing.
    bool temporalJitter = false;
};

// A pipeline is a list of features, set up in stage order (and, within a stage, in the
// order they were added). Features may keep state; setup() only runs on the render thread.
struct RenderFeature {
    virtual ~RenderFeature() = default;

    virtual FeatureStage stage() const noexcept = 0;
    virtual FeatureRequirements requirements() const noexcept { return FeatureRequirements{}; }
    virtual void setup(FrameGraphBuilder &graph, ViewContext const &view) = 0;
};

} // namespace lr::render
