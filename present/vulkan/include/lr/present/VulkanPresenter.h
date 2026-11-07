#pragma once

#include <lr/render/Presenter.h>
#include <lr/render/View.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace lr::present {

enum class PresentMode : std::uint8_t {
    Fifo,      // one frame per vertical blank, waiting for it; every surface has it
    Mailbox,   // the newest frame at each vertical blank, without waiting; Fifo where missing
    Immediate, // at once, tearing; Fifo where missing
};

struct VulkanPresenterConfig {
    // The instance extensions the window system's surface needs, such as those
    // SDL_Vulkan_GetInstanceExtensions() lists.
    std::vector<std::string> instanceExtensions;
    // Makes the window's surface on the presenter's instance, which then owns it. Without
    // it the presenter shows nothing (tests, measurements): it presents to a surface of
    // VK_EXT_headless_surface, which only Mesa's drivers have (NVIDIA's crashes when asked
    // about one), or, `offscreen`, copies into BGRA images of its own in place of a
    // swapchain's, which every driver can.
    std::function<VkSurfaceKHR(VkInstance)> createSurface;
    bool offscreen = false;
    // The first GPU whose name contains this and that presents to the surface.
    std::string deviceNameContains;
    PresentMode mode = PresentMode::Fifo;
    // Offers the renderer the SharedBuffer path where its device is the presenter's GPU.
    bool shareBuffers = true;
};

struct PresentStats {
    std::uint64_t shown = 0;
    std::uint64_t sharedShown = 0; // of `shown`, through the SharedBuffer path
    // Replaced by a newer frame before the swapchain took them, or handed over with no
    // buffer free.
    std::uint64_t dropped = 0;
    // Host time from handing a frame over (present(), presentShared()) to presenting it.
    double meanLatencyMs = 0.0;
    double maxLatencyMs = 0.0;
};

// Shows one window target's frames through a Vulkan device and swapchain of its own.
// Where the renderer's device is the same GPU and exports sync files, frames take the
// SharedBuffer path: the renderer writes them into buffers this presenter exported, and
// its own queue waits for the frame's fence, copies the buffer into a swapchain image
// (blitting it when the swapchain's format or extent differs) and presents. Otherwise the
// Readback path copies the pixels into a staging buffer first. A thread of its own takes
// the newest frame handed over each time the swapchain can take one, so a renderer faster
// than the display drops frames and never waits for the presenter.
struct VulkanPresenter : render::Presenter {
    virtual std::string deviceName() const = 0;
    virtual PresentStats stats() const = 0;
    // The pixels of the next frame shown, read back from the swapchain image, as
    // ImageRgba8; the future fails where the swapchain's images cannot be read.
    virtual std::future<render::ImageRgba8> captureNext() = 0;
    // Waits until every frame handed over so far has been shown or dropped, and the
    // device has finished with the frames shown.
    virtual void flush() = 0;
};

// Throws std::runtime_error when there is no such GPU, or it cannot make the surface or
// present to it.
std::shared_ptr<VulkanPresenter> makeVulkanPresenter(VulkanPresenterConfig config);

} // namespace lr::present
