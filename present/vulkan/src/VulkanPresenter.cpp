#include <lr/present/VulkanPresenter.h>

#include "VulkanDevice.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lr::present {

namespace {

using detail::check;
using detail::Owned;
using detail::vkStruct;
using Clock = std::chrono::steady_clock;

constexpr auto kSlots = std::size_t{3};
constexpr auto kForever = std::numeric_limits<std::uint64_t>::max();

enum class SlotState : std::uint8_t {
    Free,    // the presenter's last copy from it may still run
    Writing, // the renderer, or present(), writes a frame into it
    Ready,   // the frame waits for the swapchain
    Taken,   // the present thread copies it into a swapchain image
    Shown,   // like Free, with the release fence of the copy (SharedBuffer path)
};

enum class SlotMemory : std::uint8_t {
    None,
    Shared,  // device memory exported to the renderer
    Staging, // host memory the Readback path copies the pixels into
};

struct Slot {
    SlotState state = SlotState::Free;
    SlotMemory kind = SlotMemory::None;
    core::ivec2 extent{0};
    std::uint64_t generation = 0;
    Owned<VkBuffer> buffer;
    Owned<VkDeviceMemory> memory;
    VkDeviceSize allocationBytes = 0;
    std::uint32_t memoryType = 0;
    void *mapped = nullptr; // Staging
    // Between the buffer and swapchain images of another format or extent.
    Owned<VkImage> image;
    Owned<VkDeviceMemory> imageMemory;

    VkCommandBuffer commands = VK_NULL_HANDLE;
    Owned<VkFence> copiedFence; // of the slot's last submission
    bool submitted = false;
    Owned<VkSemaphore> imageAcquired;
    Owned<VkSemaphore> frameWritten; // the renderer's fence, imported for one wait
    Owned<VkSemaphore> copied;       // exported as `release` after each submission

    // Ready.
    std::uint64_t frameIndex = 0;
    tc::SyncFence acquire;
    Clock::time_point handedOver;
    // Shown: signalled once the copy has read the buffer.
    tc::SyncFence release;
};

struct OwnedImage {
    Owned<VkImage> image;
    Owned<VkDeviceMemory> memory;
};

// A swapchain's images, or offscreen images of the presenter's own.
struct Swapchain {
    Owned<VkSwapchainKHR> handle;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{0, 0};
    VkExtent2D requested{0, 0}; // the frames' extent it was made for
    bool surfaceSized = false;  // the surface decides the extent, not the frames
    bool readable = false;      // TRANSFER_SRC: captureNext()
    std::vector<VkImage> images;
    std::vector<Owned<VkSemaphore>> presentable; // per image
    std::vector<OwnedImage> offscreen;
    std::uint32_t next = 0; // the offscreen image the next frame goes to
};

std::string messageOf(std::exception_ptr const &failure) {
    try {
        std::rethrow_exception(failure);
    } catch (std::exception const &error) {
        return error.what();
    } catch (...) {
        return "an unknown failure";
    }
}

bool operator==(VkExtent2D const a, VkExtent2D const b) noexcept {
    return a.width == b.width and a.height == b.height;
}

VkExtent2D extentOf(core::ivec2 const extent) noexcept {
    return VkExtent2D{static_cast<std::uint32_t>(extent.x), static_cast<std::uint32_t>(extent.y)};
}

Owned<VkSemaphore> makeSemaphore(VkDevice const device, bool const exportable) {
    auto exportInfo = vkStruct<VkExportSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    auto info = vkStruct<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    info.pNext = exportable ? &exportInfo : nullptr;
    auto semaphore = VkSemaphore{};
    check(vkCreateSemaphore(device, &info, nullptr, &semaphore), "vkCreateSemaphore");
    return Owned<VkSemaphore>{device, semaphore, vkDestroySemaphore};
}

void imageBarrier(VkCommandBuffer const commands, VkImage const image, VkImageLayout const from, VkImageLayout const to,
                  VkAccessFlags const fromAccess, VkAccessFlags const toAccess, VkPipelineStageFlags const fromStage,
                  VkPipelineStageFlags const toStage) {
    auto barrier = vkStruct<VkImageMemoryBarrier>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
    barrier.srcAccessMask = fromAccess;
    barrier.dstAccessMask = toAccess;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(commands, fromStage, toStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

// The shared buffer passes between the renderer's device and this one.
void bufferHandOver(VkCommandBuffer const commands, VkBuffer const buffer, std::uint32_t const from,
                    std::uint32_t const to, VkAccessFlags const fromAccess, VkAccessFlags const toAccess,
                    VkPipelineStageFlags const fromStage, VkPipelineStageFlags const toStage) {
    auto barrier = vkStruct<VkBufferMemoryBarrier>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
    barrier.srcAccessMask = fromAccess;
    barrier.dstAccessMask = toAccess;
    barrier.srcQueueFamilyIndex = from;
    barrier.dstQueueFamilyIndex = to;
    barrier.buffer = buffer;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(commands, fromStage, toStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

VkBufferImageCopy wholeImage(VkExtent2D const extent) {
    auto region = VkBufferImageCopy{};
    region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = VkExtent3D{extent.width, extent.height, 1};
    return region;
}

VkPresentModeKHR presentModeOf(PresentMode const mode) noexcept {
    switch (mode) {
    case PresentMode::Mailbox:
        return VK_PRESENT_MODE_MAILBOX_KHR;
    case PresentMode::Immediate:
        return VK_PRESENT_MODE_IMMEDIATE_KHR;
    case PresentMode::Fifo:
        break;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

// Unorm formats only: the frames hold sRGB-encoded bytes already.
std::optional<VkSurfaceFormatKHR> chooseFormat(std::vector<VkSurfaceFormatKHR> const &formats) {
    for (auto const wanted : {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM}) {
        for (auto const &format : formats) {
            if (format.format == wanted and format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
    }
    return std::nullopt;
}

class VulkanPresenterImpl final : public VulkanPresenter {
public:
    explicit VulkanPresenterImpl(VulkanPresenterConfig config_) : config(std::move(config_)), gpu(config) {
        auto const device = gpu.device();
        auto poolInfo = vkStruct<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = gpu.family();
        auto pool = VkCommandPool{};
        check(vkCreateCommandPool(device, &poolInfo, nullptr, &pool), "vkCreateCommandPool");
        commandPool = Owned<VkCommandPool>{device, pool, vkDestroyCommandPool};
        auto buffers = std::array<VkCommandBuffer, kSlots>{};
        auto allocateInfo = vkStruct<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        allocateInfo.commandPool = pool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = static_cast<std::uint32_t>(kSlots);
        check(vkAllocateCommandBuffers(device, &allocateInfo, buffers.data()), "vkAllocateCommandBuffers");
        for (auto index = std::size_t{0}; index < kSlots; ++index) {
            auto &slot = slots[index];
            slot.commands = buffers[index];
            auto fenceInfo = vkStruct<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
            auto fence = VkFence{};
            check(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence");
            slot.copiedFence = Owned<VkFence>{device, fence, vkDestroyFence};
            slot.imageAcquired = makeSemaphore(device, false);
            slot.frameWritten = makeSemaphore(device, false);
            if (gpu.shares()) {
                slot.copied = makeSemaphore(device, true);
            }
        }
        thread = std::thread{[this] { run(); }};
    }

    ~VulkanPresenterImpl() override {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            stopping = true;
        }
        work.notify_all();
        thread.join();
        vkDeviceWaitIdle(gpu.device());
    }

    VulkanPresenterImpl(VulkanPresenterImpl const &) = delete;
    VulkanPresenterImpl &operator=(VulkanPresenterImpl const &) = delete;

    std::string deviceName() const override { return gpu.name(); }

    bool sharesBuffersWith(tc::DeviceInfo const &device) const override {
        return config.shareBuffers and gpu.shares() and device.gpu and *device.gpu == gpu.identity();
    }

    render::SharedBufferSlot acquireShared(core::ivec2 const extent) override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto *const slot = pick(true);
        if (slot == nullptr) {
            throw std::runtime_error{"lr::present: every shared buffer is in use"};
        }
        if (slot->state == SlotState::Ready) {
            ++counters.dropped;
            slot->acquire = tc::SyncFence{};
        }
        auto release = copyFinished(*slot) ? tc::SyncFence{} : std::move(slot->release);
        slot->release = tc::SyncFence{};
        if (slot->kind != SlotMemory::Shared or slot->extent != extent) {
            release = tc::SyncFence{};
            allocate(*slot, SlotMemory::Shared, extent);
        }
        slot->state = SlotState::Writing;
        return render::SharedBufferSlot{indexOf(*slot), slot->generation, std::move(release)};
    }

    tc::ExternalBufferMemory exportShared(std::uint32_t const index) override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto const &slot = slots.at(index);
        auto info = vkStruct<VkMemoryGetFdInfoKHR>(VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR);
        info.memory = slot.memory.get();
        info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto fd = -1;
        check(gpu.getMemoryFd(gpu.device(), &info, &fd), "vkGetMemoryFdKHR");
        auto memory = tc::ExternalBufferMemory{};
        memory.fd = tc::MemoryFd{fd};
        memory.allocationBytes = static_cast<std::size_t>(slot.allocationBytes);
        memory.memoryType = slot.memoryType;
        return memory;
    }

    void presentShared(render::SharedBufferFrame frame) override {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            auto &slot = slots.at(frame.slot);
            slot.state = SlotState::Ready;
            slot.frameIndex = frame.frameIndex;
            slot.acquire = std::move(frame.acquire);
            slot.handedOver = Clock::now();
        }
        work.notify_one();
    }

    void discardShared(std::uint32_t const index) override {
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            slots.at(index).state = SlotState::Free;
        }
        changed.notify_all();
    }

    void present(render::PresentFrame const &frame) override {
        auto const extent = frame.pixels.extent;
        auto *slot = static_cast<Slot *>(nullptr);
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            slot = pick(false);
            if (slot == nullptr) {
                ++counters.dropped;
                return;
            }
            if (slot->state == SlotState::Ready) {
                ++counters.dropped;
            }
            slot->acquire = tc::SyncFence{};
            slot->release = tc::SyncFence{};
            if (slot->kind != SlotMemory::Staging or slot->extent != extent) {
                allocate(*slot, SlotMemory::Staging, extent);
            }
            slot->state = SlotState::Writing;
        }
        std::memcpy(slot->mapped, frame.pixels.pixels,
                    static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y) * 4u);
        {
            auto const lock = std::lock_guard<std::mutex>{mutex};
            slot->state = SlotState::Ready;
            slot->frameIndex = frame.frameIndex;
            slot->handedOver = Clock::now();
        }
        work.notify_one();
    }

    PresentStats stats() const override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        auto result = counters;
        result.meanLatencyMs = counters.shown == 0 ? 0.0 : latencySum / static_cast<double>(counters.shown);
        return result;
    }

    std::future<render::ImageRgba8> captureNext() override {
        auto const lock = std::lock_guard<std::mutex>{mutex};
        captures.emplace_back();
        return captures.back().get_future();
    }

    void flush() override {
        auto lock = std::unique_lock<std::mutex>{mutex};
        changed.wait(lock, [this] {
            return std::none_of(slots.begin(), slots.end(), [](Slot const &slot) {
                return slot.state == SlotState::Ready or slot.state == SlotState::Taken;
            });
        });
        for (auto &slot : slots) {
            if (slot.state != SlotState::Writing) {
                waitCopied(slot);
            }
        }
    }

private:
    std::uint32_t indexOf(Slot const &slot) const noexcept {
        return static_cast<std::uint32_t>(&slot - slots.data());
    }

    // Under the lock, for a slot the present thread does not hold.
    bool copyFinished(Slot const &slot) const {
        return not slot.submitted or vkGetFenceStatus(gpu.device(), slot.copiedFence.get()) == VK_SUCCESS;
    }

    void waitCopied(Slot const &slot) const {
        if (slot.submitted) {
            auto const fence = slot.copiedFence.get();
            check(vkWaitForFences(gpu.device(), 1, &fence, VK_TRUE, kForever), "vkWaitForFences");
        }
    }

    // Under the lock: where a new frame goes. A slot no copy reads any more; else, where
    // the renderer's device waits (`deviceWaits`), one whose copy it waits for; else the
    // oldest frame the swapchain has not taken, which is dropped.
    Slot *pick(bool const deviceWaits) {
        auto *running = static_cast<Slot *>(nullptr);
        auto *ready = static_cast<Slot *>(nullptr);
        for (auto &slot : slots) {
            if (slot.state == SlotState::Free or slot.state == SlotState::Shown) {
                if (copyFinished(slot)) {
                    return &slot;
                }
                if (deviceWaits and not slot.release.empty() and running == nullptr) {
                    running = &slot;
                }
            } else if (slot.state == SlotState::Ready and (ready == nullptr or slot.frameIndex < ready->frameIndex)) {
                ready = &slot;
            }
        }
        return running != nullptr ? running : ready;
    }

    // Under the lock: new memory for the slot, once its last copy has finished.
    void allocate(Slot &slot, SlotMemory const kind, core::ivec2 const extent) {
        waitCopied(slot);
        auto const device = gpu.device();
        slot.image.reset();
        slot.imageMemory.reset();
        slot.mapped = nullptr;
        slot.buffer.reset();
        slot.memory.reset();
        auto const bytes = static_cast<VkDeviceSize>(extent.x) * static_cast<VkDeviceSize>(extent.y) * 4u;
        auto const shared = kind == SlotMemory::Shared;
        auto externalBuffer = vkStruct<VkExternalMemoryBufferCreateInfo>(VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
        externalBuffer.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto bufferInfo = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        bufferInfo.pNext = shared ? &externalBuffer : nullptr;
        bufferInfo.size = bytes;
        bufferInfo.usage =
            shared ? detail::kSharedBufferUsage : VkBufferUsageFlags{VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        auto buffer = VkBuffer{};
        check(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");
        slot.buffer = Owned<VkBuffer>{device, buffer, vkDestroyBuffer};
        auto requirements = VkMemoryRequirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        auto const wanted = shared ? VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}
                                   : VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                           | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
        auto const type = gpu.memoryType(requirements.memoryTypeBits, wanted);
        if (not type) {
            throw std::runtime_error{"lr::present: no memory type for the presenter's buffers"};
        }
        auto exportInfo = vkStruct<VkExportMemoryAllocateInfo>(VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO);
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto allocateInfo = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        allocateInfo.pNext = shared ? &exportInfo : nullptr;
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = *type;
        auto memory = VkDeviceMemory{};
        check(vkAllocateMemory(device, &allocateInfo, nullptr, &memory), "vkAllocateMemory");
        slot.memory = Owned<VkDeviceMemory>{device, memory, vkFreeMemory};
        check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
        if (not shared) {
            check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &slot.mapped), "vkMapMemory");
        }
        slot.kind = kind;
        slot.extent = extent;
        slot.allocationBytes = requirements.size;
        slot.memoryType = *type;
        slot.generation = ++generations;
    }

    // A device image that copies and blits read and write.
    OwnedImage makeImage(VkFormat const format, VkExtent2D const extent) const {
        auto const device = gpu.device();
        auto info = vkStruct<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = VkExtent3D{extent.width, extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto image = VkImage{};
        check(vkCreateImage(device, &info, nullptr, &image), "vkCreateImage");
        auto owned = OwnedImage{Owned<VkImage>{device, image, vkDestroyImage}, {}};
        auto requirements = VkMemoryRequirements{};
        vkGetImageMemoryRequirements(device, image, &requirements);
        auto const type = gpu.memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (not type) {
            throw std::runtime_error{"lr::present: no memory type for the presenter's images"};
        }
        auto allocateInfo = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = *type;
        auto memory = VkDeviceMemory{};
        check(vkAllocateMemory(device, &allocateInfo, nullptr, &memory), "vkAllocateMemory");
        owned.memory = Owned<VkDeviceMemory>{device, memory, vkFreeMemory};
        check(vkBindImageMemory(device, image, memory, 0), "vkBindImageMemory");
        return owned;
    }

    // Present thread: the slot's image, made for its extent where missing.
    VkImage blitSource(Slot &slot) {
        if (not slot.image) {
            auto made = makeImage(VK_FORMAT_R8G8B8A8_UNORM, extentOf(slot.extent));
            slot.image = std::move(made.image);
            slot.imageMemory = std::move(made.memory);
        }
        return slot.image.get();
    }

    bool offscreen() const noexcept { return gpu.surface() == VK_NULL_HANDLE; }

    // Present thread: three images of the frames' extent take turns.
    void createOffscreen(VkExtent2D const requested) {
        check(vkQueueWaitIdle(gpu.queue()), "vkQueueWaitIdle");
        auto next = Swapchain{};
        next.format = VK_FORMAT_B8G8R8A8_UNORM;
        next.extent = requested;
        next.requested = requested;
        next.readable = true;
        for (auto index = 0; index < 3; ++index) {
            next.offscreen.push_back(makeImage(next.format, requested));
            next.images.push_back(next.offscreen.back().image.get());
        }
        swapchain = std::move(next);
        outOfDate = false;
    }

    // Present thread. Leaves the swapchain empty while the surface has no area.
    void createSwapchain(VkExtent2D const requested) {
        if (offscreen()) {
            createOffscreen(requested);
            return;
        }
        auto const physical = gpu.physical();
        auto const surface = gpu.surface();
        auto capabilities = VkSurfaceCapabilitiesKHR{};
        check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &capabilities),
              "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        auto const surfaceSized = capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max();
        auto extent = surfaceSized ? capabilities.currentExtent : requested;
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height =
            std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        check(vkQueueWaitIdle(gpu.queue()), "vkQueueWaitIdle");
        if (extent.width == 0 or extent.height == 0) {
            swapchain = Swapchain{};
            return;
        }
        if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
            throw std::runtime_error{"lr::present: the surface's images cannot be copied to"};
        }
        auto count = std::uint32_t{0};
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, nullptr),
              "vkGetPhysicalDeviceSurfaceFormatsKHR");
        auto formats = std::vector<VkSurfaceFormatKHR>(count);
        check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &count, formats.data()),
              "vkGetPhysicalDeviceSurfaceFormatsKHR");
        auto const format = chooseFormat(formats);
        if (not format) {
            throw std::runtime_error{"lr::present: the surface has no 8-bit RGBA or BGRA unorm format"};
        }
        auto formatProperties = VkFormatProperties{};
        vkGetPhysicalDeviceFormatProperties(physical, format->format, &formatProperties);
        if ((formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) == 0) {
            throw std::runtime_error{"lr::present: the surface's format cannot be blitted to"};
        }
        check(vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &count, nullptr),
              "vkGetPhysicalDeviceSurfacePresentModesKHR");
        auto modes = std::vector<VkPresentModeKHR>(count);
        check(vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface, &count, modes.data()),
              "vkGetPhysicalDeviceSurfacePresentModesKHR");
        auto const wantedMode = presentModeOf(config.mode);
        auto const mode = std::find(modes.begin(), modes.end(), wantedMode) != modes.end() ? wantedMode
                                                                                          : VK_PRESENT_MODE_FIFO_KHR;
        auto images = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount != 0) {
            images = std::min(images, capabilities.maxImageCount);
        }
        auto const readable = (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
        auto const opaque = (capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0;

        auto info = vkStruct<VkSwapchainCreateInfoKHR>(VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);
        info.surface = surface;
        info.minImageCount = images;
        info.imageFormat = format->format;
        info.imageColorSpace = format->colorSpace;
        info.imageExtent = extent;
        info.imageArrayLayers = 1;
        info.imageUsage = VkImageUsageFlags{VK_IMAGE_USAGE_TRANSFER_DST_BIT}
                        | (readable ? VkImageUsageFlags{VK_IMAGE_USAGE_TRANSFER_SRC_BIT} : VkImageUsageFlags{0});
        info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.preTransform = capabilities.currentTransform;
        info.compositeAlpha = opaque ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                     : static_cast<VkCompositeAlphaFlagBitsKHR>(
                                           capabilities.supportedCompositeAlpha
                                           & (~capabilities.supportedCompositeAlpha + 1u));
        info.presentMode = mode;
        info.clipped = VK_TRUE;
        info.oldSwapchain = swapchain.handle.get();
        auto const device = gpu.device();
        auto handle = VkSwapchainKHR{};
        check(vkCreateSwapchainKHR(device, &info, nullptr, &handle), "vkCreateSwapchainKHR");
        auto next = Swapchain{};
        next.handle = Owned<VkSwapchainKHR>{device, handle, vkDestroySwapchainKHR};
        next.format = format->format;
        next.extent = extent;
        next.requested = requested;
        next.surfaceSized = surfaceSized;
        next.readable = readable;
        check(vkGetSwapchainImagesKHR(device, handle, &count, nullptr), "vkGetSwapchainImagesKHR");
        next.images.resize(count);
        check(vkGetSwapchainImagesKHR(device, handle, &count, next.images.data()), "vkGetSwapchainImagesKHR");
        for (auto index = std::uint32_t{0}; index < count; ++index) {
            next.presentable.push_back(makeSemaphore(device, false));
        }
        swapchain = std::move(next);
        outOfDate = false;
    }

    // Present thread: a swapchain image for a frame of `requested` pixels, its semaphore
    // signalled through the slot; nullopt while the surface has no area.
    std::optional<std::uint32_t> acquireImage(Slot &slot, VkExtent2D const requested) {
        for (auto attempt = 0; attempt < 3; ++attempt) {
            if (outOfDate or swapchain.images.empty()
                or (not swapchain.surfaceSized and not(swapchain.requested == requested))) {
                createSwapchain(requested);
            }
            if (swapchain.images.empty()) {
                return std::nullopt;
            }
            if (offscreen()) {
                return swapchain.next++ % static_cast<std::uint32_t>(swapchain.images.size());
            }
            auto image = std::uint32_t{0};
            auto const result = vkAcquireNextImageKHR(gpu.device(), swapchain.handle.get(), kForever,
                                                      slot.imageAcquired.get(), VK_NULL_HANDLE, &image);
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                outOfDate = true;
                continue;
            }
            if (result == VK_SUBOPTIMAL_KHR) {
                outOfDate = true;
            } else {
                check(result, "vkAcquireNextImageKHR");
            }
            return image;
        }
        return std::nullopt;
    }

    struct CaptureBuffer {
        Owned<VkBuffer> buffer;
        Owned<VkDeviceMemory> memory;
        void *mapped = nullptr;
    };

    CaptureBuffer captureBuffer(VkExtent2D const extent) const {
        auto const device = gpu.device();
        auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        info.size = VkDeviceSize{extent.width} * extent.height * 4u;
        info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto buffer = VkBuffer{};
        check(vkCreateBuffer(device, &info, nullptr, &buffer), "vkCreateBuffer");
        auto capture = CaptureBuffer{};
        capture.buffer = Owned<VkBuffer>{device, buffer, vkDestroyBuffer};
        auto requirements = VkMemoryRequirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        auto const type = gpu.memoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (not type) {
            throw std::runtime_error{"lr::present: no host memory to capture into"};
        }
        auto allocateInfo = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        allocateInfo.allocationSize = requirements.size;
        allocateInfo.memoryTypeIndex = *type;
        auto memory = VkDeviceMemory{};
        check(vkAllocateMemory(device, &allocateInfo, nullptr, &memory), "vkAllocateMemory");
        capture.memory = Owned<VkDeviceMemory>{device, memory, vkFreeMemory};
        check(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
        check(vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &capture.mapped), "vkMapMemory");
        return capture;
    }

    // Present thread: the copy of the slot into swapchain image `image`, and into the
    // capture buffer when there is one.
    void record(Slot &slot, std::uint32_t const image, CaptureBuffer const *const capture) {
        auto const commands = slot.commands;
        check(vkResetCommandBuffer(commands, 0), "vkResetCommandBuffer");
        auto begin = vkStruct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(commands, &begin), "vkBeginCommandBuffer");
        auto const shared = slot.kind == SlotMemory::Shared;
        auto const buffer = slot.buffer.get();
        auto const target = swapchain.images[image];
        auto const transfer = VkPipelineStageFlags{VK_PIPELINE_STAGE_TRANSFER_BIT};
        if (shared) {
            bufferHandOver(commands, buffer, VK_QUEUE_FAMILY_EXTERNAL, gpu.family(), 0, VK_ACCESS_TRANSFER_READ_BIT,
                           VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, transfer);
        }
        imageBarrier(commands, target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                     VK_ACCESS_TRANSFER_WRITE_BIT, transfer, transfer);
        auto const frameExtent = extentOf(slot.extent);
        if (swapchain.format == VK_FORMAT_R8G8B8A8_UNORM and swapchain.extent == frameExtent) {
            auto const region = wholeImage(frameExtent);
            vkCmdCopyBufferToImage(commands, buffer, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        } else {
            auto const source = blitSource(slot);
            imageBarrier(commands, source, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                         VK_ACCESS_TRANSFER_WRITE_BIT, transfer, transfer);
            auto const region = wholeImage(frameExtent);
            vkCmdCopyBufferToImage(commands, buffer, source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            imageBarrier(commands, source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, transfer, transfer);
            auto blit = VkImageBlit{};
            blit.srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = VkOffset3D{slot.extent.x, slot.extent.y, 1};
            blit.dstSubresource = blit.srcSubresource;
            blit.dstOffsets[1] = VkOffset3D{static_cast<std::int32_t>(swapchain.extent.width),
                                            static_cast<std::int32_t>(swapchain.extent.height), 1};
            auto const filter = swapchain.extent == frameExtent ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            vkCmdBlitImage(commands, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, filter);
        }
        auto layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        auto access = VkAccessFlags{VK_ACCESS_TRANSFER_WRITE_BIT};
        if (capture != nullptr) {
            imageBarrier(commands, target, layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, access,
                         VK_ACCESS_TRANSFER_READ_BIT, transfer, transfer);
            auto const region = wholeImage(swapchain.extent);
            vkCmdCopyImageToBuffer(commands, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture->buffer.get(), 1,
                                   &region);
            layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            access = VK_ACCESS_TRANSFER_READ_BIT;
        }
        auto const shown = offscreen() ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        imageBarrier(commands, target, layout, shown, access, 0, transfer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        if (shared) {
            bufferHandOver(commands, buffer, gpu.family(), VK_QUEUE_FAMILY_EXTERNAL, VK_ACCESS_TRANSFER_READ_BIT, 0,
                           transfer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        }
        check(vkEndCommandBuffer(commands), "vkEndCommandBuffer");
    }

    // Present thread: whether the queue waits for the renderer's fence; otherwise this
    // thread did.
    bool importWritten(Slot &slot) {
        if (slot.acquire.empty()) {
            return false;
        }
        if (gpu.shares()) {
            auto import = vkStruct<VkImportSemaphoreFdInfoKHR>(VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR);
            import.semaphore = slot.frameWritten.get();
            import.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
            import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            import.fd = slot.acquire.get();
            if (gpu.importSemaphoreFd(gpu.device(), &import) == VK_SUCCESS) {
                slot.acquire.release(); // the semaphore owns it now
                return true;
            }
        }
        slot.acquire.wait();
        slot.acquire = tc::SyncFence{};
        return false;
    }

    // Present thread: shows the slot's frame; false when it was dropped instead.
    bool show(Slot &slot, std::optional<std::promise<render::ImageRgba8>> &capture) {
        auto const device = gpu.device();
        waitCopied(slot);
        auto const image = acquireImage(slot, extentOf(slot.extent));
        if (not image) {
            slot.acquire = tc::SyncFence{};
            return false;
        }
        auto const presents = not offscreen();
        auto waits = std::vector<VkSemaphore>{};
        auto stages = std::vector<VkPipelineStageFlags>{};
        if (presents) {
            waits.push_back(slot.imageAcquired.get());
            stages.push_back(VK_PIPELINE_STAGE_TRANSFER_BIT);
        }
        if (importWritten(slot)) {
            waits.push_back(slot.frameWritten.get());
            stages.push_back(VK_PIPELINE_STAGE_TRANSFER_BIT);
        }
        auto const exportsRelease = slot.kind == SlotMemory::Shared and slot.copied;
        auto signals = std::vector<VkSemaphore>{};
        if (presents) {
            signals.push_back(swapchain.presentable[*image].get());
        }
        if (exportsRelease) {
            signals.push_back(slot.copied.get());
        }
        auto captured = std::optional<CaptureBuffer>{};
        if (capture and swapchain.readable) {
            captured = captureBuffer(swapchain.extent);
        }
        record(slot, *image, captured ? &*captured : nullptr);
        auto const fence = slot.copiedFence.get();
        if (slot.submitted) {
            check(vkResetFences(device, 1, &fence), "vkResetFences");
        }
        auto submit = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        submit.waitSemaphoreCount = static_cast<std::uint32_t>(waits.size());
        submit.pWaitSemaphores = waits.data();
        submit.pWaitDstStageMask = stages.data();
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &slot.commands;
        submit.signalSemaphoreCount = static_cast<std::uint32_t>(signals.size());
        submit.pSignalSemaphores = signals.data();
        check(vkQueueSubmit(gpu.queue(), 1, &submit, fence), "vkQueueSubmit");
        slot.submitted = true;
        if (exportsRelease) {
            auto info = vkStruct<VkSemaphoreGetFdInfoKHR>(VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);
            info.semaphore = slot.copied.get();
            info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            auto fd = -1;
            check(gpu.getSemaphoreFd(device, &info, &fd), "vkGetSemaphoreFdKHR");
            slot.release = tc::SyncFence{fd};
        }
        if (presents) {
            presentImage(*image);
        }
        if (captured) {
            waitCopied(slot);
            capture->set_value(pixelsOf(*captured));
            capture.reset();
        }
        return true;
    }

    void presentImage(std::uint32_t const image) {
        auto const presentable = swapchain.presentable[image].get();
        auto const handle = swapchain.handle.get();
        auto present = vkStruct<VkPresentInfoKHR>(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR);
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &presentable;
        present.swapchainCount = 1;
        present.pSwapchains = &handle;
        present.pImageIndices = &image;
        auto const presented = vkQueuePresentKHR(gpu.queue(), &present);
        if (presented == VK_ERROR_OUT_OF_DATE_KHR or presented == VK_SUBOPTIMAL_KHR) {
            outOfDate = true;
        } else {
            check(presented, "vkQueuePresentKHR");
        }
    }

    render::ImageRgba8 pixelsOf(CaptureBuffer const &capture) const {
        auto const extent = swapchain.extent;
        auto image = render::ImageRgba8{core::ivec2{static_cast<int>(extent.width), static_cast<int>(extent.height)},
                                        std::vector<std::uint32_t>(std::size_t{extent.width} * extent.height)};
        std::memcpy(image.pixels.data(), capture.mapped, image.pixels.size() * sizeof(std::uint32_t));
        if (swapchain.format == VK_FORMAT_B8G8R8A8_UNORM) {
            for (auto &pixel : image.pixels) {
                pixel = (pixel & 0xFF00FF00u) | ((pixel & 0xFFu) << 16) | ((pixel >> 16) & 0xFFu);
            }
        }
        return image;
    }

    // Under the lock: the newest frame waiting, with the older ones dropped.
    Slot *takeNewest() {
        auto *newest = static_cast<Slot *>(nullptr);
        for (auto &slot : slots) {
            if (slot.state == SlotState::Ready and (newest == nullptr or slot.frameIndex > newest->frameIndex)) {
                newest = &slot;
            }
        }
        for (auto &slot : slots) {
            if (slot.state == SlotState::Ready and &slot != newest) {
                slot.state = SlotState::Free;
                slot.acquire = tc::SyncFence{};
                ++counters.dropped;
            }
        }
        if (newest != nullptr) {
            newest->state = SlotState::Taken;
        }
        return newest;
    }

    void run() {
        auto lock = std::unique_lock<std::mutex>{mutex};
        for (;;) {
            work.wait(lock, [this] {
                return stopping or std::any_of(slots.begin(), slots.end(), [](Slot const &slot) {
                           return slot.state == SlotState::Ready;
                       });
            });
            auto *const slot = takeNewest();
            if (slot == nullptr) {
                return;
            }
            auto capture = std::optional<std::promise<render::ImageRgba8>>{};
            if (not captures.empty()) {
                capture = std::move(captures.front());
                captures.erase(captures.begin());
            }
            auto const shared = slot->kind == SlotMemory::Shared;
            auto const handedOver = slot->handedOver;
            auto const working = failed.empty();
            lock.unlock();
            auto shown = false;
            auto failure = std::exception_ptr{};
            if (working) {
                try {
                    shown = show(*slot, capture);
                } catch (std::exception const &) {
                    failure = std::current_exception();
                }
            }
            auto const latency = std::chrono::duration<double, std::milli>(Clock::now() - handedOver).count();
            lock.lock();
            slot->state = shown ? SlotState::Shown : SlotState::Free;
            if (shown) {
                ++counters.shown;
                counters.sharedShown += shared ? 1u : 0u;
                latencySum += latency;
                counters.maxLatencyMs = std::max(counters.maxLatencyMs, latency);
            } else {
                ++counters.dropped;
            }
            if (failure) {
                failed = messageOf(failure);
            }
            if (capture) {
                if (shown or not failed.empty()) {
                    auto const why = failed.empty() ? std::string{"the swapchain's images cannot be read"} : failed;
                    capture->set_exception(std::make_exception_ptr(std::runtime_error{"lr::present: " + why}));
                } else {
                    captures.insert(captures.begin(), std::move(*capture));
                }
            }
            changed.notify_all();
        }
    }

    VulkanPresenterConfig config;
    detail::VulkanDevice gpu;
    Owned<VkCommandPool> commandPool;
    std::array<Slot, kSlots> slots;
    Swapchain swapchain; // present thread
    bool outOfDate = false;

    mutable std::mutex mutex;
    std::condition_variable work;    // a frame is ready, or the presenter goes
    std::condition_variable changed; // a slot stopped being Ready or Taken
    std::uint64_t generations = 0;
    PresentStats counters;
    double latencySum = 0.0;
    std::vector<std::promise<render::ImageRgba8>> captures;
    std::string failed; // why the present thread stopped presenting
    bool stopping = false;
    std::thread thread;
};

} // namespace

std::shared_ptr<VulkanPresenter> makeVulkanPresenter(VulkanPresenterConfig config) {
    return std::make_shared<VulkanPresenterImpl>(std::move(config));
}

} // namespace lr::present
