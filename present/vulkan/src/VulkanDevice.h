#pragma once

#include <lr/present/VulkanPresenter.h>

#include <tc/external_memory.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace lr::present::detail {

// Throws std::runtime_error naming the call unless `result` is VK_SUCCESS.
void check(VkResult result, char const *call);

// A zeroed Vulkan structure of type `type`.
template <class T>
T vkStruct(VkStructureType const type) noexcept {
    auto value = T{};
    value.sType = type;
    return value;
}

// A Vulkan object of a device, destroyed with it.
template <class T>
class Owned {
public:
    using Destroy = void (*)(VkDevice, T, VkAllocationCallbacks const *);

    Owned() noexcept = default;
    Owned(VkDevice const device_, T const handle_, Destroy const destroy_) noexcept
        : device(device_), handle(handle_), destroy(destroy_) {}
    Owned(Owned &&other) noexcept
        : device(other.device), handle(std::exchange(other.handle, VK_NULL_HANDLE)), destroy(other.destroy) {}
    Owned &operator=(Owned &&other) noexcept {
        if (this != &other) {
            reset();
            device = other.device;
            handle = std::exchange(other.handle, VK_NULL_HANDLE);
            destroy = other.destroy;
        }
        return *this;
    }
    Owned(Owned const &) = delete;
    Owned &operator=(Owned const &) = delete;
    ~Owned() { reset(); }

    T get() const noexcept { return handle; }
    explicit operator bool() const noexcept { return handle != VK_NULL_HANDLE; }
    void reset() noexcept {
        if (handle != VK_NULL_HANDLE) {
            destroy(device, handle, nullptr);
        }
        handle = VK_NULL_HANDLE;
    }

private:
    VkDevice device = VK_NULL_HANDLE;
    T handle = VK_NULL_HANDLE;
    Destroy destroy = nullptr;
};

// The presenter's instance, surface (none when offscreen) and device, with the queue it
// submits to: one of a family that has graphics (blits need them) and presents to the
// surface.
class VulkanDevice {
public:
    explicit VulkanDevice(VulkanPresenterConfig const &config);
    VulkanDevice(VulkanDevice const &) = delete;
    VulkanDevice &operator=(VulkanDevice const &) = delete;
    // Waits for the device, then destroys it, the surface and the instance.
    ~VulkanDevice();

    VkPhysicalDevice physical() const noexcept { return physicalDevice; }
    VkDevice device() const noexcept { return logical; }
    VkQueue queue() const noexcept { return deviceQueue; }
    std::uint32_t family() const noexcept { return queueFamily; }
    VkSurfaceKHR surface() const noexcept { return windowSurface; }
    std::string const &name() const noexcept { return deviceName; }
    tc::GpuIdentity const &identity() const noexcept { return gpu; }
    // Buffers of the SharedBuffer path can be exported, and sync files imported and exported.
    bool shares() const noexcept { return sharing; }

    // The first memory type in `types` with every flag of `wanted`; nullopt when none.
    std::optional<std::uint32_t> memoryType(std::uint32_t types, VkMemoryPropertyFlags wanted) const;

    PFN_vkGetMemoryFdKHR getMemoryFd = nullptr;
    PFN_vkGetSemaphoreFdKHR getSemaphoreFd = nullptr;
    PFN_vkImportSemaphoreFdKHR importSemaphoreFd = nullptr;

private:
    void createInstance(VulkanPresenterConfig const &config);
    void choosePhysical(std::string const &nameContains);
    void createDevice();
    void destroy() noexcept;

    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR windowSurface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice logical = VK_NULL_HANDLE;
    VkQueue deviceQueue = VK_NULL_HANDLE;
    std::uint32_t queueFamily = 0;
    std::string deviceName;
    tc::GpuIdentity gpu;
    bool sharing = false;
    VkPhysicalDeviceMemoryProperties memory{};
};

// What buffers of the SharedBuffer path are made with: what tc::Device::importBuffer gives
// the buffer it makes in the imported memory, so both need the same memory types.
constexpr VkBufferUsageFlags kSharedBufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

} // namespace lr::present::detail
