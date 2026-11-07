#include "VulkanDevice.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace lr::present::detail {

namespace {

std::vector<VkExtensionProperties> instanceExtensions() {
    auto count = std::uint32_t{0};
    check(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr), "vkEnumerateInstanceExtensionProperties");
    auto extensions = std::vector<VkExtensionProperties>(count);
    check(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()),
          "vkEnumerateInstanceExtensionProperties");
    return extensions;
}

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice const physical) {
    auto count = std::uint32_t{0};
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr),
          "vkEnumerateDeviceExtensionProperties");
    auto extensions = std::vector<VkExtensionProperties>(count);
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()),
          "vkEnumerateDeviceExtensionProperties");
    return extensions;
}

bool has(std::vector<VkExtensionProperties> const &extensions, char const *const name) {
    return std::any_of(extensions.begin(), extensions.end(),
                       [name](VkExtensionProperties const &extension) { return std::strcmp(extension.extensionName, name) == 0; });
}

// The family of the first queue with graphics that presents to the surface, if there is one.
std::optional<std::uint32_t> presentingFamily(VkPhysicalDevice const physical, VkSurfaceKHR const surface) {
    auto count = std::uint32_t{0};
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    auto families = std::vector<VkQueueFamilyProperties>(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    for (auto family = std::uint32_t{0}; family < count; ++family) {
        if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
            continue;
        }
        auto presents = VkBool32{VK_TRUE};
        if (surface != VK_NULL_HANDLE
            and vkGetPhysicalDeviceSurfaceSupportKHR(physical, family, surface, &presents) != VK_SUCCESS) {
            continue;
        }
        if (presents == VK_TRUE) {
            return family;
        }
    }
    return std::nullopt;
}

// Exports opaque file descriptors of buffers like tc's and imports and exports sync files.
bool canShare(VkPhysicalDevice const physical, std::vector<VkExtensionProperties> const &extensions) {
    if (not has(extensions, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)
        or not has(extensions, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME)) {
        return false;
    }
    auto bufferInfo = vkStruct<VkPhysicalDeviceExternalBufferInfo>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO);
    bufferInfo.usage = kSharedBufferUsage;
    bufferInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto buffer = vkStruct<VkExternalBufferProperties>(VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES);
    vkGetPhysicalDeviceExternalBufferProperties(physical, &bufferInfo, &buffer);
    auto const memoryFeatures = buffer.externalMemoryProperties.externalMemoryFeatures;
    auto semaphoreInfo =
        vkStruct<VkPhysicalDeviceExternalSemaphoreInfo>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO);
    semaphoreInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    auto semaphore = vkStruct<VkExternalSemaphoreProperties>(VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES);
    vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &semaphoreInfo, &semaphore);
    auto const semaphoreFeatures = semaphore.externalSemaphoreFeatures;
    return (memoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) != 0
       and (memoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) == 0
       and (semaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0
       and (semaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
}

} // namespace

void check(VkResult const result, char const *const call) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error{std::string{"lr::present: "} + call + " failed (VkResult "
                                 + std::to_string(static_cast<int>(result)) + ")"};
    }
}

VulkanDevice::VulkanDevice(VulkanPresenterConfig const &config) {
    try {
        createInstance(config);
        choosePhysical(config.deviceNameContains);
        createDevice();
    } catch (...) {
        destroy();
        throw;
    }
}

VulkanDevice::~VulkanDevice() { destroy(); }

void VulkanDevice::destroy() noexcept {
    if (logical != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(logical);
        vkDestroyDevice(logical, nullptr);
        logical = VK_NULL_HANDLE;
    }
    if (windowSurface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, windowSurface, nullptr);
        windowSurface = VK_NULL_HANDLE;
    }
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }
}

void VulkanDevice::createInstance(VulkanPresenterConfig const &config) {
    auto const available = instanceExtensions();
    auto const offscreen = not config.createSurface and config.offscreen;
    auto names = offscreen ? std::vector<std::string>{} : std::vector<std::string>{VK_KHR_SURFACE_EXTENSION_NAME};
    names.insert(names.end(), config.instanceExtensions.begin(), config.instanceExtensions.end());
    if (not config.createSurface and not offscreen) {
        names.emplace_back(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
    }
    auto enabled = std::vector<char const *>{};
    for (auto const &name : names) {
        if (not has(available, name.c_str())) {
            throw std::runtime_error{"lr::present: the Vulkan instance has no " + name};
        }
        enabled.push_back(name.c_str());
    }
    auto app = vkStruct<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    app.pApplicationName = "lrender presenter";
    app.apiVersion = VK_API_VERSION_1_1;
    auto info = vkStruct<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = static_cast<std::uint32_t>(enabled.size());
    info.ppEnabledExtensionNames = enabled.data();
    check(vkCreateInstance(&info, nullptr, &instance), "vkCreateInstance");
    if (offscreen) {
        return;
    }
    if (config.createSurface) {
        windowSurface = config.createSurface(instance);
        if (windowSurface == VK_NULL_HANDLE) {
            throw std::runtime_error{"lr::present: the window made no Vulkan surface"};
        }
        return;
    }
    auto const createHeadless = reinterpret_cast<PFN_vkCreateHeadlessSurfaceEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT"));
    auto headless = vkStruct<VkHeadlessSurfaceCreateInfoEXT>(VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT);
    check(createHeadless(instance, &headless, nullptr, &windowSurface), "vkCreateHeadlessSurfaceEXT");
}

void VulkanDevice::choosePhysical(std::string const &nameContains) {
    auto count = std::uint32_t{0};
    check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
    auto devices = std::vector<VkPhysicalDevice>(count);
    check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
    for (auto const candidate : devices) {
        auto ids = vkStruct<VkPhysicalDeviceIDProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
        auto properties = vkStruct<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        properties.pNext = &ids;
        vkGetPhysicalDeviceProperties2(candidate, &properties);
        auto const name = std::string{properties.properties.deviceName};
        if (name.find(nameContains) == std::string::npos or properties.properties.apiVersion < VK_API_VERSION_1_1) {
            continue;
        }
        auto const extensions = deviceExtensions(candidate);
        if (windowSurface != VK_NULL_HANDLE and not has(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            continue;
        }
        auto const family = presentingFamily(candidate, windowSurface);
        if (not family) {
            continue;
        }
        physicalDevice = candidate;
        queueFamily = *family;
        deviceName = name;
        std::copy(std::begin(ids.deviceUUID), std::end(ids.deviceUUID), gpu.device.begin());
        std::copy(std::begin(ids.driverUUID), std::end(ids.driverUUID), gpu.driver.begin());
        sharing = canShare(candidate, extensions);
        vkGetPhysicalDeviceMemoryProperties(candidate, &memory);
        return;
    }
    throw std::runtime_error{"lr::present: no Vulkan device whose name contains '" + nameContains
                             + (windowSurface != VK_NULL_HANDLE ? "' presents to the surface" : "' has graphics")};
}

void VulkanDevice::createDevice() {
    auto const priority = 1.0f;
    auto queueInfo = vkStruct<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;
    auto extensions = std::vector<char const *>{};
    if (windowSurface != VK_NULL_HANDLE) {
        extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    if (sharing) {
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
    }
    auto info = vkStruct<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queueInfo;
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physicalDevice, &info, nullptr, &logical), "vkCreateDevice");
    vkGetDeviceQueue(logical, queueFamily, 0, &deviceQueue);
    if (sharing) {
        getMemoryFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(logical, "vkGetMemoryFdKHR"));
        getSemaphoreFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(logical, "vkGetSemaphoreFdKHR"));
        importSemaphoreFd =
            reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(vkGetDeviceProcAddr(logical, "vkImportSemaphoreFdKHR"));
        sharing = getMemoryFd != nullptr and getSemaphoreFd != nullptr and importSemaphoreFd != nullptr;
    }
}

std::optional<std::uint32_t> VulkanDevice::memoryType(std::uint32_t const types,
                                                      VkMemoryPropertyFlags const wanted) const {
    for (auto index = std::uint32_t{0}; index < memory.memoryTypeCount; ++index) {
        if ((types & (1u << index)) != 0 and (memory.memoryTypes[index].propertyFlags & wanted) == wanted) {
            return index;
        }
    }
    return std::nullopt;
}

} // namespace lr::present::detail
