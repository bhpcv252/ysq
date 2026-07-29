#include <Compute/Vulkan/VulkanBackend.hpp>

#include <Core/Logger.hpp>

#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

namespace ysq {

namespace {

bool hasComputeCapableDevice(VkInstance instance) {
    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const VkPhysicalDevice device : devices) {
        std::uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());
        for (const VkQueueFamilyProperties& family : families) {
            if ((family.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                return true;
            }
        }
    }
    return false;
}

}  // namespace

std::unique_ptr<ComputeBackend> VulkanBackend::create() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "ysq";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
        logging::debug("Vulkan compute backend unavailable: vkCreateInstance failed ({})",
                       static_cast<int>(result));
        return nullptr;
    }

    const bool hasDevice = hasComputeCapableDevice(instance);
    vkDestroyInstance(instance, nullptr);
    if (!hasDevice) {
        logging::debug("Vulkan compute backend unavailable: no compute-capable device");
        return nullptr;
    }
    return std::make_unique<VulkanBackend>();
}

void VulkanBackend::saxpy(std::span<const float>, std::span<float>, float) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

float VulkanBackend::sum(std::span<const float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
    return 0.0f;
}

}  // namespace ysq
