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

void VulkanBackend::linearCombine(std::span<const std::span<const float>>,
                                  std::span<const float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::gravitationalNBody(std::span<const float>, std::span<const float>,
                                       std::span<const float>, std::span<const float>,
                                       float, std::span<float>, std::span<float>,
                                       std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::electricFieldNBody(std::span<const float>, std::span<const float>,
                                       std::span<const float>, std::span<const float>,
                                       float, std::span<float>, std::span<float>,
                                       std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::magneticFieldNBody(std::span<const float>, std::span<const float>,
                                       std::span<const float>, std::span<const float>,
                                       std::span<const float>, std::span<const float>,
                                       std::span<const float>, float, std::span<float>,
                                       std::span<float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::sphDensityPressure(std::span<const float>, std::span<const float>,
                                       std::span<const float>, std::span<const float>,
                                       float, float, float, std::span<float>,
                                       std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::sphPressureAcceleration(
    std::span<const float>, std::span<const float>, std::span<const float>,
    std::span<const float>, std::span<const float>, std::span<const float>, float,
    std::span<float>, std::span<float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::heatEquation3DStep(std::span<const float>, std::size_t, std::size_t,
                                       std::size_t, float, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::acoustic3DStep(std::span<const float>, std::span<const float>,
                                   std::span<const float>, std::span<const float>,
                                   std::size_t, std::size_t, std::size_t, float, float,
                                   std::span<float>, std::span<float>, std::span<float>,
                                   std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::maxwell3DStep(std::span<const float>, std::span<const float>,
                                  std::span<const float>, std::span<const float>,
                                  std::span<const float>, std::span<const float>,
                                  std::size_t, std::size_t, std::size_t, float, float,
                                  std::span<float>, std::span<float>, std::span<float>,
                                  std::span<float>, std::span<float>,
                                  std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::eulerianFluid3DSweep(std::span<const float>, std::span<const float>,
                                         std::span<const float>, std::span<const float>,
                                         std::span<const float>, std::size_t, std::size_t,
                                         std::size_t, int, float, float, std::span<float>,
                                         std::span<float>, std::span<float>,
                                         std::span<float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::fftBatched(std::span<const float>, std::span<const float>,
                               std::size_t, std::size_t, bool, std::span<float>,
                               std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::matVec(std::span<const float>, std::size_t, std::size_t,
                           std::span<const float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::matMul(std::span<const float>, std::size_t, std::size_t,
                           std::span<const float>, std::size_t, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

bool VulkanBackend::luDecomposeGpu(std::span<const float>, std::size_t, std::span<float>,
                                   std::span<std::uint32_t>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
    return false;
}

bool VulkanBackend::choleskyDecomposeGpu(std::span<const float>, std::size_t,
                                         std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
    return false;
}

void VulkanBackend::qrDecomposeGpu(std::span<const float>, std::size_t, std::size_t,
                                   std::span<float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::jacobiEigenSymmetricGpu(std::span<const float>, std::size_t, int,
                                            float, std::span<float>,
                                            std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::jacobiSvdGpu(std::span<const float>, std::size_t, std::size_t, int,
                                 float, std::span<float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchErf(std::span<const float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchErfc(std::span<const float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchGamma(std::span<const float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchLogGamma(std::span<const float>, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchLegendreP(unsigned, unsigned, std::span<const float>,
                                   std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchPolynomialEval(std::span<const float>, std::span<const float>,
                                        std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchCubicSplineEval(std::span<const float>, std::span<const float>,
                                         std::span<const float>, std::span<const float>,
                                         std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchUniformReal(std::uint64_t, std::uint64_t,
                                     std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::batchNormal(std::uint64_t, std::uint64_t, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::sortAscending(std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::multigridRestrict3D(std::span<const float>, std::size_t, std::size_t,
                                        std::size_t, std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

void VulkanBackend::multigridProlongateAndAdd3D(std::span<const float>,
                                                std::span<const float>, std::size_t,
                                                std::size_t, std::size_t,
                                                std::span<float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
}

std::size_t VulkanBackend::minIndex(std::span<const float>) const {
    assert(false &&
           "VulkanBackend kernels are not implemented yet; see Compute/README.md");
    return 0;
}

}  // namespace ysq
