#include <Renderer/RayTracer.hpp>

#include <Renderer/shaders/Raytrace.frag.hpp>
#include <Renderer/shaders/Raytrace.vert.hpp>

#include <glad/gl.h>

#include <algorithm>
#include <cmath>

namespace ysq {

namespace {

/// Builds the "structure of arrays" a Material list uploads as, one call per
/// primitive type, so RayTracer::render() is not seven near-identical loops
/// written out by hand for spheres, seven more for planes, seven more for
/// disks.
struct MaterialArrays {
    std::vector<Vec3f> albedo;
    std::vector<Vec3f> emissive;
    std::vector<float> ambient;
    std::vector<float> diffuse;
    std::vector<float> specular;
    std::vector<float> shininess;
    std::vector<float> reflectivity;

    void push(const Material& material) {
        albedo.push_back(material.albedo);
        emissive.push_back(material.emissive);
        ambient.push_back(material.ambient);
        diffuse.push_back(material.diffuse);
        specular.push_back(material.specular);
        shininess.push_back(material.shininess);
        reflectivity.push_back(material.reflectivity);
    }

    void upload(const Shader& shader, std::string_view prefix) const {
        shader.setUniformArray(std::string{prefix} + "Albedo", albedo);
        shader.setUniformArray(std::string{prefix} + "Emissive", emissive);
        shader.setUniformArray(std::string{prefix} + "Ambient", ambient);
        shader.setUniformArray(std::string{prefix} + "Diffuse", diffuse);
        shader.setUniformArray(std::string{prefix} + "Specular", specular);
        shader.setUniformArray(std::string{prefix} + "Shininess", shininess);
        shader.setUniformArray(std::string{prefix} + "Reflectivity", reflectivity);
    }
};

}  // namespace

std::optional<RayTracer> RayTracer::create(std::string* error) {
    std::optional<Shader> shader = Shader::compile(shaders::kRaytraceVertSource,
                                                   shaders::kRaytraceFragSource, error);
    if (!shader) {
        return std::nullopt;
    }
    std::optional<Mesh> quad = Mesh::quad(2.0f);
    if (!quad) {
        return std::nullopt;
    }
    return std::optional<RayTracer>{RayTracer{std::move(*shader), std::move(*quad)}};
}

void RayTracer::render(const RaytracedScene& scene, const Camera& camera, float aspect,
                       int viewportWidth, int viewportHeight, int maxBounces) {
    glViewport(0, 0, viewportWidth, viewportHeight);
    // A full-screen pass determines every pixel outright; anything already
    // in the depth buffer must not occlude it, and it writes no depth of its
    // own to occlude anything drawn after.
    glDisable(GL_DEPTH_TEST);

    m_shader.use();

    const Vec3f forward = camera.forward();
    const Vec3f right = normalized(cross(forward, camera.up));
    const Vec3f up = cross(right, forward);

    m_shader.setUniform("uCameraPosition", camera.position);
    m_shader.setUniform("uCameraForward", forward);
    m_shader.setUniform("uCameraRight", right);
    m_shader.setUniform("uCameraUp", up);
    // RayTracer always generates a perspective frustum: orthographic ray
    // tracing (parallel rays) is not a case any scene here needs yet.
    m_shader.setUniform("uTanHalfFovY",
                        std::tan(camera.perspectiveSettings.fovYRadians / 2.0f));
    m_shader.setUniform("uAspect", aspect);
    m_shader.setUniform("uMaxBounces", std::min(maxBounces, maxBounceDepth()));
    m_shader.setUniform("uBackgroundColor", scene.backgroundColor);
    m_shader.setUniform("uHasEnvironment", scene.environment != nullptr ? 1 : 0);
    if (scene.environment != nullptr) {
        scene.environment->bind(0);
        m_shader.setUniform("uEnvironment", 0);
    }

    const std::size_t sphereCount = std::min(scene.spheres.size(), maxSpheres());
    std::vector<Vec3f> sphereCenter;
    std::vector<float> sphereRadius;
    MaterialArrays sphereMaterial;
    for (std::size_t i = 0; i < sphereCount; ++i) {
        sphereCenter.push_back(scene.spheres[i].center);
        sphereRadius.push_back(scene.spheres[i].radius);
        sphereMaterial.push(scene.spheres[i].material);
    }
    m_shader.setUniform("uSphereCount", static_cast<int>(sphereCount));
    m_shader.setUniformArray("uSphereCenter", sphereCenter);
    m_shader.setUniformArray("uSphereRadius", sphereRadius);
    sphereMaterial.upload(m_shader, "uSphere");

    const std::size_t planeCount = std::min(scene.planes.size(), maxPlanes());
    std::vector<Vec3f> planePoint;
    std::vector<Vec3f> planeNormal;
    MaterialArrays planeMaterial;
    for (std::size_t i = 0; i < planeCount; ++i) {
        planePoint.push_back(scene.planes[i].point);
        planeNormal.push_back(scene.planes[i].normal);
        planeMaterial.push(scene.planes[i].material);
    }
    m_shader.setUniform("uPlaneCount", static_cast<int>(planeCount));
    m_shader.setUniformArray("uPlanePoint", planePoint);
    m_shader.setUniformArray("uPlaneNormal", planeNormal);
    planeMaterial.upload(m_shader, "uPlane");

    const std::size_t diskCount = std::min(scene.disks.size(), maxDisks());
    std::vector<Vec3f> diskCenter;
    std::vector<Vec3f> diskNormal;
    std::vector<float> diskInnerRadius;
    std::vector<float> diskOuterRadius;
    MaterialArrays diskMaterial;
    for (std::size_t i = 0; i < diskCount; ++i) {
        diskCenter.push_back(scene.disks[i].center);
        diskNormal.push_back(scene.disks[i].normal);
        diskInnerRadius.push_back(scene.disks[i].innerRadius);
        diskOuterRadius.push_back(scene.disks[i].outerRadius);
        diskMaterial.push(scene.disks[i].material);
    }
    m_shader.setUniform("uDiskCount", static_cast<int>(diskCount));
    m_shader.setUniformArray("uDiskCenter", diskCenter);
    m_shader.setUniformArray("uDiskNormal", diskNormal);
    m_shader.setUniformArray("uDiskInnerRadius", diskInnerRadius);
    m_shader.setUniformArray("uDiskOuterRadius", diskOuterRadius);
    diskMaterial.upload(m_shader, "uDisk");

    const std::size_t pointLightCount =
        std::min(scene.pointLights.size(), maxPointLights());
    std::vector<Vec3f> pointLightPosition;
    std::vector<Vec3f> pointLightColor;
    std::vector<float> pointLightIntensity;
    std::vector<float> pointLightRadius;
    for (std::size_t i = 0; i < pointLightCount; ++i) {
        pointLightPosition.push_back(scene.pointLights[i].position);
        pointLightColor.push_back(scene.pointLights[i].color);
        pointLightIntensity.push_back(scene.pointLights[i].intensity);
        pointLightRadius.push_back(scene.pointLights[i].radius);
    }
    m_shader.setUniform("uPointLightCount", static_cast<int>(pointLightCount));
    m_shader.setUniformArray("uPointLightPosition", pointLightPosition);
    m_shader.setUniformArray("uPointLightColor", pointLightColor);
    m_shader.setUniformArray("uPointLightIntensity", pointLightIntensity);
    m_shader.setUniformArray("uPointLightRadius", pointLightRadius);

    const std::size_t directionalLightCount =
        std::min(scene.directionalLights.size(), maxDirectionalLights());
    std::vector<Vec3f> directionalLightDirection;
    std::vector<Vec3f> directionalLightColor;
    std::vector<float> directionalLightIntensity;
    for (std::size_t i = 0; i < directionalLightCount; ++i) {
        directionalLightDirection.push_back(scene.directionalLights[i].direction);
        directionalLightColor.push_back(scene.directionalLights[i].color);
        directionalLightIntensity.push_back(scene.directionalLights[i].intensity);
    }
    m_shader.setUniform("uDirectionalLightCount",
                        static_cast<int>(directionalLightCount));
    m_shader.setUniformArray("uDirectionalLightDirection", directionalLightDirection);
    m_shader.setUniformArray("uDirectionalLightColor", directionalLightColor);
    m_shader.setUniformArray("uDirectionalLightIntensity", directionalLightIntensity);

    m_quad.draw();

    glEnable(GL_DEPTH_TEST);
}

}  // namespace ysq
