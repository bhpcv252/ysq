#include <Applications/KeplerSolarSystem/Scenario.hpp>

#include <Core/Config.hpp>
#include <Core/Logger.hpp>
#include <Core/Timer.hpp>
#include <Core/UUID.hpp>

#include <Math/Matrix4.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <Physics/Gravity/Kepler.hpp>
#include <Physics/Optics/Illumination.hpp>

#include <Platform/Input.hpp>
#include <Platform/Platform.hpp>
#include <Platform/Window.hpp>

#include <Renderer/Camera.hpp>
#include <Renderer/Light.hpp>
#include <Renderer/Material.hpp>
#include <Renderer/Mesh.hpp>
#include <Renderer/Renderer.hpp>
#include <Renderer/SceneCameraController.hpp>

#include <UI/CameraOverlay.hpp>
#include <UI/ImGuiLayer.hpp>
#include <UI/Panel.hpp>
#include <UI/StatsOverlay.hpp>

#include <Units/Length.hpp>
#include <Units/Luminosity.hpp>
#include <Units/Time.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace ysq::kepler_solar_system;

/// One trail sample: render position, and the simulation time it was taken
/// at, the same convention Applications::SolarSystem's own main.cpp uses.
struct TrailPoint {
    double time;
    ysq::Vec3f position;
};

/// Every body's absolute position at `simulationTime`, real meters, walked
/// from the Sun (fixed at the origin -- this app has no reflex wobble to
/// give it, unlike a real barycentric N-body frame) down through however
/// many parent links a body has. Memoized per body index rather than
/// assuming the catalog happens to list every parent before its own
/// children: `loadKeplerBodyCatalog` preserves the CSV's own row order and
/// says nothing about what order that has to be in.
class BodyPositions {
public:
    explicit BodyPositions(const std::vector<ysq::applications::KeplerCatalogBody>& bodies,
                           double simulationTime)
        : m_bodies(bodies), m_simulationTime(simulationTime), m_resolved(bodies.size(), false),
          m_positions(bodies.size(), ysq::Vec3::zero()) {}

    const ysq::Vec3& at(std::size_t index) {
        if (!m_resolved[index]) {
            resolve(index);
        }
        return m_positions[index];
    }

private:
    void resolve(std::size_t index) {
        const ysq::applications::KeplerCatalogBody& body = m_bodies[index];
        if (!body.elements.has_value()) {
            m_positions[index] = ysq::Vec3::zero();  // the Sun, fixed
        } else {
            const ysq::Vec3& parentPosition = at(static_cast<std::size_t>(body.parentIndex));
            const ysq::KeplerStateVector local =
                ysq::stateVectorAtTime(*body.elements, body.parentGm, m_simulationTime);
            m_positions[index] = parentPosition + rotate(body.frameRotation, local.position);
        }
        m_resolved[index] = true;
    }

    const std::vector<ysq::applications::KeplerCatalogBody>& m_bodies;
    double m_simulationTime;
    std::vector<bool> m_resolved;
    std::vector<ysq::Vec3> m_positions;
};

}  // namespace

int main() {
    ysq::Logger::init();

    const ysq::UUID runId = ysq::UUID::generate();
    ysq::logging::info("kepler-solar-system run {}", runId.toString());

    const auto platform = ysq::Platform::initialize();
    if (!platform) {
        ysq::logging::error("no windowing system available");
        return EXIT_FAILURE;
    }

    ysq::WindowSettings windowSettings;
    windowSettings.title = "YSQ - Kepler Solar System";
    ysq::WindowError windowError;
    auto window = ysq::Window::create(windowSettings, &windowError);
    if (!window) {
        ysq::logging::error("failed to create window: {}", windowError.message);
        return EXIT_FAILURE;
    }

    std::string rendererError;
    std::optional<ysq::Renderer> rendererOpt = ysq::Renderer::create(&rendererError);
    if (!rendererOpt) {
        ysq::logging::error("failed to create renderer: {}", rendererError);
        return EXIT_FAILURE;
    }
    ysq::Renderer renderer = std::move(*rendererOpt);

    std::string uiError;
    std::optional<ysq::ImGuiLayer> uiOpt = ysq::ImGuiLayer::create(*window, {}, &uiError);
    if (!uiOpt) {
        ysq::logging::error("failed to create UI layer: {}", uiError);
        return EXIT_FAILURE;
    }
    ysq::ImGuiLayer ui = std::move(*uiOpt);

    std::optional<ysq::Mesh> sphereMeshOpt = ysq::Mesh::sphere();
    if (!sphereMeshOpt) {
        ysq::logging::error("failed to build sphere mesh");
        return EXIT_FAILURE;
    }
    ysq::Mesh sphereMesh = std::move(*sphereMeshOpt);

    // Asteroid-belt/Kuiper-belt particles are single dots, not bodies worth
    // real geometric detail: a low-poly sphere costs far less per instance
    // across several thousand of them.
    std::optional<ysq::Mesh> particleMeshOpt = ysq::Mesh::sphere(1.0f, 4, 8);
    if (!particleMeshOpt) {
        ysq::logging::error("failed to build particle mesh");
        return EXIT_FAILURE;
    }
    ysq::Mesh particleMesh = std::move(*particleMeshOpt);

    const ysq::Config config =
        ysq::Config::load("kepler-solar-system.ini").value_or(ysq::Config{});

    // Simulated time per real second: the same control
    // Applications::SolarSystem's own main.cpp uses, except here there is
    // no accuracy/performance cost to raising it -- every body's position
    // is a direct evaluation of its own Kepler orbit at `simulationTime`,
    // not a step count that has to keep up with how far that time jumped.
    // 1 year/sec, or far beyond it, costs the same as 1 second/sec.
    constexpr std::array<const char*, 7> kSimSpeedUnitNames{
        "second", "minute", "hour", "day", "week", "month", "year"};
    constexpr std::array<double, 7> kSecondsPerUnit{
        ysq::units::second.value(), ysq::units::minute.value(),
        ysq::units::hour.value(),   ysq::units::day.value(),
        ysq::units::week.value(),   ysq::units::month.value(),
        ysq::units::year.value()};

    float simSpeedValue =
        static_cast<float>(config.get<double>("physics.simSpeedValue", 1.0));
    const std::string configuredSimSpeedUnit = config.get("physics.simSpeedUnit", "day");
    int simSpeedUnitSelection = 3;  // day: this app's own default, not hour
    for (std::size_t i = 0; i < kSimSpeedUnitNames.size(); ++i) {
        if (configuredSimSpeedUnit == kSimSpeedUnitNames[i]) {
            simSpeedUnitSelection = static_cast<int>(i);
            break;
        }
    }
    double timeScale = static_cast<double>(simSpeedValue) *
                       kSecondsPerUnit[static_cast<std::size_t>(simSpeedUnitSelection)];

    std::string scenarioError;
    const std::optional<Scenario> scenarioOpt = makeScenario(&scenarioError);
    if (!scenarioOpt) {
        ysq::logging::error("failed to load Kepler solar system data: {}", scenarioError);
        return EXIT_FAILURE;
    }
    const Scenario& scenario = *scenarioOpt;

    ysq::logging::info(
        "scenario: {} bodies, {} asteroid-belt particles, {} Kuiper-belt particles, {} rings",
        scenario.bodies.size(), scenario.asteroidBelt.size(), scenario.kuiperBelt.size(),
        scenario.rings.size());

    // The 8 real planets, looked up once by name: a Sun-parented particle
    // (an asteroid or Kuiper belt object) checks its own real eclipse
    // fraction against each of these below, the only bodies big and near
    // enough to ever plausibly occlude the Sun from one. A ring particle
    // does not need this list at all -- its own parent (already carried on
    // the particle itself) is the only occluder that matters for it.
    std::array<std::size_t, 8> planetIndices{};
    {
        constexpr std::array<const char*, 8> kPlanetNames{
            "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"};
        for (std::size_t p = 0; p < kPlanetNames.size(); ++p) {
            const auto it = std::find_if(
                scenario.bodies.begin(), scenario.bodies.end(),
                [&](const ysq::applications::KeplerCatalogBody& body) {
                    return body.name == kPlanetNames[p];
                });
            planetIndices[p] =
                static_cast<std::size_t>(std::distance(scenario.bodies.begin(), it));
        }
    }

    // Real geometric albedo -- the actual fraction of incoming sunlight a
    // body reflects back, not a made-up "diffuse" constant -- for every
    // body this catalog has a real published value for; a small, dark
    // 0.1 (typical for an unmeasured icy/rocky small body, the same
    // status the catalog's own "5 km, 1500 kg/m^3" estimate already has
    // for its many small, uncharacterized moons) for everything else.
    // Precomputed once, not per frame: a real physical property does not
    // change with simulation time.
    const std::vector<float> albedos = [&] {
        constexpr std::array<std::pair<const char*, float>, 24> kRealAlbedos{{
            {"Mercury", 0.142f}, {"Venus", 0.689f}, {"Earth", 0.434f}, {"Mars", 0.170f},
            {"Jupiter", 0.538f}, {"Saturn", 0.499f}, {"Uranus", 0.488f}, {"Neptune", 0.442f},
            {"Moon", 0.12f}, {"Phobos", 0.071f}, {"Deimos", 0.068f},
            {"Io", 0.63f}, {"Europa", 0.67f}, {"Ganymede", 0.43f}, {"Callisto", 0.22f},
            {"Titan", 0.22f},
            {"Ceres", 0.09f}, {"Pluto", 0.52f}, {"Haumea", 0.66f}, {"Makemake", 0.82f},
            {"Eris", 0.96f},
            {"Halley", 0.04f}, {"Encke", 0.05f}, {"Swift-Tuttle", 0.05f},
        }};
        constexpr float kDefaultAlbedo = 0.1f;

        std::vector<float> result;
        result.reserve(scenario.bodies.size());
        for (const ysq::applications::KeplerCatalogBody& body : scenario.bodies) {
            const auto it = std::find_if(
                kRealAlbedos.begin(), kRealAlbedos.end(),
                [&](const std::pair<const char*, float>& entry) {
                    return body.name == entry.first;
                });
            result.push_back(it != kRealAlbedos.end() ? it->second : kDefaultAlbedo);
        }
        return result;
    }();

    ysq::Timer frameTimer;
    double simulationTime = 0.0;

    std::vector<std::deque<TrailPoint>> trails(scenario.bodies.size());
    constexpr int kTargetTrailPoints = 300;
    double lastTrailSampleTime = -std::numeric_limits<double>::infinity();

    ysq::Camera camera;
    ysq::SceneCameraController sceneCamera;
    sceneCamera.orbit.distance = 40.0f;
    sceneCamera.orbit.elevationRadians = 0.5f;
    // Always on, not a user-facing choice: standing inside the POV body's
    // own sphere looking out is the only sensible way to use POV at all.
    sceneCamera.hidePov = true;

    std::vector<ysq::NamedSphere> povNameSeeds;
    povNameSeeds.reserve(scenario.bodies.size());
    for (const ysq::applications::KeplerCatalogBody& body : scenario.bodies) {
        const bool isPlanetOrSun = body.parent.empty() || body.parent == "Sun";
        const std::string displayName =
            isPlanetOrSun ? body.name : std::format("{} ({})", body.name, body.parent);
        povNameSeeds.push_back(ysq::NamedSphere{displayName, ysq::Vec3f::zero(), 0.0f});
    }
    const std::vector<std::string> povOptions = sceneCamera.povOptions(povNameSeeds);
    std::vector<std::string> focusOptionsLive{"Free"};

    std::vector<std::string> cameraModeOptions{"Orbit", "Free fly"};
    int cameraModeSelection = 0;
    int povSelection = 0;
    int focusSelection = 0;
    int previousFocusSelection = -1;

    ysq::Panel controls("Simulation");
    std::vector<std::string> simSpeedUnitOptions(kSimSpeedUnitNames.begin(),
                                                 kSimSpeedUnitNames.end());
    bool paused = false;
    bool showTrails = true;
    bool showOrbitLines = true;
    bool showLabels = true;
    float trailDurationDays = 30.0f;
    bool showReferencePlane = true;
    bool showRings = true;
    bool showAsteroidBelt = true;
    bool showKuiperBelt = true;
    // On: every body, ring and belt particle's own light is compensated
    // back to what it would be at 1 AU (see sunDistanceCompensation
    // below), so distance never makes something genuinely too dim to
    // see. Off: the real, uncompensated inverse-square law -- distant
    // bodies genuinely read dark, the same real dimming a real camera
    // would see without a longer exposure. Real eclipse/shadow geometry
    // applies either way; this toggle is only about distance, not
    // occlusion.
    bool artificialLight = true;
    // Real physics, not a description of how bright things look: the
    // inverse-square law (irradiance = luminosity / (4 pi distance^2))
    // applied to the Sun's own real luminosity and whichever body is
    // currently focused/POV'd own real distance, in real W/m^2. Computed
    // fresh every frame below, once bodyPositions (real meters, not
    // render units) is available.
    std::string sunlightReadoutText = "(select a POV or Focus body)";
    controls.combo("Camera mode", cameraModeOptions, cameraModeSelection);
    controls.combo("POV", povOptions, povSelection);
    controls.comboLive("Focus", focusOptionsLive, focusSelection);
    controls.inputFloat("Simulated time per real second", simSpeedValue, 0.0f, 0.0f,
                        "%.6g");
    controls.combo("Unit", simSpeedUnitOptions, simSpeedUnitSelection);
    controls.checkbox("Paused", paused);
    controls.checkbox("Show trails", showTrails);
    controls.slider("Trail length (days)", trailDurationDays, 1.0f, 4400.0f);
    controls.checkbox("Show orbits", showOrbitLines);
    controls.checkbox("Show labels", showLabels);
    controls.checkbox("Show reference plane", showReferencePlane);
    controls.checkbox("Show rings", showRings);
    controls.checkbox("Show asteroid belt", showAsteroidBelt);
    controls.checkbox("Show Kuiper belt", showKuiperBelt);
    controls.checkbox("Artificial light", artificialLight);
    controls.text("Real sunlight (irradiance)", sunlightReadoutText);

    ysq::StatsOverlay statsOverlay;
    ysq::CameraOverlay cameraOverlay;

    while (!window->shouldClose()) {
        window->input().newFrame();
        ysq::Platform::pollEvents();
        if (ui.wantsMouseCapture()) {
            window->input().suppressMouseThisFrame();
        }

        const double frameSeconds = frameTimer.lap().count();
        if (!paused) {
            simulationTime += frameSeconds * timeScale;
        }

        // Every body's position, evaluated directly at simulationTime: no
        // stepping, no catch-up, the same cost whether simulationTime moved
        // by a second or by a thousand years since last frame.
        BodyPositions bodyPositions(scenario.bodies, simulationTime);

        const ysq::Extent framebufferSize = window->framebufferSize();
        if (framebufferSize.width <= 0 || framebufferSize.height <= 0) {
            window->swapBuffers();
            continue;
        }

        std::vector<ysq::Vec3f> renderPositions(scenario.bodies.size());
        std::vector<float> renderRadii(scenario.bodies.size());
        for (std::size_t i = 0; i < scenario.bodies.size(); ++i) {
            renderPositions[i] = toRenderPosition(bodyPositions.at(i));
            renderRadii[i] = toRenderRadius(scenario.bodies[i].radiusMeters);
        }

        if (showTrails) {
            const double trailDurationSeconds =
                static_cast<double>(trailDurationDays) * 24.0 * 3600.0;
            const double trailSampleIntervalSeconds =
                trailDurationSeconds / static_cast<double>(kTargetTrailPoints);
            if (simulationTime - lastTrailSampleTime >= trailSampleIntervalSeconds) {
                lastTrailSampleTime = simulationTime;
                for (std::size_t i = 0; i < scenario.bodies.size(); ++i) {
                    std::deque<TrailPoint>& trail = trails[i];
                    trail.push_back(TrailPoint{simulationTime, renderPositions[i]});
                    while (!trail.empty() &&
                           trail.front().time < simulationTime - trailDurationSeconds) {
                        trail.pop_front();
                    }
                }
            }
        }

        std::vector<ysq::NamedSphere> objects;
        objects.reserve(scenario.bodies.size());
        for (std::size_t i = 0; i < scenario.bodies.size(); ++i) {
            objects.push_back(
                ysq::NamedSphere{scenario.bodies[i].name, renderPositions[i], renderRadii[i]});
        }

        sceneCamera.povIndex = ysq::SceneCameraController::indexFromPovSelection(povSelection);
        sceneCamera.focusIndex = sceneCamera.indexFromFocusSelection(focusSelection, objects);
        sceneCamera.mode = (cameraModeSelection == 0) ? ysq::CameraMode::Orbit
                                                      : ysq::CameraMode::FreeFly;

        if (focusSelection != previousFocusSelection) {
            previousFocusSelection = focusSelection;
            if (sceneCamera.focusIndex >= 0) {
                sceneCamera.orbit.distance = std::max(
                    objects[static_cast<std::size_t>(sceneCamera.focusIndex)].radius * 4.0f,
                    1.0e-5f);
            }
        }

        sceneCamera.update(camera, objects, window->input(), static_cast<float>(frameSeconds));
        focusOptionsLive = sceneCamera.focusOptions(objects);

        // Real physics: irradiance = luminosity / (4 pi distance^2), the
        // Sun's own real luminosity and this body's own real distance
        // from it (bodyPositions, real meters, not the render-space
        // renderPositions everything else here uses) -- not inferred from
        // how bright the shading looks, an actual number. Focus wins over
        // POV when both are set, since Focus is "the body I'm looking at."
        const int sunlightBodyIndex =
            sceneCamera.focusIndex >= 0 ? sceneCamera.focusIndex : sceneCamera.povIndex;
        if (sunlightBodyIndex >= 0) {
            // Floored at the body's own real radius: distance from the
            // Sun to itself is exactly zero, which the real inverse-square
            // law has no finite answer for (a true point source's own
            // irradiance at distance zero is genuinely infinite). Flooring
            // at the radius instead asks a different, real, and
            // well-defined question -- the flux at the source's own
            // surface -- rather than showing "inf".
            const double distanceMeters = std::max(
                length(bodyPositions.at(static_cast<std::size_t>(sunlightBodyIndex))),
                scenario.bodies[static_cast<std::size_t>(sunlightBodyIndex)].radiusMeters);
            const double irradiance = ysq::units::solarLuminosity.value() /
                                      (4.0 * ysq::kPi<double> * distanceMeters * distanceMeters);
            const double earthIrradiance =
                ysq::units::solarLuminosity.value() /
                (4.0 * ysq::kPi<double> *
                 ysq::units::astronomicalUnit.value() * ysq::units::astronomicalUnit.value());
            sunlightReadoutText =
                std::format("{:.3g} W/m^2 ({:.4g}x Earth's) at {}", irradiance,
                           irradiance / earthIrradiance,
                           scenario.bodies[static_cast<std::size_t>(sunlightBodyIndex)].name);
        } else {
            sunlightReadoutText = "(select a POV or Focus body)";
        }

        float cameraDistance = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < objects.size(); ++i) {
            if (sceneCamera.isHidden(i)) {
                continue;
            }
            cameraDistance = std::min(cameraDistance, length(camera.position - objects[i].position));
        }
        cameraDistance = std::max(cameraDistance, 1.0e-7f);

        // The Kuiper belt (up to 50 AU) always sits well inside Eris's own
        // ~68 AU orbit, so the real bodies already loaded set the far
        // plane correctly without asking the belts themselves.
        float systemExtent = 0.0f;
        for (const ysq::Vec3f& position : renderPositions) {
            systemExtent = std::max(systemExtent, length(position - renderPositions[0]));
        }
        systemExtent += renderRadii[0] + 10.0f;

        camera.perspectiveSettings.nearPlane = std::max(cameraDistance * 0.001f, 1.0e-7f);
        camera.perspectiveSettings.farPlane = std::max(cameraDistance * 1000.0f, systemExtent);

        const float aspect = static_cast<float>(framebufferSize.width) /
                             static_cast<float>(framebufferSize.height);
        renderer.beginFrame(camera, aspect, framebufferSize.width, framebufferSize.height);

        constexpr float kLabelPixelHeight = 40.0f;
        const float verticalHalfFovTangent =
            std::tan(camera.perspectiveSettings.fovYRadians * 0.5f);
        const auto worldSizeForPixels = [&](const ysq::Vec3f& worldPosition, float pixels) {
            const float distance = length(camera.position - worldPosition);
            return distance * verticalHalfFovTangent *
                   (pixels / static_cast<float>(framebufferSize.height));
        };

        ysq::PointLight sunLight;
        sunLight.position = renderPositions[0];
        sunLight.color = scenario.bodies[0].color;
        // Renderer/Light.hpp's own PointLight already applies real
        // inverse-square falloff using each surface's own real distance
        // to the light -- that part is real physics, not tuned per body.
        // The one free parameter a non-HDR renderer cannot avoid is
        // exposure: kSunReferenceExposure names it explicitly (what
        // "correctly exposed sunlight" should read as at 1 AU) instead of
        // leaving intensity a bare, undocumented number; every other
        // body's own brightness still falls out of the real 1/distance^2
        // law from there, not from any further tuning.
        constexpr float kSunReferenceExposure = 3.0f;
        sunLight.intensity = kSunReferenceExposure * kRenderUnitsPerAu * kRenderUnitsPerAu;
        const std::array<ysq::PointLight, 1> pointLights{sunLight};
        renderer.setLights(pointLights, {});

        // Real inverse-square is, correctly, very dim by an outer
        // planet's, moon's or ring's real distance -- about 1% of
        // Earth's own sunlight at Saturn, exactly the reason a real
        // photograph out there needs a far longer exposure than one of
        // Earth to look properly lit, not a shorter one. With
        // artificialLight on, this is that same real compensation
        // computed from each point's own real distance from the Sun,
        // taken all the way to canceling the real 1/distance^2 falloff
        // outright (exponent 2), so anything -- a planet, a moon, a
        // comet, a ring or belt particle -- reads the same brightness
        // it would at 1 AU regardless of how far out it actually is.
        // With it off, this returns 1.0: no compensation, the real,
        // uncompensated law, distant things genuinely read dark. Either
        // way this only scales the light a point receives before
        // falloff is un-done or not; real eclipse/shadow geometry
        // (computed separately, per body or per particle) still darkens
        // whatever it darkens on top of this.
        const auto sunDistanceCompensation = [&](double distanceFromSunMeters) {
            if (!artificialLight) {
                return 1.0;
            }
            return std::pow(distanceFromSunMeters / ysq::units::astronomicalUnit.value(), 2.0);
        };

        for (std::size_t i = 0; i < scenario.bodies.size(); ++i) {
            if (sceneCamera.isHidden(i)) {
                continue;
            }

            const ysq::applications::KeplerCatalogBody& body = scenario.bodies[i];
            const ysq::Vec3f& renderPosition = renderPositions[i];
            const float renderRadius = renderRadii[i];

            ysq::Material material;
            material.albedo = body.color;
            if (i == 0) {
                material.emissive = body.color;
            } else {
                // Real geometric albedo: how much of the light that
                // actually reaches this body it reflects back, not an
                // arbitrary constant. No real per-body gloss/shininess
                // data exists for any of these (a rocky, icy or gaseous
                // surface is overwhelmingly a diffuse scatterer, not a
                // polished one), so specular stays near zero rather than
                // inventing a number -- honest about what is not known,
                // not just what is.
                material.diffuse = albedos[i];
                material.specular = 0.02f;
                material.shininess = 8.0f;
                // Low, not zero: a real night side is not perfectly
                // black (starlight, a moon's own earthshine), but it
                // should read as genuinely dark, not a dim gray version
                // of the day side.
                material.ambient = 0.02f;

                const double distanceFromSunMeters = length(bodyPositions.at(i));
                const double exposureCompensation =
                    sunDistanceCompensation(distanceFromSunMeters);

                // Real eclipse geometry: a moon (parentIndex > 0 means
                // its own parent is a planet, not the Sun) checks itself
                // against its own parent's real shadow -- the actual
                // "size and distance" test, not a binary in/out one.
                // Nothing else here gets a real, common occluder of its
                // own: a planet is not routinely eclipsed by anything in
                // this system.
                double eclipse = 1.0;
                if (body.parentIndex > 0) {
                    eclipse = ysq::discOcclusionFraction(
                        bodyPositions.at(i), bodyPositions.at(0),
                        scenario.bodies[0].radiusMeters,
                        bodyPositions.at(static_cast<std::size_t>(body.parentIndex)),
                        scenario.bodies[static_cast<std::size_t>(body.parentIndex)]
                            .radiusMeters);
                }
                material.lightMultiplier =
                    static_cast<float>(eclipse * exposureCompensation);
            }

            renderer.draw(sphereMesh, material,
                          ysq::Matrix4<float>::translation(renderPosition) *
                              ysq::Matrix4<float>::scale(ysq::Vec3f::splat(renderRadius)));

            if (showLabels) {
                const float labelWorldHeight = worldSizeForPixels(renderPosition, kLabelPixelHeight);
                renderer.debugDraw().text(
                    renderPosition + ysq::Vec3f{0.0f, renderRadius + labelWorldHeight * 0.5f, 0.0f},
                    body.name, labelWorldHeight);
            }

            if (showTrails) {
                const std::deque<TrailPoint>& trail = trails[i];
                for (std::size_t k = 1; k < trail.size(); ++k) {
                    renderer.debugDraw().line(trail[k - 1].position, trail[k].position,
                                              body.color);
                }
            }

            // The whole real orbit, not just a recent trail: sampled
            // directly from this body's own elements at its own current
            // (possibly precessing) argument of periapsis, one closed
            // ellipse around wherever its own parent is right now, not a
            // fading window of recent motion.
            if (showOrbitLines && body.elements.has_value()) {
                constexpr int kOrbitSegments = 128;
                const double currentArgumentOfPeriapsis =
                    body.elements->argumentOfPeriapsis +
                    body.elements->precessionRatePerSecond * simulationTime;
                ysq::OrbitalElements orbitShape{};
                orbitShape.semiMajorAxis = body.elements->semiMajorAxis;
                orbitShape.eccentricity = body.elements->eccentricity;
                orbitShape.inclination = body.elements->inclination;
                orbitShape.longitudeOfAscendingNode = body.elements->longitudeOfAscendingNode;
                orbitShape.argumentOfPeriapsis = currentArgumentOfPeriapsis;

                const ysq::Vec3f& parentRenderPosition =
                    renderPositions[static_cast<std::size_t>(body.parentIndex)];
                ysq::Vec3f previousPoint{};
                for (int s = 0; s <= kOrbitSegments; ++s) {
                    orbitShape.trueAnomaly =
                        ysq::kTau<double> * static_cast<double>(s) / kOrbitSegments;
                    const ysq::KeplerStateVector local =
                        ysq::stateVectorFromElements(orbitShape, body.parentGm);
                    const ysq::Vec3f point =
                        parentRenderPosition +
                        toRenderPosition(rotate(body.frameRotation, local.position));
                    if (s > 0) {
                        renderer.debugDraw().line(previousPoint, point, body.color);
                    }
                    previousPoint = point;
                }
            }
        }

        // The Sun's own emissive sphere above is real to scale -- at true
        // AU distances its rendered radius drops under a pixel from
        // anywhere past the inner planets, and a flat-shaded mesh that
        // small simply does not rasterize. This additive glow is what
        // keeps it a locatable point of light at any distance instead:
        // its on-screen *size* bleeds visibly past the sphere's own edge
        // up close (a real corona does too) and falls back to a fixed
        // pixel footprint once the sphere itself has shrunk past that,
        // and its *brightness* falls off with the camera's own real
        // distance to the Sun -- gently (1/d, not 1/d^2: that steeper,
        // truly physical law is already what the light in the scene
        // itself uses, and would put this glow's own useful range
        // entirely inside 2 AU, reading as static -- a "never really
        // changes" look -- everywhere past it) -- floored so it never
        // actually reaches zero. Both the gentler law and the floor are
        // a stated visibility choice, not real physics: a non-HDR
        // renderer cannot represent the true, enormous dynamic range
        // between "next to the Sun" and "at Eris" any other way.
        if (!sceneCamera.isHidden(0)) {
            const float cameraToSunDistance =
                std::max(length(camera.position - renderPositions[0]), 0.1f);
            constexpr float kSunGlowReferenceIntensity = 20.0f;
            constexpr float kSunGlowMinimumIntensity = 0.02f;
            const float sunGlowIntensity = std::max(
                kSunGlowReferenceIntensity / cameraToSunDistance, kSunGlowMinimumIntensity);
            const float sunGlowRadius =
                std::max(worldSizeForPixels(renderPositions[0], 24.0f), renderRadii[0] * 1.8f);
            renderer.drawGlow(renderPositions[0], sunGlowRadius, scenario.bodies[0].color,
                              sunGlowIntensity);
        }

        // Asteroids, Kuiper belt objects and ring grains are all the same
        // kind of thing to draw: real, independently orbiting particles,
        // each evaluated at simulationTime from its own elements around
        // its own parent (the Sun for the two belts, the ring's own
        // planet for a ring -- Kepler's third law then gives a ring's
        // inner particles a real, visibly faster orbit than its outer
        // ones, the same differential rotation a real ring shows) rather
        // than one piece of static geometry. Always drawn true to scale,
        // floored at roughly a pixel: a real individual particle's true
        // size genuinely is sub-pixel from any orbital distance (a
        // correct physical fact, not a bug), but a rasterizer still
        // cannot draw less than about a pixel, so the floor is what keeps
        // it visible at all -- the same rendering-necessity floor the
        // Sun's own glow already uses for the same reason, not a second,
        // artistic size to choose instead of the real one.
        const ysq::Vec3& sunPositionMeters = bodyPositions.at(0);
        const double sunRadiusMeters = scenario.bodies[0].radiusMeters;

        // Real eclipse geometry per particle: a ring particle checks
        // itself against its own parent planet only (the real, common
        // case -- Saturn's own shadow really does fall across its own
        // rings); a Sun-parented one (an asteroid or Kuiper belt object,
        // parentIndex == 0) checks itself against each of the 8 real
        // planets and keeps the darkest result, since nothing else in
        // this system is ever plausibly between it and the Sun. Both are
        // the same closed-form discOcclusionFraction call, just against a
        // different, small set of candidate occluders. Combined with the
        // same sunDistanceCompensation the body-drawing loop above uses,
        // for the same reason: a ring or belt particle is just as
        // genuinely dim under the real, uncompensated law at a planet's
        // own distance as the planet itself is.
        const auto particleLightMultiplier = [&](const ysq::applications::KeplerParticle& particle,
                                              const ysq::Vec3& positionMeters) {
            double eclipse = 1.0;
            if (particle.parentIndex != 0) {
                const std::size_t parent = static_cast<std::size_t>(particle.parentIndex);
                eclipse = ysq::discOcclusionFraction(positionMeters, sunPositionMeters,
                                              sunRadiusMeters, bodyPositions.at(parent),
                                              scenario.bodies[parent].radiusMeters);
            } else {
                for (std::size_t planetIndex : planetIndices) {
                    eclipse = std::min(
                        eclipse, ysq::discOcclusionFraction(positionMeters, sunPositionMeters,
                                                     sunRadiusMeters,
                                                     bodyPositions.at(planetIndex),
                                                     scenario.bodies[planetIndex].radiusMeters));
                }
            }
            const double distanceFromSunMeters = length(positionMeters - sunPositionMeters);
            return eclipse * sunDistanceCompensation(distanceFromSunMeters);
        };

        // One real lighting scheme for every particle population -- real
        // eclipse geometry plus sunDistanceCompensation, same as every
        // other body -- just with each population's own real ambient
        // and diffuse. `ringAmbient`/`ringDiffuse` only apply to a ring:
        // a real ring is optically thick enough that a particle in
        // another one's shadow still receives real forward-scattered
        // light from its neighbors and reflected planetshine off its
        // own planet, never literally black the way an isolated body's
        // own night side is (0.12 is a floor for that, not a
        // measurement -- no per-ring optical-depth data is modeled
        // here); 0.9 is close to fresh water ice's own real, high
        // geometric albedo, what Saturn's rings are mostly made of. A
        // belt particle gets the same low, not-zero ambient a body gets
        // (isolated in real vacuum, no neighbor to scatter light off
        // of) and a representative real albedo for its own kind of
        // small body (typical, not a measurement, the same honest-
        // estimate status these populations' own sizes already have).
        const auto drawParticles =
            [&](const std::vector<ysq::applications::KeplerParticle>& particles, float ambient,
               float diffuse) {
                if (particles.empty()) {
                    return;
                }
                std::vector<ysq::Matrix4<float>> transforms;
                std::vector<float> lightMultipliers;
                transforms.reserve(particles.size());
                lightMultipliers.reserve(particles.size());
                for (const ysq::applications::KeplerParticle& particle : particles) {
                    const ysq::KeplerStateVector local =
                        ysq::stateVectorAtTime(
                            particle.elements, particle.parentGm, simulationTime);
                    const ysq::Vec3 positionMeters =
                        bodyPositions.at(static_cast<std::size_t>(particle.parentIndex)) +
                        local.position;
                    const ysq::Vec3f position = toRenderPosition(positionMeters);
                    const float scale = std::max(toRenderRadius(particle.realRadiusMeters),
                                                 worldSizeForPixels(position, 1.0f));
                    transforms.push_back(ysq::Matrix4<float>::translation(position) *
                                         ysq::Matrix4<float>::scale(ysq::Vec3f::splat(scale)));
                    lightMultipliers.push_back(static_cast<float>(
                        particleLightMultiplier(particle, positionMeters)));
                }
                particleMesh.setInstanceTransforms(transforms);
                particleMesh.setInstanceLightMultipliers(lightMultipliers);

                ysq::Material material;
                material.albedo = particles.front().color;
                material.ambient = ambient;
                material.diffuse = diffuse;
                renderer.drawInstanced(particleMesh, material);
            };
        if (showAsteroidBelt) {
            // 0.15: a real, typical main-belt geometric albedo (dark
            // carbonaceous C-types dominate the real belt by count, at
            // ~0.03-0.09; brighter S-types run ~0.15-0.25), not a
            // measurement of any one asteroid.
            drawParticles(scenario.asteroidBelt, /*ambient=*/0.02f, /*diffuse=*/0.15f);
        }
        if (showKuiperBelt) {
            // 0.12: a real, typical classical-Kuiper-belt-object albedo
            // for the generic, unnamed population this draws (the few
            // real, bright ones -- Pluto, Eris and the like -- are
            // already their own catalogued bodies elsewhere in this
            // scenario, not part of this generic population).
            drawParticles(scenario.kuiperBelt, /*ambient=*/0.02f, /*diffuse=*/0.12f);
        }
        if (showRings) {
            for (const RingPopulation& ring : scenario.rings) {
                if (ring.particles.empty() ||
                    sceneCamera.isHidden(
                        static_cast<std::size_t>(ring.particles.front().parentIndex))) {
                    continue;
                }
                drawParticles(ring.particles, /*ambient=*/0.12f, /*diffuse=*/0.9f);
            }
        }

        if (showReferencePlane) {
            renderer.debugDraw().grid(60.0f, 30);
        }
        renderer.endFrame();

        statsOverlay.update(static_cast<float>(frameSeconds), renderer.drawCallCount());
        cameraOverlay.update(sceneCamera.statusText(camera, objects));

        ui.beginFrame();
        controls.draw();
        simSpeedValue = std::max(simSpeedValue, 0.0f);
        const std::size_t simSpeedUnitIndex =
            (simSpeedUnitSelection >= 0 &&
             static_cast<std::size_t>(simSpeedUnitSelection) < kSecondsPerUnit.size())
                ? static_cast<std::size_t>(simSpeedUnitSelection)
                : 3;  // day
        timeScale = static_cast<double>(simSpeedValue) * kSecondsPerUnit[simSpeedUnitIndex];
        statsOverlay.draw();
        cameraOverlay.draw();
        ui.endFrame();

        window->swapBuffers();
    }

    return EXIT_SUCCESS;
}
