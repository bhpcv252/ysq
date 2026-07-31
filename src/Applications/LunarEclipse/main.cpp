#include <Applications/LunarEclipse/Scenario.hpp>

#include <Compute/ComputeBackend.hpp>
#include <Core/Clock.hpp>
#include <Core/Config.hpp>
#include <Core/Logger.hpp>
#include <Core/Timer.hpp>
#include <Core/UUID.hpp>

#include <Math/Integrators/Symplectic.hpp>
#include <Math/Matrix4.hpp>
#include <Math/Quaternion.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Physics/Mechanics/RigidBody.hpp>
#include <Physics/Optics/Illumination.hpp>

#include <Platform/Input.hpp>
#include <Platform/Platform.hpp>
#include <Platform/Window.hpp>

#include <Renderer/Camera.hpp>
#include <Renderer/CameraController.hpp>
#include <Renderer/Light.hpp>
#include <Renderer/Material.hpp>
#include <Renderer/Mesh.hpp>
#include <Renderer/Renderer.hpp>

#include <UI/ImGuiLayer.hpp>
#include <UI/Panel.hpp>
#include <UI/StatsOverlay.hpp>

#include <Units/Length.hpp>
#include <Units/Mass.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ysq::lunar_eclipse;

/// Everything Optics/Illumination.hpp needs about the current geometry, and
/// a display-only eclipse phase label derived from its result: labels are
/// for the UI panel alone, never fed back into physics.
struct IlluminationState {
    ysq::IlluminationResult result;
    std::string phaseLabel = "None";
};

std::string phaseLabelFor(const ysq::IlluminationResult& result) {
    if (0.99 <= result.geometricVisibility) {
        return "None";
    }
    if (0.6 <= result.geometricVisibility) {
        return "Penumbral";
    }
    if (0.01 <= result.geometricVisibility) {
        return "Partial";
    }
    const double totalTransmission =
        result.transmission.x + result.transmission.y + result.transmission.z;
    return (1.0e-6 < totalTransmission) ? "Total (refracted light visible)"
                                        : "Total (dark)";
}

/// The Moon's position relative to Earth's shadow axis, in real metres:
/// pure geometry, no atmosphere ray-tracing, cheap enough to recompute every
/// frame (unlike ysq::illuminate()) so a live "how close" readout and a fast
/// search loop both have something inexpensive to check against. The same
/// straight-line occlusion geometry Optics/Illumination.hpp resolves more
/// carefully (with refraction) once something is actually inside the
/// penumbra; this only asks whether it is, not what color reaches it there.
struct ShadowGeometry {
    double axialDistance = 0.0;  // Moon's depth behind Earth, along the anti-solar axis
    double perpendicularDistance = 0.0;  // Moon's offset from that axis
    double penumbraRadius = 0.0;
    double umbraRadius = 0.0;  // clamped to zero past the umbra's own apex
};

ShadowGeometry computeShadowGeometry(const ysq::Body& sun, const ysq::Body& earth,
                                     const ysq::Body& moon) {
    const ysq::Vec3 earthToSun = sun.position.value() - earth.position.value();
    const ysq::Vec3 antiSolar = -normalized(earthToSun);
    const double sunEarthDistance = length(earthToSun);

    const ysq::Vec3 earthToMoon = moon.position.value() - earth.position.value();
    const double axialDistance = dot(earthToMoon, antiSolar);
    const ysq::Vec3 perpendicular = earthToMoon - antiSolar * axialDistance;

    const double penumbraHalfAngle =
        std::atan((sun.radius.value() + earth.radius.value()) / sunEarthDistance);
    const double umbraHalfAngle =
        std::atan((sun.radius.value() - earth.radius.value()) / sunEarthDistance);

    ShadowGeometry geometry;
    geometry.axialDistance = axialDistance;
    geometry.perpendicularDistance = length(perpendicular);
    geometry.penumbraRadius =
        earth.radius.value() + axialDistance * std::tan(penumbraHalfAngle);
    geometry.umbraRadius =
        std::max(0.0, earth.radius.value() - axialDistance * std::tan(umbraHalfAngle));
    return geometry;
}

/// One fixed physics step: translational n-body plus Earth's own rotation,
/// factored out so the normal frame loop and the fast "jump to next
/// eclipse" search below run the exact same physics, not two
/// implementations that could drift apart.
void stepBodies(std::vector<ysq::Body>& bodies, ysq::Length softening, double dt,
                std::size_t earthIndex,
                ysq::VelocityVerletStepper<ysq::NBodyState>& stepper) {
    const ysq::NewtonianField field(bodies, softening);
    const ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                                 ysq::velocitiesOf(bodies)};
    ysq::PhaseState<ysq::NBodyState> next;
    stepper.step(field, 0.0, state, dt, next);
    ysq::applyState(bodies, next.position, next.velocity);

    std::vector<ysq::Body> perturbersOfEarth;
    perturbersOfEarth.reserve(bodies.size() - 1);
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (i != earthIndex) {
            perturbersOfEarth.push_back(bodies[i]);
        }
    }
    ysq::stepRigidBody(bodies[earthIndex], perturbersOfEarth, dt);
}

}  // namespace

int main() {
    ysq::Logger::init();

    const ysq::UUID runId = ysq::UUID::generate();
    ysq::logging::info("lunar-eclipse run {}", runId.toString());

    const auto platform = ysq::Platform::initialize();
    if (!platform) {
        ysq::logging::error("no windowing system available");
        return EXIT_FAILURE;
    }

    ysq::WindowSettings windowSettings;
    windowSettings.title = "YSQ - Lunar Eclipse";
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

    const std::unique_ptr<ysq::ComputeBackend> computeBackend =
        ysq::selectComputeBackend();
    ysq::logging::info("compute backend: {}", ysq::toString(computeBackend->kind()));

    const ysq::Config config =
        ysq::Config::load("lunar-eclipse.ini").value_or(ysq::Config{});
    const double timeScale = config.get<double>("physics.timeScale", 2.0e6);
    const ysq::Length softening{config.get<double>("physics.softeningMeters", 1.0e6)};

    const Scenario scenario = makeScenario();
    std::vector<ysq::Body> bodies = scenario.allBodies();
    // Sun, Earth, Moon, Jupiter, matching Scenario::allBodies()'s own order.
    constexpr std::size_t kSun = 0;
    constexpr std::size_t kEarth = 1;
    constexpr std::size_t kMoon = 2;

    // The Moon's own orbital period, the fastest-moving body that actually
    // matters to this scenario's dynamics, sizes the fixed step the same
    // way SolarSystem sizes its own against Mercury's.
    const double gmEarth = ysq::constants::G.value() * bodies[kEarth].mass.value();
    const double moonOrbitRadius =
        length(bodies[kMoon].position.value() - bodies[kEarth].position.value());
    const double moonPeriod =
        ysq::kTau<double> *
        std::sqrt(moonOrbitRadius * moonOrbitRadius * moonOrbitRadius / gmEarth);
    const double fixedStep = moonPeriod / 2000.0;

    ysq::Clock::Settings clockSettings;
    clockSettings.fixedStep = fixedStep;
    clockSettings.timeScale = timeScale;
    clockSettings.maxStepsPerAdvance = 2000;
    ysq::Clock clock{clockSettings};
    ysq::Timer frameTimer;
    ysq::Timer illuminationTimer;

    ysq::VelocityVerletStepper<ysq::NBodyState> stepper;

    // True to scale (Scenario::toRenderRadius uses the exact factor
    // toRenderPosition does), so the dynamic range a body's own radius and
    // the whole system's extent span is enormous: a fixed near/far plane
    // cannot cover both being able to get close to the Moon and seeing the
    // Sun-Earth distance. Both are recomputed every frame from the orbit
    // controller's current distance instead; see the render loop below.
    ysq::Camera camera;
    ysq::OrbitCameraController orbitController;
    orbitController.minDistance = 1.0e-6f;

    // Which body the orbit camera is centered on. Index matches kSun /
    // kEarth / kMoon directly. target tracks the focused body's position
    // every frame (so it stays centered as the body orbits); distance
    // resets to a close-up on the body only when the selection actually
    // changes, not every frame, so scrolling to zoom still works normally
    // in between.
    std::vector<std::string> focusOptions{"Sun", "Earth", "Moon"};
    int focusIndex = 1;  // Earth: the most immediately interesting body to start on
    int previousFocusIndex = -1;

    // Locked alignment views, alongside the default free-orbit camera: each
    // fixes look-direction to the live Earth-Moon line, so which hemisphere
    // of the Moon is visible (and whether it is the sunlit, reddening one
    // during an eclipse, or the dark far side) depends on which of these is
    // picked, not on manual navigation. See src/Applications/README.md's
    // sibling docs for why this composition lives here, not in the engine:
    // it is a camera convenience, not a physical law.
    std::vector<std::string> cameraModeOptions{"Free orbit", "Behind Earth",
                                               "Behind Moon", "Between Earth and Moon"};
    int cameraModeIndex = 0;

    ysq::Panel controls("Simulation");
    float timeScaleThousandsPerSecond = static_cast<float>(timeScale / 1.0e3);
    bool paused = false;
    bool showShadowCones = true;
    bool showLabels = true;
    controls.combo("Focus", focusOptions, focusIndex);
    controls.combo("Camera mode", cameraModeOptions, cameraModeIndex);
    controls.slider("Time scale (Ks/s)", timeScaleThousandsPerSecond, 0.0f, 5000.0f);
    controls.checkbox("Paused", paused);
    controls.checkbox("Show shadow cones", showShadowCones);
    controls.checkbox("Show labels", showLabels);

    ysq::Panel readout("Eclipse");
    std::string phaseText = "None";
    std::string transmissionText = "1.00, 1.00, 1.00";
    std::string distanceText;
    std::string shadowMissText;
    readout.text("Phase", phaseText);
    readout.text("Moon light (R,G,B)", transmissionText);
    readout.text("Distances (AU)", distanceText);
    readout.text("Shadow miss (km)", shadowMissText);
    bool jumpRequested = false;
    readout.button("Jump to next eclipse", [&]() { jumpRequested = true; });

    ysq::StatsOverlay statsOverlay;

    IlluminationState illumination;
    // illuminate() walks a null geodesic through Earth's atmosphere, which
    // is not free: recomputed a few times a second of *real* time, not
    // every rendered frame, is indistinguishable to the eye for something
    // that changes on a timescale of minutes, and stays inside a real-time
    // frame budget. Precision here trades against that budget through
    // sourceSamples and stepBudget alike; a one-off, higher-fidelity answer
    // is what tests/integration and the unit tests already ask for
    // directly.
    constexpr double kIlluminationUpdateInterval = 0.5;
    constexpr int kIlluminationSourceSamples = 4;
    constexpr int kIlluminationStepBudget = 600;
    const std::array<double, 3> rgbWavelengths{630.0e-9, 532.0e-9, 465.0e-9};

    const auto updateIllumination = [&]() {
        ysq::RefractingOccluder earthOccluder;
        earthOccluder.center = bodies[kEarth].position.value();
        earthOccluder.opaqueRadius = bodies[kEarth].radius.value();
        earthOccluder.medium = scenario.earthAtmosphere;
        earthOccluder.surfaceNumberDensity = scenario.earthSurfaceNumberDensity;
        earthOccluder.scatteringScaleHeight = scenario.earthScatteringScaleHeight;

        illumination.result = ysq::illuminate(
            bodies[kSun].position.value(), bodies[kSun].radius.value(), {},
            &earthOccluder, bodies[kMoon].position.value(), rgbWavelengths,
            kIlluminationSourceSamples, kIlluminationStepBudget);
        illumination.phaseLabel = phaseLabelFor(illumination.result);
    };

    // A search window generous enough to cross at least one real eclipse
    // season (they recur roughly every 173 days) even from an unlucky
    // starting point in the Moon's 18.6-year nodal precession cycle.
    const int jumpMaxSteps = static_cast<int>(5.0 * 365.25 * 86400.0 / fixedStep);

    while (!window->shouldClose()) {
        window->input().newFrame();
        ysq::Platform::pollEvents();

        const double frameSeconds = frameTimer.lap().count();
        clock.advance(frameSeconds);

        while (clock.consumeStep()) {
            stepBodies(bodies, softening, clock.fixedStep(), kEarth, stepper);
        }

        if (jumpRequested) {
            jumpRequested = false;
            for (int i = 0; i < jumpMaxSteps; ++i) {
                clock.stepOnce();
                stepBodies(bodies, softening, clock.fixedStep(), kEarth, stepper);
                const ShadowGeometry geometry =
                    computeShadowGeometry(bodies[kSun], bodies[kEarth], bodies[kMoon]);
                if (geometry.perpendicularDistance <= geometry.penumbraRadius) {
                    break;
                }
            }
            updateIllumination();
            illuminationTimer.reset();
        }

        if (kIlluminationUpdateInterval <= illuminationTimer.elapsed().count()) {
            illuminationTimer.reset();
            updateIllumination();
        }

        const ShadowGeometry liveShadowGeometry =
            computeShadowGeometry(bodies[kSun], bodies[kEarth], bodies[kMoon]);

        const ysq::Extent framebufferSize = window->framebufferSize();
        if (framebufferSize.width <= 0 || framebufferSize.height <= 0) {
            window->swapBuffers();
            continue;
        }

        const std::array<ysq::Vec3f, 3> renderPositions{
            toRenderPosition(bodies[kSun].position),
            toRenderPosition(bodies[kEarth].position),
            toRenderPosition(bodies[kMoon].position)};
        const std::array<float, 3> renderRadii{toRenderRadius(bodies[kSun].radius),
                                               toRenderRadius(bodies[kEarth].radius),
                                               toRenderRadius(bodies[kMoon].radius)};

        // cameraDistance stands in for "how zoomed in is the camera right
        // now" regardless of mode, feeding the near/far planes and label
        // scaling below the same way in either case.
        float cameraDistance = orbitController.distance;

        if (cameraModeIndex == 0) {
            orbitController.target =
                renderPositions[static_cast<std::size_t>(focusIndex)];
            if (focusIndex != previousFocusIndex) {
                orbitController.distance = std::max(
                    renderRadii[static_cast<std::size_t>(focusIndex)] * 4.0f, 1.0e-5f);
                previousFocusIndex = focusIndex;
            }
            orbitController.update(camera, window->input());
            cameraDistance = orbitController.distance;
        } else {
            // The three locked alignment views: look-direction always
            // follows the live Earth-Moon line, so which of the Moon's
            // hemispheres is visible (the sunlit, reddening one, or the
            // dark far side) is a property of which mode is picked, not of
            // manual navigation. "Behind Moon" specifically shows the dark
            // far side (Earth blocking the Sun, viewed from beyond the
            // Moon), the other two show the lit near side.
            ysq::Vec3f earthToMoonDirection =
                renderPositions[2] - renderPositions[1];  // Moon minus Earth
            const float earthMoonSeparation = length(earthToMoonDirection);
            if (earthMoonSeparation > 0.0f) {
                earthToMoonDirection = earthToMoonDirection / earthMoonSeparation;
            }

            if (cameraModeIndex == 1) {  // Behind Earth, looking at the Moon
                cameraDistance = renderRadii[1] * 6.0f;
                camera.position =
                    renderPositions[1] - earthToMoonDirection * cameraDistance;
                camera.target = renderPositions[2];
            } else if (cameraModeIndex == 2) {  // Behind the Moon, looking back at Earth
                cameraDistance = renderRadii[2] * 6.0f;
                camera.position =
                    renderPositions[2] + earthToMoonDirection * cameraDistance;
                camera.target = renderPositions[1];
            } else {  // Between Earth and Moon, looking back at Earth
                cameraDistance = renderRadii[2] * 6.0f;
                camera.position =
                    renderPositions[2] - earthToMoonDirection * cameraDistance;
                camera.target = renderPositions[1];
            }
            // A degenerate view matrix if the Earth-Moon line is ever
            // exactly parallel to the usual up axis; falls back to a
            // different hint in that one case rather than producing NaNs.
            camera.up = (std::abs(dot(earthToMoonDirection, ysq::Vec3f::unitY())) < 0.99f)
                            ? ysq::Vec3f::unitY()
                            : ysq::Vec3f::unitX();
        }

        // True to scale means an enormous dynamic range: the near plane
        // has to stay well inside whatever body is currently focused (its
        // true radius can be smaller than a fixed near plane would ever
        // allow), and the far plane has to reach at least the whole
        // system's own extent, not just wherever the camera happens to be
        // zoomed, or the Sun itself would clip out of view while examining
        // the Moon up close.
        const float systemExtent =
            length(renderPositions[0] - renderPositions[1]) + renderRadii[0] + 10.0f;
        camera.perspectiveSettings.nearPlane = std::max(cameraDistance * 0.001f, 1.0e-7f);
        camera.perspectiveSettings.farPlane =
            std::max(cameraDistance * 1000.0f, systemExtent);

        const float aspect = static_cast<float>(framebufferSize.width) /
                             static_cast<float>(framebufferSize.height);

        renderer.beginFrame(camera, aspect, framebufferSize.width,
                            framebufferSize.height);

        const ysq::Vec3f& sunRenderPosition = renderPositions[0];
        const ysq::Vec3f& earthRenderPosition = renderPositions[1];
        const ysq::Vec3f& moonRenderPosition = renderPositions[2];
        const float sunRenderRadius = renderRadii[0];
        const float earthRenderRadius = renderRadii[1];
        const float moonRenderRadius = renderRadii[2];

        // A locator pip for whichever body has shrunk below a few pixels
        // at the current zoom: DebugDraw's points are a fixed pixel size
        // regardless of distance, so this is purely a "here it is" aid,
        // never a substitute for the body's own true-scale sphere, which
        // still draws at whatever size that actually is.
        const auto apparentPixelDiameter = [&](const ysq::Vec3f& renderPosition,
                                               float renderRadius) {
            const float distanceToCamera = length(camera.position - renderPosition);
            if (distanceToCamera <= renderRadius) {
                return 1.0e6f;  // camera is inside the body
            }
            const float angularDiameter =
                2.0f * std::atan(renderRadius / distanceToCamera);
            const float pixelsPerRadian = static_cast<float>(framebufferSize.height) /
                                          camera.perspectiveSettings.fovYRadians;
            return angularDiameter * pixelsPerRadian;
        };
        constexpr float kLocatorThresholdPixels = 3.0f;
        for (std::size_t i = 0; i < renderPositions.size(); ++i) {
            if (apparentPixelDiameter(renderPositions[i], renderRadii[i]) <
                kLocatorThresholdPixels) {
                renderer.debugDraw().point(renderPositions[i],
                                           ysq::Vec3f{1.0f, 1.0f, 1.0f});
            }
        }

        // Sun and Earth: full, ordinary sunlight. Earth's own day/night
        // terminator is nothing but this light's diffuse falloff; there is
        // no separate "shading" step for it.
        ysq::PointLight sunLight;
        sunLight.position = sunRenderPosition;
        sunLight.color = scenario.sunColor;
        sunLight.intensity = 3.0f;
        sunLight.radius = 200.0f;
        const std::array<ysq::PointLight, 1> fullSunlight{sunLight};
        renderer.setLights(fullSunlight, {});

        ysq::Material sunMaterial;
        sunMaterial.albedo = scenario.sunColor;
        sunMaterial.emissive = scenario.sunColor;
        renderer.draw(sphereMesh, sunMaterial,
                      ysq::Matrix4<float>::translation(sunRenderPosition) *
                          ysq::Matrix4<float>::scale(ysq::Vec3f::splat(sunRenderRadius)));

        ysq::Material earthMaterial;
        earthMaterial.albedo = scenario.earthColor;
        earthMaterial.ambient = 0.03f;  // low, so the night side actually reads as dark
        earthMaterial.diffuse = 0.9f;
        earthMaterial.specular = 0.15f;
        renderer.draw(
            sphereMesh, earthMaterial,
            ysq::Matrix4<float>::translation(earthRenderPosition) *
                ysq::Matrix4<float>::scale(ysq::Vec3f::splat(earthRenderRadius)));

        // The Moon: lit by whatever illuminate() actually found reaches it,
        // full sunlight outside any shadow, dim and reddened light bent
        // through Earth's atmosphere during totality, by whatever
        // proportion in between during a partial phase. Still positioned
        // at the Sun, a reasonable approximation for shading purposes: the
        // real totality glow arrives from all around Earth's rim, not from
        // one direction, which the boosted ambient term below stands in
        // for rather than modeling directly.
        ysq::PointLight moonLight;
        moonLight.position = sunRenderPosition;
        moonLight.color =
            ysq::Vec3f{static_cast<float>(illumination.result.transmission.x),
                       static_cast<float>(illumination.result.transmission.y),
                       static_cast<float>(illumination.result.transmission.z)};
        moonLight.intensity = 3.0f;
        moonLight.radius = 200.0f;
        const std::array<ysq::PointLight, 1> moonIllumination{moonLight};
        renderer.setLights(moonIllumination, {});

        ysq::Material moonMaterial;
        moonMaterial.albedo = scenario.moonColor;
        moonMaterial.ambient = 0.08f;
        moonMaterial.diffuse = 0.85f;
        moonMaterial.specular = 0.05f;
        renderer.draw(
            sphereMesh, moonMaterial,
            ysq::Matrix4<float>::translation(moonRenderPosition) *
                ysq::Matrix4<float>::scale(ysq::Vec3f::splat(moonRenderRadius)));

        if (showLabels) {
            // True-scale bodies range from sub-pixel to filling the view
            // depending on zoom, so a label's size and offset are tied to
            // the current orbit distance (roughly "screen-space constant
            // size") rather than to the body's own radius: a fixed offset
            // like the old cosmetic scale used would either vanish inside
            // the Sun or dwarf the Moon depending on which body is in view.
            const float labelHeight = cameraDistance * 0.06f;
            const float labelOffset = cameraDistance * 0.1f;
            renderer.debugDraw().text(
                sunRenderPosition + ysq::Vec3f{0.0f, sunRenderRadius + labelOffset, 0.0f},
                "Sun", labelHeight);
            renderer.debugDraw().text(
                earthRenderPosition +
                    ysq::Vec3f{0.0f, earthRenderRadius + labelOffset, 0.0f},
                "Earth", labelHeight);
            renderer.debugDraw().text(
                moonRenderPosition +
                    ysq::Vec3f{0.0f, moonRenderRadius + labelOffset, 0.0f},
                "Moon", labelHeight);
        }

        if (showShadowCones) {
            // The umbra's own axis: straight out from Earth, directly away
            // from the Sun, the same raw occlusion geometry
            // Optics/Illumination.hpp tests against, drawn rather than
            // computed a second way: a wireframe cylinder standing in for
            // the umbra's true (near-cylindrical, over these distances)
            // cross-section, one Earth radius wide, out to the Moon's own
            // distance.
            const ysq::Vec3 earthToSun =
                bodies[kSun].position.value() - bodies[kEarth].position.value();
            const ysq::Vec3 antiSolar = -normalized(earthToSun);
            const double shadowLength =
                length(bodies[kMoon].position.value() - bodies[kEarth].position.value()) *
                1.3;
            const float coneRadius = earthRenderRadius;

            ysq::Vec3 arbitrary =
                (std::abs(antiSolar.x) < 0.9) ? ysq::Vec3::unitX() : ysq::Vec3::unitY();
            const ysq::Vec3 side = normalized(cross(arbitrary, antiSolar));
            const ysq::Vec3 up = cross(antiSolar, side);

            const ysq::Vec3f farEnd = toRenderPosition(
                ysq::Length3{bodies[kEarth].position.value() + antiSolar * shadowLength});

            constexpr int kConeSegments = 16;
            for (int i = 0; i < kConeSegments; ++i) {
                const double angle =
                    ysq::kTau<double> * static_cast<double>(i) / kConeSegments;
                const ysq::Vec3 rim3 = side * (coneRadius * std::cos(angle)) +
                                       up * (coneRadius * std::sin(angle));
                const ysq::Vec3f nearPoint =
                    earthRenderPosition + ysq::Vec3f{static_cast<float>(rim3.x),
                                                     static_cast<float>(rim3.y),
                                                     static_cast<float>(rim3.z)};
                const ysq::Vec3f farPoint =
                    farEnd + ysq::Vec3f{static_cast<float>(rim3.x),
                                        static_cast<float>(rim3.y),
                                        static_cast<float>(rim3.z)};
                renderer.debugDraw().line(nearPoint, farPoint,
                                          ysq::Vec3f{0.5f, 0.1f, 0.1f});
            }
        }

        renderer.endFrame();

        statsOverlay.update(static_cast<float>(frameSeconds), renderer.drawCallCount());

        ui.beginFrame();
        controls.draw();
        clock.setTimeScale(static_cast<double>(timeScaleThousandsPerSecond) * 1.0e3);
        if (paused) {
            clock.pause();
        } else {
            clock.resume();
        }

        phaseText = illumination.phaseLabel;
        transmissionText =
            std::to_string(illumination.result.transmission.x).substr(0, 4) + ", " +
            std::to_string(illumination.result.transmission.y).substr(0, 4) + ", " +
            std::to_string(illumination.result.transmission.z).substr(0, 4);
        {
            const double auScale = 1.0 / ysq::units::astronomicalUnit.value();
            const double earthSunAu =
                length(bodies[kEarth].position.value() - bodies[kSun].position.value()) *
                auScale;
            const double earthMoonAu =
                length(bodies[kMoon].position.value() - bodies[kEarth].position.value()) *
                auScale;
            distanceText = "Sun-Earth " + std::to_string(earthSunAu).substr(0, 5) +
                           ", Earth-Moon " + std::to_string(earthMoonAu).substr(0, 6);
        }
        {
            // Negative means the Moon is already inside the penumbra; this
            // updates every frame, unlike Phase above, so it is the number
            // to actually watch shrink while looking for an eclipse rather
            // than waiting for the throttled label to change.
            const double missKm = (liveShadowGeometry.perpendicularDistance -
                                   liveShadowGeometry.penumbraRadius) /
                                  1000.0;
            const double penumbraRadiusKm = liveShadowGeometry.penumbraRadius / 1000.0;
            shadowMissText = std::to_string(missKm).substr(0, 9) + " (penumbra radius " +
                             std::to_string(penumbraRadiusKm).substr(0, 6) + " km)";
        }
        readout.draw();

        statsOverlay.draw();
        ui.endFrame();

        window->swapBuffers();
    }

    return EXIT_SUCCESS;
}
