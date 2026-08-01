#include <Applications/SolarSystem/Scenario.hpp>

#include <Compute/ComputeBackend.hpp>
#include <Core/Clock.hpp>
#include <Core/Config.hpp>
#include <Core/Event.hpp>
#include <Core/Logger.hpp>
#include <Core/Timer.hpp>
#include <Core/UUID.hpp>

#include <Math/Integrators/Symplectic.hpp>
#include <Math/Matrix4.hpp>
#include <Math/ODE.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <Physics/Body.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>

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
#include <UI/PlotPanel.hpp>
#include <UI/StatsOverlay.hpp>

#include <Units/Length.hpp>
#include <Units/Mass.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ysq::solar_system;

/// Diagnostics for one completed fixed step, decoupled from who consumes
/// them: the plots and the drift check both subscribe rather than being
/// called inline from the integration loop.
struct StepCompleted {
    double simulationTime;
    double totalEnergy;
    double totalMomentum;
};

/// One trail sample: render position, and the simulation time it was taken
/// at. The time is what lets a trail be pruned by how much simulated time
/// it spans rather than by how many points happen to be in it; see the
/// "Trail length (days)" control and the trail-pruning loop below.
struct TrailPoint {
    double time;
    ysq::Vec3f position;
};

}  // namespace

int main() {
    ysq::Logger::init();

    const ysq::UUID runId = ysq::UUID::generate();
    ysq::logging::info("solar-system run {}", runId.toString());

    const auto platform = ysq::Platform::initialize();
    if (!platform) {
        ysq::logging::error("no windowing system available");
        return EXIT_FAILURE;
    }

    ysq::WindowSettings windowSettings;
    windowSettings.title = "YSQ - Solar System";
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

    // Optional: tunables read from a config file next to the working
    // directory if present, with defaults otherwise. Config::load never
    // throws and Config::get falls back cleanly, so an absent file is not a
    // special case here.
    const ysq::Config config =
        ysq::Config::load("solar-system.ini").value_or(ysq::Config{});
    const double timeScale = config.get<double>("physics.timeScale", 2.0e7);
    const ysq::Length softening{config.get<double>("physics.softeningMeters", 1.0e8)};

    const Scenario scenario = makeScenario();
    std::vector<ysq::Body> bodies = scenario.allBodies();

    ysq::logging::info("scenario: Sun + {} planets", scenario.planets.size());
    for (const Planet& planet : scenario.planets) {
        ysq::logging::info("  {}: {:.3f} AU", planet.name,
                           length(planet.body.position.value()) /
                               ysq::units::astronomicalUnit.value());
    }

    // Mercury's own orbital period: the fastest-moving body, so the fixed
    // step -- sized relative to it -- stays accurate for every body,
    // independent of timeScale.
    const double gmSun = ysq::constants::G.value() * ysq::units::solarMass.value();
    const auto orbitalPeriod = [gmSun](double radius) {
        return 2.0 * ysq::kPi<double> * std::sqrt(radius * radius * radius / gmSun);
    };
    const double mercuryPeriod =
        orbitalPeriod(length(scenario.planets.front().body.position.value()));
    const double fixedStep = mercuryPeriod / 2000.0;

    ysq::Clock::Settings clockSettings;
    clockSettings.fixedStep = fixedStep;
    clockSettings.timeScale = timeScale;
    clockSettings.maxStepsPerAdvance = 2000;
    ysq::Clock clock{clockSettings};
    ysq::Timer frameTimer;

    ysq::VelocityVerletStepper<ysq::NBodyState> stepper;

    ysq::EventBus bus;
    ysq::TimeSeriesPlot energyPlot("Energy drift");
    ysq::TimeSeriesPlot momentumPlot("Momentum drift");
    double initialEnergy = 0.0;
    bool haveInitialEnergy = false;
    const ysq::Subscription stepSubscription =
        bus.subscribe<StepCompleted>([&](const StepCompleted& event) {
            energyPlot.addSample(event.simulationTime, event.totalEnergy);
            momentumPlot.addSample(event.simulationTime, event.totalMomentum);
            if (!haveInitialEnergy) {
                initialEnergy = event.totalEnergy;
                haveInitialEnergy = true;
            } else if (std::abs(event.totalEnergy - initialEnergy) >
                       std::abs(initialEnergy) * 0.01) {
                ysq::logging::warn("energy drift exceeded 1% at t={:.0f}s",
                                   event.simulationTime);
            }
        });

    std::vector<std::deque<TrailPoint>> trails(scenario.planets.size());

    ysq::Camera camera;
    ysq::SceneCameraController sceneCamera;
    sceneCamera.orbit.distance = 40.0f;
    sceneCamera.orbit.elevationRadians = 0.5f;

    // POV's option list never changes at runtime (the body list is fixed
    // for this scenario), so it's built once here rather than every frame;
    // positions/radii are irrelevant for naming it, so a throwaway seed
    // list is enough. Focus's list does change at runtime -- it must
    // exclude whichever body is currently POV -- so it's rebuilt from the
    // real `objects` list every frame below and bound live via
    // Panel::comboLive.
    std::vector<ysq::NamedSphere> povNameSeeds;
    povNameSeeds.reserve(1 + scenario.planets.size());
    povNameSeeds.push_back(ysq::NamedSphere{"Sun", ysq::Vec3f::zero(), 0.0f});
    for (const Planet& planet : scenario.planets) {
        povNameSeeds.push_back(ysq::NamedSphere{planet.name, ysq::Vec3f::zero(), 0.0f});
    }
    const std::vector<std::string> povOptions = sceneCamera.povOptions(povNameSeeds);
    std::vector<std::string> focusOptionsLive{"Free"};

    std::vector<std::string> cameraModeOptions{"Orbit", "Free fly"};
    int cameraModeSelection = 0;
    int povSelection = 0;    // "Free"
    int focusSelection = 0;  // "Free"
    int previousFocusSelection = -1;

    ysq::Panel controls("Simulation");
    float timeScaleMillionsPerSecond = static_cast<float>(timeScale / 1.0e6);
    bool paused = false;
    bool showTrails = true;
    bool showLabels = true;
    // Trails span this many simulated days for every planet, not one orbit
    // each: a shared window is what actually answers "how far did each
    // planet move in the same N days", and keeps the trail an honest,
    // modest arc instead of, say, Jupiter eventually filling in a
    // near-complete, camera-distorting loop of its own. 30 days is enough
    // to show Mercury a third of an orbit and the outer planets a small,
    // physically honest stub; the slider covers up to a bit past Jupiter's
    // own ~12-year period for anyone who wants to see a slow orbit close.
    float trailDurationDays = 30.0f;
    controls.combo("Camera mode", cameraModeOptions, cameraModeSelection);
    controls.combo("POV", povOptions, povSelection);
    controls.comboLive("Focus", focusOptionsLive, focusSelection);
    controls.checkbox("Hide POV body", sceneCamera.hidePov);
    controls.slider("Time scale (Ms/s)", timeScaleMillionsPerSecond, 0.0f, 100.0f);
    controls.checkbox("Paused", paused);
    controls.checkbox("Show trails", showTrails);
    controls.slider("Trail length (days)", trailDurationDays, 1.0f, 4400.0f);
    controls.checkbox("Show labels", showLabels);

    ysq::StatsOverlay statsOverlay;
    ysq::CameraOverlay cameraOverlay;

    while (!window->shouldClose()) {
        window->input().newFrame();
        ysq::Platform::pollEvents();
        if (ui.wantsMouseCapture()) {
            // A panel widget already claimed this click/drag; without this,
            // dragging a slider would also drag the 3D camera underneath it.
            window->input().suppressMouseThisFrame();
        }

        const double frameSeconds = frameTimer.lap().count();
        clock.advance(frameSeconds);

        while (clock.consumeStep()) {
            const ysq::NewtonianField field(bodies, softening);
            const ysq::PhaseState<ysq::NBodyState> state{ysq::positionsOf(bodies),
                                                         ysq::velocitiesOf(bodies)};
            ysq::PhaseState<ysq::NBodyState> next;
            stepper.step(field, clock.simulationTime(), state, clock.fixedStep(), next);
            ysq::applyState(bodies, next.position, next.velocity);

            std::vector<float> kineticEnergies;
            kineticEnergies.reserve(bodies.size());
            for (const ysq::Body& body : bodies) {
                const double speed = length(body.velocity().value());
                kineticEnergies.push_back(
                    static_cast<float>(0.5 * body.mass.value() * speed * speed));
            }
            const double totalKinetic =
                static_cast<double>(computeBackend->sum(kineticEnergies));
            const double totalPotential =
                ysq::newtonianPotentialEnergy(bodies, softening).value();

            ysq::Vec3 totalMomentum = ysq::Vec3::zero();
            for (const ysq::Body& body : bodies) {
                totalMomentum += body.momentum.value();
            }

            bus.publish(StepCompleted{clock.simulationTime(),
                                      totalKinetic + totalPotential,
                                      length(totalMomentum)});

            if (showTrails) {
                const double trailDurationSeconds =
                    static_cast<double>(trailDurationDays) * 24.0 * 3600.0;
                for (std::size_t i = 0; i < scenario.planets.size(); ++i) {
                    std::deque<TrailPoint>& trail = trails[i];
                    trail.push_back(TrailPoint{clock.simulationTime(),
                                               toRenderPosition(bodies[i + 1].position)});
                    while (!trail.empty() &&
                           trail.front().time <
                               clock.simulationTime() - trailDurationSeconds) {
                        trail.pop_front();
                    }
                }
            }
        }

        const ysq::Extent framebufferSize = window->framebufferSize();
        if (framebufferSize.width <= 0 || framebufferSize.height <= 0) {
            // Minimized, or a transient zero-size resize: nothing to draw
            // into, and dividing by a zero height below would hand the
            // projection matrix a NaN aspect ratio. ImGui's own
            // NewFrame() also expects a non-empty display size, so skip
            // rendering and UI entirely for this frame rather than only
            // guarding the 3D view.
            window->swapBuffers();
            continue;
        }

        const ysq::Vec3f sunRenderPosition = toRenderPosition(bodies[0].position);
        std::vector<ysq::Vec3f> planetRenderPositions(scenario.planets.size());
        for (std::size_t i = 0; i < scenario.planets.size(); ++i) {
            planetRenderPositions[i] = toRenderPosition(bodies[i + 1].position);
        }

        std::vector<ysq::NamedSphere> objects;
        objects.reserve(1 + scenario.planets.size());
        objects.push_back(
            ysq::NamedSphere{"Sun", sunRenderPosition, scenario.sunRenderRadius});
        for (std::size_t i = 0; i < scenario.planets.size(); ++i) {
            objects.push_back(ysq::NamedSphere{scenario.planets[i].name,
                                               planetRenderPositions[i],
                                               scenario.planets[i].renderRadius});
        }

        sceneCamera.povIndex =
            ysq::SceneCameraController::indexFromPovSelection(povSelection);
        sceneCamera.focusIndex =
            sceneCamera.indexFromFocusSelection(focusSelection, objects);
        sceneCamera.mode = (cameraModeSelection == 0) ? ysq::CameraMode::Orbit
                                                      : ysq::CameraMode::FreeFly;

        // A fresh Focus selection re-frames the orbit distance to the new
        // target's own scale: SceneCameraController's auto-track only
        // moves orbit.target, never orbit.distance, so without this
        // whatever zoom level was set for the previous target would carry
        // over unchanged (e.g. Mercury-close, suddenly applied to the Sun).
        if (focusSelection != previousFocusSelection) {
            previousFocusSelection = focusSelection;
            if (sceneCamera.focusIndex >= 0) {
                sceneCamera.orbit.distance =
                    objects[static_cast<std::size_t>(sceneCamera.focusIndex)].radius *
                    4.0f;
            }
        }

        sceneCamera.update(camera, objects, window->input(),
                           static_cast<float>(frameSeconds));
        focusOptionsLive = sceneCamera.focusOptions(objects);

        const float aspect = static_cast<float>(framebufferSize.width) /
                             static_cast<float>(framebufferSize.height);

        renderer.beginFrame(camera, aspect, framebufferSize.width,
                            framebufferSize.height);

        ysq::PointLight sunLight;
        sunLight.position = sunRenderPosition;
        sunLight.color = scenario.sunColor;
        sunLight.intensity = 3.0f;
        sunLight.radius = 60.0f;
        const std::array<ysq::PointLight, 1> pointLights{sunLight};
        renderer.setLights(pointLights, {});

        if (!sceneCamera.isHidden(0)) {
            ysq::Material sunMaterial;
            sunMaterial.albedo = scenario.sunColor;
            sunMaterial.emissive = scenario.sunColor;
            renderer.draw(sphereMesh, sunMaterial,
                          ysq::Matrix4<float>::translation(sunRenderPosition) *
                              ysq::Matrix4<float>::scale(
                                  ysq::Vec3f::splat(scenario.sunRenderRadius)));
            if (showLabels) {
                renderer.debugDraw().text(
                    sunRenderPosition +
                        ysq::Vec3f{0.0f, scenario.sunRenderRadius + 0.6f, 0.0f},
                    "Sun");
            }
        }

        for (std::size_t i = 0; i < scenario.planets.size(); ++i) {
            const Planet& planet = scenario.planets[i];
            const ysq::Vec3f& renderPosition = planetRenderPositions[i];
            // objects[0] is the Sun, objects[i + 1] this planet -- the same
            // offset the objects list above was built with.
            const bool hidden = sceneCamera.isHidden(i + 1);

            if (!hidden) {
                ysq::Material material;
                material.albedo = planet.color;
                material.ambient = 0.15f;
                material.diffuse = 0.85f;
                material.specular = 0.2f;
                material.shininess = 16.0f;

                renderer.draw(sphereMesh, material,
                              ysq::Matrix4<float>::translation(renderPosition) *
                                  ysq::Matrix4<float>::scale(
                                      ysq::Vec3f::splat(planet.renderRadius)));
                if (showLabels) {
                    renderer.debugDraw().text(
                        renderPosition +
                            ysq::Vec3f{0.0f, planet.renderRadius + 0.4f, 0.0f},
                        planet.name, 0.4f);
                }

                if (showTrails) {
                    const std::deque<TrailPoint>& trail = trails[i];
                    for (std::size_t k = 1; k < trail.size(); ++k) {
                        renderer.debugDraw().line(trail[k - 1].position,
                                                  trail[k].position, planet.color);
                    }
                }
            }
        }

        renderer.debugDraw().grid(60.0f, 30);
        renderer.endFrame();

        statsOverlay.update(static_cast<float>(frameSeconds), renderer.drawCallCount());
        cameraOverlay.update(sceneCamera.statusText(camera, objects));

        ui.beginFrame();
        controls.draw();
        clock.setTimeScale(static_cast<double>(timeScaleMillionsPerSecond) * 1.0e6);
        if (paused) {
            clock.pause();
        } else {
            clock.resume();
        }
        statsOverlay.draw();
        cameraOverlay.draw();
        energyPlot.draw();
        momentumPlot.draw();
        ui.endFrame();

        window->swapBuffers();
    }

    return EXIT_SUCCESS;
}
