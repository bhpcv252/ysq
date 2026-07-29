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
#include <Renderer/CameraController.hpp>
#include <Renderer/Light.hpp>
#include <Renderer/Material.hpp>
#include <Renderer/Mesh.hpp>
#include <Renderer/Renderer.hpp>

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
    ysq::log::info("solar-system run {}", runId.toString());

    const auto platform = ysq::Platform::initialize();
    if (!platform) {
        ysq::log::error("no windowing system available");
        return EXIT_FAILURE;
    }

    ysq::WindowSettings windowSettings;
    windowSettings.title = "YSQ - Solar System";
    ysq::WindowError windowError;
    auto window = ysq::Window::create(windowSettings, &windowError);
    if (!window) {
        ysq::log::error("failed to create window: {}", windowError.message);
        return EXIT_FAILURE;
    }

    std::string rendererError;
    std::optional<ysq::Renderer> rendererOpt = ysq::Renderer::create(&rendererError);
    if (!rendererOpt) {
        ysq::log::error("failed to create renderer: {}", rendererError);
        return EXIT_FAILURE;
    }
    ysq::Renderer renderer = std::move(*rendererOpt);

    std::string uiError;
    std::optional<ysq::ImGuiLayer> uiOpt = ysq::ImGuiLayer::create(*window, {}, &uiError);
    if (!uiOpt) {
        ysq::log::error("failed to create UI layer: {}", uiError);
        return EXIT_FAILURE;
    }
    ysq::ImGuiLayer ui = std::move(*uiOpt);

    std::optional<ysq::Mesh> sphereMeshOpt = ysq::Mesh::sphere();
    if (!sphereMeshOpt) {
        ysq::log::error("failed to build sphere mesh");
        return EXIT_FAILURE;
    }
    ysq::Mesh sphereMesh = std::move(*sphereMeshOpt);

    const std::unique_ptr<ysq::ComputeBackend> computeBackend =
        ysq::selectComputeBackend();
    ysq::log::info("compute backend: {}", ysq::toString(computeBackend->kind()));

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

    ysq::log::info("scenario: Sun + {} planets", scenario.planets.size());
    for (const Planet& planet : scenario.planets) {
        ysq::log::info("  {}: {:.3f} AU", planet.name,
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
                ysq::log::warn("energy drift exceeded 1% at t={:.0f}s",
                               event.simulationTime);
            }
        });

    std::vector<std::deque<TrailPoint>> trails(scenario.planets.size());

    ysq::Camera camera;
    ysq::OrbitCameraController orbitController;
    orbitController.distance = 40.0f;
    orbitController.elevationRadians = 0.5f;

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
    controls.slider("Time scale (Ms/s)", timeScaleMillionsPerSecond, 0.0f, 100.0f);
    controls.checkbox("Paused", paused);
    controls.checkbox("Show trails", showTrails);
    controls.slider("Trail length (days)", trailDurationDays, 1.0f, 4400.0f);
    controls.checkbox("Show labels", showLabels);

    ysq::StatsOverlay statsOverlay;

    while (!window->shouldClose()) {
        window->input().newFrame();
        ysq::Platform::pollEvents();

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

        orbitController.update(camera, window->input());

        const float aspect = static_cast<float>(framebufferSize.width) /
                             static_cast<float>(framebufferSize.height);

        renderer.beginFrame(camera, aspect, framebufferSize.width,
                            framebufferSize.height);

        const ysq::Vec3f sunRenderPosition = toRenderPosition(bodies[0].position);
        ysq::PointLight sunLight;
        sunLight.position = sunRenderPosition;
        sunLight.color = scenario.sunColor;
        sunLight.intensity = 3.0f;
        sunLight.radius = 60.0f;
        const std::array<ysq::PointLight, 1> pointLights{sunLight};
        renderer.setLights(pointLights, {});

        ysq::Material sunMaterial;
        sunMaterial.albedo = scenario.sunColor;
        sunMaterial.emissive = scenario.sunColor;
        renderer.draw(
            sphereMesh, sunMaterial,
            ysq::Matrix4<float>::translation(sunRenderPosition) *
                ysq::Matrix4<float>::scale(ysq::Vec3f::splat(scenario.sunRenderRadius)));
        if (showLabels) {
            renderer.debugDraw().text(
                sunRenderPosition +
                    ysq::Vec3f{0.0f, scenario.sunRenderRadius + 0.6f, 0.0f},
                "Sun");
        }

        for (std::size_t i = 0; i < scenario.planets.size(); ++i) {
            const Planet& planet = scenario.planets[i];
            const ysq::Vec3f renderPosition = toRenderPosition(bodies[i + 1].position);

            ysq::Material material;
            material.albedo = planet.color;
            material.ambient = 0.15f;
            material.diffuse = 0.85f;
            material.specular = 0.2f;
            material.shininess = 16.0f;

            renderer.draw(
                sphereMesh, material,
                ysq::Matrix4<float>::translation(renderPosition) *
                    ysq::Matrix4<float>::scale(ysq::Vec3f::splat(planet.renderRadius)));
            if (showLabels) {
                renderer.debugDraw().text(
                    renderPosition + ysq::Vec3f{0.0f, planet.renderRadius + 0.4f, 0.0f},
                    planet.name, 0.4f);
            }

            if (showTrails) {
                const std::deque<TrailPoint>& trail = trails[i];
                for (std::size_t k = 1; k < trail.size(); ++k) {
                    renderer.debugDraw().line(trail[k - 1].position, trail[k].position,
                                              planet.color);
                }
            }
        }

        renderer.debugDraw().grid(60.0f, 30);
        renderer.endFrame();

        statsOverlay.update(static_cast<float>(frameSeconds), renderer.drawCallCount());

        ui.beginFrame();
        controls.draw();
        clock.setTimeScale(static_cast<double>(timeScaleMillionsPerSecond) * 1.0e6);
        if (paused) {
            clock.pause();
        } else {
            clock.resume();
        }
        statsOverlay.draw();
        energyPlot.draw();
        momentumPlot.draw();
        ui.endFrame();

        window->swapBuffers();
    }

    return EXIT_SUCCESS;
}
