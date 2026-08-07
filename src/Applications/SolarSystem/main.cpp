#include <Applications/SolarSystem/Scenario.hpp>

#include <Compute/ComputeBackend.hpp>
#include <Core/Config.hpp>
#include <Core/Event.hpp>
#include <Core/Logger.hpp>
#include <Core/Timer.hpp>
#include <Core/UUID.hpp>

#include <Math/Matrix4.hpp>
#include <Math/Scalar.hpp>
#include <Math/Vector3.hpp>

#include <Physics/Body.hpp>
#include <Physics/Gravity/Kepler.hpp>
#include <Physics/Gravity/Newtonian.hpp>
#include <Physics/Gravity/PostNewtonian.hpp>
#include <Physics/Mechanics/Dynamics.hpp>
#include <Physics/Mechanics/Hermite.hpp>

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
#include <Units/Time.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
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

    // Simulated time per real second, entered as a value and a unit rather
    // than a single raw number: "100 days/sec" says what it means, where a
    // bare seconds-per-second figure does not. kSecondsPerUnit is
    // kSimSpeedUnitNames' own index paired with how many seconds that unit
    // is, from Units/Time.hpp -- the same conversion the "Simulated time per
    // real second" control below multiplies by every frame.
    constexpr std::array<const char*, 7> kSimSpeedUnitNames{
        "second", "minute", "hour", "day", "week", "month", "year"};
    constexpr std::array<double, 7> kSecondsPerUnit{
        ysq::units::second.value(), ysq::units::minute.value(),
        ysq::units::hour.value(),   ysq::units::day.value(),
        ysq::units::week.value(),   ysq::units::month.value(),
        ysq::units::year.value()};

    float simSpeedValue =
        static_cast<float>(config.get<double>("physics.simSpeedValue", 1.0));
    const std::string configuredSimSpeedUnit = config.get("physics.simSpeedUnit", "hour");
    int simSpeedUnitSelection = 2;  // hour: the default, and the config fallback
    for (std::size_t i = 0; i < kSimSpeedUnitNames.size(); ++i) {
        if (configuredSimSpeedUnit == kSimSpeedUnitNames[i]) {
            simSpeedUnitSelection = static_cast<int>(i);
            break;
        }
    }
    double timeScale = static_cast<double>(simSpeedValue) *
                       kSecondsPerUnit[static_cast<std::size_t>(simSpeedUnitSelection)];
    // Well below the closest real separation in the catalog (an inner ring
    // moon a little over 1e8 m from its planet), so softening never
    // dominates a real body's own gravity.
    const ysq::Length softening{config.get<double>("physics.softeningMeters", 1.0e6)};
    // The Aarseth criterion's own accuracy knob (dt = sqrt(eta * |a| /
    // |jerk|)): smaller shrinks every body's own step and buys more
    // accuracy at more cost. See Physics/Mechanics/Hermite.hpp.
    const double eta = config.get<double>("physics.eta", 0.01);

    std::string scenarioError;
    const std::optional<Scenario> scenarioOpt = makeScenario(&scenarioError);
    if (!scenarioOpt) {
        ysq::logging::error("failed to load solar system data: {}", scenarioError);
        return EXIT_FAILURE;
    }
    const Scenario& scenario = *scenarioOpt;
    std::vector<ysq::Body> bodies = scenario.allBodies();

    ysq::logging::info("scenario: {} bodies (Sun, 8 planets, and every real moon)",
                       bodies.size());

    // Name -> index: used below both for the fastest-body step-size search
    // and to look up a moon's own parent when qualifying its POV/Focus
    // list entry.
    std::unordered_map<std::string, std::size_t> indexByName;
    for (std::size_t i = 0; i < scenario.bodies.size(); ++i) {
        indexByName[scenario.bodies[i].name] = i;
    }

    // Each body's own dominant local primary, for
    // RelativisticNBodySystem's 1PN correction: a planet's own primary is
    // the Sun, a moon's is its own planet, never the Sun directly. -1 (the
    // Sun itself, index 0) means no correction -- nothing more dominant
    // sits near it.
    std::vector<int> primaryIndex(scenario.bodies.size(), -1);
    for (std::size_t i = 1; i < scenario.bodies.size(); ++i) {
        primaryIndex[i] = static_cast<int>(indexByName.at(scenario.bodies[i].parent));
    }

    // The slowest body's own orbital period sets a sane ceiling on any
    // body's own step (`baseInterval` below): IndividualTimestepScheduler
    // gives every body its own step from its own current acceleration and
    // jerk (the Aarseth criterion), so the fastest body does not set the
    // pace for every other one. T = 2 pi sqrt(a^3 / gm), with `a` and `gm`
    // each body's own distance from, and its own parent's real mass -- not
    // the Sun's, for a moon.
    double slowestPeriod = 0.0;
    for (std::size_t i = 1; i < bodies.size(); ++i) {
        const std::size_t parentIndex = indexByName.at(scenario.bodies[i].parent);
        const double gm = ysq::constants::G.value() * bodies[parentIndex].mass.value();
        const double a =
            length(bodies[i].position.value() - bodies[parentIndex].position.value());
        const double period = ysq::keplerOrbitalPeriod(gm, a);
        slowestPeriod = std::max(slowestPeriod, period);
    }
    const double baseInterval = slowestPeriod / 2000.0;

    ysq::Timer frameTimer;
    double simulationTime = 0.0;

    // On by default: the real physics (visible perihelion precession).
    // IndividualTimestepScheduler is not symplectic either way (see
    // Physics/Mechanics/Hermite.hpp's own header comment), so turning this
    // off costs only the accuracy of the 1PN term itself, not an
    // integrator downgrade. See src/Physics/README.md's gravity ladder
    // section for what the term itself still trades off.
    bool useRelativity = true;

    // Both jerk-providing force laws are built once, not rebuilt per
    // update or per frame: their precomputed values depend only on mass,
    // which never changes here. advanceTo() is called with whichever one
    // the checkbox currently selects.
    const ysq::NewtonianJerkField newtonianJerkField(bodies, softening);
    const ysq::RelativisticNBodyJerkSystem relativisticJerkSystem(bodies, primaryIndex,
                                                                  softening);

    const ysq::NBodyState initialPositions = ysq::positionsOf(bodies);
    const ysq::NBodyState initialVelocities = ysq::velocitiesOf(bodies);
    ysq::NBodyState initialAccelerations(bodies.size());
    ysq::NBodyState initialJerks(bodies.size());
    for (std::size_t i = 0; i < bodies.size(); ++i) {
        const auto [acceleration, jerk] =
            useRelativity ? relativisticJerkSystem(i, initialPositions, initialVelocities)
                         : newtonianJerkField(i, initialPositions, initialVelocities);
        initialAccelerations[i] = acceleration;
        initialJerks[i] = jerk;
    }
    ysq::IndividualTimestepScheduler scheduler(initialPositions, initialVelocities,
                                               initialAccelerations, initialJerks, 0.0, eta,
                                               baseInterval);

    ysq::EventBus bus;
    ysq::TimeSeriesPlot energyPlot("Energy drift");
    ysq::TimeSeriesPlot momentumPlot("Momentum drift");
    double initialEnergy = 0.0;
    bool haveInitialEnergy = false;
    // Edge-triggered, not level-triggered: without `wasDrifting`, a run
    // that stays past 1% drift for any length of time logs this every
    // single frame for as long as it remains true, which is what flooded
    // the log the one time this actually fired. One line when it starts,
    // one when it recovers, is what a human reading the log actually
    // wants.
    bool wasDrifting = false;
    const ysq::Subscription stepSubscription =
        bus.subscribe<StepCompleted>([&](const StepCompleted& event) {
            energyPlot.addSample(event.simulationTime, event.totalEnergy);
            momentumPlot.addSample(event.simulationTime, event.totalMomentum);
            if (!haveInitialEnergy) {
                initialEnergy = event.totalEnergy;
                haveInitialEnergy = true;
                return;
            }
            const bool isDrifting = std::abs(event.totalEnergy - initialEnergy) >
                                    std::abs(initialEnergy) * 0.01;
            if (isDrifting && !wasDrifting) {
                ysq::logging::warn("energy drift exceeded 1% at t={:.0f}s",
                                   event.simulationTime);
            } else if (!isDrifting && wasDrifting) {
                ysq::logging::info("energy drift back under 1% at t={:.0f}s",
                                   event.simulationTime);
            }
            wasDrifting = isDrifting;
        });

    std::vector<std::deque<TrailPoint>> trails(bodies.size());
    // How many points a trail keeps, regardless of its current duration;
    // see the sampling-cadence comment where this is used, in the physics
    // loop below.
    constexpr int kTargetTrailPoints = 300;
    double lastTrailSampleTime = -std::numeric_limits<double>::infinity();

    ysq::Camera camera;
    ysq::SceneCameraController sceneCamera;
    sceneCamera.orbit.distance = 40.0f;
    sceneCamera.orbit.elevationRadians = 0.5f;

    // POV/Focus names: unqualified for the Sun and each planet, but
    // "Name (Parent)" for a moon -- a flat list of every real body stays
    // searchable and scannable this way without new list-grouping UI
    // infrastructure. The on-screen label text (drawn separately below)
    // stays each body's own short, unqualified name; only the dropdown
    // entries carry the parent qualifier.
    std::vector<ysq::NamedSphere> povNameSeeds;
    povNameSeeds.reserve(scenario.bodies.size());
    for (const ysq::applications::CatalogBody& catalogBody : scenario.bodies) {
        const bool isPlanetOrSun = catalogBody.parent.empty() || catalogBody.parent == "Sun";
        const std::string displayName =
            isPlanetOrSun ? catalogBody.name
                         : std::format("{} ({})", catalogBody.name, catalogBody.parent);
        povNameSeeds.push_back(ysq::NamedSphere{displayName, ysq::Vec3f::zero(), 0.0f});
    }
    const std::vector<std::string> povOptions = sceneCamera.povOptions(povNameSeeds);
    std::vector<std::string> focusOptionsLive{"Free"};

    std::vector<std::string> cameraModeOptions{"Orbit", "Free fly"};
    int cameraModeSelection = 0;
    int povSelection = 0;    // "Free"
    int focusSelection = 0;  // "Free"
    int previousFocusSelection = -1;

    ysq::Panel controls("Simulation");
    std::vector<std::string> simSpeedUnitOptions(kSimSpeedUnitNames.begin(),
                                                 kSimSpeedUnitNames.end());
    bool paused = false;
    bool showTrails = true;
    bool showLabels = true;
    // Trails span this many simulated days for every body, not one orbit
    // each: a shared window is what actually answers "how far did each
    // body move in the same N days", and keeps a trail an honest, modest
    // arc instead of, say, Jupiter eventually filling in a near-complete,
    // camera-distorting loop of its own. 30 days is enough to show
    // Mercury a third of an orbit and the outer planets a small,
    // physically honest stub; the slider covers up to a bit past
    // Jupiter's own ~12-year period for anyone who wants to see a slow
    // orbit close.
    float trailDurationDays = 30.0f;
    bool showReferencePlane = true;
    controls.combo("Camera mode", cameraModeOptions, cameraModeSelection);
    controls.combo("POV", povOptions, povSelection);
    controls.comboLive("Focus", focusOptionsLive, focusSelection);
    controls.checkbox("Hide POV body", sceneCamera.hidePov);
    controls.inputFloat("Simulated time per real second", simSpeedValue, 0.0f, 0.0f,
                        "%.6g");
    controls.combo("Unit", simSpeedUnitOptions, simSpeedUnitSelection);
    controls.checkbox("Paused", paused);
    controls.checkbox("Relativistic corrections", useRelativity);
    // How many single-body updates the scheduler spent catching up this
    // frame -- a rough proxy for how much headroom this machine has at the
    // current requested speed.
    std::string updatesThisFrameText;
    controls.text("Individual updates this frame", updatesThisFrameText);
    controls.checkbox("Show trails", showTrails);
    controls.slider("Trail length (days)", trailDurationDays, 1.0f, 4400.0f);
    controls.checkbox("Show labels", showLabels);
    controls.checkbox("Show reference plane", showReferencePlane);

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
        if (!paused) {
            simulationTime += frameSeconds * timeScale;
        }

        // No cap on how many single-body updates one frame may spend
        // catching up: this app always drives the scheduler all the way to
        // simulationTime rather than letting it fall behind under a budget.
        constexpr int kUnboundedUpdates = std::numeric_limits<int>::max();
        if (useRelativity) {
            scheduler.advanceTo(relativisticJerkSystem, simulationTime, kUnboundedUpdates);
        } else {
            scheduler.advanceTo(newtonianJerkField, simulationTime, kUnboundedUpdates);
        }

        // Honors how far the scheduler actually got rather than asking
        // predictedState() below to extrapolate a shortfall -- relevant
        // whenever targetTime lands between two bodies' own scheduled
        // updates, not because of any update budget. A no-op whenever the
        // scheduler landed exactly on simulationTime, since currentTime()
        // already equals it then.
        simulationTime = scheduler.currentTime();
        updatesThisFrameText = std::format("{}", scheduler.lastAdvanceUpdateCount());

        // Every body's predicted "now" (see
        // IndividualTimestepScheduler::predictedState()'s own comment on
        // why this, and not each body's own last true update, is what
        // rendering, trails and the diagnostics below all have to share),
        // written back into `bodies` so everything downstream keeps
        // reading it exactly as before.
        {
            ysq::NBodyState predictedPositions(bodies.size());
            ysq::NBodyState predictedVelocities(bodies.size());
            for (std::size_t i = 0; i < bodies.size(); ++i) {
                const auto [position, velocity] =
                    scheduler.predictedState(i, simulationTime);
                predictedPositions[i] = position;
                predictedVelocities[i] = velocity;
            }
            ysq::applyState(bodies, predictedPositions, predictedVelocities);
        }

        std::vector<float> kineticEnergies;
        kineticEnergies.reserve(bodies.size());
        for (const ysq::Body& body : bodies) {
            const double speed = length(body.velocity().value());
            kineticEnergies.push_back(
                static_cast<float>(0.5 * body.mass.value() * speed * speed));
        }
        const double totalKinetic = static_cast<double>(computeBackend->sum(kineticEnergies));
        const double totalPotential =
            ysq::newtonianPotentialEnergy(bodies, softening).value();

        ysq::Vec3 totalMomentum = ysq::Vec3::zero();
        for (const ysq::Body& body : bodies) {
            totalMomentum += body.momentum.value();
        }

        bus.publish(
            StepCompleted{simulationTime, totalKinetic + totalPotential, length(totalMomentum)});

        if (showTrails) {
            const double trailDurationSeconds =
                static_cast<double>(trailDurationDays) * 24.0 * 3600.0;
            // A trail point is sampled on a fixed real-world (simulated)
            // cadence, not once per frame: a trail is a visual aid, not a
            // physical quantity -- sampling it every frame for a 30-day
            // (or longer) window would accumulate far more points across
            // 175 bodies than is ever visible. kTargetTrailPoints keeps
            // every trail's own resolution proportional to its own current
            // duration instead: a short trail and a long one both look
            // equally smooth.
            const double trailSampleIntervalSeconds =
                trailDurationSeconds / static_cast<double>(kTargetTrailPoints);
            if (simulationTime - lastTrailSampleTime >= trailSampleIntervalSeconds) {
                lastTrailSampleTime = simulationTime;
                for (std::size_t i = 0; i < bodies.size(); ++i) {
                    std::deque<TrailPoint>& trail = trails[i];
                    trail.push_back(
                        TrailPoint{simulationTime, toRenderPosition(bodies[i].position)});
                    while (!trail.empty() &&
                           trail.front().time < simulationTime - trailDurationSeconds) {
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

        std::vector<ysq::Vec3f> renderPositions(bodies.size());
        std::vector<float> renderRadii(bodies.size());
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            renderPositions[i] = toRenderPosition(bodies[i].position);
            renderRadii[i] = toRenderRadius(bodies[i].radius);
        }

        std::vector<ysq::NamedSphere> objects;
        objects.reserve(scenario.bodies.size());
        for (std::size_t i = 0; i < scenario.bodies.size(); ++i) {
            objects.push_back(
                ysq::NamedSphere{scenario.bodies[i].name, renderPositions[i], renderRadii[i]});
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
        // over unchanged (e.g. a small moon-close zoom, suddenly applied
        // to the Sun).
        if (focusSelection != previousFocusSelection) {
            previousFocusSelection = focusSelection;
            if (sceneCamera.focusIndex >= 0) {
                sceneCamera.orbit.distance = std::max(
                    objects[static_cast<std::size_t>(sceneCamera.focusIndex)].radius * 4.0f,
                    1.0e-5f);
            }
        }

        sceneCamera.update(camera, objects, window->input(),
                           static_cast<float>(frameSeconds));
        focusOptionsLive = sceneCamera.focusOptions(objects);

        // True to scale means an enormous dynamic range: distance to the
        // nearest *visible* body sets how close the near plane can safely
        // sit (a small moon's true radius can be far smaller than a fixed
        // near plane would ever allow), and the far plane has to reach
        // the whole system's own real extent -- Neptune's orbit, not just
        // wherever the camera happens to be zoomed -- or the outer
        // planets would clip out of view while examining an inner moon up
        // close. Both follow the same pattern LunarEclipse's own
        // main.cpp already uses, generalized from two bodies to however
        // many the scenario actually has.
        float cameraDistance = std::numeric_limits<float>::max();
        for (std::size_t i = 0; i < objects.size(); ++i) {
            if (sceneCamera.isHidden(i)) {
                continue;
            }
            cameraDistance =
                std::min(cameraDistance, length(camera.position - objects[i].position));
        }
        cameraDistance = std::max(cameraDistance, 1.0e-7f);

        float systemExtent = 0.0f;
        for (const ysq::Vec3f& position : renderPositions) {
            systemExtent = std::max(systemExtent, length(position - renderPositions[0]));
        }
        systemExtent += renderRadii[0] + 10.0f;

        camera.perspectiveSettings.nearPlane = std::max(cameraDistance * 0.001f, 1.0e-7f);
        camera.perspectiveSettings.farPlane =
            std::max(cameraDistance * 1000.0f, systemExtent);

        const float aspect = static_cast<float>(framebufferSize.width) /
                             static_cast<float>(framebufferSize.height);

        renderer.beginFrame(camera, aspect, framebufferSize.width,
                            framebufferSize.height);

        // A label's own world-space size has to track the camera's
        // distance to whatever it is labeling, or it is either invisible
        // or gigantic: true-to-scale positions and radii span an enormous
        // dynamic range (the whole system's ~150-unit extent down to a
        // small moon's real, tiny render radius up close), and a single
        // fixed world size cannot read well at more than one of those at
        // once. `kLabelPixelHeight` is the one constant that is actually
        // meaningful across that whole range: how tall the label reads on
        // screen, in pixels, wherever the camera happens to be.
        constexpr float kLabelPixelHeight = 14.0f;
        const float verticalHalfFovTangent =
            std::tan(camera.perspectiveSettings.fovYRadians * 0.5f);
        const auto labelWorldHeightAt = [&](const ysq::Vec3f& worldPosition) {
            const float distance = length(camera.position - worldPosition);
            return 2.0f * distance * verticalHalfFovTangent *
                   (kLabelPixelHeight / static_cast<float>(framebufferSize.height));
        };

        ysq::PointLight sunLight;
        sunLight.position = renderPositions[0];
        sunLight.color = scenario.bodies[0].color;
        sunLight.intensity = 3.0f;
        const std::array<ysq::PointLight, 1> pointLights{sunLight};
        renderer.setLights(pointLights, {});

        for (std::size_t i = 0; i < bodies.size(); ++i) {
            if (sceneCamera.isHidden(i)) {
                continue;
            }

            const ysq::applications::CatalogBody& catalogBody = scenario.bodies[i];
            const ysq::Vec3f& renderPosition = renderPositions[i];
            const float renderRadius = renderRadii[i];

            ysq::Material material;
            material.albedo = catalogBody.color;
            if (i == 0) {
                // The Sun alone is a light source, not a lit surface.
                material.emissive = catalogBody.color;
            } else {
                material.ambient = 0.15f;
                material.diffuse = 0.85f;
                material.specular = 0.2f;
                material.shininess = 16.0f;
            }

            renderer.draw(sphereMesh, material,
                          ysq::Matrix4<float>::translation(renderPosition) *
                              ysq::Matrix4<float>::scale(ysq::Vec3f::splat(renderRadius)));

            if (showLabels) {
                const float labelWorldHeight = labelWorldHeightAt(renderPosition);
                renderer.debugDraw().text(
                    renderPosition +
                        ysq::Vec3f{0.0f, renderRadius + labelWorldHeight * 0.5f, 0.0f},
                    catalogBody.name, labelWorldHeight);
            }

            if (showTrails) {
                const std::deque<TrailPoint>& trail = trails[i];
                for (std::size_t k = 1; k < trail.size(); ++k) {
                    renderer.debugDraw().line(trail[k - 1].position, trail[k].position,
                                              catalogBody.color);
                }
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
        // combo() itself does not clamp an out-of-range selection (see
        // Panel.hpp), so this indexes defensively rather than assuming the
        // UI can only ever produce a value inside kSecondsPerUnit.
        const std::size_t simSpeedUnitIndex =
            (simSpeedUnitSelection >= 0 &&
             static_cast<std::size_t>(simSpeedUnitSelection) < kSecondsPerUnit.size())
                ? static_cast<std::size_t>(simSpeedUnitSelection)
                : 2;  // hour
        timeScale = static_cast<double>(simSpeedValue) * kSecondsPerUnit[simSpeedUnitIndex];
        statsOverlay.draw();
        cameraOverlay.draw();
        energyPlot.draw();
        momentumPlot.draw();
        ui.endFrame();

        window->swapBuffers();
    }

    return EXIT_SUCCESS;
}
