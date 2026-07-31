# Physics: the engine's theories, one subdirectory each. Nothing lower depends
# on this; see the root README's Project structure section.

add_library(ysq_physics STATIC
    Body.hpp

    Mechanics/Frame.hpp
    Mechanics/Kinematics.hpp
    Mechanics/Dynamics.hpp
    Mechanics/Dynamics.cpp
    Mechanics/RigidBody.hpp
    Mechanics/RigidBody.cpp

    Gravity/Newtonian.hpp
    Gravity/Newtonian.cpp
    Gravity/PostNewtonian.hpp
    Gravity/PostNewtonian.cpp
    Gravity/BarnesHut.hpp
    Gravity/BarnesHut.cpp

    Spacetime/Metric.hpp
    Spacetime/Minkowski.hpp
    Spacetime/Schwarzschild.hpp
    Spacetime/Kerr.hpp
    Spacetime/FLRW.hpp
    Spacetime/Geodesic.hpp

    Optics/Propagation.hpp
    Optics/Lensing.hpp
    Optics/FrequencyShift.hpp
    Optics/RefractiveMedium.hpp
    Optics/RayleighScattering.hpp
    Optics/Illumination.hpp
    Optics/Illumination.cpp

    Electromagnetism/Field.hpp
    Electromagnetism/Field.cpp
    Electromagnetism/Lorentz.hpp
    Electromagnetism/Lorentz.cpp
    Electromagnetism/Maxwell.hpp
    Electromagnetism/Maxwell.cpp

    Fluids/SPH.hpp
    Fluids/SPH.cpp
    Fluids/Eulerian.hpp
    Fluids/Eulerian.cpp

    Thermodynamics/Thermodynamics.hpp
    Thermodynamics/HeatEquation.hpp
    Thermodynamics/HeatEquation.cpp)
add_library(ysq::Physics ALIAS ysq_physics)

target_include_directories(ysq_physics PUBLIC ${YSQ_SOURCE_INCLUDE_DIR})

# Math and Units are PUBLIC: every Physics header hands back a Quantity or a
# Math type, so a consumer of Physics needs them too. Core and Compute are
# PRIVATE: nothing in a Physics header names either yet. The Compute edge is
# wired up now even though nothing calls through it, so the dependency graph
# in the root README's Project structure section is true rather than
# aspirational; a GPU-accelerated N-body kernel lands in Compute and gets
# called from here when a rung of the gravity ladder actually needs one, per
# src/Compute/README.md.
target_link_libraries(ysq_physics
    PUBLIC ysq::Math ysq::Units
    PRIVATE ysq::Core ysq::Compute ysq::warnings_strict)
