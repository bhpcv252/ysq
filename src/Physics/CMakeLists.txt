# Physics: the engine's theories, one subdirectory each. Nothing lower depends
# on this; see the root README's Project structure section.

add_library(ysq_physics STATIC
    Body.hpp

    Acoustics/Acoustic.hpp
    Acoustics/Acoustic.cpp
    Acoustics/Acoustic3D.hpp
    Acoustics/Acoustic3D.cpp

    Continuum/Elasticity.hpp
    Continuum/ElasticChain1D.hpp
    Continuum/ElasticChain1D.cpp

    QuantumMechanics/Schrodinger.hpp
    QuantumMechanics/Schrodinger.cpp
    QuantumMechanics/Schrodinger3D.hpp
    QuantumMechanics/Schrodinger3D.cpp
    QuantumMechanics/WavePacket.hpp
    QuantumMechanics/WavePacket.cpp
    QuantumMechanics/WavePacket3D.hpp
    QuantumMechanics/WavePacket3D.cpp

    Mechanics/Frame.hpp
    Mechanics/Kinematics.hpp
    Mechanics/Dynamics.hpp
    Mechanics/Dynamics.cpp
    Mechanics/RigidBody.hpp
    Mechanics/RigidBody.cpp
    Mechanics/Hermite.hpp
    Mechanics/Hermite.cpp
    Mechanics/Spring.hpp
    Mechanics/Drag.hpp
    Mechanics/Collision.hpp
    Mechanics/Collision.cpp
    Mechanics/Friction.hpp
    Mechanics/Constraints.hpp
    Mechanics/Constraints.cpp

    Gravity/Newtonian.hpp
    Gravity/Newtonian.cpp
    Gravity/PostNewtonian.hpp
    Gravity/PostNewtonian.cpp
    Gravity/BarnesHut.hpp
    Gravity/BarnesHut.cpp
    Gravity/Kepler.hpp
    Gravity/Kepler.cpp
    Gravity/SphericalHarmonics.hpp
    Gravity/SphericalHarmonics.cpp

    Spacetime/Metric.hpp
    Spacetime/Minkowski.hpp
    Spacetime/Schwarzschild.hpp
    Spacetime/Kerr.hpp
    Spacetime/FLRW.hpp
    Spacetime/Geodesic.hpp
    Spacetime/ADM.hpp
    Spacetime/Bssn.hpp
    Spacetime/Bssn.cpp
    Spacetime/PunctureInitialData.hpp
    Spacetime/PunctureInitialData.cpp

    Optics/Propagation.hpp
    Optics/Lensing.hpp
    Optics/FrequencyShift.hpp
    Optics/RefractiveMedium.hpp
    Optics/RayleighScattering.hpp
    Optics/Illumination.hpp
    Optics/Illumination.cpp
    Optics/RadiationPressure.hpp
    Optics/Diffraction.hpp
    Optics/Diffraction.cpp
    Optics/Aberration.hpp

    Electromagnetism/Field.hpp
    Electromagnetism/Field.cpp
    Electromagnetism/Lorentz.hpp
    Electromagnetism/Lorentz.cpp
    Electromagnetism/Maxwell.hpp
    Electromagnetism/Maxwell.cpp
    Electromagnetism/Maxwell3D.hpp
    Electromagnetism/Maxwell3D.cpp

    Fluids/SPH.hpp
    Fluids/SPH.cpp
    Fluids/Eulerian.hpp
    Fluids/Eulerian.cpp
    Fluids/Eulerian3D.hpp
    Fluids/Eulerian3D.cpp

    Thermodynamics/Thermodynamics.hpp
    Thermodynamics/StatisticalMechanics.hpp
    Thermodynamics/RadiativeTransfer.hpp
    Thermodynamics/HeatEquation.hpp
    Thermodynamics/HeatEquation.cpp
    Thermodynamics/HeatEquation3D.hpp
    Thermodynamics/HeatEquation3D.cpp)
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
