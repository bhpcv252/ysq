#include <Physics/Continuum/Elasticity.hpp>
#include <Units/Elasticity.hpp>
#include <Units/Force.hpp>
#include <Units/Length.hpp>
#include <support/UnitsApprox.hpp>

#include <gtest/gtest.h>

namespace {

using ysq::Area;
using ysq::Force;
using ysq::Length;
using ysq::SpringConstant;
using ysq::Strain;
using ysq::Stress;

}  // namespace

TEST(PhysicsContinuumElasticity, HookeStressMatchesTheClosedForm) {
    const Stress youngsModulus{200e9};  // roughly steel, Pa
    const Strain strain{0.001};

    EXPECT_QUANTITY_NEAR(ysq::hookeStress(youngsModulus, strain), Stress{2e8},
                         Stress{1.0});
}

TEST(PhysicsContinuumElasticity, HookeStressIsZeroAtZeroStrain) {
    EXPECT_QUANTITY_NEAR(ysq::hookeStress(Stress{200e9}, Strain{0.0}), Stress{0.0},
                         Stress{1e-9});
}

TEST(PhysicsContinuumElasticity, AxialExtensionMatchesTheClosedForm) {
    const Force load{1000.0};
    const Length naturalLength{2.0};
    const Area crossSection{0.0001};
    const Stress youngsModulus{200e9};

    const Length extension =
        ysq::axialExtension(load, naturalLength, crossSection, youngsModulus);
    const double expected = load.value() * naturalLength.value() /
                            (crossSection.value() * youngsModulus.value());

    EXPECT_NEAR(extension.value(), expected, expected * 1e-12);
}

TEST(PhysicsContinuumElasticity, EquivalentSpringConstantMatchesTheClosedForm) {
    const Stress youngsModulus{200e9};
    const Area crossSection{0.0001};
    const Length naturalLength{2.0};

    const SpringConstant k =
        ysq::equivalentSpringConstant(youngsModulus, crossSection, naturalLength);
    const double expected =
        youngsModulus.value() * crossSection.value() / naturalLength.value();

    EXPECT_NEAR(k.value(), expected, expected * 1e-12);
}

TEST(PhysicsContinuumElasticity, AxialExtensionAndSpringConstantAgreeWithEachOther) {
    // The same physics reached two ways: extension = load / k, where k is
    // exactly equivalentSpringConstant's own result.
    const Force load{1500.0};
    const Length naturalLength{1.5};
    const Area crossSection{0.0002};
    const Stress youngsModulus{70e9};  // roughly aluminum

    const Length viaExtensionFormula =
        ysq::axialExtension(load, naturalLength, crossSection, youngsModulus);
    const SpringConstant k =
        ysq::equivalentSpringConstant(youngsModulus, crossSection, naturalLength);
    const Length viaSpringConstant{load.value() / k.value()};

    EXPECT_NEAR(viaExtensionFormula.value(), viaSpringConstant.value(),
                viaExtensionFormula.value() * 1e-12);
}
