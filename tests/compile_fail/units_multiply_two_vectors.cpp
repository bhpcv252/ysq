/// Multiplying two vector quantities. There is no product of two vectors that
/// is a vector or a scalar without saying which is meant, so operator* does
/// not exist for them and dot and cross do.
///
/// Positive form:
/// UnitsDimensions.VectorQuantitiesRefuseTheOperationsVectorsDoNotHave.

#include <Units/Force.hpp>
#include <Units/Length.hpp>

int main() {
    const ysq::Length3 lever{ysq::Vec3{2.0, 0.0, 0.0}};
    const ysq::Force3 force{ysq::Vec3{0.0, 3.0, 0.0}};
    const auto nonsense = lever * force;
    return static_cast<int>(nonsense.value().x);
}
