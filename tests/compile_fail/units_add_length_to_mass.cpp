/// A distance plus a mass. The canonical dimensional error, and the one this
/// module exists to make impossible.
///
/// Positive form: UnitsDimensions.AdditionRequiresTheSameDimension.

#include <Units/Length.hpp>
#include <Units/Mass.hpp>

int main() {
    const ysq::Length distance{5.0};
    const ysq::Mass mass{2.0};
    const auto nonsense = distance + mass;
    return static_cast<int>(nonsense.value());
}
