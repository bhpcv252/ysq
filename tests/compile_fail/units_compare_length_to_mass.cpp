/// Ordering across dimensions. Comparison is the operation most likely to be
/// written by reflex against the wrong variable.
///
/// Positive form: UnitsDimensions.ComparisonRequiresTheSameDimension.

#include <Units/Length.hpp>
#include <Units/Mass.hpp>

int main() {
    const ysq::Length distance{5.0};
    const ysq::Mass mass{2.0};
    return (distance < mass) ? 0 : 1;
}
