/// The square root of a length. An area has one; a length does not, in a
/// system with integer dimension exponents.
///
/// Positive form: UnitsDimensions.RootsAreConstrainedAtTheCallSite, and
/// UnitsQuantity.RootsHalveOrDivideTheExponents for the area that works.

#include <Units/Length.hpp>

int main() {
    const ysq::Length distance{4.0};
    const auto nonsense = sqrt(distance);
    return static_cast<int>(nonsense.value());
}
