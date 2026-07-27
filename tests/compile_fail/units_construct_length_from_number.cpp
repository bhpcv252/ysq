/// A bare number becoming a length by implicit conversion. Construction is
/// explicit precisely so this needs a deliberate act.
///
/// Positive form: UnitsDimensions.ARawNumberIsNotAQuantity, which asserts that
/// the explicit form does work.

#include <Units/Length.hpp>

int main() {
    const ysq::Length distance = 5.0;
    return static_cast<int>(distance.value());
}
