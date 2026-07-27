/// Assigning one dimension to another. Both are a double in storage, so
/// nothing but the type system stops this.
///
/// Positive form: UnitsDimensions.ARawNumberIsNotAQuantity.

#include <Units/Length.hpp>
#include <Units/Time.hpp>

int main() {
    ysq::Length distance{5.0};
    const ysq::Time elapsed{2.0};
    distance = elapsed;
    return static_cast<int>(distance.value());
}
