# Optics

Light propagation, lensing, and frequency shift: one idea, looked at three
ways.

## The idea

In general relativity, light isn't a special case that needs its own
physics. It's a massless particle following a geodesic (see
[docs/physics/spacetime.md](spacetime.md)) through whatever metric it's
traveling through, exactly like a free-falling object, except its path is
*null*: the spacetime interval along it is exactly zero, rather than
negative the way a massive particle's timelike path is. Nothing else
distinguishes light from anything else moving through curved spacetime.

That one idea is why three things that sound like separate phenomena are
actually the same computation:

- **Propagation** is just running the geodesic solver with a null starting
  direction.
- **Gravitational lensing** (light bending near a mass) is what a null
  geodesic does when it passes close to something massive; there's nothing
  extra to compute beyond following the path.
- **Frequency shift** (Doppler, gravitational redshift, cosmological
  redshift) is the ratio of a photon's momentum measured by two different
  observers, at emission and at reception. Which name you use depends only
  on who those observers are (relatively moving, near a mass, or comoving
  in an expanding universe), not on a different formula.

## What YSQ gives you

| Header | Purpose |
| --- | --- |
| `Optics/Propagation.hpp` | Build a light ray's starting direction (`nullTangent`) and run it (`propagate`) |
| `Optics/Lensing.hpp` | `deflectionAngle`: how much a ray bends |
| `Optics/FrequencyShift.hpp` | `frequencyShift`: one formula for Doppler, gravitational, and cosmological shift |
| `Optics/RefractiveMedium.hpp` | A graded-index medium (air, glass, anything) as a metric: refraction is a null geodesic too |
| `Optics/RayleighScattering.hpp` | A gas's scattering cross-section and the wavelength-dependent transmission along a path |
| `Optics/Illumination.hpp` | How much of an extended light source is visible from a point, through occlusion and one refracting medium |

**A subtlety worth knowing about, because it's a real bug that was caught
this way.** Measuring the deflection angle by comparing a ray's total
sweep against a flat `pi` sounds right and isn't: for any *finite* starting
radius, a perfectly straight line in flat space already sweeps less than
`pi`. Comparing against bare `pi` gives a deflection estimate that looks
like it's converging as you refine the integration, but actually converges
to the wrong number, one that depends on where you started measuring from.
`deflectionAngle` compares against the correct flat-space sweep instead,
and this is exactly the kind of subtle error a good test suite exists to
catch rather than a hypothetical concern.

**Refraction is the same idea again, against a different metric.** A ray
through an ordinary medium of refractive index `n(r)` extremizes optical
path length (Fermat's principle), which turns out to be exactly the
spatial path of a null geodesic of a particular "ultra-static" metric built
from `n(r)`. `RefractiveMedium.hpp` is that metric; nothing about
`nullTangent`, `propagate`, or `deflectionAngle` changes to use it, because
none of them know or care which metric they're handed. This is how
atmospheric refraction (the reason the sun is still visible for a few
minutes after it's geometrically below the horizon, or the reason a total
lunar eclipse isn't pitch black) gets computed in this engine: the same
solver, a different metric, not a special case.

`RayleighScattering.hpp` supplies the one law refraction alone doesn't:
why the light that survives a long grazing path through an atmosphere
comes out reddened, not just dimmer. `Illumination.hpp` composes both with
straight-line occlusion into one answer, "how much of an extended source
reaches this point", for any scene shaped that way, not only an eclipse.

## Using it

```cpp
#include <Physics/Optics/FrequencyShift.hpp>
#include <Physics/Optics/Propagation.hpp>

const ysq::Vec4 k = ysq::nullTangent(schwarzschild, emissionEvent, direction);
const ysq::PhaseState<ysq::Vec4> end =
    ysq::propagate(schwarzschild, {emissionEvent, k}, affineInterval, steps);

const double shift = ysq::frequencyShift(
    schwarzschild, emissionEvent, k, emitterFourVelocity,
    end.position, end.velocity, observerFourVelocity);
```

`shift` is `1.0` for no shift, greater than `1.0` for blueshift, less than
`1.0` for redshift, whatever the actual cause: a moving source, a source
deep in a gravity well, or an expanding universe between source and
observer.

## Go deeper

[docs/api/physics/optics.md](../api/physics/optics.md) has every signature:
`nullTangent`/`propagate`, `deflectionAngle` and its helpers, and
`frequencyShift`/`staticObserverFourVelocity`.

[src/Physics/README.md](../../src/Physics/README.md) has the full
derivations: `nullTangent`'s quadratic solve for a future-directed photon,
the exact flat-space sweep formula the lensing correction above uses, and
`staticObserverFourVelocity`, which covers both the gravitational and
cosmological observer cases from one function.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+physics/optics)
and let us know.
