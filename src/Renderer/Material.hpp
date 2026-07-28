#pragma once

#include <Math/Vector3.hpp>

namespace ysq {

/// Blinn-Phong surface parameters, shared by the rasterizer and RayTracer.
///
/// This is not a physically-based material system: nothing this engine
/// renders needs one yet (analytic bodies, fields, grids), and a BRDF-based
/// pipeline would be complexity spent on a problem that does not exist here.
/// See docs/rendering.md.
struct Material {
    Vec3f albedo = Vec3f::splat(0.8f);
    /// Self-lit contribution, independent of any light: what makes a star a
    /// light source in the scene rather than merely a lit sphere.
    Vec3f emissive = Vec3f::zero();

    float ambient = 0.1f;
    float diffuse = 0.9f;
    float specular = 0.3f;
    float shininess = 32.0f;

    /// RayTracer only: fraction of a reflection ray's color mixed into this
    /// surface's own, [0, 1]. The rasterizer has no reflection pass and
    /// ignores this.
    float reflectivity = 0.0f;
};

}  // namespace ysq
