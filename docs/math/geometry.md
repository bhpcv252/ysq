# Euclidean geometry and spatial partitioning

Shapes, and the questions you ask about them: does this ray hit that
sphere, what's the closest point on this triangle to some point in space,
which of these thousand boxes are anywhere near this one? None of it has
a mass or a material — that's what makes it geometry rather than physics,
and why it lives in `Math` rather than `Physics`.

## The idea

Two different jobs share this corner of `Math`.

**Shapes and the tests between them** (rays, spheres, planes, triangles,
boxes — both axis-aligned and oriented) answer questions like "does this
light ray hit this object" or "where's the nearest point on this surface
to the camera." These are the building blocks a renderer's picking logic,
a physics engine's collision detection, or a level-of-detail system all
need, and they don't care what the shape represents.

**Spatial partitioning** answers a different question: given a *lot* of
objects, which ones are actually worth testing against? Testing a ray
against every triangle in a scene, or every pair of particles for
collision, is `O(n)` or `O(n^2)` — fine for a handful of objects, hopeless
for thousands. A spatial index — a k-d tree, a bounding volume hierarchy,
an octree — narrows "which objects" down to a handful of candidates before
any expensive test runs at all, which is what actually makes a large
scene or a large particle system tractable.

## What YSQ gives you

| Header | What it's for |
| --- | --- |
| `Geometry/Primitives.hpp` | `Ray3`, `Sphere3`, `Plane3`, `Segment3`, `AABB3`, `Triangle3`, `OBB3` (oriented box) |
| `Geometry/Intersection.hpp` | Ray-vs-shape and shape-vs-shape overlap tests |
| `Geometry/Queries.hpp` | Closest point on a shape to an arbitrary point; point-in-polygon |
| `Geometry/ConvexHull.hpp` | The convex hull of a 2D or 3D point cloud |
| `SpatialPartition/KdTree.hpp` | Nearest-neighbor and radius queries over a cloud of points |
| `SpatialPartition/Bvh.hpp` | Overlap and ray queries over a set of boxed objects, splitting the *objects* |
| `SpatialPartition/Octree.hpp` | The same queries, splitting the *space* the objects live in instead |

**Why both a BVH and an octree**, when they answer the same two query
types. A BVH's node bounds are always the tight fit around whatever
objects that subtree holds, rebuilt from the data; an octree's cells are
fixed geometric regions, chosen once and never adjusted to the data. That
makes an octree the better fit when objects move or get added over time
(the cells stay meaningful without a rebuild) and a BVH the tighter fit
for a static scene queried many times.

## Using it

A ray hitting the nearest of several spheres:

```cpp
#include <Math/Geometry/Intersection.hpp>

const ysq::Ray3<double> ray{cameraPosition, viewDirection};
std::optional<double> closestHit;
for (const ysq::Sphere3<double>& sphere : scene) {
    const std::optional<double> hit = ysq::intersect(ray, sphere);
    if (hit && (!closestHit || *hit < *closestHit)) {
        closestHit = hit;
    }
}
```

The same query, sped up for thousands of objects by asking a bounding
volume hierarchy which ones are even worth testing first:

```cpp
#include <Math/SpatialPartition/Bvh.hpp>

const ysq::Bvh3<double> bvh(objectBoundingBoxes);
for (std::size_t candidate : bvh.rayQuery(ray)) {
    // only test the real shape of objects the broad phase didn't already rule out
}
```

## Go deeper

[docs/api/math/geometry.md](../api/math/geometry.md) has every signature:
every intersection and query function, and the exact preconditions on
`OBB3`'s axes staying orthonormal and a polygon being simple and open.

[src/Math/README.md](../../src/Math/README.md) has the algorithm behind
each one in full — the Separating Axis Theorem's fifteen candidate axes
for oriented boxes, Möller-Trumbore's barycentric solve for ray-triangle,
Ericson's region-based closest-point-on-triangle, and why a k-d tree and a
BVH prune their search the way they do.

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+math/geometry)
and let us know.
