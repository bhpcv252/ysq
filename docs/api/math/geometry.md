# Math API reference: Euclidean geometry and spatial partitioning

Shapes and operations with no physical meaning — nothing here has a mass,
a material, or an owner — plus the two structures for "what's near this?"
queries. Start with [docs/math/geometry.md](../../math/geometry.md) for the
ideas; this page is the lookup table.
[src/Math/README.md](../../../src/Math/README.md) has every algorithm's
derivation and validation in full.

## `Math/Geometry/Primitives.hpp`

The shapes every other header on this page operates on: aggregates with
public members, no invariants enforced beyond what the caller documents
(a `Plane3`'s `normal` must stay unit length, an `OBB3`'s `axes` must stay
orthonormal — nothing here renormalizes either, the same convention
`Quaternion` leaves to its own caller).

```cpp
template <Numeric T> struct Ray3 { Vector3<T> origin{}; Vector3<T> direction{1, 0, 0}; };
template <Numeric T> struct Sphere3 { Vector3<T> center{}; T radius{}; };
template <Numeric T> struct Plane3 { Vector3<T> normal{0, 1, 0}; T distance{}; };  // dot(normal, p) == distance
template <Numeric T> struct Segment3 { Vector3<T> start{}; Vector3<T> end{}; };
template <Numeric T> struct AABB3 { Vector3<T> min{}; Vector3<T> max{}; };
template <Numeric T> struct Triangle3 { Vector3<T> a{}; Vector3<T> b{}; Vector3<T> c{}; };

template <Numeric T> struct OBB3 {
    Vector3<T> center{};
    std::array<Vector3<T>, 3> axes{Vector3<T>::unitX(), Vector3<T>::unitY(), Vector3<T>::unitZ()};
    Vector3<T> halfExtents{};
};
```

## `Math/Geometry/Intersection.hpp`

```cpp
template <Numeric T> std::optional<T> intersect(const Ray3<T>&, const Sphere3<T>&);
template <Numeric T> bool segmentIntersectsSphere(const Vector3<T>& from, const Vector3<T>& to, const Sphere3<T>&);
template <Numeric T> std::optional<T> intersect(const Ray3<T>&, const Plane3<T>&);
template <Numeric T> std::optional<T> intersect(const Ray3<T>&, const Triangle3<T>&);
template <Numeric T> std::optional<T> intersect(const Ray3<T>&, const AABB3<T>&);
template <Numeric T> std::optional<T> intersect(const Ray3<T>&, const OBB3<T>&);

template <Numeric T> bool intersects(const Sphere3<T>&, const Sphere3<T>&);
template <Numeric T> bool intersects(const AABB3<T>&, const AABB3<T>&);
template <Numeric T> bool intersects(const Plane3<T>&, const Sphere3<T>&);
template <Numeric T> bool intersects(const OBB3<T>&, const OBB3<T>&);
template <Numeric T> bool intersects(const OBB3<T>&, const AABB3<T>&);
```

| Function | Description |
| --- | --- |
| `intersect(Ray3, Sphere3)`/`Plane3`/`Triangle3`/`AABB3`/`OBB3` | Nearest hit distance along `ray.direction` from `ray.origin`, `nullopt` on a miss or a hit entirely behind the origin. `Triangle3` uses Möller-Trumbore (barycentric coordinates, no explicit plane); `AABB3` the slab method; `OBB3` transforms into the box's own frame and reuses the `AABB3` slab test. |
| `segmentIntersectsSphere` | Whether a sphere blocks the line of sight between two points — an intersection strictly between them, not merely somewhere along the infinite ray. |
| `intersects(OBB3, OBB3)` | The Separating Axis Theorem over fifteen candidate axes: each box's own three face normals, plus the nine pairwise cross products — the axes that catch a genuinely edge-to-edge separation a naive six-axis test misses. |
| `intersects(OBB3, AABB3)` | Builds the AABB's equivalent axis-aligned `OBB3` and reuses the same SAT routine. |

```cpp
const ysq::Ray3<double> ray{origin, direction};
const std::optional<double> hit = ysq::intersect(ray, sphere);
if (hit) { const ysq::Vec3 point = ray.origin + ray.direction * (*hit); }
```

## `Math/Geometry/Queries.hpp`

```cpp
template <Numeric T> Vector3<T> closestPoint(const Segment3<T>&, const Vector3<T>& point);
template <Numeric T> Vector3<T> closestPoint(const Plane3<T>&, const Vector3<T>& point);
template <Numeric T> Vector3<T> closestPoint(const AABB3<T>&, const Vector3<T>& point);
template <Numeric T> Vector3<T> closestPoint(const Triangle3<T>&, const Vector3<T>& point);
template <Numeric T> Vector3<T> closestPoint(const OBB3<T>&, const Vector3<T>& point);

template <Numeric T> bool pointInPolygon(std::span<const Vector2<T>> polygon, const Vector2<T>& point);
```

| Function | Description |
| --- | --- |
| `closestPoint(Triangle3, point)` | Ericson's region-based method: classifies the point against each vertex/edge Voronoi region in turn, falling through to the face interior only once every region is ruled out. |
| `closestPoint(OBB3, point)` | Projects onto each of the box's own axes, clamps to that axis's half-extent, reconstructs in world space — the oriented-box counterpart of `AABB3`'s per-axis clamp. |
| `pointInPolygon` | The even-odd (PNPOLY) rule: casts a ray along `+x`, counts edge crossings. `polygon` must be simple (non-self-intersecting) and open (no repeated first/last point); winding order doesn't matter. |

```cpp
const ysq::Vec3 nearest = ysq::closestPoint(triangle, queryPoint);
```

## `Math/Geometry/ConvexHull.hpp`

```cpp
template <Numeric T> std::vector<Vector2<T>> convexHull2D(std::span<const Vector2<T>> points);
template <Numeric T> std::vector<Triangle3<T>> convexHull3D(std::span<const Vector3<T>> points);
```

| Function | Description |
| --- | --- |
| `convexHull2D` | Andrew's monotone chain: sort by `x` (then `y`), build the lower and upper chains in one linear pass each. `O(n log n)`, dominated by the sort. Returns hull vertices in order. |
| `convexHull3D` | The incremental algorithm: start from a non-degenerate tetrahedron (found by farthest-pair, then farthest-from-that-line, then farthest-from-that-plane), then add each remaining point, removing every face it can see past and patching the hole from the horizon. Returns the hull as a list of outward-facing triangles. |

```cpp
const std::vector<ysq::Vec2> hull = ysq::convexHull2D<double>(pointCloud2D);
const std::vector<ysq::Triangle3<double>> hull3D = ysq::convexHull3D<double>(pointCloud3D);
```

## `Math/SpatialPartition/KdTree.hpp`

Point-based radius and k-nearest-neighbor queries over a fixed point cloud.

```cpp
template <std::floating_point T> class KdTree3 {
public:
    explicit KdTree3(std::vector<Vector3<T>> points);
    std::vector<std::size_t> radiusQuery(const Vector3<T>& queryPoint, T radius) const;
    std::vector<std::size_t> nearestNeighbors(const Vector3<T>& queryPoint, std::size_t k) const;
};
```

Splits points by their median along an axis cycling `x`/`y`/`z` with tree
depth. `radiusQuery` and `nearestNeighbors` both prune the far side of a
splitting plane whenever nothing there could possibly qualify.
`Physics/Fluids/SPH.cpp`'s neighbor search is the reference consumer: a
radius query at the kernel's own compact-support distance, replacing a
brute-force all-pairs loop.

```cpp
const ysq::KdTree3<double> tree(points);
for (std::size_t index : tree.radiusQuery(queryPoint, radius)) { /* ... */ }
```

## `Math/SpatialPartition/Bvh.hpp`

Box-based overlap and ray queries over a set of `AABB3`s, one per object:
**object partitioning** — the tree splits the *objects* at their median.

```cpp
template <std::floating_point T> class Bvh3 {
public:
    explicit Bvh3(std::vector<AABB3<T>> bounds);
    std::vector<std::size_t> overlapQuery(const AABB3<T>& region) const;
    std::vector<std::size_t> rayQuery(const Ray3<T>& ray) const;
};
```

Builds top-down, splitting each range of objects along whichever axis its
combined bounds are longest on, at the median object center. Node bounds
are always the tight union of their objects' own boxes, so a query prunes
a subtree the moment its bounds miss.

```cpp
const ysq::Bvh3<double> bvh(boxes);
for (std::size_t index : bvh.overlapQuery(region)) { /* candidate objects */ }
```

## `Math/SpatialPartition/Octree.hpp`

The same overlap/ray query interface as `Bvh3`, but **space
partitioning**: the tree splits a fixed geometric region into eight equal
octants, so cell boundaries never move once chosen — what you want when
cells need to stay meaningful across incremental updates, rather than
only for a single build-once query pass.

```cpp
template <std::floating_point T> class Octree3 {
public:
    Octree3(std::vector<AABB3<T>> bounds, AABB3<T> worldBounds,
           std::size_t maxObjectsPerNode = 8, int maxDepth = 8);
    std::vector<std::size_t> overlapQuery(const AABB3<T>& region) const;
    std::vector<std::size_t> rayQuery(const Ray3<T>& ray) const;
};
```

An object is handed down to a single child only if its box fits entirely
inside that child's octant; a box straddling a split stays at the node
doing the splitting rather than being duplicated. Unlike `Bvh3`, a cell's
bounds are a fixed region, not a tight fit around its contents, so a
query still tests each candidate object's own box before accepting it.

```cpp
const ysq::AABB3<double> worldBounds{Vec3{-100, -100, -100}, Vec3{100, 100, 100}};
const ysq::Octree3<double> octree(boxes, worldBounds);
for (std::size_t index : octree.rayQuery(ray)) { /* candidate objects */ }
```

---
Notice something missing or wrong on this page?
[Open an issue](https://github.com/bhpcv252/ysq/issues/new?title=docs:+api/math/geometry)
and let us know.
