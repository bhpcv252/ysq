#version 410 core

// A fragment-shader ray tracer, not a compute shader: compute shaders are a
// 4.3 feature and macOS never gets past 4.1, so this is the form that stays
// portable everywhere the rasterizer already runs. See src/Renderer/README.md.
//
// Scene data arrives as plain "structure of arrays" uniform arrays rather
// than a UBO: no std140 padding to get right by hand, and it is exactly as
// GL-4.1-portable as a UBO would be. Counts (uSphereCount etc.) say how much
// of each array is actually valid; RayTracer never uploads more than the
// Max* below, so a scene that outgrows one frame's arrays is truncated
// rather than corrupting memory.

#define MAX_SPHERES 32
#define MAX_PLANES 8
#define MAX_DISKS 8
#define MAX_POINT_LIGHTS 8
#define MAX_DIRECTIONAL_LIGHTS 4
#define MAX_BOUNCES 8

in vec2 vNdc;
out vec4 fragColor;

uniform vec3 uCameraPosition;
uniform vec3 uCameraForward;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uTanHalfFovY;
uniform float uAspect;
uniform int uMaxBounces;

uniform vec3 uBackgroundColor;
uniform bool uHasEnvironment;
uniform samplerCube uEnvironment;

uniform int uSphereCount;
uniform vec3 uSphereCenter[MAX_SPHERES];
uniform float uSphereRadius[MAX_SPHERES];
uniform vec3 uSphereAlbedo[MAX_SPHERES];
uniform vec3 uSphereEmissive[MAX_SPHERES];
uniform float uSphereAmbient[MAX_SPHERES];
uniform float uSphereDiffuse[MAX_SPHERES];
uniform float uSphereSpecular[MAX_SPHERES];
uniform float uSphereShininess[MAX_SPHERES];
uniform float uSphereReflectivity[MAX_SPHERES];

uniform int uPlaneCount;
uniform vec3 uPlanePoint[MAX_PLANES];
uniform vec3 uPlaneNormal[MAX_PLANES];
uniform vec3 uPlaneAlbedo[MAX_PLANES];
uniform vec3 uPlaneEmissive[MAX_PLANES];
uniform float uPlaneAmbient[MAX_PLANES];
uniform float uPlaneDiffuse[MAX_PLANES];
uniform float uPlaneSpecular[MAX_PLANES];
uniform float uPlaneShininess[MAX_PLANES];
uniform float uPlaneReflectivity[MAX_PLANES];

uniform int uDiskCount;
uniform vec3 uDiskCenter[MAX_DISKS];
uniform vec3 uDiskNormal[MAX_DISKS];
uniform float uDiskInnerRadius[MAX_DISKS];
uniform float uDiskOuterRadius[MAX_DISKS];
uniform vec3 uDiskAlbedo[MAX_DISKS];
uniform vec3 uDiskEmissive[MAX_DISKS];
uniform float uDiskAmbient[MAX_DISKS];
uniform float uDiskDiffuse[MAX_DISKS];
uniform float uDiskSpecular[MAX_DISKS];
uniform float uDiskShininess[MAX_DISKS];
uniform float uDiskReflectivity[MAX_DISKS];

uniform int uPointLightCount;
uniform vec3 uPointLightPosition[MAX_POINT_LIGHTS];
uniform vec3 uPointLightColor[MAX_POINT_LIGHTS];
uniform float uPointLightIntensity[MAX_POINT_LIGHTS];

uniform int uDirectionalLightCount;
uniform vec3 uDirectionalLightDirection[MAX_DIRECTIONAL_LIGHTS];
uniform vec3 uDirectionalLightColor[MAX_DIRECTIONAL_LIGHTS];
uniform float uDirectionalLightIntensity[MAX_DIRECTIONAL_LIGHTS];

const float kMaxDistance = 1e30;
const float kEpsilon = 1e-3;

struct Hit {
    float t;
    vec3 point;
    vec3 normal;
    vec3 albedo;
    vec3 emissive;
    float ambient;
    float diffuse;
    float specular;
    float shininess;
    float reflectivity;
};

bool intersectSphere(vec3 origin, vec3 dir, vec3 center, float radius, out float t) {
    vec3 oc = origin - center;
    float b = dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        return false;
    }
    float sq = sqrt(disc);
    float t0 = -b - sq;
    float t1 = -b + sq;
    t = (t0 > kEpsilon) ? t0 : t1;
    return t > kEpsilon;
}

bool intersectPlane(vec3 origin, vec3 dir, vec3 point, vec3 normal, out float t) {
    float denom = dot(normal, dir);
    if (abs(denom) < 1e-6) {
        return false;
    }
    t = dot(point - origin, normal) / denom;
    return t > kEpsilon;
}

bool intersectDisk(vec3 origin, vec3 dir, vec3 center, vec3 normal, float innerRadius,
                   float outerRadius, out float t) {
    if (!intersectPlane(origin, dir, center, normal, t)) {
        return false;
    }
    float d = length((origin + dir * t) - center);
    return d >= innerRadius && d <= outerRadius;
}

bool traceScene(vec3 origin, vec3 dir, out Hit hit) {
    hit.t = kMaxDistance;
    bool found = false;

    for (int i = 0; i < uSphereCount; ++i) {
        float t;
        if (intersectSphere(origin, dir, uSphereCenter[i], uSphereRadius[i], t) && t < hit.t) {
            hit.t = t;
            hit.point = origin + dir * t;
            hit.normal = normalize(hit.point - uSphereCenter[i]);
            hit.albedo = uSphereAlbedo[i];
            hit.emissive = uSphereEmissive[i];
            hit.ambient = uSphereAmbient[i];
            hit.diffuse = uSphereDiffuse[i];
            hit.specular = uSphereSpecular[i];
            hit.shininess = uSphereShininess[i];
            hit.reflectivity = uSphereReflectivity[i];
            found = true;
        }
    }
    for (int i = 0; i < uPlaneCount; ++i) {
        float t;
        if (intersectPlane(origin, dir, uPlanePoint[i], uPlaneNormal[i], t) && t < hit.t) {
            hit.t = t;
            hit.point = origin + dir * t;
            hit.normal = uPlaneNormal[i];
            hit.albedo = uPlaneAlbedo[i];
            hit.emissive = uPlaneEmissive[i];
            hit.ambient = uPlaneAmbient[i];
            hit.diffuse = uPlaneDiffuse[i];
            hit.specular = uPlaneSpecular[i];
            hit.shininess = uPlaneShininess[i];
            hit.reflectivity = uPlaneReflectivity[i];
            found = true;
        }
    }
    for (int i = 0; i < uDiskCount; ++i) {
        float t;
        if (intersectDisk(origin, dir, uDiskCenter[i], uDiskNormal[i], uDiskInnerRadius[i],
                          uDiskOuterRadius[i], t) &&
            t < hit.t) {
            hit.t = t;
            hit.point = origin + dir * t;
            hit.normal = uDiskNormal[i];
            hit.albedo = uDiskAlbedo[i];
            hit.emissive = uDiskEmissive[i];
            hit.ambient = uDiskAmbient[i];
            hit.diffuse = uDiskDiffuse[i];
            hit.specular = uDiskSpecular[i];
            hit.shininess = uDiskShininess[i];
            hit.reflectivity = uDiskReflectivity[i];
            found = true;
        }
    }
    return found;
}

bool occluded(vec3 origin, vec3 dir, float maxDistance) {
    for (int i = 0; i < uSphereCount; ++i) {
        float t;
        if (intersectSphere(origin, dir, uSphereCenter[i], uSphereRadius[i], t) &&
            t < maxDistance) {
            return true;
        }
    }
    for (int i = 0; i < uPlaneCount; ++i) {
        float t;
        if (intersectPlane(origin, dir, uPlanePoint[i], uPlaneNormal[i], t) && t < maxDistance) {
            return true;
        }
    }
    for (int i = 0; i < uDiskCount; ++i) {
        float t;
        if (intersectDisk(origin, dir, uDiskCenter[i], uDiskNormal[i], uDiskInnerRadius[i],
                          uDiskOuterRadius[i], t) &&
            t < maxDistance) {
            return true;
        }
    }
    return false;
}

vec3 shade(Hit hit, vec3 viewDir) {
    vec3 color = hit.ambient * hit.albedo + hit.emissive;

    for (int i = 0; i < uPointLightCount; ++i) {
        vec3 toLight = uPointLightPosition[i] - hit.point;
        float dist = length(toLight);
        vec3 L = toLight / max(dist, 1e-4);
        if (occluded(hit.point + hit.normal * kEpsilon, L, dist)) {
            continue;
        }

        // Real inverse-square falloff; the max() is a numerical floor
        // against the true 1/d^2 singularity as a surface approaches the
        // light itself, not a physical parameter to tune.
        float attenuation = uPointLightIntensity[i] / max(dist * dist, 1e-4);
        float diffuseTerm = max(dot(hit.normal, L), 0.0);
        vec3 halfway = normalize(L + viewDir);
        float specularTerm = pow(max(dot(hit.normal, halfway), 0.0), hit.shininess);
        color += attenuation *
                (hit.diffuse * diffuseTerm * hit.albedo + hit.specular * specularTerm) *
                uPointLightColor[i];
    }

    for (int i = 0; i < uDirectionalLightCount; ++i) {
        vec3 L = normalize(-uDirectionalLightDirection[i]);
        if (occluded(hit.point + hit.normal * kEpsilon, L, kMaxDistance)) {
            continue;
        }

        float diffuseTerm = max(dot(hit.normal, L), 0.0);
        vec3 halfway = normalize(L + viewDir);
        float specularTerm = pow(max(dot(hit.normal, halfway), 0.0), hit.shininess);
        color += uDirectionalLightIntensity[i] *
                (hit.diffuse * diffuseTerm * hit.albedo + hit.specular * specularTerm) *
                uDirectionalLightColor[i];
    }

    return color;
}

vec3 environment(vec3 dir) {
    return uHasEnvironment ? texture(uEnvironment, dir).rgb : uBackgroundColor;
}

void main() {
    vec3 rayDir = normalize(uCameraForward +
                            uCameraRight * (vNdc.x * uTanHalfFovY * uAspect) +
                            uCameraUp * (vNdc.y * uTanHalfFovY));
    vec3 rayOrigin = uCameraPosition;

    vec3 color = vec3(0.0);
    vec3 throughput = vec3(1.0);
    int bounces = min(uMaxBounces, MAX_BOUNCES);

    for (int bounce = 0; bounce <= bounces; ++bounce) {
        Hit hit;
        if (!traceScene(rayOrigin, rayDir, hit)) {
            color += throughput * environment(rayDir);
            break;
        }

        vec3 viewDir = normalize(-rayDir);
        color += throughput * (1.0 - hit.reflectivity) * shade(hit, viewDir);

        if (hit.reflectivity <= 0.0 || bounce == bounces) {
            break;
        }

        throughput *= hit.reflectivity;
        rayDir = reflect(rayDir, hit.normal);
        rayOrigin = hit.point + hit.normal * kEpsilon;
    }

    fragColor = vec4(color, 1.0);
}
