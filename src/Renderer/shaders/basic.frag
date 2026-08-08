#version 410 core

#define MAX_POINT_LIGHTS 8
#define MAX_DIRECTIONAL_LIGHTS 4

in vec3 vWorldPosition;
in vec3 vNormal;
in vec2 vUV;
// Real eclipse/shadow support: scales every light's own contribution here,
// not the ambient/emissive terms above -- a fully eclipsed surface still
// has its own ambient light and its own emission, it just receives none
// of a light this factor covers. See basic.vert/instanced.vert for where
// this comes from.
in float vLightMultiplier;

out vec4 fragColor;

uniform vec3 uCameraPosition;

uniform vec3 uAlbedo;
uniform vec3 uEmissive;
uniform float uAmbient;
uniform float uDiffuse;
uniform float uSpecular;
uniform float uShininess;

uniform int uPointLightCount;
uniform vec3 uPointLightPosition[MAX_POINT_LIGHTS];
uniform vec3 uPointLightColor[MAX_POINT_LIGHTS];
uniform float uPointLightIntensity[MAX_POINT_LIGHTS];

uniform int uDirectionalLightCount;
uniform vec3 uDirectionalLightDirection[MAX_DIRECTIONAL_LIGHTS];
uniform vec3 uDirectionalLightColor[MAX_DIRECTIONAL_LIGHTS];
uniform float uDirectionalLightIntensity[MAX_DIRECTIONAL_LIGHTS];

// L points from the surface toward the light.
vec3 blinnPhong(vec3 normal, vec3 viewDir, vec3 L, vec3 lightColor, float attenuation) {
    float diffuseTerm = max(dot(normal, L), 0.0);
    vec3 halfway = normalize(L + viewDir);
    float specularTerm = pow(max(dot(normal, halfway), 0.0), uShininess);
    vec3 diffuse = uDiffuse * diffuseTerm * uAlbedo * lightColor;
    vec3 specular = uSpecular * specularTerm * lightColor;
    return attenuation * (diffuse + specular);
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);

    vec3 color = uAmbient * uAlbedo + uEmissive;

    for (int i = 0; i < uPointLightCount; ++i) {
        vec3 toLight = uPointLightPosition[i] - vWorldPosition;
        float dist = length(toLight);
        vec3 L = toLight / max(dist, 1e-4);
        // Real inverse-square falloff; the max() is a numerical floor
        // against the true 1/d^2 singularity as a surface approaches the
        // light itself, not a physical parameter to tune.
        float attenuation = uPointLightIntensity[i] / max(dist * dist, 1e-4) * vLightMultiplier;
        color += blinnPhong(normal, viewDir, L, uPointLightColor[i], attenuation);
    }

    for (int i = 0; i < uDirectionalLightCount; ++i) {
        vec3 L = normalize(-uDirectionalLightDirection[i]);
        color += blinnPhong(normal, viewDir, L, uDirectionalLightColor[i],
                           uDirectionalLightIntensity[i] * vLightMultiplier);
    }

    fragColor = vec4(color, 1.0);
}
