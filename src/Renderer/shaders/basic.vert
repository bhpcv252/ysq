#version 410 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uViewProjection;
// Real eclipse/shadow support (Physics/Optics/Illumination.hpp's own
// discOcclusionFraction): scales the light-dependent part of this draw call's
// own shading. 1.0 (fully lit) unless a caller sets it otherwise --
// Material::lightMultiplier's own default, so this is a no-op for every
// existing draw() call. instanced.vert's per-instance counterpart is the
// same idea, one attribute instead of one uniform, since one instanced
// draw call covers many differently-shadowed instances at once.
uniform float uLightMultiplier;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec2 vUV;
out float vLightMultiplier;

void main() {
    vec4 worldPosition = uModel * vec4(aPosition, 1.0);
    vWorldPosition = worldPosition.xyz;
    // Rotation and uniform scale only: a non-uniform scale needs the
    // inverse-transpose instead, which nothing this engine draws yet does.
    vNormal = normalize(mat3(uModel) * aNormal);
    vUV = aUV;
    vLightMultiplier = uLightMultiplier;
    gl_Position = uViewProjection * worldPosition;
}
