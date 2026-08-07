#version 410 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

uniform mat4 uViewProjection;
uniform vec3 uCenter;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;
uniform float uWorldRadius;

out vec2 vLocal;

void main() {
    // aPosition is Mesh::quad(1.0)'s own unit quad, [-0.5, 0.5] in x/y: the
    // same camera-facing billboard technique text.vert uses, offsetting
    // uCenter along the camera's own right/up rather than the model's,
    // scaled to a diameter of 2 * uWorldRadius.
    vec3 worldPos =
        uCenter + (uCameraRight * aPosition.x + uCameraUp * aPosition.y) * (2.0 * uWorldRadius);
    gl_Position = uViewProjection * vec4(worldPos, 1.0);
    vLocal = aPosition.xy * 2.0;  // [-1, 1], for a circular falloff in the fragment shader
}
