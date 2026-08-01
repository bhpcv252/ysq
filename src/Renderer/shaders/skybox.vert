#version 410 core

layout(location = 0) in vec3 aPosition;

// Rotation only: the skybox must not translate as the camera moves, so
// Renderer uploads uView with its translation column zeroed.
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vDirection;

void main() {
    vDirection = aPosition;
    vec4 clipPosition = uProjection * uView * vec4(aPosition, 1.0);
    // Forces depth to the far plane so the skybox never occludes real
    // geometry and is never occluded by the depth buffer's clear value.
    // Reversed-Z (see Renderer.cpp / Camera::projectionMatrix()) maps far to
    // z/w == -1, not +1, so this pins z to -w rather than the more familiar
    // xyww ("z == w") trick a standard depth convention would use.
    gl_Position = vec4(clipPosition.xy, -clipPosition.w, clipPosition.w);
}
