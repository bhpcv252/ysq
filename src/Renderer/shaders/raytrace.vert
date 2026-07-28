#version 410 core

// RayTracer draws a -1..1 quad (Mesh::quad(2.0)), so clip space needs no
// transform at all: this pass only exists to hand the fragment shader a
// per-pixel NDC coordinate to build a camera ray from.
layout(location = 0) in vec3 aPosition;

out vec2 vNdc;

void main() {
    vNdc = aPosition.xy;
    gl_Position = vec4(aPosition.xy, 0.0, 1.0);
}
