#version 410 core

layout(location = 0) in vec3 aAnchor;
layout(location = 1) in vec2 aLocalOffset;
layout(location = 2) in vec2 aUV;
layout(location = 3) in vec3 aColor;

uniform mat4 uViewProjection;
uniform vec3 uCameraRight;
uniform vec3 uCameraUp;

out vec2 vUV;
out vec3 vColor;

void main() {
    // Billboard text leaves aLocalOffset as the glyph's actual local
    // position and lets this add it along the camera's own right/up each
    // frame. Fixed-orientation text has already resolved the offset on the
    // CPU against its own fixed right/up and passes aLocalOffset = (0, 0),
    // so this term drops out and aAnchor alone is the final position: one
    // shader, two placement modes, selected by what DebugDraw::text() vs
    // textFixed() bake into the vertex before it ever reaches here.
    vec3 worldPos = aAnchor + uCameraRight * aLocalOffset.x + uCameraUp * aLocalOffset.y;
    gl_Position = uViewProjection * vec4(worldPos, 1.0);
    vUV = aUV;
    vColor = aColor;
}
