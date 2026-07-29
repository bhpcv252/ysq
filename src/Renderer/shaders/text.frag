#version 410 core

in vec2 vUV;
in vec3 vColor;

uniform sampler2D uFontAtlas;

out vec4 fragColor;

void main() {
    float coverage = texture(uFontAtlas, vUV).a;
    if (coverage < 0.05) {
        discard;
    }
    fragColor = vec4(vColor, coverage);
}
