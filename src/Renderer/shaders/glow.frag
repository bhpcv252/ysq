#version 410 core

in vec2 vLocal;
out vec4 fragColor;

uniform vec3 uColor;
uniform float uIntensity;

void main() {
    float r = length(vLocal);
    if (r > 1.0) {
        discard;
    }
    // Exponential, not linear: `1 - smoothstep` falls off gradually enough
    // across the whole quad that its own edge still reads as a distinct
    // ring against a black background (a real glow has no edge at all).
    // exp(-r^2 * k) concentrates almost all of the brightness close to
    // the center and decays smoothly the rest of the way to the edge.
    float falloff = exp(-r * r * 4.0);
    vec3 raw = uColor * uIntensity * falloff;
    // A real star's own core is far brighter than an 8-bit backbuffer can
    // represent -- it reads as white-hot near the center, its true color
    // only visible further out where the (real, physical) brightness has
    // actually dropped enough not to saturate. A hard "mix to white
    // inside some fixed radius" was tried here first and rejected: it
    // draws a visible second ring at that radius no matter how it is
    // tuned, since it is a genuine discontinuity, not a gradient. This
    // reaches the same result -- white-hot core, true color at the edge
    // -- as a smooth, single continuous function of the real brightness
    // instead: each channel saturates toward 1 at its own rate (red
    // reaches 1 before blue does, since uColor's own red component is
    // larger), so the color itself warms toward white exactly where the
    // real brightness is highest, with no second radius anywhere in the
    // math to show up as an edge.
    vec3 color = vec3(1.0) - exp(-raw);
    float alpha = max(max(color.r, color.g), color.b);
    fragColor = vec4(color, alpha);
}
