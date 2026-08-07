#version 410 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;
// One instance's model matrix, one vec4 per column: GL has no single mat4
// vertex attribute, only four consecutive vec4 locations. See
// Mesh::setInstanceTransforms.
layout(location = 3) in vec4 aModelCol0;
layout(location = 4) in vec4 aModelCol1;
layout(location = 5) in vec4 aModelCol2;
layout(location = 6) in vec4 aModelCol3;
// Real eclipse/shadow support, one value per instance rather than one per
// draw call: Mesh::setInstanceLightMultipliers, defaulted to 1.0 (fully lit)
// by setInstanceTransforms itself when a caller never sets it. See
// basic.vert's own uLightMultiplier for the non-instanced counterpart.
layout(location = 7) in float aLightMultiplier;

uniform mat4 uViewProjection;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec2 vUV;
out float vLightMultiplier;

void main() {
    mat4 model = mat4(aModelCol0, aModelCol1, aModelCol2, aModelCol3);
    vec4 worldPosition = model * vec4(aPosition, 1.0);
    vWorldPosition = worldPosition.xyz;
    vNormal = normalize(mat3(model) * aNormal);
    vUV = aUV;
    vLightMultiplier = aLightMultiplier;
    gl_Position = uViewProjection * worldPosition;
}
