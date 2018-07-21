#version 400
#extension GL_ARB_explicit_uniform_location : enable // required in version lower4.3
#extension GL_ARB_shading_language_420pack : enable // required in version lower 4.2

layout(location = 0) uniform float scale;
layout(location = 1, binding = 0) uniform sampler2D srcTexture;

/* layout(origin_upper_left) */in vec4 gl_FragCoord;

out vec4 color;

void main(void) {
    color = texelFetch(srcTexture, ivec2(gl_FragCoord.xy / scale), 0);
}