#version 400
#extension GL_ARB_explicit_uniform_location : enable // required in version lower 4.3
#extension GL_ARB_shading_language_420pack : enable // required in version lower 4.2

layout(location = 0) uniform int srcFullWidth;
layout(location = 1) uniform float shrinkCoeff;
layout(location = 2) uniform float brightness;
layout(location = 3) uniform int enableDebugRendering;
layout(location = 4, binding = 0) uniform sampler2D srcTexture;

layout(origin_upper_left) in vec4 gl_FragCoord;

out vec4 color;

vec3 sRGB_gamma(vec3 v) {
    vec3 ret;
    ret.r = v.r < 0.0031308 ? (12.92 * v.r) : (1.055 * pow(v.r, 1 / 2.4) - 0.055);
    ret.g = v.g < 0.0031308 ? (12.92 * v.g) : (1.055 * pow(v.g, 1 / 2.4) - 0.055);
    ret.b = v.b < 0.0031308 ? (12.92 * v.b) : (1.055 * pow(v.b, 1 / 2.4) - 0.055);
    return ret;
}

void main(void) {
    vec2 srcPixel = gl_FragCoord.xy / shrinkCoeff;
    vec4 opResult = texelFetch(srcTexture, ivec2(srcPixel), 0);
    opResult.rgb = max(opResult.rgb, 0.0f);
    if (bool(enableDebugRendering)) {
        opResult.rgb = min(opResult.rgb, 1.0f);
    }
    else {
        opResult.rgb *= brightness;
        opResult.rgb = 1 - exp(-opResult.rgb);
        // opResult.rgb = sRGB_gamma(opResult.rgb);
    }
    color = opResult;
}