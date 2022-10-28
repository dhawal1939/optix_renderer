#pragma once
#include <vector_types.h>
#include <common.cuh>

__device__
void fetchLtcMat(float alpha, float theta, owl::common::vec3f ltc_mat[3], float& amplitude)
{
    theta = theta * 0.99f / (0.5 * PI);

    float4 r1 = tex2D<float4>(optixLaunchParams.ltc_1, theta, alpha);
    float4 r2 = tex2D<float4>(optixLaunchParams.ltc_2, theta, alpha);
    float4 r3 = tex2D<float4>(optixLaunchParams.ltc_3, theta, alpha);

    ltc_mat[0] = owl::common::vec3f(r1.x, r1.y, r1.z);
    ltc_mat[1] = owl::common::vec3f(r2.x, r2.y, r2.z);
    ltc_mat[2] = owl::common::vec3f(r3.x, r3.y, r3.z);

    amplitude = r3.w;
}

__device__
owl::common::vec3f integrateEdgeVec(owl::common::vec3f v1, owl::common::vec3f v2)
{
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985f + (0.4965155f + 0.0145206f * y) * y;
    float b = 3.4175940f + (4.1616724f + y) * y;
    float v = a / b;

    float theta_sintheta = (x > 0.0f) ? v : 0.5 * (1.0f / sqrt(max(1.0f - x * x, 1e-7))) - v;

    return cross(v1, v2) * theta_sintheta;
}

__device__
float integrateEdge(owl::common::vec3f v1, owl::common::vec3f v2)
{
    return integrateEdgeVec(v1, v2).z;
}