#pragma once
#include <common.cuh>
#include <utils.cuh>
#include <vector_types.h>
#include <helper_math.cuh>
#include <texture_fetch_functions.h>
#include <ltc/polygon_utils.cuh>

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

__device__
owl::common::vec3f integrateOverPolygon(SurfaceInteraction& si, owl::common::vec3f ltc_mat[3],
	owl::common::vec3f ltc_mat_inv[3], float amplitude,
	owl::common::vec3f iso_frame[3], TriLight& triLight)
{
	owl::common::vec3f lv1 = triLight.v1;
	owl::common::vec3f lv2 = triLight.v2;
	owl::common::vec3f lv3 = triLight.v3;
	owl::common::vec3f lemit = triLight.emit;
	owl::common::vec3f lnormal = triLight.normal;

	// Move to origin and normalize
	lv1 = owl::normalize(lv1 - si.p);
	lv2 = owl::normalize(lv2 - si.p);
	lv3 = owl::normalize(lv3 - si.p);

	owl::common::vec3f cg = normalize(lv1 + lv2 + lv3);
	if (owl::dot(-cg, lnormal) < 0.f)
		return owl::common::vec3f(0.f);

	lv1 = owl::normalize(apply_mat(si.to_local, lv1));
	lv2 = owl::normalize(apply_mat(si.to_local, lv2));
	lv3 = owl::normalize(apply_mat(si.to_local, lv3));

	lv1 = owl::normalize(apply_mat(iso_frame, lv1));
	lv2 = owl::normalize(apply_mat(iso_frame, lv2));
	lv3 = owl::normalize(apply_mat(iso_frame, lv3));

	float diffuse_shading = 0.f;
	float ggx_shading = 0.f;

	owl::common::vec3f diff_clipped[5] = { lv1, lv2, lv3, lv1, lv1 };
	int diff_vcount = clipPolygon(3, diff_clipped);

	if (diff_vcount == 3) {
		diffuse_shading = integrateEdge(diff_clipped[0], diff_clipped[1]);
		diffuse_shading += integrateEdge(diff_clipped[1], diff_clipped[2]);
		diffuse_shading += integrateEdge(diff_clipped[2], diff_clipped[0]);
		diffuse_shading = owl::abs(diffuse_shading);
	}
	else if (diff_vcount == 4) {
		diffuse_shading = integrateEdge(diff_clipped[0], diff_clipped[1]);
		diffuse_shading += integrateEdge(diff_clipped[1], diff_clipped[2]);
		diffuse_shading += integrateEdge(diff_clipped[2], diff_clipped[3]);
		diffuse_shading += integrateEdge(diff_clipped[3], diff_clipped[0]);
		diffuse_shading = owl::abs(diffuse_shading);
	}

	diff_clipped[0] = owl::normalize(apply_mat(ltc_mat_inv, lv1));
	diff_clipped[1] = owl::normalize(apply_mat(ltc_mat_inv, lv2));
	diff_clipped[2] = owl::normalize(apply_mat(ltc_mat_inv, lv3));
	diff_clipped[3] = owl::normalize(apply_mat(ltc_mat_inv, lv1));
	diff_clipped[4] = owl::normalize(apply_mat(ltc_mat_inv, lv1));

	owl::common::vec3f ltc_clipped[5] = { diff_clipped[0], diff_clipped[1], diff_clipped[2], diff_clipped[3], diff_clipped[4] };
	int ltc_vcount = clipPolygon(diff_vcount, ltc_clipped);

	if (ltc_vcount == 3) {
		ggx_shading = integrateEdge(ltc_clipped[0], ltc_clipped[1]);
		ggx_shading += integrateEdge(ltc_clipped[1], ltc_clipped[2]);
		ggx_shading += integrateEdge(ltc_clipped[2], ltc_clipped[0]);
		ggx_shading = owl::abs(ggx_shading);
	}
	else if (ltc_vcount == 4) {
		ggx_shading = integrateEdge(ltc_clipped[0], ltc_clipped[1]);
		ggx_shading += integrateEdge(ltc_clipped[1], ltc_clipped[2]);
		ggx_shading += integrateEdge(ltc_clipped[2], ltc_clipped[3]);
		ggx_shading += integrateEdge(ltc_clipped[3], ltc_clipped[0]);
		ggx_shading = owl::abs(ggx_shading);
	}
	else if (ltc_vcount == 5) {
		ggx_shading = integrateEdge(ltc_clipped[0], ltc_clipped[1]);
		ggx_shading += integrateEdge(ltc_clipped[1], ltc_clipped[2]);
		ggx_shading += integrateEdge(ltc_clipped[2], ltc_clipped[3]);
		ggx_shading += integrateEdge(ltc_clipped[3], ltc_clipped[4]);
		ggx_shading += integrateEdge(ltc_clipped[4], ltc_clipped[0]);
		ggx_shading = owl::abs(ggx_shading);
	}

	owl::common::vec3f color = (si.diffuse * lemit * diffuse_shading) + (amplitude * lemit * ggx_shading);
	return color;
}