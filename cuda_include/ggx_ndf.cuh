#pragma once
#include <common.h>
#include <helper_math.cuh>
#include <utils.cuh>
#include <cuda_runtime.h>
#include <owl/common/math/vec.h>


#define EPS float(1e-4)


//owl::common::vec3f sampleVndf(owl::common::vec3f V, float roughness, vec2 rng, mat3 onb) {
//	V = owl::common::vec3f(dot(V, onb[0]), dot(V, onb[2]), dot(V, onb[1]));
//
//	// Transforming the view direction to the hemisphere configuration
//	V = normalize(owl::common::vec3f(roughness * V.x, roughness * V.y, V.z));
//
//	// Orthonormal basis (with special case if cross product is zero)
//	float lensq = V.x * V.x + V.y * V.y;
//	owl::common::vec3f T1 =
//		lensq > 0. ? owl::common::vec3f(-V.y, V.x, 0) * inversesqrt(lensq) : owl::common::vec3f(1, 0, 0);
//	owl::common::vec3f T2 = cross(V, T1);
//
//	// Parameterization of the projected area
//	float r = sqrt(rng.x);
//	float phi = 2.0 * pi * rng.y;
//	float t1 = r * cos(phi);
//	float t2 = r * sin(phi);
//	float s = 0.5 * (1.0 + V.z);
//	t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
//
//	// Reprojection onto hemisphere
//	owl::common::vec3f Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * V;
//	// Transforming the normal back to the ellipsoid configuration
//	owl::common::vec3f Ne = normalize(owl::common::vec3f(roughness * Nh.x, max(0.0, Nh.z), roughness * Nh.y));
//	return normalize(onb * Ne);
//}
//
//float sampleVndfPdf(owl::common::vec3f V, owl::common::vec3f H, float D) {
//	float VdotH = clampDot(V, H, true);
//	return VdotH > 0 ? D / (4 * VdotH) : 0;
//}


__device__
float clampDot(owl::common::vec3f a, owl::common::vec3f b, bool zero) {
	return max(owl::common::dot(a, b), zero ? 0 : EPS);
}

__device__
// Reflective GGX
float D(owl::common::vec3f N, owl::common::vec3f H, float roughness) {
	float a2 = roughness * roughness;
	float NdotH = clampDot(N, H, true);
	float denom = ((NdotH * NdotH) * (a2 * a2 - 1) + 1);
	return a2 / (PI * denom * denom);
}


__device__
float G(owl::common::vec3f N, owl::common::vec3f V, owl::common::vec3f L, float roughness) {
	float NdotV = clampDot(N, V, false);
	float NdotL = clampDot(N, L, false);
	float k = roughness * roughness / 2;
	float G_V = NdotV / (NdotV * (1.0 - k) + k);
	float G_L = NdotL / (NdotL * (1.0 - k) + k);

	return G_V * G_L;
}


__device__
owl::common::vec3f F(owl::common::vec3f L, owl::common::vec3f H, owl::common::vec3f F0) {
	float LdotH = clampDot(L, H, true);
	return F0 + (1.0f - F0) * pow(1.0f - LdotH, 5.0f);
}

__device__
owl::common::vec3f sampleNdf(float roughness, owl::common::vec2f rng, owl::common::vec3f* onb) {
	// GGX NDF sampling
	float a2 = roughness * roughness;
	float cosThetaH = sqrt(max(0.0, (1.0 - rng.x) / ((a2 - 1.0) * rng.x + 1.0)));
	float sinThetaH = sqrt(max(0.0, 1.0 - cosThetaH * cosThetaH));
	float phiH = rng.y * PI * 2.0;

	// Get our GGX NDF sample (i.e., the half vector)
	owl::common::vec3f H = owl::common::vec3f(sinThetaH * cos(phiH), cosThetaH, sinThetaH * sin(phiH));
	return normalize(apply_mat(onb, H));
}

__device__
float sampleNdfPdf(owl::common::vec3f L, owl::common::vec3f H, owl::common::vec3f N, float alpha) {
	return D(N, H, alpha) * clampDot(N, H, true) / (4 * clampDot(H, L, false));
}


__device__
owl::common::vec3f evalBSDF(float alpha, 
	owl::common::vec3f diffuse_color,
	owl::common::vec3f N, owl::common::vec3f H, 
	owl::common::vec3f V, owl::common::vec3f L, 
	owl::common::vec3f pos)
{
	float d = D(N, H, alpha);
	float g = G(N, V, L, alpha);
	owl::common::vec3f color = diffuse_color;
	owl::common::vec3f f = F(L, H, color);
	owl::common::vec3f brdf = d * g * f;
	brdf = brdf / (4 * clampDot(N, H, false) * clampDot(N, L, false));

	return brdf;
}