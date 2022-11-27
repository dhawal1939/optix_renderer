#pragma once
#include <common.h>
#include <helper_math.cuh>
#include <cuda_runtime.h>
#include <owl/common/math/vec.h>

#define EPS float(1e-5)


__device__
float D(owl::common::vec3f N, owl::common::vec3f H, float roughness) {
	float a2 = roughness * roughness;
	float NdotH = owl::common::max(owl::common::dot(N, H), 0.f);
	float denom = ((NdotH * NdotH) * (a2 * a2 - 1) + 1);
	return a2 / (PI * denom * denom);
}


__device__
float G(owl::common::vec3f N, owl::common::vec3f V, owl::common::vec3f L, float roughness) {
	float NdotV = owl::common::max(owl::common::dot(N, V), EPS);
	float NdotL = owl::common::max(owl::common::dot(N, L), EPS);
	float k = roughness * roughness / 2;
	float G_V = NdotV / (NdotV * (1.0 - k) + k);
	float G_L = NdotL / (NdotL * (1.0 - k) + k);

	return G_V * G_L;
}


__device__
owl::common::vec3f F(owl::common::vec3f L, owl::common::vec3f H, owl::common::vec3f F0) {
	float LdotH = owl::common::max(owl::common::dot(L, H), 0.f);
	return F0 + (1.0f - F0) * pow(1.0f - LdotH, 5.0f);
}

__device__
owl::common::vec3f sample_GGX(owl::common::vec2f rng, float alpha, owl::common::vec3f V) {
	// GGX NDF sampling
	float a2 = alpha * alpha;
	float cosThetaH = sqrt(max(0.0, (1.0 - rng.x) / ((a2 - 1.0) * rng.x + 1.0)));
	float sinThetaH = sqrt(max(0.0, 1.0 - cosThetaH * cosThetaH));
	float phiH = rng.y * PI * 2.0;

	// Get our GGX NDF sample (i.e., the half vector)
	owl::common::vec3f H = owl::common::vec3f(sinThetaH * cos(phiH), 
											  cosThetaH, 
											  sinThetaH * sin(phiH));

	owl::common::vec3f L = -V + 2.0f * H * owl::common::dot(V, H);

	return normalize(L);
}

__device__
float get_brdf_pdf(float alpha, owl::common::vec3f L,
	owl::common::vec3f Ne) {
	owl::common::vec3f N = owl::common::vec3f(0., 0., 1.), H = Ne;
	return D(N, H, alpha) * owl::common::max(owl::common::dot(N, H), 0.f) /
		(4 * owl::common::max(owl::common::dot(H, L), EPS));
}

__device__
owl::common::vec3f evaluate_brdf(owl::common::vec3f V, // wo
	owl::common::vec3f L, // wi
	owl::common::vec3f diffuse_color, //
	float alpha) {
	owl::common::vec3f N = owl::common::vec3f(0., 0., 1.); // Local space
	owl::common::vec3f H = owl::common::normalize(V + L);
	float d = D(N, H, alpha);
	float g = G(N, V, L, alpha);

	
	owl::common::vec3f f = F(L, H, diffuse_color);
	owl::common::vec3f brdf = d * g * f;
	brdf = brdf / (4 * owl::common::max(owl::common::dot(N, H), EPS) *
		owl::common::max(owl::common::dot(N, L), EPS));
	return f;
	return brdf;
}
