#pragma once
#include <utils.cuh>
#include <frostbite.cuh>
#include <helper_math.cuh>

__device__
owl::common::vec3f evaluate(owl::common::vec3f& wi, owl::common::vec3f& wo,
	owl::common::vec3f& base_color, float alpha, owl::common::vec3f emittance)
{
	float metalness = 1.;
	float eta = 0.f;
	float alpha2 = alpha * alpha;
	owl::common::vec3f f0 = base_color;  // lerp between schlickf from eta

	owl::common::vec3f diffuse = diffuse_Lambert(wi, wo, base_color);
	owl::common::vec3f specular = microfacetReflection_GGX(wi, wo, f0, eta, alpha2);

	float diffuseweight = 0.5f;
	float specularweight = 0.5f;

	return diffuseweight * diffuse + specularweight * specular;
}


__device__
void computeLobeProbabilities(owl::common::vec3f& wo, float& pDiffuse, float& pSpecular, owl::common::vec3f base_colors,
	float metalness)
{
	float eta = 0.;
	owl::common::vec3f f0 = base_colors;
	owl::common::vec3f fresnel = Fr_Schlick(abs(cosTheta(wo)), f0);

	float diffuseWeight = 0.5;
	
	pDiffuse = max(base_colors.x, max(base_colors.y, base_colors.z)) * diffuseWeight;
	pSpecular = max(f0.x, max(f0.y, f0.z));

	float normFactor = 1.0f / (pDiffuse + pSpecular);
	pDiffuse *= normFactor;
	pSpecular *= normFactor;
}

__device__
float remap(float value, float low1, float high1, float low2, float high2) {
	float remapped = low2 + (value - low1) * (high2 - low2) / (high1 - low1);
	return clamp(remapped, low2, high2);
}

__device__
owl::common::vec3f sample_direction(owl::common::vec3f& wo, float u1, float u2, float* pdf, 
	owl::common::vec3f base_colors, float metalness, float alpha)
{
	float pDiffuse, pSpecular;

	computeLobeProbabilities(wo, pDiffuse, pSpecular, base_colors, metalness);
	owl::common::vec3f wi;
	pDiffuse = 1.f;
	if (u1 < pDiffuse)
	{
		u1 = remap(u1,
			0.0f, pDiffuse - EPS,
			0.0f, 1 - EPS);
		wi = copysign(1.f, cosTheta(wo)) * sampleCosineHemisphere(u1, u2);
		wi = owl::common::normalize(wi);
	}
	else
	{
		float low1 = 0.0, low2 = 0.0;
		float high1 = pDiffuse - EPS, high2 = 1 - EPS;
		u1 = remap(u1,
			pDiffuse, pDiffuse + pSpecular - EPS,
			0.0f, 1 - EPS);

		owl::common::vec3f wo_upper = (float)copysign(1., cosTheta(wo)) * wo; // sign(+wo) * +wo = +wo, sign(-wo) * -wo = +wo
		owl::common::vec3f wh = (float)copysign(1., cosTheta(wo)) * sampleGGX_VNDF(wo_upper, alpha, u1, u2);
		if (dot(wo, wh) < 0.0f) {
			return owl::common::vec3f(0.0f);
		}
		// reflect
		wi = 2.0f * wh * dot(wh, wo) - wh;
		if (!sameHemisphere(wi, wo)) {
			return owl::common::vec3f(0.0f);
		}
	}
	if (pdf)
	{
		*pdf = pDiffuse * pdfCosineHemisphere(wi, wo)
			+ pSpecular * pdfGGX_VNDF_reflection(wi, wo, alpha);
	}
	return wi;
}

__device__
float pdf(owl::common::vec3f& wi, owl::common::vec3f& wo,
	owl::common::vec3f base_colors, float metalness, float alpha)
{
	float eta = 1.f;

	float pDiffuse, pSpecular;
	computeLobeProbabilities(wo, pDiffuse, pSpecular, base_colors, metalness);

	return pDiffuse * pdfCosineHemisphere(wi, wo)
		+ pSpecular * pdfGGX_VNDF_reflection(wi, wo, alpha);
}