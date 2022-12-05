#pragma once
#include <common.h>
#include <helper_math.cuh>
#include <cuda_runtime.h>
#include <owl/common/math/vec.h>
#include <utils.cuh>

#define EPS float(1e-5)


// ALL ARE IN GLOBAL

__device__
float clampDot(owl::common::vec3f a, owl::common::vec3f b, bool zero) {
    return max(owl::common::dot(a, b), zero ? 0 : EPS);
}

__device__
float ggxNormalDistribution(float NdotH, float roughness)
{
    float a2 = roughness * roughness;
    float d = ((NdotH * a2 - NdotH) * NdotH + 1);
    return a2 / max(0.001f, (d * d * M_PI));
}

__device__
float schlickMaskingTerm(float NdotL, float NdotV, float roughness)
{
    // Karis notes they use alpha / 2 (or roughness^2 / 2)
    float k = roughness * roughness / 2;

    // Karis also notes they can use the following equation, but only for analytical lights
    //float k = (roughness + 1)*(roughness + 1) / 8; 

    // Compute G(v) and G(l).  These equations directly from Schlick 1994
    //     (Though note, Schlick's notation is cryptic and confusing.)
    float g_v = NdotV / (NdotV * (1 - k) + k);
    float g_l = NdotL / (NdotL * (1 - k) + k);

    // Return G(v) * G(l)
    return g_v * g_l;
}

__device__
owl::common::vec3f schlickFresnel(owl::common::vec3f f0, float lDotH)
{
    return f0 + (owl::common::vec3f(1.0f, 1.0f, 1.0f) - f0) * pow(1.0f - lDotH, 5.0f);
}


/*! Evaluates the full BRDF with both diffuse and specular terms.
    The specular BRDF is GGX specular (Taken from Eric Heitz's JCGT paper).
    Fresnel is not used (commented).
    Evaluates only f (i.e. BRDF without cosine foreshortening) */
__device__
owl::common::vec3f evaluate_brdf(owl::common::vec3f V, owl::common::vec3f N, owl::common::vec3f L, 
    owl::common::vec3f diffuse_color, float alpha, owl::common::vec3f f0) {
    owl::common::vec3f brdf = owl::common::vec3f(0.0f);

    owl::common::vec3f H = owl::common::normalize(V + L);
    float NdotH = saturate(owl::common::dot(N, H));
    float NdotV = saturate(owl::common::dot(N, V));
    float NdotL = saturate(owl::common::dot(N, L));
    float LdotH = saturate(owl::common::dot(L, H));

    float  D = ggxNormalDistribution(NdotH, alpha);
    float  G = schlickMaskingTerm(NdotL, NdotV, alpha);
    owl::common::vec3f F = schlickFresnel(f0, LdotH);
    brdf = D * G * F / (4 * NdotL * NdotV);

    // Diffuse pdf == 1 /PI; Use metalicity for weighing metalicity = 0.5
    brdf += diffuse_color * float(1 / PI);
    return brdf / 2.f;
}

__device__
float get_brdf_pdf(float alpha, owl::common::vec3f V, owl::common::vec3f N, owl::common::vec3f H, owl::common::vec3f L) {
    /*float cosT = Ne.z;
    float alphasq = alpha * alpha;

    float num = alphasq * cosT;
    float denom = PI * pow((alphasq - 1.f) * cosT * cosT + 1.f, 2.f);

    float pdf = num / denom;
    return pdf / (4.f * clampDot(V, Ne, false));*/
    float NdotH = owl::common::clamp(owl::common::dot(N, H), 0.f, 1.f);
    float HdotV = owl::common::clamp(owl::common::dot(H, V), 0.f, 1.f);
    float LdotH = owl::common::clamp(owl::common::dot(L, V), 0.f, 1.f);

    float D = ggxNormalDistribution(NdotH, alpha);

    //Evaluate this
    return D * NdotH / (4.f * HdotV);
}


// Picked and altered from ChrisWaymann's Tutorial14 Siggraph 21
__device__
owl::common::vec3f sample_GGX(owl::common::vec2f rand, float alpha, owl::common::vec3f hitNorm)
{
    owl::common::vec2f randVal = rand;

    // Get an orthonormal basis from the normal
    // reflection -- assume 0.5
    // f0 = 0.16f * reflectance *  reflectance  * (float3(1.0f, 1.0f, 1.0f)  - metalness) + albedo * metalness;
    //check whether B T N form a right handed system. also check normalization normalize and then check right handed or not.
    // Get an orthonormal basis from the normal
    owl::common::vec3f B = getPerpendicularVector(hitNorm);
    owl::common::vec3f T = owl::common::cross(B, hitNorm);

    // GGX NDF sampling
    float a2 = alpha * alpha;
    float cosThetaH = sqrt(max(0.0f, (1.0 - randVal.x) / ((a2 - 1.0) * randVal.x + 1)));
    float sinThetaH = sqrt(max(0.0f, 1.0f - cosThetaH * cosThetaH));
    float phiH = randVal.y * M_PI * 2.0f;

    // Get our GGX NDF sample (i.e., the half vector)
    return T * (sinThetaH * cos(phiH)) + B * (sinThetaH * sin(phiH)) + hitNorm * cosThetaH;
}

// __device__
// float get_brdf_pdf(float alpha, owl::common::vec3f V, owl::common::vec3f Ne) {
//     owl::common::vec3f L = -V + 2.0f * Ne * dot(V, Ne);
//     L = normalize(L);
// 
//     return L.z / PI;
// }
// 
// __device__
// owl::common::vec3f sample_GGX(vec2f rand, float alpha, owl::common::vec3f V) {
//     return CosineSampleHemisphere(rand);
// }