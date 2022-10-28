#pragma once
#include <common.h>
#include <helper_math.cuh>
#include <cuda_runtime.h>
#include <owl/common/math/vec.h>

__device__
float G2(float alpha, owl::common::vec3f V, owl::common::vec3f L) {
    float alphasq = alpha * alpha;

    float cosV = V.z;
    float cosV2 = cosV * cosV;

    float cosL = L.z;
    float cosL2 = cosL * cosL;

    float num = 2.f * cosV * cosL;
    float denom = cosV * owl::sqrt(alphasq + (1.f - alphasq) * cosL2) + cosL * owl::sqrt(alphasq + (1.f - alphasq) * cosV2);

    return num / denom;
}

__device__
float D(float alpha, owl::common::vec3f H) {
    float alphasq = alpha * alpha;

    float num = alphasq;
    float denom = PI * pow(H.z * H.z * (alphasq - 1.f) + 1, 2.f);

    return num / denom;
}

__device__
float GGX(float alpha, owl::common::vec3f V, owl::common::vec3f L) {
    owl::common::vec3f H = normalize(V + L);
    float value = D(alpha, H) * G2(alpha, V, L) / 4.0f / V.z / L.z;

    return value;
}

/*! Evaluates the full BRDF with both diffuse and specular terms.
    The specular BRDF is GGX specular (Taken from Eric Heitz's JCGT paper).
    Fresnel is not used (commented).
    Evaluates only f (i.e. BRDF without cosine foreshortening) */
__device__
owl::common::vec3f evaluate_brdf(owl::common::vec3f wo, owl::common::vec3f wi, owl::common::vec3f diffuse_color, float alpha) {
    owl::common::vec3f brdf = owl::common::vec3f(0.0f);

    // Diffuse + specular
    brdf += diffuse_color / owl::common::vec3f(PI);
    brdf += GGX(alpha, wo, wi);

    return brdf;
}

__device__
float get_brdf_pdf(float alpha, owl::common::vec3f V, owl::common::vec3f Ne) {
    float cosT = Ne.z;
    float alphasq = alpha * alpha;

    float num = alphasq * cosT;
    float denom = PI * pow((alphasq - 1.f) * cosT * cosT + 1.f, 2.f);

    float pdf = num / denom;
    return pdf / (4.f * dot(V, Ne));
}

__device__
owl::common::vec3f sample_GGX(owl::common::vec2f rand, float alpha, owl::common::vec3f V) {
    float num = 1.f - rand.x;
    float denom = rand.x * (alpha * alpha - 1.f) + 1;
    float t = acos(owl::sqrt(num / denom));
    float p = 2.f * PI * rand.y;

    owl::common::vec3f N(sin(t) * cos(p), sin(t) * sin(p), cos(t));
    owl::common::vec3f L = -V + 2.0f * N * dot(V, N);

    return normalize(L);
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
// owl::common::vec3f sample_GGX(owl::common::vec2f rand, float alpha, owl::common::vec3f V) {
//     return CosineSampleHemisphere(rand);
// }