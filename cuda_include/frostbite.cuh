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


__device__ float cosTheta(const owl::common::vec3f& w) { return w.z; }
__device__ float cosTheta2(const owl::common::vec3f& w) { return w.z * w.z; }
__device__ float sinTheta2(const owl::common::vec3f& w) { return max(0.0f, 1.0f - cosTheta2(w)); }
__device__ float sinTheta(const owl::common::vec3f& w) { return std::sqrt(sinTheta2(w)); }
__device__ float tanTheta(const owl::common::vec3f& w) { return sinTheta(w) / cosTheta(w); }
__device__ float tanTheta2(const owl::common::vec3f& w) { return sinTheta2(w) / cosTheta2(w); }

__device__ bool sameHemisphere(const owl::common::vec3f& w, const owl::common::vec3f& wp) {
    return w.z * wp.z > 0.0f;
}


__device__ owl::common::vec3f schlickF0FromRelativeIOR(float eta) {
    float a = (1 - eta) / (1 + eta);
    return owl::common::vec3f(a * a);
}

__device__ owl::common::vec3f Fr_Schlick(float cosThetaI, const owl::common::vec3f& f0) {
    float a = max(0.0f, 1.0f - cosThetaI);
    float a2 = a * a;
    float a5 = a2 * a2 * a;
    return f0 + (owl::common::vec3f(1.0f) - f0) * a5;
}

__device__ float D_GGX(const owl::common::vec3f& wh, float alpha) {
    float alpha2 = alpha * alpha;
    float a = 1.0f + cosTheta2(wh) * (alpha2 - 1.0f);
    return alpha2 / ((float)PI *a * a);
}

__device__ float G1_Smith_GGX(const owl::common::vec3f& w, float alpha) {
    float tan2ThetaW = tanTheta2(w);
    if (tan2ThetaW > 1e5) return 0.0f;
    float alpha2 = alpha * alpha;
    assert(alpha2 * tan2ThetaW >= -1.0f);
    float lambda = (-1.0f + std::sqrt(alpha2 * tan2ThetaW + 1.0f)) / 2.0f;
    return 1.0f / (1.0f + lambda);
}

__device__ float G2_SmithUncorrelated_GGX(const owl::common::vec3f& wi, const owl::common::vec3f& wo, float alpha) {
    return G1_Smith_GGX(wi, alpha) * G1_Smith_GGX(wo, alpha);
}

__device__ float G2_SmithHeightCorrelated_GGX(const owl::common::vec3f& wi, const owl::common::vec3f& wo, float alpha) {
    float tan2ThetaO = tanTheta2(wo);
    float tan2ThetaI = tanTheta2(wi);
    if (tan2ThetaO > 1e-5 || tan2ThetaI > 1e-5) return 0;
    float alpha2 = alpha * alpha;
    assert(alpha2 * tan2ThetaO >= -1.0f);
    assert(alpha2 * tan2ThetaI >= -1.0f);
    float lambda_wo = (-1.0f + std::sqrt(alpha2 * tan2ThetaO + 1.0f)) / 2.0f;
    float lambda_wi = (-1.0f + std::sqrt(alpha2 * tan2ThetaI + 1.0f)) / 2.0f;
    return 1.0f / (1.0f + lambda_wo + lambda_wi);
}

__device__ float G2_None(const owl::common::vec3f& wi, const owl::common::vec3f& wo, float alpha) {
    return 1.0f;
}


// BxDF functions
__device__ owl::common::vec3f diffuse_Lambert(const owl::common::vec3f& wi, const owl::common::vec3f& wo, const owl::common::vec3f& diffuseColor) {
    if (!sameHemisphere(wi, wo)) {
        return owl::common::vec3f(0.0f);
    }

    return diffuseColor / (float)PI;
}

__device__ owl::common::vec3f microfacetReflection_GGX(const owl::common::vec3f& wi, const owl::common::vec3f& wo,
    const owl::common::vec3f& f0, float eta, float alpha) {
    if (!sameHemisphere(wi, wo) || cosTheta(wi) == 0.0f || cosTheta(wo) == 0.0f) {
        return owl::common::vec3f(0.0f);
    }

    owl::common::vec3f wh = wi + wo;
    if (wh.x == 0.0f && wh.y == 0.0f && wh.z == 0.0f) {
        return owl::common::vec3f(0.0f);
    }
    wh = normalize(wh);

    owl::common::vec3f F;
    if (eta < 1.0f) {
        float cosThetaT = dot(wi, wh);
        float cos2ThetaT = cosThetaT * cosThetaT;
        F = cos2ThetaT > 0.0f ? Fr_Schlick(abs(cosThetaT), f0) : owl::common::vec3f(1.0f);
    }
    else {
        F = Fr_Schlick(abs(dot(wh, wo)), f0);
    }

    float G = G2_SmithHeightCorrelated_GGX(wi, wo, alpha);
    float D = D_GGX(wh, alpha);
    return F * G * D / (4.0f * abs(cosTheta(wi)) * abs(cosTheta(wo)));
}

__device__ owl::common::vec3f microfacetTransmission_GGX(const owl::common::vec3f& wi, const owl::common::vec3f& wo, const owl::common::vec3f& f0, float eta, float alpha) {
    if (sameHemisphere(wi, wo) || cosTheta(wi) == 0.0f || cosTheta(wo) == 0.0f) {
        return owl::common::vec3f(0.0f);
    }

    owl::common::vec3f wh = normalize(wi + eta * wo);
    if (cosTheta(wh) < 0.0f) {
        wh = -wh;
    }

    bool sameSide = dot(wo, wh) * dot(wi, wh) > 0.0f;
    if (sameSide) {
        return owl::common::vec3f(0.0f);
    }

    owl::common::vec3f F;
    if (eta < 1.0f) {
        float cosThetaT = dot(wi, wh);
        float cos2ThetaT = cosThetaT * cosThetaT;
        F = cos2ThetaT > 0.0f ? Fr_Schlick(abs(cosThetaT), f0) : owl::common::vec3f(1.0f);
    }
    else {
        F = Fr_Schlick(abs(dot(wh, wo)), f0);
    }

    float G = G2_SmithHeightCorrelated_GGX(wi, wo, alpha);
    float D = D_GGX(wh, alpha);
    float denomSqrt = dot(wi, wh) + eta * dot(wo, wh);
    return (owl::common::vec3f(1.0f) - F) * D * G * abs(dot(wi, wh)) * abs(dot(wo, wh))
        / (denomSqrt * denomSqrt * abs(cosTheta(wi)) * abs(cosTheta(wo)));
}


__device__ owl::common::vec3f sampleUniformSphere(float u1, float u2) {
    float cosTheta = 1.0f - 2.0f * u1;
    float sinTheta = std::sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * (float)PI *u2;
    return owl::common::vec3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}

__device__ float pdfUniformSphere(const owl::common::vec3f& wi, const owl::common::vec3f& wo) {
    return 1.0f / (4.0f * (float)PI);
}


__device__ owl::common::vec3f sampleCosineHemisphere(float u1, float u2) {
    float cosTheta = std::sqrt(max(0.0f, 1.0f - u1));
    float sinTheta = std::sqrt(u1);
    float phi = 2.0f * (float)PI *u2;
    return owl::common::vec3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
}

__device__ float pdfCosineHemisphere(const owl::common::vec3f& wi, const owl::common::vec3f& wo) {
    return sameHemisphere(wi, wo) ? cosTheta(wi) / (float)PI : 0.0f;
}


__device__ owl::common::vec3f sampleGGX(float alpha, float u1, float u2) {
    float phi = 2.0f * (float)PI * u1;
    float cosTheta2 = (1.0f - u2) / ((alpha * alpha - 1.0f) * u2 + 1.0f);
    float sinTheta = std::sqrt(max(0.0f, 1.0f - cosTheta2));
    owl::common::vec3f wh(sinTheta * std::cos(phi), sinTheta * std::sin(phi), std::sqrt(cosTheta2));
    return wh;
}

__device__ float pdfGGX_reflection(const owl::common::vec3f& wi, const owl::common::vec3f& wo, float alpha) {
    if (!sameHemisphere(wi, wo)) {
        return 0.0f;
    }

    owl::common::vec3f wh = normalize(wi + wo);
    float pdf_h = D_GGX(wh, alpha) * abs(cosTheta(wh));
    float dwh_dwi = 1.0f / (4.0f * dot(wi, wh));
    return pdf_h * dwh_dwi;
}

__device__ float pdfGGX_transmission(const owl::common::vec3f& wi, const owl::common::vec3f& wo, float eta, float alpha) {
    if (sameHemisphere(wi, wo)) {
        return 0.0f;
    }

    owl::common::vec3f wh = normalize(wi + eta * wo);
    bool sameSide = dot(wo, wh) * dot(wi, wh) > 0.0f;
    if (sameSide) return 0.0f;

    float pdf_h = D_GGX(wh, alpha) * abs(cosTheta(wh));
    float sqrtDenom = dot(wi, wh) + eta * dot(wo, wh);
    float dwh_dwi = abs(dot(wi, wh)) / (sqrtDenom * sqrtDenom);
    return pdf_h * dwh_dwi;
}


// See: http://jcgt.org/published/0007/04/01/paper.pdf
__device__ owl::common::vec3f sampleGGX_VNDF(const owl::common::vec3f& wo, float alpha, float u1, float u2) {
    // Transform view direction to hemisphere configuration
    owl::common::vec3f woHemi = normalize(owl::common::vec3f(alpha * wo.x, alpha * wo.y, wo.z));

    // Create orthonormal basis
    float length2 = woHemi.x * woHemi.x + woHemi.y * woHemi.y;
    owl::common::vec3f b1 = length2 > 0.0f
        ? owl::common::vec3f(-woHemi.y, woHemi.x, 0.0f) * (1.0f / std::sqrt(length2))
        : owl::common::vec3f(1.0f, 0.0f, 0.0f);
    owl::common::vec3f b2 = cross(woHemi, b1);

    // Parameterization of projected area
    float r = std::sqrt(u1);
    float phi = 2.0f * (float)PI *u2;
    float t1 = r * std::cos(phi);
    float t2 = r * std::sin(phi);
    float s = 0.5f * (1.0f + woHemi.z);
    t2 = (1.0f - s) * std::sqrt(1.0f - t1 * t1) + s * t2;

    // Reprojection onto hemisphere
    owl::common::vec3f whHemi = t1 * b1 + t2 * b2 + (float)std::sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * woHemi;

    // Transforming half vector back to ellipsoid configuration
    return normalize(owl::common::vec3f(alpha * whHemi.x, alpha * whHemi.y, max(0.0f, whHemi.z)));
}

__device__ float pdfGGX_VNDF_reflection(const owl::common::vec3f& wi, const owl::common::vec3f& wo, float alpha) {
    if (!sameHemisphere(wi, wo)) {
        return 0.0f;
    }

    owl::common::vec3f wh = normalize(wi + wo);
    float pdf_h = G1_Smith_GGX(wo, alpha) * D_GGX(wh, alpha) * abs(dot(wh, wo)) / abs(cosTheta(wo));
    float dwh_dwi = 1.0f / (4.0f * dot(wi, wh));
    return pdf_h * dwh_dwi;
}

__device__ float pdfGGX_VNDF_transmission(const owl::common::vec3f& wi, const owl::common::vec3f& wo, float eta, float alpha) {
    if (sameHemisphere(wi, wo)) {
        return 0.0f;
    }

    owl::common::vec3f wh = normalize(wi + eta * wo);
    bool sameSide = dot(wo, wh) * dot(wi, wh) > 0.0f;
    if (sameSide) return 0.0f;

    float pdf_h = G1_Smith_GGX(wo, alpha) * D_GGX(wh, alpha) * abs(dot(wh, wo)) / abs(cosTheta(wo));
    float sqrtDenom = dot(wi, wh) + eta * dot(wo, wh);
    float dwh_dwi = abs(dot(wi, wh)) / (sqrtDenom * sqrtDenom);
    return pdf_h * dwh_dwi;
}