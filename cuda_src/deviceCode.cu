// // ======================================================================== //
// // Copyright 2019-2020 Ingo Wald                                            //
// //                                                                          //
// // Licensed under the Apache License, Version 2.0 (the "License");          //
// // you may not use this file except in compliance with the License.         //
// // You may obtain a copy of the License at                                  //
// //                                                                          //
// //     http://www.apache.org/licenses/LICENSE-2.0                           //
// //                                                                          //
// // Unless required by applicable law or agreed to in writing, software      //
// // distributed under the License is distributed on an "AS IS" BASIS,        //
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// // See the License for the specific language governing permissions and      //
// // limitations under the License.                                           //
// // ======================================================================== //

#include <utils.cuh>
#include <common.cuh>
#include <frostbite.cuh>
#include <lcg_random.cuh>
#include <optix_device.h>
#include <owl/owl_device.h>
#include <ltc/ltc_utils.cuh>
#include <ltc/polygon_utils.cuh>

#include <vector_types.h>
#include <texture_fetch_functions.h>




OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCHShadow)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    const owl::common::vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];
    ShadowRayData& srd = owl::getPRD<ShadowRayData>();

    if (self.isLight) {
        srd.visibility = owl::common::vec3f(1.f);
        srd.point = barycentricInterpolate(self.vertex, primitiveIndices);
        srd.normal = normalize(barycentricInterpolate(self.normal, primitiveIndices));
        srd.emit = self.emit;

        owl::common::vec3f v1 = self.vertex[primitiveIndices.x];
        owl::common::vec3f v2 = self.vertex[primitiveIndices.y];
        owl::common::vec3f v3 = self.vertex[primitiveIndices.z];
        srd.area = 0.5f * length(cross(v1 - v2, v3 - v2));

        srd.cg = (v1 + v2 + v3) / 3.f;
    }
    else {
        srd.visibility = owl::common::vec3f(0.f);
        srd.point = barycentricInterpolate(self.vertex, primitiveIndices);
        srd.normal = normalize(barycentricInterpolate(self.normal, primitiveIndices));
        srd.emit = owl::common::vec3f(0.);

        owl::common::vec3f v1 = self.vertex[primitiveIndices.x];
        owl::common::vec3f v2 = self.vertex[primitiveIndices.y];
        owl::common::vec3f v3 = self.vertex[primitiveIndices.z];
        srd.area = 0.5f * length(cross(v1 - v2, v3 - v2));

        srd.cg = (v1 + v2 + v3) / 3.f;
    }

}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    const owl::common::vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();

    // Exact hit point on the triangle
    si.p = barycentricInterpolate(self.vertex, primitiveIndices);

    // Out going direction pointing toward the pixel location
    si.wo = owl::normalize(optixLaunchParams.camera.pos - si.p);

    // UV coordinate of the hit point
    si.uv = barycentricInterpolate(self.texCoord, primitiveIndices);

    // geometric normal 
    si.n_geom = normalize(barycentricInterpolate(self.normal, primitiveIndices));

    // Initializes to_local from n_geo then obtains to_world by taking inverse of the to_local
    orthonormalBasis(si.n_geom, si.to_local, si.to_world);

    // obtain wo is in world space cam_pos - hit_loc_world get local frame of the wo as wo_local
    si.wo_local = normalize(apply_mat(si.to_local, si.wo));

    // axix independet prop
    si.diffuse = self.diffuse;
    if (self.hasDiffuseTexture)
        si.diffuse = (owl::common::vec3f)tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);

    si.alpha = self.alpha;
    if (self.hasAlphaTexture)
        si.alpha = tex2D<float4>(self.alpha_texture, si.uv.x, si.uv.y).x;
    si.alpha = clamp(si.alpha, 0.01f, 1.f);

    si.emit = self.emit;
    si.isLight = self.isLight;

    si.hit = true;
}

OPTIX_MISS_PROGRAM(miss)()
{
    const owl::common::vec2i pixelId = owl::getLaunchIndex();
    const MissProgData& self = owl::getProgramData<MissProgData>();

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
    si.hit = false;
    si.diffuse = self.const_color;
}


__device__
owl::common::vec3f integrateOverPolygon(SurfaceInteraction& si, owl::common::vec3f ltc_mat[3], owl::common::vec3f ltc_mat_inv[3], float amplitude,
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

__device__
owl::common::vec3f sampleLightSource(SurfaceInteraction si, int lightIdx, float lightSelectionPdf, owl::common::vec2f rand, bool mis)
{
    owl::common::vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;
    TriLight triLight = optixLaunchParams.triLights[lightIdx];

    owl::common::vec3f lv1 = triLight.v1;
    owl::common::vec3f lv2 = triLight.v2;
    owl::common::vec3f lv3 = triLight.v3;
    owl::common::vec3f lnormal = triLight.normal;
    owl::common::vec3f lemit = triLight.emit;
    float larea = triLight.area;

    owl::common::vec3f lpoint = samplePointOnTriangle(lv1, lv2, lv3, rand.x, rand.y);
    owl::common::vec3f wi = normalize(lpoint - si.p);
    owl::common::vec3f wi_local = normalize(apply_mat(si.to_local, wi));

    float xmy = pow(owl::length(lpoint - si.p), 2.f);
    float lDotWi = owl::abs(owl::dot(lnormal, -wi));

    light_pdf = lightSelectionPdf * (xmy / (larea * lDotWi));

    ShadowRay ray;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = wi;

    ShadowRayData srd;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    if (si.wo_local.z > 0.f && wi_local.z > 0.f && srd.visibility != owl::common::vec3f(0.f) && light_pdf > 0.f && owl::dot(-wi, lnormal) > 0.f) {
        owl::common::vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        if (mis && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, light_pdf, 1, brdf_pdf);
            color += brdf * lemit * owl::abs(wi_local.z) * weight / light_pdf;
        }
        else if (!mis) {
            color += brdf * lemit * owl::abs(wi_local.z) / light_pdf;
        }
    }

    return color;
}

__device__
owl::common::vec3f sampleLightSourceNoNLTest(SurfaceInteraction si, int lightIdx, float lightSelectionPdf, owl::common::vec2f rand, bool mis)
{
    owl::common::vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;
    TriLight triLight = optixLaunchParams.triLights[lightIdx];

    owl::common::vec3f lv1 = triLight.v1;
    owl::common::vec3f lv2 = triLight.v2;
    owl::common::vec3f lv3 = triLight.v3;
    owl::common::vec3f lnormal = triLight.normal;
    owl::common::vec3f lemit = triLight.emit;
    float larea = triLight.area;

    owl::common::vec3f lpoint = samplePointOnTriangle(lv1, lv2, lv3, rand.x, rand.y);
    owl::common::vec3f wi = normalize(lpoint - si.p);
    owl::common::vec3f wi_local = normalize(apply_mat(si.to_local, wi));

    float xmy = pow(owl::length(lpoint - si.p), 2.f);
    float lDotWi = owl::abs(owl::dot(lnormal, -wi));

    light_pdf = lightSelectionPdf * (xmy / (larea * lDotWi));

    ShadowRay ray;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = wi;

    ShadowRayData srd;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    if (si.wo_local.z > 0.f && wi_local.z > 0.f && srd.visibility != owl::common::vec3f(0.f) && light_pdf > 0.f) {
        owl::common::vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        if (mis && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, light_pdf, 1, brdf_pdf);
            color += brdf * lemit * owl::abs(wi_local.z) * weight / light_pdf;
        }
        else if (!mis) {
            color += brdf * lemit * owl::abs(wi_local.z) / light_pdf;
        }
    }

    return color;
}

__device__
owl::common::vec3f sampleLightSourceNoVis(SurfaceInteraction si, int lightIdx, float lightSelectionPdf, owl::common::vec2f rand, bool mis)
{
    owl::common::vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;
    TriLight triLight = optixLaunchParams.triLights[lightIdx];

    owl::common::vec3f lv1 = triLight.v1;
    owl::common::vec3f lv2 = triLight.v2;
    owl::common::vec3f lv3 = triLight.v3;
    owl::common::vec3f lnormal = triLight.normal;
    owl::common::vec3f lemit = triLight.emit;
    float larea = triLight.area;

    owl::common::vec3f lpoint = samplePointOnTriangle(lv1, lv2, lv3, rand.x, rand.y);
    owl::common::vec3f wi = normalize(lpoint - si.p);
    owl::common::vec3f wi_local = normalize(apply_mat(si.to_local, wi));

    float xmy = pow(owl::length(lpoint - si.p), 2.f);
    float lDotWi = owl::abs(owl::dot(lnormal, -wi));

    light_pdf = lightSelectionPdf * (xmy / (larea * lDotWi));

    ShadowRay ray;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = wi;

    ShadowRayData srd;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    if (si.wo_local.z > 0.f && wi_local.z > 0.f && light_pdf > 0.f) {
        owl::common::vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        if (mis && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, light_pdf, 1, brdf_pdf);
            color += brdf * lemit * owl::abs(wi_local.z) * weight / light_pdf;
        }
        else if (!mis) {
            color += brdf * lemit * owl::abs(wi_local.z) / light_pdf;
        }
    }

    return color;
}

__device__
owl::common::vec3f sampleBRDF(SurfaceInteraction si, float lightSelectionPdf, owl::common::vec2f rand, bool mis)
{
    owl::common::vec3f wi_local = sample_GGX(rand, si.alpha, si.wo_local);
    owl::common::vec3f wi = normalize(apply_mat(si.to_world, wi_local));

    ShadowRay ray;
    ShadowRayData srd;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = wi;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    owl::common::vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;

    if (wi_local.z > 0.f && si.wo_local.z > 0.f && srd.visibility != owl::common::vec3f(0.f)) {
        float xmy = pow(owl::length(srd.point - si.p), 2.f);
        float lDotWi = owl::abs(owl::dot(srd.normal, -wi));
        light_pdf = lightSelectionPdf * (xmy / (srd.area * lDotWi));

        owl::common::vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        if (mis && light_pdf > 0.f && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, brdf_pdf, 1, light_pdf);
            color += brdf * srd.emit * owl::abs(wi_local.z) * weight / brdf_pdf;
        }
        else if (!mis && brdf_pdf > 0.f) {
            color += brdf * srd.emit * owl::abs(wi_local.z) / brdf_pdf;
        }
    }

    return color;
}

__device__
owl::common::vec3f sampleBRDFNoNLTest(SurfaceInteraction si, float lightSelectionPdf, owl::common::vec2f rand, bool mis)
{
    owl::common::vec3f wi_local = sample_GGX(rand, si.alpha, si.wo_local);
    owl::common::vec3f wi = normalize(apply_mat(si.to_world, wi_local));

    ShadowRay ray;
    ShadowRayData srd;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = wi;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    owl::common::vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;

    if (srd.visibility != owl::common::vec3f(0.f)) {
        float xmy = pow(owl::length(srd.point - si.p), 2.f);
        float lDotWi = owl::abs(owl::dot(srd.normal, -wi));
        light_pdf = lightSelectionPdf * (xmy / (srd.area * lDotWi));

        owl::common::vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        if (mis && light_pdf > 0.f && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, brdf_pdf, 1, light_pdf);
            color += brdf * srd.emit * owl::abs(wi_local.z) * weight / brdf_pdf;
        }
        else if (!mis && brdf_pdf > 0.f) {
            color += brdf * srd.emit * owl::abs(wi_local.z) / brdf_pdf;
        }
    }

    return color;
}

__device__
owl::common::vec3f sampleBRDFNoVis(SurfaceInteraction si, float lightSelectionPdf, owl::common::vec2f rand, bool mis)
{
    owl::common::vec3f wi_local = sample_GGX(rand, si.alpha, si.wo_local);
    owl::common::vec3f wi = normalize(apply_mat(si.to_world, wi_local));

    ShadowRay ray;
    ShadowRayData srd;
    ray.origin = si.p + 1e-3f * si.n_geom;
    ray.direction = wi;
    owl::traceRay(optixLaunchParams.world, ray, srd);

    owl::common::vec3f color(0.f, 0.f, 0.f);
    float light_pdf = 0.f, brdf_pdf = 0.f;

    if (true) {
        float xmy = pow(owl::length(srd.point - si.p), 2.f);
        float lDotWi = owl::abs(owl::dot(srd.normal, -wi));
        light_pdf = lightSelectionPdf * (xmy / (srd.area * lDotWi));

        owl::common::vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        if (mis && light_pdf > 0.f && brdf_pdf > 0.f) {
            float weight = PowerHeuristic(1, brdf_pdf, 1, light_pdf);
            color += brdf * srd.emit * owl::abs(wi_local.z) * weight / brdf_pdf;
        }
        else if (!mis && brdf_pdf > 0.f) {
            color += brdf * srd.emit * owl::abs(wi_local.z) / brdf_pdf;
        }
    }

    return color;
}

__device__
owl::common::vec3f estimateDirectLighting(SurfaceInteraction& si, LCGRand& rng, int type)
{
    owl::common::vec2f rand1 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
    owl::common::vec2f rand2 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));

    owl::common::vec3f lightSample = owl::common::vec3f(0.f);
    owl::common::vec3f brdfSample = owl::common::vec3f(0.f);
    owl::common::vec3f color = owl::common::vec3f(0.f);

    if (type == 0) {
        int selectedTriLight = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
        float lightSelectionPdf = 1.f / optixLaunchParams.numTriLights;

        lightSample = sampleLightSource(si, selectedTriLight, lightSelectionPdf, rand1, false);

        color = lightSample;
    }
    else if (type == 1) {
        brdfSample = sampleBRDF(si, 0.f, rand2, false);

        color = brdfSample;
    }
    else if (type == 2) {
        int selectedTriLight = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
        float lightSelectionPdf = 1.f / optixLaunchParams.numTriLights;

        brdfSample = sampleBRDF(si, lightSelectionPdf, rand1, true);
        lightSample = sampleLightSource(si, selectedTriLight, lightSelectionPdf, rand2, true);

        color = brdfSample + lightSample;
    }
    else if (type == 3) {
        int selectedTriLight = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
        float lightSelectionPdf = 1.f / optixLaunchParams.numTriLights;

        brdfSample = sampleBRDFNoNLTest(si, lightSelectionPdf, rand1, true);
        lightSample = sampleLightSourceNoNLTest(si, selectedTriLight, lightSelectionPdf, rand2, true);

        color = brdfSample + lightSample;
    }
    else if (type == 4) {
        int selectedTriLight = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
        float lightSelectionPdf = 1.f / optixLaunchParams.numTriLights;

        brdfSample = sampleBRDFNoVis(si, lightSelectionPdf, rand1, true);
        lightSample = sampleLightSourceNoVis(si, selectedTriLight, lightSelectionPdf, rand2, true);

        color = brdfSample + lightSample;
    }

    // Make sure there are no negative colors!
    color.x = owl::max(0.f, color.x);
    color.y = owl::max(0.f, color.y);
    color.z = owl::max(0.f, color.z);

    return color;
}

__device__
owl::common::vec3f estimatePathTracing(SurfaceInteraction& si, LCGRand& rng)
{
    owl::common::vec3f color = owl::common::vec3f(0.f);
    SurfaceInteraction _dummy_si = si;
    if (si.isLight)
        return si.emit;
    owl::common::vec3f tp(1., 1., 1.);
    for (int ray_depth = 0; ray_depth < 8; ray_depth++)
    {   
        owl::common::vec2f rand1 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        owl::common::vec2f rand2 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));

        int selectedTriLight = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
        float lightSelectionPdf = 1.f / optixLaunchParams.numTriLights;

        owl::common::vec3f light_sample(0.f, 0.f, 0.f), brdf_sample(0., 0., 0.);
        owl::common::vec2i pixel_id = owl::getLaunchIndex();
        bool mis = false; //  CHECk
        // Light sampling
        //{
        //    float light_pdf = 0.f, brdf_pdf = 0.f;
        //    TriLight triLight = optixLaunchParams.triLights[selectedTriLight];

        //    owl::common::vec3f lv1 = triLight.v1;
        //    owl::common::vec3f lv2 = triLight.v2;
        //    owl::common::vec3f lv3 = triLight.v3;
        //    owl::common::vec3f lnormal = triLight.normal;
        //    owl::common::vec3f lemit = triLight.emit;
        //    float larea = triLight.area;

        //    owl::common::vec3f lpoint = samplePointOnTriangle(lv1, lv2, lv3, rand1.x, rand1.y);
        //    owl::common::vec3f wi = normalize(lpoint - si.p);
        //    owl::common::vec3f wi_local = normalize(apply_mat(si.to_local, wi));

        //    float xmy = pow(owl::length(lpoint - si.p), 2.f);
        //    float lDotWi = owl::abs(owl::dot(lnormal, -wi));

        //    light_pdf = lightSelectionPdf * (xmy / (larea * lDotWi));

        //    ShadowRay ray;
        //    ray.origin = si.p + 1e-3f * si.n_geom;
        //    ray.direction = wi;
        //    //if(pixel_id.x == 1024 && pixel_id.y == 1024)
        //    //    printf("light %f, %f, %f\n", si.p.x, si.p.x, si.p.x);
        //    ShadowRayData srd;
        //    owl::traceRay(optixLaunchParams.world, ray, srd);

        //    if (si.wo_local.z > 0.f && wi_local.z > 0.f && srd.visibility != owl::common::vec3f(0.f) && light_pdf > 0.f && owl::dot(-wi, lnormal) > 0.f) {
        //        owl::common::vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
        //        brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));

        //        if (mis && brdf_pdf > 0.f) {
        //            float weight = PowerHeuristic(1, light_pdf, 1, brdf_pdf);
        //            color += tp * brdf * lemit * owl::abs(wi_local.z) * weight / light_pdf;
        //        }
        //        else if (!mis) {
        //            color += tp * brdf * lemit * owl::abs(wi_local.z) / light_pdf;
        //        }
        //    }
        //    //light_sample *= owl::abs(owl::dot(owl::normalize(si.n_geom), owl::normalize(srd.point - si.p)));
        //}
        // BRDF sampling
        float lidotN = 0.;
        float lDotWi = 0.;
        SurfaceInteraction _si;
        {


            owl::common::vec3f wi_local = sample_GGX(rand2, si.alpha, si.wo_local);
            owl::common::vec3f wi = normalize(apply_mat(si.to_world, wi_local));

            ShadowRayData srd;

            ShadowRay shadow_ray;
            RadianceRay rad_ray;
            rad_ray.origin = si.p + 1e-3f * si.n_geom;
            rad_ray.direction = wi;

            /*if (pixel_id.x == 1024 && pixel_id.y == 1024)
                printf("brdf %f, %f, %f\n", si.p.x, si.p.x, si.p.x);
            */
            shadow_ray.origin = si.p + 1e-3f * si.n_geom;
            shadow_ray.direction = wi;
            owl::traceRay(optixLaunchParams.world, rad_ray, _si);
            owl::traceRay(optixLaunchParams.world, shadow_ray, srd);

            float light_pdf = 0.f, brdf_pdf = 0.f;
            
            lDotWi = owl::abs(owl::dot(_si.n_geom, -wi));
            lidotN = max(1e-6, owl::dot(_si.n_geom, -wi));
            owl::common::vec3f brdf = evaluate_brdf(si.wo_local, wi_local, si.diffuse, si.alpha);
            brdf_pdf = get_brdf_pdf(si.alpha, si.wo_local, normalize(si.wo_local + wi_local));
            //return owl::common::vec3f(lDotWi);
            if (wi_local.z > 0.f && si.wo_local.z > 0.f && srd.visibility != owl::common::vec3f(0.f)) {
                float xmy = pow(owl::length(srd.point - si.p), 2.f);

                light_pdf = lightSelectionPdf * (xmy / (srd.area * lDotWi));

                if (mis && light_pdf > 0.f && brdf_pdf > 0.f) {
                    float weight = PowerHeuristic(1, brdf_pdf, 1, light_pdf);
                    color += tp * brdf * srd.emit * owl::abs(wi_local.z) * weight / brdf_pdf;
                }
                else if (!mis && brdf_pdf > 0.f) {
                    color += tp * brdf * srd.emit * owl::abs(wi_local.z) / brdf_pdf;
                }
                break;
            }
            tp *= lidotN * brdf / brdf_pdf;

            if (!_si.hit)
                break;
            si = _si;
        }
    }
    si = _dummy_si;
    return color;
}


__device__
owl::common::vec3f ltcDirectLighingBaseline(SurfaceInteraction& si, LCGRand& rng)
{
    owl::common::vec3f wo_local = normalize(apply_mat(si.to_local, si.wo));
    if (wo_local.z < 0.f)
        return owl::common::vec3f(0.f);

    owl::common::vec3f normal_local(0.f, 0.f, 1.f);
    owl::common::vec3f color(0.0, 0.0, 0.0);

    /* Analytic shading via LTCs */
    owl::common::vec3f ltc_mat[3], ltc_mat_inv[3];
    float alpha = si.alpha;
    float theta = sphericalTheta(wo_local);

    float amplitude = 1.f;
    fetchLtcMat(alpha, theta, ltc_mat, amplitude);
    matrixInverse(ltc_mat, ltc_mat_inv);

    owl::common::vec3f iso_frame[3];

    iso_frame[0] = wo_local;
    iso_frame[0].z = 0.f;
    iso_frame[0] = normalize(iso_frame[0]);
    iso_frame[2] = normal_local;
    iso_frame[1] = normalize(owl::cross(iso_frame[2], iso_frame[0]));

    for (int lidx = 0; lidx < optixLaunchParams.numTriLights; lidx++) {
        color += integrateOverPolygon(si, ltc_mat, ltc_mat_inv, amplitude, iso_frame,
            optixLaunchParams.triLights[lidx]);
    }

    return color;
}


OPTIX_RAYGEN_PROGRAM(rayGen)()
{
    const RayGenData& self = owl::getProgramData<RayGenData>();
    const owl::common::vec2i pixelId = owl::getLaunchIndex();
    const int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;

    LCGRand rng = get_rng(optixLaunchParams.accumId + 10007, make_uint2(pixelId.x, pixelId.y),
        make_uint2(self.frameBufferSize.x, self.frameBufferSize.y));

    const owl::common::vec2f screen = (owl::common::vec2f(pixelId) + owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng))) / owl::common::vec2f(self.frameBufferSize);
    RadianceRay ray;
    ray.origin
        = optixLaunchParams.camera.pos;
    ray.direction
        = normalize(optixLaunchParams.camera.dir_00
            + screen.u * optixLaunchParams.camera.dir_du
            + screen.v * optixLaunchParams.camera.dir_dv);

    SurfaceInteraction si;
    owl::traceRay(optixLaunchParams.world, ray, si);

    owl::common::vec3f color(0.f, 0.f, 0.f);
    //printf("%d\n", optixLaunchParams.rendererType);
    if (si.hit == false)
    {
        color = si.diffuse;
        color = si.n_geom;
    }
    else if (optixLaunchParams.rendererType == MASK)
        color = owl::common::vec3f(1., 1., 1.);
    else if (optixLaunchParams.rendererType == POSITION)
        color = si.p;
    else if (optixLaunchParams.rendererType == DIFFUSE)
        color = si.diffuse;
    else if (optixLaunchParams.rendererType == ALPHA)
        color = si.alpha;
    else if (optixLaunchParams.rendererType == NORMALS)
        color = si.n_geom;
    // Direct lighting with MC
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_LSAMPLE) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLighting(si, rng, 0);
    }
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_BRDFSAMPLE) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLighting(si, rng, 1);
    }
    else if (optixLaunchParams.rendererType == DIRECT_LIGHT_MIS) {
        if (si.isLight)
            color = si.emit;
        else
            color = estimateDirectLighting(si, rng, 2);
    }
    // Direct lighting with LTC
    else if (optixLaunchParams.rendererType == LTC_BASELINE) {
        if (si.isLight)
            color = si.emit;
        else
            color = ltcDirectLighingBaseline(si, rng);
    }
    else if (optixLaunchParams.rendererType == RATIO) {
        owl::common::vec3f ltc_color = owl::common::vec3f(0.);
        owl::common::vec3f sto_S = owl::common::vec3f(0.);
        owl::common::vec3f sto_U = owl::common::vec3f(0.);

        if (si.isLight)
            color = si.emit;
        else {
            ltc_color = ltcDirectLighingBaseline(si, rng);
            for (int i = 0; i < 4; i++)
            {
                sto_S += estimateDirectLighting(si, rng, 3);
                sto_U += estimateDirectLighting(si, rng, 4);
            }

            color.x = (sto_U.x < 1e-4) ? 0. : ltc_color.x * sto_S.x / sto_U.x;
            color.y = (sto_U.y < 1e-4) ? 0. : ltc_color.y * sto_S.y / sto_U.y;
            color.z = (sto_U.z < 1e-4) ? 0. : ltc_color.z * sto_S.z / sto_U.z;
            //color = ltc_color * sto_S / sto_U;
        }

    }
    else if (optixLaunchParams.rendererType == PATH)
    {
        int n = 1;
        for(int i=0;i<n;i++)
           color += estimatePathTracing(si, rng);
        color /= n;
    }
    else {
        color = owl::common::vec3f(1., 0., 0.);
    }

    if (optixLaunchParams.accumId > 0)
        color = color + owl::common::vec3f(optixLaunchParams.accumBuffer[fbOfs].x, optixLaunchParams.accumBuffer[fbOfs].y,
            optixLaunchParams.accumBuffer[fbOfs].z);

    optixLaunchParams.accumBuffer[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
    color = (1.f / (optixLaunchParams.accumId + 1)) * color;
    self.frameBuffer[fbOfs] = owl::make_rgba(color);

    //self.frameBuffer[fbOfs] = owl::make_rgba(owl::common::vec3f(0., 1., 0.));
}
