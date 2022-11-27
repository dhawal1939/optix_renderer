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

#include <hit_miss.cuh>


owl::common::vec3f sampleLight(int selectedLightIdx, owl::common::vec2f rand) {

    TriLight triLight = optixLaunchParams.triLights[selectedLightIdx];
    owl::common::vec3f lv1 = triLight.v1;
    owl::common::vec3f lv2 = triLight.v2;
    owl::common::vec3f lv3 = triLight.v3;
    owl::common::vec3f lnormal = triLight.normal;
    owl::common::vec3f lemit = triLight.emit;
    float larea = triLight.area;
    return samplePointOnTriangle(lv1, lv2, lv3, rand.x, rand.y);
}

float sampleLightPdf(int selectedLightIdx) 
{ 
    // Area calucation
    TriLight triLight = optixLaunchParams.triLights[selectedLightIdx];
    return 1 / (triLight.area * optixLaunchParams.numTriLights); 
}

// Convert from area measure to angle measure
float pdfA2W(float pdf, float dist2, float cos_theta) {
    float abs_cos_theta = abs(cos_theta);
    if (abs_cos_theta < 1e-8) {
        return 0.0;
    }

    return pdf * dist2 / abs_cos_theta;
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
owl::common::vec3f estimatePathTracing(SurfaceInteraction& si, LCGRand& rng, int max_ray_depth = 4)
{
    for (int ray_depth = 0; ray_depth < max_ray_depth; ray_depth++)
    {
        owl::common::vec2f rand1 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        owl::common::vec2f rand2 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));

        owl::common::vec3f V = si.wo; // Global direction
        // MIS
        {
            int selectedLightIdx = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
            
            float lightPdf = sampleLightPdf(selectedLightIdx);

            owl::common::vec3f pos = sampleLight(selectedLightIdx, rand1);
            owl::common::vec3f newPos = si.p;

            owl::common::vec3f L = normalize(pos - newPos);  // incoming from light
            float dist = owl::common::length(pos - newPos);
            dist = dist * dist;

            owl::common::vec3f tNormal;
            if (si.hit && si.isLight) {
                owl::common::vec3f H = normalize(V + L);

                float brdfPdf;
                owl::common::vec3f brdf = evalBSDF(mat, normal, H, V, L, newPos, brdfPdf);

                float lightPdfW = pdfA2W(lightPdf, dist, dot(-L, tNormal));

                float misW = lightPdfW / (lightPdfW + brdfPdf);

                contrib += misW * tMat.color * tp * brdf * clampDot(normal, L, true) /
                    lightPdfW;
            }
        }
    }
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
    ray.origin = optixLaunchParams.camera.pos;
    ray.direction = normalize(optixLaunchParams.camera.dir_00
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
           /* for (int i = 0; i < 4; i++)
            {
                sto_S += estimateDirectLighting(si, rng, 3);
                sto_U += estimateDirectLighting(si, rng, 4);
            }*/

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
