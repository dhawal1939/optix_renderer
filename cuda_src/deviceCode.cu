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


#include <path/path.cuh>
#include <ratio/ratio.cuh>
#include <ltc/ltc_utils.cuh>


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

    {
        // Out going direction pointing towards the pixel location
        si.wo = owl::normalize(ray.origin - si.p);
        // Initializes to_local from n_geo then obtains to_world by taking inverse of the to_local
        orthonormalBasis(si.n_geom, si.to_local, si.to_world);

        // obtain wo is in world space cam_pos - hit_loc_world get local frame of the wo as wo_local
        si.wo_local = normalize(apply_mat(si.to_local, si.wo));
    }

    owl::common::vec3f color(0.f, 0.f, 0.f);
    struct triColor colors;
    colors.colors[0] = owl::common::vec3f(0.f);
    colors.colors[1] = owl::common::vec3f(0.f);
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
    else if (optixLaunchParams.rendererType == SHADE_NORMALS)
        color = si.n_shad;
    // Direct lighting with LTC
    else if (optixLaunchParams.rendererType == LTC_BASELINE) {
        if (si.isLight)
            color = si.emit;
        else
            color = ltcDirectLighingBaseline(si, rng);
    }
    else if (optixLaunchParams.rendererType == RATIO) {
        if (si.isLight)
        {
            color = si.emit;
            optixLaunchParams.ltc_buffer[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
            optixLaunchParams.stoDirectRatio[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
            optixLaunchParams.stoNoVisRatio[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
        }
        else
        {
            owl::common::vec3f ltc_color = ltcDirectLighingBaseline(si, rng);
            int n = 4;
            for (int i = 0; i < n; i++)
            {
                struct triColor temp_colors = ratio_based_shadow(si, rng, 1);
                colors.colors[0] += temp_colors.colors[0];
                colors.colors[1] += temp_colors.colors[1];
            }
            colors.colors[0] /= n;
            colors.colors[1] /= n;
            color = ltc_color;
            
            optixLaunchParams.ltc_buffer[fbOfs] = make_float4(ltc_color.x, ltc_color.y, ltc_color.z, 1.f);
            float color_direct = (colors.colors[0].x + colors.colors[0].y + colors.colors[0].z) / 3.f;
            float color_noVis = (colors.colors[1].x + colors.colors[1].y + colors.colors[1].z) / 3.f;
            optixLaunchParams.stoDirectRatio[fbOfs] = make_float4(color_direct, color_direct, color_direct, 1.f);
            optixLaunchParams.stoNoVisRatio[fbOfs] = make_float4(color_noVis, color_noVis, color_noVis, 1.f);
        }
    }
    else if (optixLaunchParams.rendererType == PATH)
    {   
        if (si.isLight)
            color = si.emit;
        else
           color = estimatePathTracing(si, rng, ray, 2);

    }
    else {
        color = owl::common::vec3f(1., 0., 0.);
    }
  
    if (optixLaunchParams.accumId > 0)
        color = color + owl::common::vec3f(optixLaunchParams.accumBuffer[fbOfs].x, optixLaunchParams.accumBuffer[fbOfs].y,
            optixLaunchParams.accumBuffer[fbOfs].z);
    optixLaunchParams.accumBuffer[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
    optixLaunchParams.albedo[fbOfs] = make_float4(si.diffuse.x, si.diffuse.y, si.diffuse.z, 1.f);
    optixLaunchParams.normal[fbOfs] = make_float4(si.n_geom.x, si.n_geom.y, si.n_geom.z, 1.f);

    color = (1.f / (optixLaunchParams.accumId + 1.f)) * color;

    self.frameBuffer[fbOfs] = owl::make_rgba(color);
}
