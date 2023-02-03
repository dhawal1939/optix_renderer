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
VEC3f ltcDirectLighingBaseline(SurfaceInteraction& si, LCGRand& rng)
{
    VEC3f wo_local = normalize(apply_mat(si.to_local, si.wo));
    if (wo_local.z < 0.f)
        return VEC3f(0.f);

    VEC3f normal_local(0.f, 0.f, 1.f);
    VEC3f color(0.0, 0.0, 0.0);

    /* Analytic shading via LTCs */
    VEC3f ltc_mat[3], ltc_mat_inv[3];
    float alpha = si.alpha;
    float theta = sphericalTheta(wo_local);

    float amplitude = 1.f;
    fetchLtcMat(alpha, theta, ltc_mat, amplitude);
    matrixInverse(ltc_mat, ltc_mat_inv);

    VEC3f iso_frame[3];

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
    const VEC2i pixelId = owl::getLaunchIndex();
    const int fbOfs = pixelId.x + self.frameBufferSize.x * pixelId.y;

    LCGRand rng = get_rng(optixLaunchParams.accumId + 10007, make_uint2(pixelId.x, pixelId.y),
        make_uint2(self.frameBufferSize.x, self.frameBufferSize.y));

    const VEC2f screen = (VEC2f(pixelId) + VEC2f(lcg_randomf(rng), lcg_randomf(rng))) / VEC2f(self.frameBufferSize);
    RadianceRay ray;
    ray.origin = optixLaunchParams.camera.pos;
    ray.direction = normalize(optixLaunchParams.camera.dir_00
        + screen.u * optixLaunchParams.camera.dir_du
        + screen.v * optixLaunchParams.camera.dir_dv);

    SurfaceInteraction si;
    owl::traceRay(optixLaunchParams.world, ray, si);

    {
        // Out going direction pointing towards the pixel location
        si.wo = normalize(ray.origin - si.p);
        // Initializes to_local from n_geo then obtains to_world by taking inverse of the to_local
        orthonormalBasis(si.n_geom, si.to_local, si.to_world);
        // obtain wo is in world space cam_pos - hit_loc_world get local frame of the wo as wo_local
        si.wo_local = normalize(apply_mat(si.to_local, si.wo));
    }

    VEC3f color(0.f, 0.f, 0.f);
    struct triColor colors;
    colors.colors[0] = VEC3f(0.f);
    colors.colors[1] = VEC3f(0.f);
    if (si.hit == false)
    {
        color = si.diffuse;
        color = si.n_geom;
    }
    else if (optixLaunchParams.rendererType == MASK)
        color = VEC3f(1., 1., 1.);
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
    else if (optixLaunchParams.rendererType == MATERIAL_ID)
        color = owl::vec3f(si.materialID);
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
            optixLaunchParams.ltc_screen_buffer[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
            optixLaunchParams.sto_direct_ratio_screen_buffer[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
            optixLaunchParams.sto_no_vis_ratio_screen_buffer[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
        }
        else
        {
            VEC3f ltc_color = ltcDirectLighingBaseline(si, rng);
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
            
            optixLaunchParams.ltc_screen_buffer[fbOfs] = make_float4(ltc_color.x, ltc_color.y, ltc_color.z, 1.f);
            float color_direct = (colors.colors[0].x + colors.colors[0].y + colors.colors[0].z) / 3.f;
            float color_noVis = (colors.colors[1].x + colors.colors[1].y + colors.colors[1].z) / 3.f;
            optixLaunchParams.sto_direct_ratio_screen_buffer[fbOfs] = make_float4(color_direct, color_direct, color_direct, 1.f);
            optixLaunchParams.sto_no_vis_ratio_screen_buffer[fbOfs] = make_float4(color_noVis, color_noVis, color_noVis, 1.f);
        }
    }
    else if (optixLaunchParams.rendererType == PATH)
    {   
        if (si.isLight)
            color = si.emit;
        else
            color = estimatePathTracing(si, rng, ray, 1);

    }
    else {
        color = VEC3f(1., 0., 0.);
    }
  
    if (optixLaunchParams.accumId > 0)
        color = color + VEC3f(optixLaunchParams.accum_screen_buffer[fbOfs].x, optixLaunchParams.accum_screen_buffer[fbOfs].y,
            optixLaunchParams.accum_screen_buffer[fbOfs].z) * optixLaunchParams.accumId;
    color = color / (optixLaunchParams.accumId + 1.f);
    self.frameBuffer[fbOfs] = owl::make_rgba(color);

    optixLaunchParams.accum_screen_buffer[fbOfs] = make_float4(color.x, color.y, color.z, 1.f);
    optixLaunchParams.position_screen_buffer[fbOfs] = make_float4(si.p.x, si.p.y, si.p.z, 1.f);
    optixLaunchParams.normal_screen_buffer[fbOfs] = make_float4(si.n_geom.x, si.n_geom.y, si.n_geom.z, 1.f);
    optixLaunchParams.albedo_screen_buffer[fbOfs] = make_float4(si.diffuse.x, si.diffuse.y, si.diffuse.z, 1.f);
    optixLaunchParams.alpha_screen_buffer[fbOfs] = make_float4(si.alpha, 0.f, 0.f, 1.f);
    optixLaunchParams.uv_screen_buffer[fbOfs] = make_float4(si.uv.x, si.uv.y, si.diffuse.z, 1.f);
    optixLaunchParams.materialID_screen_buffer[fbOfs] = make_float4(si.materialID, 0.f, 0.f, 1.f);
}
