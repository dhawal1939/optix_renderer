#include <utils.cuh>
#include <common.cuh>
#include <ggx_ndf.cuh>
#include <lcg_random.cuh>
#include <optix_device.h>
#include <owl/owl_device.h>
#include <ltc/ltc_utils.cuh>
#include <ltc/polygon_utils.cuh>

#include <vector_types.h>
#include <texture_fetch_functions.h>

//OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCHShadow)()
//{
//    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
//    const owl::common::vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];
//    ShadowRayData& srd = owl::getPRD<ShadowRayData>();
//
//    if (self.isLight) {
//        srd.visibility = owl::common::vec3f(1.f);
//        srd.point = barycentricInterpolate(self.vertex, primitiveIndices);
//        srd.normal = normalize(barycentricInterpolate(self.normal, primitiveIndices));
//        srd.emit = self.emit;
//
//        owl::common::vec3f v1 = self.vertex[primitiveIndices.x];
//        owl::common::vec3f v2 = self.vertex[primitiveIndices.y];
//        owl::common::vec3f v3 = self.vertex[primitiveIndices.z];
//        srd.area = 0.5f * length(cross(v1 - v2, v3 - v2));
//
//        srd.cg = (v1 + v2 + v3) / 3.f;
//    }
//    else {
//        srd.visibility = owl::common::vec3f(0.f);
//        srd.point = barycentricInterpolate(self.vertex, primitiveIndices);
//        srd.normal = normalize(barycentricInterpolate(self.normal, primitiveIndices));
//        srd.emit = owl::common::vec3f(0.);
//
//        owl::common::vec3f v1 = self.vertex[primitiveIndices.x];
//        owl::common::vec3f v2 = self.vertex[primitiveIndices.y];
//        owl::common::vec3f v3 = self.vertex[primitiveIndices.z];
//        srd.area = 0.5f * length(cross(v1 - v2, v3 - v2));
//
//        srd.cg = (v1 + v2 + v3) / 3.f;
//    }
//
//}

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    const owl::common::vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
    si.hit = true;

    // Exact hit point on the triangle
    si.p = barycentricInterpolate(self.vertex, primitiveIndices);

    // Out going direction pointing toward the pixel location
    si.wo = owl::normalize(optixLaunchParams.camera.pos - si.p);

    // UV coordinate of the hit point
    si.uv = barycentricInterpolate(self.texCoord, primitiveIndices);
    si.uv.x = owl::common::abs(fmodf(si.uv.x, 1.));
    si.uv.y = owl::common::abs(fmodf(si.uv.y, 1.));

    // geometric normal 
    si.n_geom = normalize(barycentricInterpolate(self.normal, primitiveIndices));

    // Initializes to_local from n_geo then obtains to_world by taking inverse of the to_local
    orthonormalBasis(si.n_geom, si.to_local, si.to_world);

    // obtain wo is in world space cam_pos - hit_loc_world get local frame of the wo as wo_local
    si.wo_local = normalize(apply_mat(si.to_local, si.wo));

    // area of triangle

    owl::common::vec3f v1 = self.vertex[primitiveIndices.x];
    owl::common::vec3f v2 = self.vertex[primitiveIndices.y];
    owl::common::vec3f v3 = self.vertex[primitiveIndices.z];
    si.area = 0.5f * length(cross(v1 - v2, v3 - v2));

    // axix independet prop
    si.diffuse = self.diffuse;
    if (self.hasDiffuseTexture)
        si.diffuse = (owl::common::vec3f)tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);

    si.alpha = self.alpha;
    if (self.hasAlphaTexture)
    {
        float4 alpha_values = tex2D<float4>(self.alpha_texture, si.uv.x, si.uv.y);
        si.alpha = owl::common::max(alpha_values.x, owl::common::max(alpha_values.y, alpha_values.z));
        //si.alpha = owl::common::length(owl::common::vec3f(alpha_values.x, alpha_values.y, alpha_values.z));
    }
    si.alpha = clamp(si.alpha, 0.01f, 1.f);

    si.emit = self.emit;
    si.isLight = self.isLight;

}

OPTIX_MISS_PROGRAM(miss)()
{
    const owl::common::vec2i pixelId = owl::getLaunchIndex();
    const MissProgData& self = owl::getProgramData<MissProgData>();

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
    si.hit = false;
    si.diffuse = self.const_color;
}