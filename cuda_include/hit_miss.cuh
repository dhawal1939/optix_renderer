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

OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
{
    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
    const owl::common::vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
    si.hit = true;

    // area of triangle
    owl::common::vec3f v1 = self.vertex[primitiveIndices.x];
    owl::common::vec3f v2 = self.vertex[primitiveIndices.y];
    owl::common::vec3f v3 = self.vertex[primitiveIndices.z];
    si.area = 0.5f * length(cross(v1 - v2, v3 - v2));

    // Exact hit point on the triangle
    si.p = barycentricInterpolate(self.vertex, primitiveIndices);

    // UV coordinate of the hit point
    si.uv = barycentricInterpolate(self.texCoord, primitiveIndices);
    si.uv.x = owl::common::abs(fmodf(si.uv.x, 1.));
    si.uv.y = owl::common::abs(fmodf(si.uv.y, 1.));


    si.emit = self.emit;
    si.isLight = self.isLight;
    // geometric normal 
    si.n_geom = normalize(barycentricInterpolate(self.normal, primitiveIndices));
    // if bump map exists
   /* if (self.hasNormalTexture)
    {
       float4 normal_vals= tex2D<float4>(self.normal_texture, si.uv.x, si.uv.y);
       si.n_shad = owl::common::vec3f(normal_vals.x, normal_vals.y, normal_vals.z);
       si.n_shad = owl::normalize(si.n_shad);
    }*/
    // axix independet prop
    si.diffuse = self.diffuse;
    if (self.hasDiffuseTexture)
    {
        float4 diffuse_values = tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);
        si.diffuse = owl::common::vec3f(diffuse_values.x, diffuse_values.y, diffuse_values.z);
    }
    si.alpha = 1. - self.alpha;
    if (self.hasAlphaTexture)
    {
        float4 alpha_values = tex2D<float4>(self.alpha_texture, si.uv.x, si.uv.y);
        si.alpha = owl::common::max(alpha_values.x, owl::common::max(alpha_values.y, alpha_values.z));
        //si.alpha = owl::common::length(owl::common::vec3f(alpha_values.x, alpha_values.y, alpha_values.z));
    }
    si.alpha = clamp(si.alpha, 0.01f, 1.f);
}

OPTIX_MISS_PROGRAM(miss)()
{
    const owl::common::vec2i pixelId = owl::getLaunchIndex();
    const MissProgData& self = owl::getProgramData<MissProgData>();

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
    si.hit = false;
    si.diffuse = self.const_color;
}