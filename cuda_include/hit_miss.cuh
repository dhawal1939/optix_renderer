#pragma once
#include <utils.cuh>
#include <common.cuh>
#include <material.cuh>
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
    si.materialID = self.materialID;

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

    // geometric normalScreenBuffer
    si.n_geom = normalize(barycentricInterpolate(self.normal, primitiveIndices));
    si.diffuse = self.diffuse;
    if (self.hasDiffuseTexture)
    {   
        owl::common::vec4f diffuse = tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);
        si.diffuse = owl::common::vec3f(diffuse.x, diffuse.y, diffuse.z);
    }
    si.alpha = self.alpha;
    si.alpha = owl::clamp(si.alpha, 0.01f, 1.f);

    si.emit = self.emit;
    si.isLight = self.isLight;
}

OPTIX_MISS_PROGRAM(miss)()
{
    const owl::common::vec2i pixelId = owl::getLaunchIndex();
    const MissProgData& self = owl::getProgramData<MissProgData>();

    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
    si.hit = false;
    si.isLight = false;
    si.materialID = 0;
    si.n_geom = owl::vec3f(0.);
    si.diffuse = self.const_color;
}