#pragma once
#include <common.h>
#include <helper_math.cuh>

#include <optix.h>

#include <cuda_runtime.h>

#include <vector_types.h>

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>
#include <texture_fetch_functions.h>
#include <owl/common/math/constants.h>

enum RendererType {
	DIFFUSE = 0,
	ALPHA,
	NORMALS,
	DIRECT_LIGHT_LSAMPLE,
	DIRECT_LIGHT_BRDFSAMPLE,
	DIRECT_LIGHT_MIS,
	LTC_BASELINE,
	RATIO,
	NUM_RENDERER_TYPES
};

const char* rendererNames[NUM_RENDERER_TYPES] = {
													"Diffuse",
													"Alpha",
													"Normals",
													"Direct Light (Light)",
													"Direct Light (BRDF)",
													"Direct Light (MIS)",
													"LTC Baseline",
													"RATIO"
};

__inline__ __host__ bool CHECK_IF_LTC(RendererType t)
{
	switch (t) {
	case LTC_BASELINE:
	case RATIO:
		return true;
	default:
		return false;
	}
}

#define RADIANCE_RAY_TYPE 0
#define SHADOW_RAY_TYPE 1

#ifdef __CUDA_ARCH__
typedef owl::RayT<0, 2> RadianceRay;
typedef owl::RayT<1, 2> ShadowRay;
#endif


struct SurfaceInteraction {
	bool hit = false;

	owl::common::vec3f p = owl::common::vec3f(0.f);
	owl::common::vec2f uv = owl::common::vec2f(0.f);
	owl::common::vec3f wo = owl::common::vec3f(0.f), wi = owl::common::vec3f(0.f);
	owl::common::vec3f wo_local = owl::common::vec3f(0.f), wi_local = owl::common::vec3f(0.f);

	owl::common::vec3f n_geom = owl::common::vec3f(0.f), n_shad = owl::common::vec3f(0.f);

	owl::common::vec3f diffuse = owl::common::vec3f(0.f);
	float alpha = 0.f;

	owl::common::vec3f emit = owl::common::vec3f(0.f);
	bool isLight = false;

	owl::common::vec3f to_local[3], to_world[3];
};

struct LightBVH {
	owl::common::vec3f aabbMin = owl::common::vec3f(1e30f);
	owl::common::vec3f aabbMax = owl::common::vec3f(-1e30f);
	owl::common::vec3f aabbMid = owl::common::vec3f(0.f);
	float flux = 0.f;

	uint32_t left = 0, right = 0;
	uint32_t primIdx = 0, primCount = 0;
};

struct TriLight {
	owl::common::vec3f aabbMin = owl::common::vec3f(1e30f);
	owl::common::vec3f aabbMax = owl::common::vec3f(-1e30f);

	owl::common::vec3f v1, v2, v3;
	owl::common::vec3f cg;
	owl::common::vec3f normal;
	owl::common::vec3f emit;

	float flux;
	float area;
};

struct MeshLight {
	owl::common::vec3f aabbMin = owl::common::vec3f(1e30f);
	owl::common::vec3f aabbMax = owl::common::vec3f(-1e30f);
	owl::common::vec3f cg;
	float flux;

	int triIdx;
	int triCount;

	int bvhIdx;
	int bvhHeight;
};

struct LaunchParams {
	float4* accumBuffer;
	float4* UBuffer;
	float4* SBuffer;
	int accumId;

	int rendererType;
	OptixTraversableHandle world;
	cudaTextureObject_t ltc_1, ltc_2, ltc_3;

	TriLight* triLights;
	int numTriLights;

	MeshLight* meshLights;
	int numMeshLights;

	LightBVH* lightBlas;
	LightBVH* lightTlas;
	int lightTlasHeight;

	struct {
		owl::common::vec3f pos;
		owl::common::vec3f dir_00;
		owl::common::vec3f dir_du;
		owl::common::vec3f dir_dv;
	} camera;

	float lerp;
};

__constant__ LaunchParams optixLaunchParams;

struct RayGenData {
	uint32_t* frameBuffer;
	owl::common::vec2i frameBufferSize;
};

struct TriangleMeshData {
	owl::common::vec3f* vertex;
	owl::common::vec3f* normal;
	owl::common::vec3i* index;
	owl::common::vec2f* texCoord;

	bool isLight;
	owl::common::vec3f emit;

	owl::common::vec3f diffuse;
	bool hasDiffuseTexture;
	cudaTextureObject_t diffuse_texture;

	float alpha;
	bool hasAlphaTexture;
	cudaTextureObject_t alpha_texture;
};

struct MissProgData {
	owl::common::vec3f const_color;
};

struct ShadowRayData {
	owl::common::vec3f visibility = owl::common::vec3f(0.f);
	owl::common::vec3f point = owl::common::vec3f(0.f), normal = owl::common::vec3f(0.f), cg = owl::common::vec3f(0.f);
	owl::common::vec3f emit = owl::common::vec3f(0.f);
	float area = 0.f;
};

struct AABB {
	owl::common::vec3f bmin = owl::common::vec3f(1e30f);
	owl::common::vec3f bmax = owl::common::vec3f(-1e30f);

	__inline__ __device__ __host__
		void grow(owl::common::vec3f p) { bmin = owl::min(bmin, p), bmax = owl::min(bmax, p); }

	__inline__ __device__ __host__ float area()
	{
		owl::common::vec3f e = bmax - bmin; // box extent
		return e.x * e.y + e.y * e.z + e.z * e.x;
	}
};