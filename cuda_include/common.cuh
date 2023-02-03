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
	ALPHA = 1,
	NORMALS = 2,
	SHADE_NORMALS,
	POSITION,
	MASK,
	MATERIAL_ID,
	LTC_BASELINE,
	RATIO,
	PATH,
	NUM_RENDERER_TYPES
};

const char* rendererNames[NUM_RENDERER_TYPES] = {
													"Diffuse",
													"Alpha",
													"Normals",
													"Shading Normals"
													"Position",
													"Mask",
													"Material ID",
													"LTC Baseline",
													"RATIO",
													"PATH"
};

__inline__ __host__
bool CHECK_IF_LTC(RendererType t)
{
	switch (t) {
	/*case LTC_BASELINE:
	case RATIO:
		return true;*/
	default:
		return false;
	}
}

#define RADIANCE_RAY_TYPE 0
#define SHADOW_RAY_TYPE 1

#ifdef __CUDA_ARCH__
typedef owl::RayT<0, 2> RadianceRay;
//typedef owl::RayT<1, 2> ShadowRay;
#endif

struct TriLight {
	VEC3f v1, v2, v3;
	VEC3f cg;
	VEC3f normal;
	VEC3f emit;

	//float flux;
	float area;
};

struct MeshLight {
	float flux;
	int triIdx;
	int triCount;
};

struct LaunchParams {
	bool clicked;
	VEC2i pixelId;

	float4* position_screen_buffer;
	float4* normal_screen_buffer;
	float4* uv_screen_buffer;
	float4* albedo_screen_buffer;
	float4* alpha_screen_buffer;
	float4* materialID_screen_buffer;

	float4* bounce0_screen_buffer;
	float4* bounce1_screen_buffer;
	float4* bounce2_screen_buffer;

	float4* ltc_screen_buffer;
	float4* sto_direct_ratio_screen_buffer;
	float4* sto_no_vis_ratio_screen_buffer;
	
	float4* accum_screen_buffer;
	
	int accumId;

	int rendererType;
	OptixTraversableHandle world;
	cudaTextureObject_t ltc_1, ltc_2, ltc_3;

	TriLight* triLights;
	int numTriLights;

	MeshLight* meshLights;
	int numMeshLights;

	struct {
		VEC3f pos;
		VEC3f dir_00;
		VEC3f dir_du;
		VEC3f dir_dv;
	} camera;

	float lerp;
};

__constant__ LaunchParams optixLaunchParams;

struct RayGenData {
	uint32_t* frameBuffer;
	VEC2i frameBufferSize;
};

struct TriangleMeshData {
	VEC3f* vertex;
	VEC3f* normal;
	VEC3i* index;
	VEC2f* texCoord;

	bool isLight;
	VEC3f emit;
	unsigned int materialID;

	VEC3f diffuse;
	bool hasDiffuseTexture;
	cudaTextureObject_t diffuse_texture;

	float alpha;
	bool hasAlphaTexture;
	cudaTextureObject_t alpha_texture;

	float normal_map;
	bool hasNormalTexture;
	cudaTextureObject_t normal_texture;
};

struct MissProgData {
	VEC3f const_color;
};

struct ShadowRayData {
	VEC3f visibility = VEC3f(0.f);
	VEC3f point = VEC3f(0.f), normal_screen_buffer = VEC3f(0.f), cg = VEC3f(0.f);
};


struct SurfaceInteraction {
	bool hit = false;

	VEC3f p = VEC3f(0.f);
	VEC2f uv = VEC2f(0.f);
	VEC3f wo = VEC3f(0.f), wi = VEC3f(0.f);
	VEC3f wo_local = VEC3f(0.f), wi_local = VEC3f(0.f);
	
	float spec_exponent;
	float spec_intensity;

	unsigned int materialID;

	VEC3f n_geom = VEC3f(0.f), n_shad = VEC3f(0.f);

	VEC3f diffuse = VEC3f(0.f);
	float alpha = 0.f;

	VEC3f emit = VEC3f(0.f);
	bool isLight = false;

	float area = 0.f;
	VEC3f to_local[3], to_world[3];
};