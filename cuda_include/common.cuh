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
	owl::common::vec3f v1, v2, v3;
	owl::common::vec3f cg;
	owl::common::vec3f normal;
	owl::common::vec3f emit;

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
	owl::common::vec2i pixelId;

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
	unsigned int materialID;

	owl::common::vec3f diffuse;
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
	owl::common::vec3f const_color;
};

struct ShadowRayData {
	owl::common::vec3f visibility = owl::common::vec3f(0.f);
	owl::common::vec3f point = owl::common::vec3f(0.f), normal_screen_buffer = owl::common::vec3f(0.f), cg = owl::common::vec3f(0.f);
};


struct SurfaceInteraction {
	bool hit = false;

	owl::common::vec3f p = owl::common::vec3f(0.f);
	owl::common::vec2f uv = owl::common::vec2f(0.f);
	owl::common::vec3f wo = owl::common::vec3f(0.f), wi = owl::common::vec3f(0.f);
	owl::common::vec3f wo_local = owl::common::vec3f(0.f), wi_local = owl::common::vec3f(0.f);

	unsigned int materialID;

	owl::common::vec3f n_geom = owl::common::vec3f(0.f), n_shad = owl::common::vec3f(0.f);

	owl::common::vec3f diffuse = owl::common::vec3f(0.f);
	float alpha = 0.f;

	owl::common::vec3f emit = owl::common::vec3f(0.f);
	bool isLight = false;

	float area = 0.f;
	owl::common::vec3f to_local[3], to_world[3];
};