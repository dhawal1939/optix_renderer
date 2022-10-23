#pragma once

#include <owl/owl.h>
#include <cuda_runtime.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>

using namespace owl;

#define PI 3.1415926f
#define MAX_LTC_LIGHTS 20

enum RendererType {
	DIFFUSE=0,
	ALPHA=1,
	NORMALS=2,
	DIRECT_LIGHT_LSAMPLE,
	DIRECT_LIGHT_BRDFSAMPLE,
	DIRECT_LIGHT_MIS,
	LTC_BASELINE,
	RATIO,
	NUM_RENDERER_TYPES
};

const char* rendererNames[NUM_RENDERER_TYPES] = {   "Diffuse", 
                                                    "Alpha", 
                                                    "Normals", 
    												"Direct Light (Light IS)", 
                                                    "Direct Light (BRDF IS)", 
                                                    "Direct Light (Light n BRDF MIS)",
    												"LTC Baseline", 
    												"Ratio"};

__inline__ __host__
bool CHECK_IF_LTC(RendererType renderer_type)
{
	switch (renderer_type) {
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
typedef RayT<0, 2> RadianceRay;
typedef RayT<1, 2> ShadowRay;
#endif

struct TriLight {
	vec3f aabbMin = vec3f(1e30f);
	vec3f aabbMax = vec3f(-1e30f);

	vec3f v1, v2, v3;
	vec3f cg;
	vec3f normal;
	vec3f emit;

	float flux;
	float area;
};

struct MeshLight {
	vec3f aabbMin = vec3f(1e30f);
	vec3f aabbMax = vec3f(-1e30f);
	vec3f cg;
	float flux;

	int triIdx;
  	int triStartIdx;
	int triCount;

	int edgeStartIdx;
	int edgeCount;

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

	struct {
		vec3f pos;
		vec3f dir_00;
		vec3f dir_du;
		vec3f dir_dv;
	} camera;

	float lerp;
};

__constant__ LaunchParams optixLaunchParams;

struct RayGenData {
	uint32_t* frameBuffer;
	vec2i frameBufferSize;
};

struct TriangleMeshData {
	vec3f* vertex;
	vec3f* normal;
	vec3i* index;
	vec2f* texCoord;

	bool isLight;
	vec3f emit;

	vec3f diffuse;
	bool hasDiffuseTexture;
	cudaTextureObject_t diffuse_texture;

	float alpha;
	bool hasAlphaTexture;
	cudaTextureObject_t alpha_texture;
};

struct MissProgData {
	vec3f const_color;
};

struct ShadowRayData {
	vec3f visibility = vec3f(0.f);
	vec3f point = vec3f(0.f), normal = vec3f(0.f), cg = vec3f(0.f);
	vec3f emit = vec3f(0.f);
	float area = 0.f;
};

struct AABB { 
	vec3f bmin = vec3f(1e30f);
	vec3f bmax = vec3f(- 1e30f);

	__inline__ __device__ __host__
    void grow( vec3f p ) { bmin = owl::min( bmin, p ), bmax = owl::min( bmax, p ); }

	__inline__ __device__ __host__ float area() 
    { 
        vec3f e = bmax - bmin; // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x; 
    }
};

struct SurfaceInteraction {
	bool hit = false;

	vec3f p = vec3f(0.f);
	vec2f uv = vec2f(0.f);
	vec3f wo = vec3f(0.f), wi = vec3f(0.f);
	vec3f wo_local = vec3f(0.f), wi_local = vec3f(0.f);

	vec3f n_geom = vec3f(0.f), n_shad = vec3f(0.f);

	vec3f diffuse = vec3f(0.f);
	float alpha = 0.f;

	vec3f emit = vec3f(0.f);
	bool isLight = false;

	vec3f to_local[3], to_world[3];
};