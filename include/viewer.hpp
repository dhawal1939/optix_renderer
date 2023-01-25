// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

// This program sets up a single geometric object, a mesh for a cube, and
// its acceleration structure, then ray traces it.

// public owl node-graph API
// viewer base class, for window and user interaction

#include <string>
#include <fstream>


#include "owl/owl.h"
#include "owl/DeviceMemory.h"
#include "owl/common/math/vec.h"
#include <owl/helper/optix.h>
#include "owlViewer/OWLViewer.h"

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>
#include <owlViewer/OWLViewer.h>

// our device-side data structures
#include <deviceCode.cuh>

// Geometry Headers
#include <common.h>
#include <Model.h>
#include <scene.h>

// ImGUI
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// LTC LUT
#include <ltc/ltc_isotropic.h>
#include <cuda_include/common.cuh>

// const owl::common::vec2i fbSize(800,600);

// Compiled PTX code
extern "C" char deviceCode_ptx[];

const owl::common::vec3f init_lookFrom(-4.f, +3.f, -2.f);
const owl::common::vec3f init_lookAt(0.f, 0.f, 0.f);
const owl::common::vec3f init_lookUp(0.f, 1.f, 0.f);
const float init_cosFovy = 0.66f;

struct Viewer :public owl::viewer::OWLViewer
{
    Viewer(Scene& scene, owl::common::vec2i resolution, RendererType renderer_type, bool interactive, bool vsync);

    /*! gets called whenever the viewer needs us to re-render out widget */
    void render() override;
    void drawUI();

    // /*! window notifies us that we got resized. We HAVE to override
    //     this to know our actual render dimensions, and get pointer
    //     to the device frame buffer that the viewer cated for us */
    void resize(const owl::common::vec2i& newSize) override;

    int imgui_init(bool _callbacks, const char* gl_version);
    void Viewer::savebuffer(FILE* fp, OWLBuffer* buffer, int avg_factor);

    std::string gl_version;

    std::string to_save_file;
    // this function gets called whenever any camera manipulator
    // updates the camera. gets called AFTER all values have been updated
    void cameraChanged() override;

    void key(char key, const owl::common::vec2i& pos) override;
    void mouseButtonLeft(const owl::common::vec2i & where, bool pressed) override;

    void setRendererType(RendererType type);

    RendererType rendererType;
    bool sbtDirty = true;

    OWLBuffer position_screen_buffer{ 0 };
    OWLBuffer normal_screen_buffer{ 0 };
    OWLBuffer uv_screen_buffer{ 0 };
    OWLBuffer albedo_screen_buffer{ 0 };
    OWLBuffer alpha_screen_buffer{ 0 };
    OWLBuffer materialID_screen_buffer{ 0 };


    OWLBuffer accum_screen_buffer{ 0 };
    OWLBuffer bounce0_screen_buffer{ 0 };
    OWLBuffer bounce1_screen_buffer{ 0 };
    OWLBuffer bounce2_screen_buffer{ 0 };
    
    OWLBuffer ltc_screen_buffer{ 0 };
    OWLBuffer sto_direct_ratio_screen_buffer{ 0 };
    OWLBuffer sto_no_vis_ratio_screen_buffer{ 0 };
    

    int accumId = 0;

    OWLRayGen rayGen{ 0 };
    OWLMissProg missProg{ 0 };

    OWLGroup world; // TLAS

    OWLContext context{ 0 };
    OWLModule module{ 0 };

    OWLParams launchParams;

    Scene currentScene;
    std::vector<SceneCamera> recordedCameras;

    std::vector<TriLight> triLightList;
    std::vector<MeshLight> meshLightList;


    // Random controls
    float lerp = 0.5f;
    int cameraPos = 0.f;


    void denoise(const void* to_denoise);
    void denoise_setup();
    owl::common::vec2i numBlocksAndThreads;
    unsigned int denoiserScratchSize;
    unsigned int denoiserStateSize;
    OptixDenoiser denoiser;
    OWLBuffer denoiserScratch{ 0 };
    OWLBuffer denoiserState{ 0 };
    OWLBuffer denoisedBuffer{ 0 };
    OWL_float denoisedIntensity;
};

int Viewer::imgui_init(bool _callbacks, const char* gl_version)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(this->handle, _callbacks);
    return ImGui_ImplOpenGL3_Init(gl_version);
}


Viewer::Viewer(Scene& scene, owl::common::vec2i resolution, RendererType renderer_type, bool interactive = true, bool vsync = false)
    : owl::viewer::OWLViewer("Optix Viewer", resolution, interactive, vsync)
{

    this->currentScene = scene;
    this->rendererType = renderer_type;

    this->gl_version = "4.6";
    if (!this->imgui_init(true, this->gl_version.c_str()))
    {
        LOG("IMGUI Init Failed")
    }

    // Context & Module creation, accumulation buffer initialize
    this->context = owlContextCreate(nullptr, 1);
    this->module = owlModuleCreate(this->context, deviceCode_ptx);

    // Position normal albedo alpha material buffers
    this->position_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->position_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);

    this->normal_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->normal_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);

    this->uv_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->uv_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);

    this->albedo_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->albedo_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);
    
    this->alpha_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->alpha_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);

    this->materialID_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->materialID_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);


    // Bounce information
    this->bounce0_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->bounce0_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);

    this->bounce1_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->bounce1_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);

    this->bounce2_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->bounce2_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);



    // Ratio estimator ltc buffers
    this->ltc_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->ltc_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);
    
    this->sto_direct_ratio_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->sto_direct_ratio_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);
    
    this->sto_no_vis_ratio_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->sto_no_vis_ratio_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);


    // Accumulation buffer
    this->accum_screen_buffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->accum_screen_buffer, this->getWindowSize().x * this->getWindowSize().y);


    owlContextSetRayTypeCount(context, 1);

    // ====================================================
    // Area lights setup (Assume triangular area lights)
    // ====================================================

    Model* triLights = scene.triLights;

    for (auto light : triLights->meshes) {
        MeshLight meshLight;
        meshLight.flux = 0.f;
        meshLight.triIdx = this->triLightList.size();

        int numTri = 0;
        for (auto index : light->index) {
            // First, setup data foran individual triangle light source
            TriLight triLight;

            triLight.v1 = light->vertex[index.x];
            triLight.v2 = light->vertex[index.y];
            triLight.v3 = light->vertex[index.z];

            triLight.cg = (triLight.v1 + triLight.v2 + triLight.v3) / 3.f;
            triLight.normal = normalize(light->normal[index.x] + light->normal[index.y] + light->normal[index.z]);
            triLight.area = 0.5f * length(cross(triLight.v1 - triLight.v2, triLight.v3 - triLight.v2));

            triLight.emit = light->emit;

            this->triLightList.push_back(triLight); // append to a global list of all triangle light sources

            // Keep track of number of triangles in the current light mesh
            numTri++;
        }

        meshLight.triCount = numTri;
        this->meshLightList.push_back(meshLight);

    }

    // ====================================================
    // Launch Parameters setup
    // ====================================================

    OWLVarDecl launchParamsDecl[] = {
        // The actual light triangles
        {"triLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, triLights)},
        {"numTriLights", OWL_INT, OWL_OFFSETOF(LaunchParams, numTriLights)},
        // The mesh lights
        {"meshLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, meshLights)},
        {"numMeshLights", OWL_INT, OWL_OFFSETOF(LaunchParams, numMeshLights)},

        // Position normal albedo alpha material buffers
        {"position_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, position_screen_buffer)},
        {"normal_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, normal_screen_buffer)},
        {"uv_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, uv_screen_buffer)},
        {"albedo_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, albedo_screen_buffer)},
        {"alpha_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, alpha_screen_buffer)},
        {"materialID_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, materialID_screen_buffer)},

        // Bounce Info
        {"bounce0_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, bounce0_screen_buffer)},
        {"bounce1_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, bounce1_screen_buffer)},
        {"bounce2_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, bounce2_screen_buffer)},

        // Ratio Estimator
        {"ltc_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, ltc_screen_buffer)},
        {"sto_direct_ratio_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, sto_direct_ratio_screen_buffer)},
        {"sto_no_vis_ratio_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, sto_no_vis_ratio_screen_buffer)},
        
        // Accumulation buffer
        {"accum_screen_buffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accum_screen_buffer)},

        // All other parameters
        {"accumId", OWL_INT, OWL_OFFSETOF(LaunchParams, accumId)},
        {"rendererType", OWL_INT, OWL_OFFSETOF(LaunchParams, rendererType)},
        {"world", OWL_GROUP, OWL_OFFSETOF(LaunchParams, world)},
        {"ltc_1", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_1)},
        {"ltc_2", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_2)},
        {"ltc_3", OWL_TEXTURE, OWL_OFFSETOF(LaunchParams, ltc_3)},
        {"camera.pos", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.pos)},
        {"camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_00)},
        {"camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_du)},
        {"camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, camera.dir_dv)},
        // Random controls
        {"lerp", OWL_FLOAT, OWL_OFFSETOF(LaunchParams, lerp)},
        {nullptr}
    };

    this->launchParams = owlParamsCreate(context, sizeof(LaunchParams), launchParamsDecl, -1);

    // Random controls
    owlParamsSet1f(this->launchParams, "lerp", this->lerp);

    // Set LTC matrices (8x8, since only isotropic)
    OWLTexture ltc1 = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_1,
        OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);
    OWLTexture ltc2 = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_2,
        OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);
    OWLTexture ltc3 = owlTexture2DCreate(context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_3,
        OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);

    owlParamsSetTexture(this->launchParams, "ltc_1", ltc1);
    owlParamsSetTexture(this->launchParams, "ltc_2", ltc2);
    owlParamsSetTexture(this->launchParams, "ltc_3", ltc3);

    owlParamsSet1i(this->launchParams, "rendererType", this->rendererType);

    // Upload the <actual> triangle data for all area lights
    OWLBuffer triLightsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(TriLight), triLightList.size(), triLightList.data());
    owlParamsSetBuffer(this->launchParams, "triLights", triLightsBuffer);
    owlParamsSet1i(this->launchParams, "numTriLights", this->triLightList.size());

    // Upload the mesh data for all area lights
    OWLBuffer meshLightsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(MeshLight), meshLightList.size(), meshLightList.data());
    owlParamsSetBuffer(this->launchParams, "meshLights", meshLightsBuffer);
    owlParamsSet1i(this->launchParams, "numMeshLights", this->meshLightList.size());


    owlParamsSetBuffer(this->launchParams, "position_screen_buffer", this->position_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "normal_screen_buffer", this->normal_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "uv_screen_buffer", this->uv_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "albedo_screen_buffer", this->albedo_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "alpha_screen_buffer", this->alpha_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "materialID_screen_buffer", this->materialID_screen_buffer);

    owlParamsSetBuffer(this->launchParams, "bounce0_screen_buffer", this->bounce0_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "bounce1_screen_buffer", this->bounce1_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "bounce2_screen_buffer", this->bounce2_screen_buffer);

    //Ratio Estimator
    owlParamsSetBuffer(this->launchParams, "ltc_screen_buffer", this->ltc_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "sto_direct_ratio_screen_buffer", this->sto_direct_ratio_screen_buffer);
    owlParamsSetBuffer(this->launchParams, "sto_no_vis_ratio_screen_buffer", this->sto_no_vis_ratio_screen_buffer);

    // Upload accumulation buffer and ID
    owlParamsSet1i(this->launchParams, "accumId", this->accumId);
    owlParamsSetBuffer(this->launchParams, "accum_screen_buffer", this->accum_screen_buffer);

    // ====================================================
    // Scene setup (scene geometry and materials)
    // ====================================================

    // Instance level accel. struct (IAS), built over geometry accel. struct (GAS) of each individual mesh
    std::vector<OWLGroup> blasList;

    // Loop over meshes, set up data and build a GAS on it. Add it to IAS.
    Model* model = scene.model;
    for (auto mesh : model->meshes) {

        // ====================================================
        // Initial setup 
        // ====================================================

        // TriangleMeshData is a CUDA struct. This declares variables to be set on the host (var names given as 1st entry)
        OWLVarDecl triangleGeomVars[] = {
            {"vertex", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, vertex)},
            {"normal", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, normal)},
            {"index", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, index)},
            {"texCoord", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshData, texCoord)},

            {"isLight", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, isLight)},
            {"emit", OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshData, emit)},
            {"materialID", OWL_UINT, OWL_OFFSETOF(TriangleMeshData, materialID)},

            {"diffuse", OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshData, diffuse)},
            {"diffuse_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, diffuse_texture)},
            {"hasDiffuseTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasDiffuseTexture)},

            {"alpha", OWL_FLOAT, OWL_OFFSETOF(TriangleMeshData, alpha)},
            {"alpha_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, alpha_texture)},
            {"hasAlphaTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasAlphaTexture)},
            
            {"normal_map", OWL_FLOAT, OWL_OFFSETOF(TriangleMeshData, normal_map)},
            {"normal_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, hasNormalTexture)},
            {"hasNormalTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, normal_texture)},
            {nullptr}
        };

        // This defines the geometry type of the variables defined above. 
        OWLGeomType triangleGeomType = owlGeomTypeCreate(context,
            /* Geometry type, in this case, a triangle mesh */
            OWL_GEOM_TRIANGLES,
            /* Size of CUDA struct */
            sizeof(TriangleMeshData),
            /* Binding to variables on the host */
            triangleGeomVars,
            /* num of variables, -1 implies sentinel is set */
            -1);

        // Defines the function name in .cu file, to be used for closest hit processing
        owlGeomTypeSetClosestHit(triangleGeomType, RADIANCE_RAY_TYPE, module, "triangleMeshCH");
        //owlGeomTypeSetClosestHit(triangleGeomType, SHADOW_RAY_TYPE, module, "triangleMeshCH");

        // Create the actual geometry on the device
        OWLGeom triangleGeom = owlGeomCreate(context, triangleGeomType);

        // ====================================================
        // Data setup
        // ====================================================

        // Create CUDA buffers from mesh vertices, indices and UV coordinates
        OWLBuffer vertexBufferObject = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->vertex.size(), mesh->vertex.data());
        OWLBuffer normalBufferObject = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->normal.size(), mesh->normal.data());
        OWLBuffer indexBufferObject = owlDeviceBufferCreate(context, OWL_INT3, mesh->index.size(), mesh->index.data());
        OWLBuffer texCoordBufferObject = owlDeviceBufferCreate(context, OWL_FLOAT2, mesh->texcoord.size(), mesh->texcoord.data());

        // Set emission value, and more importantly, if the current mesh is a light
        owlGeomSet1b(triangleGeom, "isLight", mesh->isLight);
        owlGeomSet3f(triangleGeom, "emit", owl3f{ mesh->emit.x, mesh->emit.y, mesh->emit.z });

        // Create CUDA buffers and upload them for diffuse and alpha textures
        if (mesh->diffuseTextureID != -1) {
            Texture* diffuseTexture = model->textures[mesh->diffuseTextureID];
            OWLTexture diffuseTextureBuffer = owlTexture2DCreate(context,
                OWL_TEXEL_FORMAT_RGBA8,
                diffuseTexture->resolution.x,
                diffuseTexture->resolution.y,
                diffuseTexture->pixel,
                OWL_TEXTURE_NEAREST,
                OWL_TEXTURE_CLAMP);
            owlGeomSetTexture(triangleGeom, "diffuse_texture", diffuseTextureBuffer);
            owlGeomSet1b(triangleGeom, "hasDiffuseTexture", true);
        }
        else {
            owlGeomSet3f(triangleGeom, "diffuse", owl3f{ mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z });
            owlGeomSet1b(triangleGeom, "hasDiffuseTexture", false);
        }

        if (mesh->alphaTextureID != -1) {
            Texture* alphaTexture = model->textures[mesh->alphaTextureID];
            OWLTexture alphaTextureBuffer = owlTexture2DCreate(context,
                OWL_TEXEL_FORMAT_RGBA8,
                alphaTexture->resolution.x,
                alphaTexture->resolution.y,
                alphaTexture->pixel,
                OWL_TEXTURE_NEAREST,
                OWL_TEXTURE_CLAMP);
            owlGeomSetTexture(triangleGeom, "alpha_texture", alphaTextureBuffer);
            owlGeomSet1b(triangleGeom, "hasAlphaTexture", true);
        }
        else {
            owlGeomSet1f(triangleGeom, "alpha", mesh->alpha);
            owlGeomSet1b(triangleGeom, "hasAlphaTexture", false);
        }

        if (mesh->normalTextureID != -1) {
            Texture* normalTexture = model->textures[mesh->normalTextureID];
            OWLTexture normalTextureBuffer = owlTexture2DCreate(context,
                OWL_TEXEL_FORMAT_RGBA8,
                normalTexture->resolution.x,
                normalTexture->resolution.y,
                normalTexture->pixel,
                OWL_TEXTURE_NEAREST,
                OWL_TEXTURE_CLAMP);
            owlGeomSetTexture(triangleGeom, "normal_texture", normalTextureBuffer);
            owlGeomSet1b(triangleGeom, "hasNormalTexture", true);
        }
        else {
            owlGeomSet1b(triangleGeom, "hasNormalTexture", false);
        }

        // ====================================================
        // Send the above data to device
        // ====================================================

        // Set vertices, indices and UV coords on the device
        owlTrianglesSetVertices(triangleGeom, vertexBufferObject, mesh->vertex.size(), sizeof(owl::common::vec3f), 0);
        owlTrianglesSetIndices(triangleGeom, indexBufferObject, mesh->index.size(), sizeof(owl::common::vec3i), 0);

        owlGeomSetBuffer(triangleGeom, "vertex", vertexBufferObject);
        owlGeomSetBuffer(triangleGeom, "normal", normalBufferObject);
        owlGeomSetBuffer(triangleGeom, "index", indexBufferObject);
        owlGeomSetBuffer(triangleGeom, "texCoord", texCoordBufferObject);
        owlGeomSet1ui(triangleGeom, "materialID", mesh->materialID);

        // ====================================================
        // Build the BLAS (GAS)
        // ====================================================
        OWLGroup triangleGroup = owlTrianglesGeomGroupCreate(context, 1, &triangleGeom);
        owlGroupBuildAccel(triangleGroup);

        // Add to a list, which is later used to build the IAS
        blasList.push_back(triangleGroup);
    }

    // ====================================================
    // Build he TLAS (IAS)
    // ====================================================
    this->world = owlInstanceGroupCreate(context, blasList.size(), blasList.data());
    owlGroupBuildAccel(this->world);

    // ====================================================
    // Setup a generic miss program
    // ====================================================
    OWLVarDecl missProgVars[] = {
        {"const_color", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, const_color)},
        {nullptr}
    };

    missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData), missProgVars, -1);

    // Set a constant background color in the miss program (black for now)
    owlMissProgSet3f(missProg, "const_color", owl3f{ 0.f, 0.f, 0.f });

    // ====================================================
    // Setup a pin-hole camera ray-gen program
    // ====================================================
    OWLVarDecl rayGenVars[] = {
        {"frameBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, frameBuffer)},
        {"frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenData, frameBufferSize)},
        {nullptr}
    };

    rayGen = owlRayGenCreate(this->context, this->module, "rayGen", sizeof(RayGenData), rayGenVars, -1);
    // Set the TLAS to be used
    owlParamsSetGroup(this->launchParams, "world", world);

    // ====================================================
    // Finally, build the programs, pipeline and SBT
    // ====================================================
    owlBuildPrograms(this->context);
    owlBuildPipeline(this->context);
    owlBuildSBT(this->context);
}

void Viewer::render()
{
    if (this->sbtDirty)
    {
        owlBuildSBT(context);
        this->sbtDirty = false;
    }
    if (CHECK_IF_LTC(this->rendererType) && accumId >= 2)
        ;
    else
    {
        owlParamsSet1i(this->launchParams, "accumId", this->accumId);
        owlLaunch2D(rayGen, this->fbSize.x, this->fbSize.y, this->launchParams);
        accumId++;
    }
}

/*! window notifies us that we got resized */
void Viewer::resize(const owl::common::vec2i& newSize)
{
    // Do not resize, what is the point..
    
    // Resize framebuffer, and other ops (OWL::Viewer ops)
    OWLViewer::resize(newSize);
    
    // Resize accumulation buffer, and set to launch params
    owlBufferResize(this->position_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "position_screen_buffer", this->position_screen_buffer);
    owlBufferResize(this->normal_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "normal_screen_buffer", this->normal_screen_buffer);
    owlBufferResize(this->uv_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "uv_screen_buffer", this->uv_screen_buffer);
    owlBufferResize(this->albedo_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "albedo_screen_buffer", this->albedo_screen_buffer);
    owlBufferResize(this->alpha_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "alpha_screen_buffer", this->alpha_screen_buffer);
    owlBufferResize(this->materialID_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "materialID_screen_buffer", this->materialID_screen_buffer);


    owlBufferResize(this->accum_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "accum_screen_buffer", this->accum_screen_buffer);
    owlBufferResize(this->bounce0_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "bounce0_screen_buffer", this->bounce0_screen_buffer);
    owlBufferResize(this->bounce1_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "bounce1_screen_buffer", this->bounce1_screen_buffer);
    owlBufferResize(this->bounce2_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "bounce2_screen_buffer", this->bounce2_screen_buffer);

    owlBufferResize(this->ltc_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "ltc_screen_buffer", this->ltc_screen_buffer);
    owlBufferResize(this->sto_direct_ratio_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "sto_direct_ratio_screen_buffer", this->sto_direct_ratio_screen_buffer);
    owlBufferResize(this->sto_no_vis_ratio_screen_buffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "sto_no_vis_ratio_screen_buffer", this->sto_no_vis_ratio_screen_buffer);


    // Perform camera move i.e. set new camera parameters, and set SBT to be updated
    this->cameraChanged();
}

void Viewer::setRendererType(RendererType type)
{
    this->rendererType = type;
    owlParamsSet1i(this->launchParams, "rendererType", (int)this->rendererType);
}

/*! window notifies us that the camera has changed */
void Viewer::cameraChanged()
{
    // Reset accumulation buffer, to restart MC sampling
    this->accumId = 0;

    const owl::common::vec3f lookFrom = camera.getFrom();
    const owl::common::vec3f lookAt = camera.getAt();
    const owl::common::vec3f lookUp = camera.getUp();
    const float cosFovy = camera.getCosFovy();

    printf("Fovy, %f\n lookfrom %f %f %f\nlookat %f %f %f\nlookup %f %f %f\n",
        cosFovy, lookFrom.x, lookFrom.y, lookFrom.z, lookAt.x, lookAt.y, lookAt.z, lookUp.x, lookUp.y, lookUp.z);

    // ----------- compute variable values  ------------------
    owl::common::vec3f camera_pos = lookFrom;
    owl::common::vec3f camera_d00 = normalize(lookAt - lookFrom);
    float aspect = fbSize.x / float(fbSize.y);
    owl::common::vec3f camera_ddu = cosFovy * aspect * normalize(cross(camera_d00, lookUp));
    owl::common::vec3f camera_ddv = cosFovy * normalize(cross(camera_ddu, camera_d00));
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    // ----------- set variables  ----------------------------
    owlRayGenSet1ul(rayGen, "frameBuffer", (uint64_t)this->fbPointer);
    owlRayGenSet2i(rayGen, "frameBufferSize", (const owl2i&)this->fbSize);

    owlParamsSet3f(this->launchParams, "camera.pos", (const owl3f&)camera_pos);
    owlParamsSet3f(this->launchParams, "camera.dir_00", (const owl3f&)camera_d00);
    owlParamsSet3f(this->launchParams, "camera.dir_du", (const owl3f&)camera_ddu);
    owlParamsSet3f(this->launchParams, "camera.dir_dv", (const owl3f&)camera_ddv);

    this->sbtDirty = true;

    this->denoise_setup();
    this->cameraPos++;

}

void Viewer::drawUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
        ImGui::Begin("Controls", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
        ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        int currentType = this->rendererType;
        ImGui::Combo("Renderer", &currentType, rendererNames, NUM_RENDERER_TYPES, 0);
        if (currentType != this->rendererType)
        {
            this->rendererType = static_cast<RendererType>(currentType);
            owlParamsSet1i(this->launchParams, "rendererType", currentType);
            this->cameraChanged();
        }

        float currentLerp = this->lerp;
        ImGui::SliderFloat("LERP", &currentLerp, 0.f, 1.f);
        if (currentLerp != this->lerp)
        {
            this->lerp = currentLerp;
            owlParamsSet1f(this->launchParams, "lerp", this->lerp);
            this->cameraChanged();
        }

        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


void Viewer::savebuffer(FILE* fp, OWLBuffer* owlbuffer, int avg_factor)
{
    if (fp && (owlbuffer != NULL))
    {
        int i = 0;
        void* temp = owlbuffer;
        const void* owlbuffer_pointer = owlBufferGetPointer(*owlbuffer, 0);
        void * localMemory = calloc(this->fbSize.x * this->fbSize.y, sizeof(float4));
        cudaMemcpy(localMemory, owlbuffer_pointer, this->fbSize.x * this->fbSize.y * sizeof(float4), cudaMemcpyDeviceToHost);
        while (i < this->fbSize.x * this->fbSize.y * 4)
        {
            ((float*)localMemory)[i] = ((float*)localMemory)[i] / avg_factor;
            i++;
            temp = (void*)((float*)temp + i);
        }
        fwrite(localMemory, sizeof(float4), this->fbSize.x * this->fbSize.y, fp);
        fclose(fp);
    }
}

void Viewer::mouseButtonLeft(const owl::common::vec2i& where, bool pressed)
{
    if (pressed == true) {
        printf("framesize %d %d\n\n\n", this->fbSize.x, this->fbSize.y);
        std::string fileName;
        FILE *fp;
        if (this->rendererType == RATIO)
        {

            //this->denoise(owlBufferGetPointer(this->sto_direct_ratio_screen_buffer, 0));
            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/stoDirect.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->sto_direct_ratio_screen_buffer, 1);

            //this->denoise(owlBufferGetPointer(this->sto_no_vis_ratio_screen_buffer, 0));
            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/stoNoVis.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->sto_no_vis_ratio_screen_buffer, 1);

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/ltc.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->ltc_screen_buffer, 1);



            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/normal_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->normal_screen_buffer, 1);

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/materialID_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->materialID_screen_buffer, 1);
        }
        if (this->rendererType == PATH)
        {

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/position_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->position_screen_buffer, 1);
            

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/normal_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->normal_screen_buffer, 1);
            
            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/uv_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->uv_screen_buffer, 1);

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/alpha_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->alpha_screen_buffer, 1);
            
            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/albedo_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->albedo_screen_buffer, 1);

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/materialID_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->materialID_screen_buffer, 1);
                
            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/bounce0_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->bounce0_screen_buffer, this->accumId);

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/bounce1_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->bounce1_screen_buffer, this->accumId);

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/bounce2_screen_buffer.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->bounce2_screen_buffer, this->accumId);

            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/path.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->accum_screen_buffer, this->accumId);
        }

        if (this->rendererType == LTC_BASELINE)
        {
            fileName = "C:/Users/dhawals/repos/optix_renderer/saves/ltc_baseline.btc";
            fp = fopen(fileName.c_str(), "wb");
            savebuffer(fp, &this->accum_screen_buffer, 1);
        }
    }
}

void Viewer::key(char key, const owl::common::vec2i& pos)
{
    if (key == '1' || key == '!') {
        this->camera.setOrientation(this->camera.getFrom(), owl::common::vec3f(0.f), owl::common::vec3f(0.f, 0.f, 1.f), this->camera.getFovyInDegrees());
        this->cameraChanged();
    }
    else if (key == 'R' || key == 'r') {
        SceneCamera cam;
        cam.from = this->camera.getFrom();
        cam.at = this->camera.getAt();
        cam.up = this->camera.getUp();
        cam.cosFovy = this->camera.getCosFovy();

        this->recordedCameras.push_back(cam);
    }
    else if (key == 'F' || key == 'f') {
        nlohmann::json camerasJson;

        for (auto cam : this->recordedCameras) {
            nlohmann::json oneCameraJson;
            std::vector<float> from, at, up;

            for (int i = 0; i < 3; i++) {
                from.push_back(cam.from[i]);
                at.push_back(cam.at[i]);
                up.push_back(cam.up[i]);
            }

            oneCameraJson["from"] = from;
            oneCameraJson["to"] = at;
            oneCameraJson["up"] = up;
            oneCameraJson["cos_fovy"] = cam.cosFovy;

            camerasJson.push_back(oneCameraJson);
        }

        this->currentScene.json["cameras"] = camerasJson;
    }

    else if (key == 'P') {
        printf("Keypress");
        this->screenShot(this->to_save_file);
    }
}

void Viewer::denoise_setup()
{
    //int frameSize = this->fbSize.x * this->fbSize.y;
    //owl::common::vec2i frameRes = this->fbSize;
    //int tileSize = 256;
    //this->numBlocksAndThreads = owl::common::vec2i(frameRes.y / tileSize, frameRes.x / tileSize);
    //auto optixContext = owlContextGetOptixContext(context, 0);

    //OptixDenoiserOptions denoiserOptions = {};
    //optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_HDR, &denoiserOptions, &denoiser);

    //OptixDenoiserSizes denoiserReturnSizes;
    //optixDenoiserComputeMemoryResources(this->denoiser, frameRes.x, frameRes.y, &denoiserReturnSizes);

    ///*
    //size_t denoiserScratchSize;
    //size_t denoiserStateSize;
    //OptixDenoiser denoiser;
    //OWLBuffer denoiserScratch{ 0 };
    //OWLBuffer denoiserState{ 0 };
    //OWLBuffer denoisedBuffer{ 0 };
    //OWL_float denoisedIntensity;
    //*/

    //this->denoiserScratchSize = std::max(denoiserReturnSizes.withOverlapScratchSizeInBytes, denoiserReturnSizes.withoutOverlapScratchSizeInBytes);
    //this->denoiserStateSize = denoiserReturnSizes.stateSizeInBytes;

    //cudaMalloc(&this->denoiserScratch, this->denoiserScratchSize);
    //cudaMalloc(&this->denoiserState, this->denoiserStateSize);

    //denoiserScratch = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    //owlBufferResize(denoiserScratch, this->getWindowSize().x * this->getWindowSize().y);

    //denoiserState = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    //owlBufferResize(denoiserState, this->getWindowSize().x * this->getWindowSize().y);

    //denoisedBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    //owlBufferResize(denoisedBuffer, this->getWindowSize().x * this->getWindowSize().y);

    //optixDenoiserSetup(this->denoiser, 
    //    0,
    //    frameRes.x, frameRes.y,
    //    (CUdeviceptr)owlBufferGetPointer(this->denoiserState, 0),
    //    this->denoiserStateSize,
    //    (CUdeviceptr)owlBufferGetPointer(this->denoiserScratch, 0),
    //    this->denoiserScratchSize);
}

void Viewer::denoise(const void* to_denoise_ptr)
{
    //OptixDenoiserParams denoiserParams;
    //denoiserParams.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    //denoiserParams.hdrIntensity = (CUdeviceptr)&this->denoisedIntensity;
    //denoiserParams.blendFactor = 0.0f;

    //OptixImage2D inputLayer[1];
    //inputLayer[0].data = (CUdeviceptr)to_denoise_ptr;
    //inputLayer[0].width = this->fbSize.x;
    //inputLayer[0].height = this->fbSize.y;
    //inputLayer[0].rowStrideInBytes = this->fbSize.x * sizeof(float4);
    //inputLayer[0].pixelStrideInBytes = sizeof(float4);
    //inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    ///*
    //inputLayer[1].data = (CUdeviceptr)this->materialID_screen_buffer;
    //inputLayer[1].width = this->fbSize.x;
    //inputLayer[1].height = this->fbSize.y;
    //inputLayer[1].rowStrideInBytes = this->fbSize.x * sizeof(float4);
    //inputLayer[1].pixelStrideInBytes = sizeof(float4);
    //inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    //inputLayer[2].data = (CUdeviceptr)this->normalBuffer;
    //inputLayer[2].width = this->fbSize.x;
    //inputLayer[2].height = this->fbSize.y;
    //inputLayer[2].rowStrideInBytes = this->fbSize.x * sizeof(float4);
    //inputLayer[2].pixelStrideInBytes = sizeof(float4);
    //inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;
    //*/

    //OptixImage2D outputLayer;
    //outputLayer.data = (CUdeviceptr)owlBufferGetPointer(this->denoisedBuffer, 0);
    //outputLayer.width = this->fbSize.x;
    //outputLayer.height = this->fbSize.y;
    //outputLayer.rowStrideInBytes = this->fbSize.x * sizeof(float4);
    //outputLayer.pixelStrideInBytes = sizeof(float4);
    //outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    //OptixDenoiserGuideLayer denoiserGuideLayer = {};
    ///*denoiserGuideLayer.materialID_screen_buffer = inputLayer[1];
    //denoiserGuideLayer.normalBuffer = inputLayer[2];*/

    //OptixDenoiserLayer denoiserLayer = {};
    //denoiserLayer.input = inputLayer[0];
    //denoiserLayer.output = outputLayer;

    //optixDenoiserComputeIntensity(this->denoiser,
    //    /*stream*/0,
    //    &inputLayer[0],
    //    (CUdeviceptr)&this->denoisedIntensity,
    //    (CUdeviceptr)owlBufferGetPointer(this->denoiserScratch, 0),
    //    this->denoiserScratchSize
    //);

    //optixDenoiserInvoke(this->denoiser,
    //    /*stream*/0,
    //    &denoiserParams,
    //    (CUdeviceptr)owlBufferGetPointer(this->denoiserState, 0),
    //    this->denoiserStateSize,
    //    &denoiserGuideLayer,
    //    &denoiserLayer, 1,
    //    /*inputOffsetX*/0,
    //    /*inputOffsetY*/0,
    //    (CUdeviceptr)owlBufferGetPointer(this->denoiserScratch, 0),
    //    this->denoiserScratchSize
    //);
}