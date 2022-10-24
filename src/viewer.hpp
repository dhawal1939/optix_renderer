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
#include <owl/owl.h>
// viewer base class, for window and user interaction
#include <owlViewer/OWLViewer.h>
#include <owl/common/math/vec.h>
#include <owl/common/math/random.h>

// our device-side data structures
#include "deviceCode.h"

// Geometry Headers
#include <Model.h>
#include <scene.h>

// ImGUI
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// LTC LUT
#include <ltc/ltc_lut.h>

#include <cuda_headers/common.cuh>

using namespace owl;

#define LOG(message)                                            \
    std::cout << OWL_TERMINAL_BLUE;                             \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
    std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
    std::cout << "#owl.sample(main): " << message << std::endl; \
    std::cout << OWL_TERMINAL_DEFAULT;

// Compiled PTX code
extern "C" char deviceCode_ptx[];

// const vec2i fbSize(800,600);
const vec3f init_lookFrom(-4.f, +3.f, -2.f);
const vec3f init_lookAt(0.f, 0.f, 0.f);
const vec3f init_lookUp(0.f, 1.f, 0.f);
const float init_cosFovy = 0.66f;

struct Viewer : public owl::viewer::OWLViewer
{
    int accumId = 0;
    Scene current_scene;

    RendererType renderer_type;
    bool sbtDirty = true;

    OWLRayGen rayGen{0};
    OWLMissProg missProg{0};

    // Buffers
    OWLBuffer accumBuffer{0};
    OWLBuffer UBuffer{0};
    OWLBuffer SBuffer{0};

    OWLGroup world; // TLAS Top level accelration structure

    OWLContext context{0};
    OWLModule module{0};

    OWLParams launch_params;

    std::vector<SceneCamera> recorded_cameras;

    std::vector<TriLight> tri_light_list;
    std::vector<MeshLight> mesh_light_list;

    float lerp = 0.5f;

    Viewer(Scene &scene, vec2i resolution, bool interactive, bool vsync);

    int imgui_init(bool _callbacks);

    void setRendererType(RendererType type);


    /*! gets called whenever the viewer needs us to re-render out widget */
    void render() override;
    void drawUI();

    /*! window notifies us that we got resized. We HAVE to override
        this to know our actual render dimensions, and get pointer
        to the device frame buffer that the viewer cated for us */
    void resize(const vec2i &newSize) override;

    /*! this function gets called whenever any camera manipulator
      updates the camera. gets called AFTER all values have been updated */
    void cameraChanged() override;
};

int Viewer::imgui_init(bool _callbacks)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(this->handle, true);
    ImGui_ImplOpenGL3_Init();
    return ImGui_ImplOpenGL3_Init();
    ;
}

Viewer::Viewer(Scene &scene, vec2i resolution, bool interactive = true, bool vsync = false)
    : owl::viewer::OWLViewer("Optix Viewer", resolution, interactive, vsync)
{

    this->current_scene = scene;
    if (!this->imgui_init(true))
        LOG("IMGUI Init Failed")

    // create a context on the first device:
    this->context = owlContextCreate(nullptr, 1);
    OWLModule module = owlModuleCreate(this->context, deviceCode_ptx);

    this->accumBuffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    this->SBuffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    this->UBuffer = owlDeviceBufferCreate(this->context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(this->accumBuffer,
                    this->getWindowSize().x * this->getWindowSize().y);
    owlBufferResize(this->SBuffer, this->getWindowSize().x * this->getWindowSize().y);
    owlBufferResize(this->UBuffer, this->getWindowSize().x * this->getWindowSize().y);

    // Is it type of rays
    owlContextSetRayTypeCount(this->context, 2);

    /*
          LIGHT Geometry reader
    */

    Model *tri_lights = scene.tri_lights;

    int total_triangles;

    for (auto light : tri_lights->meshes)
    {
        MeshLight meshLight;
        meshLight.flux = 0.f;
        meshLight.triIdx = this->tri_light_list.size();
        meshLight.triStartIdx = total_triangles;

        int num_tris = 0;

        for (auto index : light->index)
        {
            TriLight triLight;

            triLight.v1 = light->vertex[index.x];
            triLight.v2 = light->vertex[index.y];
            triLight.v3 = light->vertex[index.z];

            triLight.cg = (triLight.v1 + triLight.v2 + triLight.v3) / 3.f;
            triLight.normal = normalize(light->normal[index.x] + light->normal[index.y] + light->normal[index.z]);
            triLight.area = 0.5f * length(cross(triLight.v1 - triLight.v2, triLight.v3 - triLight.v2));

            triLight.emit = light->emit;
            triLight.flux = 3.1415926f * triLight.area * (triLight.emit.x + triLight.emit.y + triLight.emit.z) / 3.f;

            meshLight.flux += triLight.flux;

            num_tris++;
        }

        total_triangles += num_tris;

        meshLight.triCount = num_tris;

        mesh_light_list.push_back(meshLight);

        std::cout << std::endl
                  << "**************" << std::endl;
    }

    OWLVarDecl launchParamsDecl[] = {
        {"triLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, triLights)},
        {"numTriLights", OWL_INT, OWL_OFFSETOF(LaunchParams, numTriLights)},
        // Light edges
        {"meshLights", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, meshLights)},
        {"numMeshLights", OWL_INT, OWL_OFFSETOF(LaunchParams, numMeshLights)},
        // All other parameters
        {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accumBuffer)},
        {"UBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, UBuffer)},
        {"SBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, SBuffer)},
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
        {nullptr}};
    this->launch_params = owlParamsCreate(this->context,
                                          sizeof(LaunchParams),
                                          launchParamsDecl, -1);

    // Random controls
    owlParamsSet1f(this->launch_params, "lerp", this->lerp);

    // Set LTC matrices (8x8, since only isotropic)
    OWLTexture ltc1 = owlTexture2DCreate(this->context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_1,
                                         OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);
    OWLTexture ltc2 = owlTexture2DCreate(this->context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_2,
                                         OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);
    OWLTexture ltc3 = owlTexture2DCreate(this->context, OWL_TEXEL_FORMAT_RGBA32F, 8, 8, ltc_iso_3,
                                         OWL_TEXTURE_LINEAR, OWL_TEXTURE_CLAMP);

    owlParamsSet1i(this->launch_params, "rendererType", (int)this->renderer_type);

    // Upload the <actual> triangle data for all area lights
    OWLBuffer triLightsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(TriLight), this->tri_light_list.size(), this->tri_light_list.data());
    owlParamsSetBuffer(this->launch_params, "triLights", triLightsBuffer);
    owlParamsSet1i(this->launch_params, "numTriLights", this->tri_light_list.size());

    // Upload the mesh data for all area lights
    OWLBuffer meshLightsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(MeshLight), this->mesh_light_list.size(), this->mesh_light_list.data());
    owlParamsSetBuffer(this->launch_params, "meshLights", meshLightsBuffer);
    owlParamsSet1i(this->launch_params, "numMeshLights", this->mesh_light_list.size());

    // Upload accumulation buffer and ID
    owlParamsSet1i(this->launch_params, "accumId", this->accumId);
    owlParamsSetBuffer(this->launch_params, "accumBuffer", this->accumBuffer);
    owlParamsSetBuffer(this->launch_params, "UBuffer", this->UBuffer);
    owlParamsSetBuffer(this->launch_params, "SBuffer", this->SBuffer);

    /*
    BLAS LIST
    */

    std::vector<OWLGroup> blasList;

    // Loop over meshes set up and Instance Accel. Structure and Geometry Accel. Structure

    Model *model = scene.model;
    for (auto mesh : model->meshes)
    {

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

            {"diffuse", OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshData, diffuse)},
            {"diffuse_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, diffuse_texture)},
            {"hasDiffuseTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasDiffuseTexture)},

            {"alpha", OWL_FLOAT, OWL_OFFSETOF(TriangleMeshData, alpha)},
            {"alpha_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, alpha_texture)},
            {"hasAlphaTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasAlphaTexture)},

            {nullptr}};

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
        owlGeomTypeSetClosestHit(triangleGeomType, SHADOW_RAY_TYPE, module, "triangleMeshCHShadow");

        // Create the actual geometry on the device
        OWLGeom triangleGeom = owlGeomCreate(context, triangleGeomType);

        // ====================================================
        // Data setup
        // ====================================================

        // Create CUDA buffers from mesh vertices, indices and UV coordinates
        OWLBuffer vertexBuffer = owlDeviceBufferCreate(this->context, OWL_FLOAT3, mesh->vertex.size(), mesh->vertex.data());
        OWLBuffer normalBuffer = owlDeviceBufferCreate(this->context, OWL_FLOAT3, mesh->normal.size(), mesh->normal.data());
        OWLBuffer indexBuffer = owlDeviceBufferCreate(this->context, OWL_INT3, mesh->index.size(), mesh->index.data());
        OWLBuffer texCoordBuffer = owlDeviceBufferCreate(this->context, OWL_FLOAT2, mesh->texcoord.size(), mesh->texcoord.data());

        // Set emission value, and more importantly, if the current mesh is a light
        owlGeomSet1b(triangleGeom, "isLight", mesh->isLight);
        owlGeomSet3f(triangleGeom, "emit", owl3f{mesh->emit.x, mesh->emit.y, mesh->emit.z});

        // Create CUDA buffers and upload them for diffuse and alpha textures
        if (mesh->diffuseTextureID != -1)
        {
            Texture *diffuseTexture = model->textures[mesh->diffuseTextureID];
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
        else
        {
            owlGeomSet3f(triangleGeom, "diffuse", owl3f{mesh->diffuse.x, mesh->diffuse.y, mesh->diffuse.z});
            owlGeomSet1b(triangleGeom, "hasDiffuseTexture", false);
        }

        if (mesh->alphaTextureID != -1)
        {
            Texture *alphaTexture = model->textures[mesh->alphaTextureID];
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
        else
        {
            owlGeomSet1f(triangleGeom, "alpha", mesh->alpha);
            owlGeomSet1b(triangleGeom, "hasAlphaTexture", false);
        }

        // ====================================================
        // Send the above data to device
        // ====================================================

        // Set vertices, indices and UV coords on the device
        owlTrianglesSetVertices(triangleGeom, vertexBuffer,
                                mesh->vertex.size(), sizeof(vec3f), 0);
        owlTrianglesSetIndices(triangleGeom, indexBuffer,
                               mesh->index.size(), sizeof(vec3i), 0);

        owlGeomSetBuffer(triangleGeom, "vertex", vertexBuffer);
        owlGeomSetBuffer(triangleGeom, "normal", normalBuffer);
        owlGeomSetBuffer(triangleGeom, "index", indexBuffer);
        owlGeomSetBuffer(triangleGeom, "texCoord", texCoordBuffer);

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
    world = owlInstanceGroupCreate(context, blasList.size(), blasList.data());
    owlGroupBuildAccel(world);

    // ====================================================
    // Setup a generic miss program
    // ====================================================
    OWLVarDecl missProgVars[] = {
        {"const_color", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, const_color)},
        {nullptr}};

    missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData), missProgVars, -1);

    // Set a constant background color in the miss program (black for now)
    owlMissProgSet3f(missProg, "const_color", owl3f{0.f, 0.f, 0.f});

    // ====================================================
    // Setup a pin-hole camera ray-gen program
    // ====================================================
    OWLVarDecl rayGenVars[] = {
        {"frameBuffer", OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData, frameBuffer)},
        {"frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenData, frameBufferSize)},
        {nullptr}};

    rayGen = owlRayGenCreate(context, module, "rayGen", sizeof(RayGenData), rayGenVars, -1);
    // Set the TLAS to be used
    owlParamsSetGroup(this->launch_params, "world", world);

    // ====================================================
    // Finally, build the programs, pipeline and SBT
    // ====================================================
    owlBuildPrograms(this->context);
    owlBuildPipeline(this->context);
    owlBuildSBT(this->context);
}

void Viewer::render()
{
    if (sbtDirty)
    {
        owlBuildSBT(context);
        sbtDirty = false;
    }
    if (CHECK_IF_LTC(this->renderer_type) && accumId >= 2)
        ;
    else
    {
        owlParamsSet1i(this->launch_params, "accumId", this->accumId);
        owlLaunch2D(rayGen, this->fbSize.x, this->fbSize.y, this->launch_params);
        accumId++;
    }
}

/*! window notifies us that we got resized */
void Viewer::resize(const vec2i &newSize)
{
    // Resize framebuffer, and other ops (OWL::Viewer ops)
    OWLViewer::resize(newSize);

    // Resize accumulation buffer, and set to launch params
    owlBufferResize(this->accumBuffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launch_params, "accumBuffer", this->accumBuffer);

    owlBufferResize(this->UBuffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launch_params, "UBuffer", this->UBuffer);

    owlBufferResize(this->SBuffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launch_params, "SBuffer", this->UBuffer);

    // Perform camera move i.e. set new camera parameters, and set SBT to be updated
    this->cameraChanged();
}

void Viewer::setRendererType(RendererType type)
{
    this->renderer_type = type;
    owlParamsSet1i(this->launch_params, "rendererType", (int)this->renderer_type);
}

/*! window notifies us that the camera has changed */
void Viewer::cameraChanged()
{
    const vec3f lookFrom = camera.getFrom();
    const vec3f lookAt = camera.getAt();
    const vec3f lookUp = camera.getUp();
    const float cosFovy = camera.getCosFovy();
    // ----------- compute variable values  ------------------
    vec3f camera_pos = lookFrom;
    vec3f camera_d00 = normalize(lookAt - lookFrom);
    float aspect = fbSize.x / float(fbSize.y);
    vec3f camera_ddu = cosFovy * aspect * normalize(cross(camera_d00, lookUp));
    vec3f camera_ddv = cosFovy * normalize(cross(camera_ddu, camera_d00));
    camera_d00 -= 0.5f * camera_ddu;
    camera_d00 -= 0.5f * camera_ddv;

    // ----------- set variables  ----------------------------
    owlRayGenSet1ul(rayGen, "fbPtr", (uint64_t)fbPointer);
    // owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
    owlRayGenSet2i(rayGen, "fbSize", (const owl2i &)fbSize);
    owlRayGenSet3f(rayGen, "camera.pos", (const owl3f &)camera_pos);
    owlRayGenSet3f(rayGen, "camera.dir_00", (const owl3f &)camera_d00);
    owlRayGenSet3f(rayGen, "camera.dir_du", (const owl3f &)camera_ddu);
    owlRayGenSet3f(rayGen, "camera.dir_dv", (const owl3f &)camera_ddv);
    sbtDirty = true;
}

void Viewer::drawUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
        ImGui::Begin("Controls", NULL, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
        ImGui::Text("%.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

        int currentType = this->renderer_type;
        ImGui::Combo("Renderer", &currentType, rendererNames, NUM_RENDERER_TYPES, 0);
        if (currentType != this->renderer_type)
        {
            this->renderer_type = static_cast<RendererType>(currentType);
            owlParamsSet1i(this->launch_params, "rendererType", currentType);
            this->cameraChanged();
        }

        float currentLerp = this->lerp;
        ImGui::SliderFloat("LERP", &currentLerp, 0.f, 1.f);
        if (currentLerp != this->lerp)
        {
            this->lerp = currentLerp;
            owlParamsSet1f(this->launch_params, "lerp", this->lerp);
            this->cameraChanged();
        }

        ImGui::End();
    }

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}