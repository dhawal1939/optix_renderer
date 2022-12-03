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
    void save_full(const std::string& fileName);

    int imgui_init(bool _callbacks, const char* gl_version);

    //void savebuffer(FILE* fp);

    std::string gl_version;

    std::string to_save_file;
    // /*! this function gets called whenever any camera manipulator
    //   updates the camera. gets called AFTER all values have been updated */
    void cameraChanged() override;

    void key(char key, const owl::common::vec2i& pos) override;
    void mouseButtonLeft(const owl::common::vec2i & where, bool pressed) override;

    void setRendererType(RendererType type);

    RendererType rendererType;
    bool sbtDirty = true;

    OWLBuffer accumBuffer{ 0 };
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

    this->gl_version = "4.5";
    if (!this->imgui_init(true, this->gl_version.c_str()))
    {
        LOG("IMGUI Init Failed")
    }

    // Context & Module creation, accumulation buffer initialize
    context = owlContextCreate(nullptr, 1);
    module = owlModuleCreate(context, deviceCode_ptx);

    accumBuffer = owlDeviceBufferCreate(context, OWL_FLOAT4, 1, nullptr);
    owlBufferResize(accumBuffer, this->getWindowSize().x * this->getWindowSize().y);

    owlContextSetRayTypeCount(context, 2);

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
        // All other parameters
        {"accumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, accumBuffer)},
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

    // Upload accumulation buffer and ID
    owlParamsSet1i(this->launchParams, "accumId", this->accumId);
    owlParamsSetBuffer(this->launchParams, "accumBuffer", this->accumBuffer);

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

            {"diffuse", OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshData, diffuse)},
            {"diffuse_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, diffuse_texture)},
            {"hasDiffuseTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasDiffuseTexture)},

            {"alpha", OWL_FLOAT, OWL_OFFSETOF(TriangleMeshData, alpha)},
            {"alpha_texture", OWL_TEXTURE, OWL_OFFSETOF(TriangleMeshData, alpha_texture)},
            {"hasAlphaTexture", OWL_BOOL, OWL_OFFSETOF(TriangleMeshData, hasAlphaTexture)},
            
            {"normal", OWL_FLOAT, OWL_OFFSETOF(TriangleMeshData, normal_map)},
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
        //owlGeomTypeSetClosestHit(triangleGeomType, SHADOW_RAY_TYPE, module, "triangleMeshCHShadow");

        // Create the actual geometry on the device
        OWLGeom triangleGeom = owlGeomCreate(context, triangleGeomType);

        // ====================================================
        // Data setup
        // ====================================================

        // Create CUDA buffers from mesh vertices, indices and UV coordinates
        OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->vertex.size(), mesh->vertex.data());
        OWLBuffer normalBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, mesh->normal.size(), mesh->normal.data());
        OWLBuffer indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, mesh->index.size(), mesh->index.data());
        OWLBuffer texCoordBuffer = owlDeviceBufferCreate(context, OWL_FLOAT2, mesh->texcoord.size(), mesh->texcoord.data());

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
        owlTrianglesSetVertices(triangleGeom, vertexBuffer,
            mesh->vertex.size(), sizeof(owl::common::vec3f), 0);
        owlTrianglesSetIndices(triangleGeom, indexBuffer,
            mesh->index.size(), sizeof(owl::common::vec3i), 0);

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

    rayGen = owlRayGenCreate(context, module, "rayGen", sizeof(RayGenData), rayGenVars, -1);
    // Set the TLAS to be used
    owlParamsSetGroup(this->launchParams, "world", world);

    // ====================================================
    // Finally, build the programs, pipeline and SBT
    // ====================================================
    owlBuildPrograms(context);
    owlBuildPipeline(context);
    owlBuildSBT(context);
}

void Viewer::render()
{
    if (sbtDirty)
    {
        owlBuildSBT(context);
        sbtDirty = false;
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
    // Resize framebuffer, and other ops (OWL::Viewer ops)
    OWLViewer::resize(newSize);

    // Resize accumulation buffer, and set to launch params
    owlBufferResize(this->accumBuffer, newSize.x * newSize.y);
    owlParamsSetBuffer(this->launchParams, "accumBuffer", this->accumBuffer);

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


void Viewer::save_full(const std::string& fileName)
{
    const uint32_t* fb
        = (const uint32_t*)fbPointer;

    std::vector<uint32_t> pixels;
    for (int y = 0; y < fbSize.y; y++) {
        const uint32_t* line = fb + (fbSize.y - 1 - y) * fbSize.x;
        for (int x = 0; x < fbSize.x; x++) {
            pixels.push_back(line[x] | (0xff << 24));
        }
    }
    printf("%d Pixels\n", pixels.size());

    std::fstream _file;
    _file.open("vector_file_2.txt", std::ios_base::out);

    std::vector<std::uint32_t>::iterator itr;

    for (itr = pixels.begin(); itr != pixels.end(); itr++)
    {
        _file << *itr << std::endl;
    }

    _file.close();

    std::cout << "#owl.viewer: frame buffer written to " << fileName << std::endl;
}

//void Viewer::savebuffer(FILE* fp)
//{
//    const uint32_t* fb
//        = (const uint32_t*)fbPointer;
//    
//    std::vector<uint32_t> pixels;
//    for (int y = 0; y < fbSize.y; y++) {
//        const uint32_t* line = fb + (fbSize.y - 1 - y) * fbSize.x;
//        for (int x = 0; x < fbSize.x; x++) {
//            pixels.push_back(line[x] | (0xff << 24));
//        }
//    }
//    std::ofstream output_file("C:/Users/dhawals/repos/optix_renderer/rendered_output.ppm");
//    std::ostream_iterator<std::string> output_iterator(output_file, "\n");
//    std::copy(pixels.begin(), pixels.end(), output_iterator);
//}

void Viewer::mouseButtonLeft(const owl::common::vec2i& where, bool pressed)
{
    if (pressed == true) {
    
        std::string fileName = "C:/Users/dhawals/repos/optix_renderer/rendered_output.btc";
        FILE *fp;
        fp = fopen(fileName.c_str(), "wb");
        const void* owlbuffer = owlBufferGetPointer(accumBuffer, 0);
        void* localMemory = calloc(this->fbSize.x * this->fbSize.y, sizeof(float4));
        cudaMemcpy(localMemory, owlbuffer, this->fbSize.x * this->fbSize.y * sizeof(float4), cudaMemcpyDeviceToHost);
        if (fp)
        {
            int i = 0;
            void* temp = localMemory;
            while (i < this->fbSize.x * this->fbSize.y * 4)
            {
                ((float*)localMemory)[i] = ((float*)localMemory)[i] / (this->accumId + 1);
                i++;
                temp = (void*)((float*)temp + i);
            }
            printf("accum id %d\n", this->accumId);
            fwrite(localMemory, sizeof(float4), this->fbSize.x * this->fbSize.y, fp);
            fclose(fp);
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