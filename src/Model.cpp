// ======================================================================== //
// Copyright 2018-2021 Ingo Wald                                            //
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

#include "Model.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

//std
#include <set>

namespace std {
    inline bool operator<(const tinyobj::index_t& a,
        const tinyobj::index_t& b)
    {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;

        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;

        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;

        return false;
    }
}

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

    /*! find vertex with given position, normal, texcoord, and return
        its vertex ID, or, if it doesn't exit, add it to the mesh, and
        its just-created index */
    int addVertex(TriangleMesh* mesh,
        tinyobj::attrib_t& attributes,
        const tinyobj::index_t& idx,
        std::map<tinyobj::index_t, int>& knownVertices)
    {
        if (knownVertices.find(idx) != knownVertices.end())
            return knownVertices[idx];

        const owl::common::vec3f* vertex_array = (const owl::common::vec3f*)attributes.vertices.data();
        const owl::common::vec3f* normal_array = (const owl::common::vec3f*)attributes.normals.data();
        const owl::common::vec2f* texcoord_array = (const owl::common::vec2f*)attributes.texcoords.data();

        int newID = (int)mesh->vertex.size();
        knownVertices[idx] = newID;

        mesh->vertex.push_back(vertex_array[idx.vertex_index]);
        if (idx.normal_index >= 0) {
            while (mesh->normal.size() < mesh->vertex.size())
                mesh->normal.push_back(normal_array[idx.normal_index]);
        }
        if (idx.texcoord_index >= 0) {
            while (mesh->texcoord.size() < mesh->vertex.size())
                mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
        }

        return newID;
    }

    /*! load a texture (if not already loaded), and return its ID in the
        model's textures[] vector. Textures that could not get loaded
        return -1 */
    int loadTexture(Model* model,
        std::map<std::string, int>& knownTextures,
        const std::string& inFileName,
        const std::string& modelPath)
    {
        if (inFileName == "")
            return -1;

        if (knownTextures.find(inFileName) != knownTextures.end())
            return knownTextures[inFileName];

        std::string fileName = inFileName;
        // first, fix backspaces:
        for (auto& c : fileName)
            if (c == '\\') c = '/';
        fileName = modelPath + "/" + fileName;

        owl::common::vec2i res;
        int   comp;
        unsigned char* image = stbi_load(fileName.c_str(),
            &res.x, &res.y, &comp, STBI_rgb_alpha);
        int textureID = -1;
        if (image) {
            textureID = (int)model->textures.size();
            Texture* texture = new Texture;
            texture->resolution = res;
            texture->pixel = (uint32_t*)image;

            /* iw - actually, it seems that stbi loads the pictures
                mirrored along the y axis - mirror them here */
            for (int y = 0; y < res.y / 2; y++) {
                uint32_t* line_y = texture->pixel + y * res.x;
                uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
                int mirror_y = res.y - 1 - y;
                for (int x = 0; x < res.x; x++) {
                    std::swap(line_y[x], mirrored_y[x]);
                }
            }

            model->textures.push_back(texture);
        }
        else {
            std::cout << "Could not load texture from " << fileName;
        }

        knownTextures[inFileName] = textureID;
        return textureID;
    }

    Model* loadOBJ(const std::string& objFile)
    {
        Model* model = new Model;

        const std::string modelDir
            = objFile.substr(0, objFile.rfind('/') + 1);

        tinyobj::attrib_t attributes;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string err = "";

        bool readOK
            = tinyobj::LoadObj(&attributes,
                &shapes,
                &materials,
                &err,
                &err,
                objFile.c_str(),
                modelDir.c_str(),
                /* triangulate */true);
        if (!readOK) {
            throw std::runtime_error("Could not read OBJ model from " + objFile + " : " + err);
        }

        if (materials.empty())
            throw std::runtime_error("could not parse materials ...");

        const owl::common::vec3f* vertex_array = (const owl::common::vec3f*)attributes.vertices.data();
        const owl::common::vec3f* normal_array = (const owl::common::vec3f*)attributes.normals.data();
        const owl::common::vec2f* texcoord_array = (const owl::common::vec2f*)attributes.texcoords.data();

        std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
        std::map<std::string, int>      knownTextures;
        for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
            tinyobj::shape_t& shape = shapes[shapeID];

            std::set<int> materialIDs;
            for (auto faceMatID : shape.mesh.material_ids)
                materialIDs.insert(faceMatID);

            std::map<tinyobj::index_t, int> knownVertices;

            for (int materialID : materialIDs) {
                TriangleMesh* mesh = new TriangleMesh;

                for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                    if (shape.mesh.material_ids[faceID] != materialID) continue;
                    tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                    tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                    tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                    // owl::common::vec3i idx(addVertex(mesh, attributes, idx0, knownVertices),
                    //     addVertex(mesh, attributes, idx1, knownVertices),
                    //     addVertex(mesh, attributes, idx2, knownVertices));

                    owl::common::vec3i vidx(mesh->vertex.size(), mesh->vertex.size() + 1, mesh->vertex.size() + 2);
                    mesh->vertex.push_back(vertex_array[idx0.vertex_index]);
                    mesh->vertex.push_back(vertex_array[idx1.vertex_index]);
                    mesh->vertex.push_back(vertex_array[idx2.vertex_index]);
                    mesh->index.push_back(vidx);

                    owl::common::vec3i nidx(mesh->normal.size(), mesh->normal.size() + 1, mesh->normal.size() + 2);
                    mesh->normal.push_back(normal_array[idx0.normal_index]);
                    mesh->normal.push_back(normal_array[idx1.normal_index]);
                    mesh->normal.push_back(normal_array[idx2.normal_index]);
                    // mesh->index.push_back(nidx);

                    owl::common::vec3i tidx(mesh->texcoord.size(), mesh->texcoord.size() + 1, mesh->texcoord.size() + 2);
                    mesh->texcoord.push_back(texcoord_array[idx0.texcoord_index]);
                    mesh->texcoord.push_back(texcoord_array[idx1.texcoord_index]);
                    mesh->texcoord.push_back(texcoord_array[idx2.texcoord_index]);
                    // mesh->index.push_back(tidx);

                    mesh->diffuse = (const owl::common::vec3f&)materials[materialID].diffuse;
                    mesh->diffuseTextureID = loadTexture(model,
                        knownTextures,
                        materials[materialID].diffuse_texname,
                        modelDir);

                    mesh->alpha = (const float)materials[materialID].shininess;
                    mesh->alphaTextureID = loadTexture(model,
                        knownTextures,
                        materials[materialID].specular_texname,
                        modelDir);


                    mesh->normalTextureID = loadTexture(model,
                        knownTextures,
                        materials[materialID].bump_texname,
                        modelDir);

                    mesh->emit = (const owl::common::vec3f&)materials[materialID].emission;
                    mesh->materialID = materialID + 1;
                }

                if (mesh->vertex.empty()) {
                    delete mesh;
                }
                else {
                    for (auto idx : mesh->index) {
                        if (idx.x < 0 || idx.x >= (int)mesh->vertex.size() ||
                            idx.y < 0 || idx.y >= (int)mesh->vertex.size() ||
                            idx.z < 0 || idx.z >= (int)mesh->vertex.size()) {
                            std::cout << ("invalid triangle indices");
                            throw std::runtime_error("invalid triangle indices");
                        }
                    }

                    model->meshes.push_back(mesh);
                }
            }
        }

        // of course, you should be using tbb::parallel_for for stuff
        // like this:
        for (auto mesh : model->meshes)
            for (auto vtx : mesh->vertex)
                model->bounds.extend(vtx);

        std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
        return model;
    }

}
