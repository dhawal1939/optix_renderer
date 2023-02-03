// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
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

#pragma once

#include <owl/owl.h>
#include <owl/common/math/AffineSpace.h>
#include <vector>
#include <common.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    /*! a simple indexed triangle mesh that our sample renderer will
        render */
    struct TriangleMesh
    {
        std::vector<VEC3f> vertex;
        std::vector<VEC3f> normal;
        std::vector<VEC2f> texcoord;
        std::vector<VEC3i> index;

        // material data:
        VEC3f diffuse;
        int diffuseTextureID{ -1 };

        float alpha; // roughness
        int alphaTextureID{ -1 };

        VEC3f normal_map_vec;
        int normalTextureID{ -1 };

        unsigned int materialID;
        VEC3f emit;

        // Is light
        bool isLight{ false };
    };

    struct QuadLight
    {
        VEC3f origin, du, dv, power;
    };

    struct Texture
    {
        ~Texture()
        {
            if (pixel)
                delete[] pixel;
        }

        uint32_t* pixel{ nullptr };
        VEC2i resolution{ -1 };
    };

    struct Model
    {
        ~Model()
        {
            for (auto mesh : meshes)
                delete mesh;
            for (auto texture : textures)
                delete texture;
        }

        std::vector<TriangleMesh*> meshes;
        std::vector<Texture*> textures;
        //! bounding box of all vertices in the model
        owl::box3f bounds;
    };

    Model* loadOBJ(const std::string& objFile);
}
