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

/*! \namespace osc - Optix Siggraph Course */
namespace osc
{
    /*! a simple indexed triangle mesh that our sample renderer will
        render */
    struct TriangleMesh
    {
        std::vector<owl::common::vec3f> vertex;
        std::vector<owl::common::vec3f> normal;
        std::vector<owl::common::vec2f> texcoord;
        std::vector<owl::common::vec3i> index;

        // material data:
        owl::common::vec3f diffuse;
        int diffuseTextureID{ -1 };

        float alpha; // roughness
        int alphaTextureID{ -1 };

        owl::common::vec3f emit;

        // Is light
        bool isLight{ false };
    };

    struct QuadLight
    {
        owl::common::vec3f origin, du, dv, power;
    };

    struct Texture
    {
        ~Texture()
        {
            if (pixel)
                delete[] pixel;
        }

        uint32_t* pixel{ nullptr };
        owl::common::vec2i resolution{ -1 };
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
