//OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCHShadow)()
//{
//    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
//    const owl::common::vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];
//    ShadowRayData& srd = owl::getPRD<ShadowRayData>();
//
//    if (self.isLight) {
//        srd.visibility = owl::common::vec3f(1.f);
//        srd.point = barycentricInterpolate(self.vertex, primitiveIndices);
//        srd.normal = normalize(barycentricInterpolate(self.normal, primitiveIndices));
//        srd.emit = self.emit;
//
//        owl::common::vec3f v1 = self.vertex[primitiveIndices.x];
//        owl::common::vec3f v2 = self.vertex[primitiveIndices.y];
//        owl::common::vec3f v3 = self.vertex[primitiveIndices.z];
//        srd.area = 0.5f * length(cross(v1 - v2, v3 - v2));
//
//        srd.cg = (v1 + v2 + v3) / 3.f;
//    }
//    else {
//        srd.visibility = owl::common::vec3f(0.f);
//    }
//
//}

//OPTIX_CLOSEST_HIT_PROGRAM(triangleMeshCH)()
//{
//    const TriangleMeshData& self = owl::getProgramData<TriangleMeshData>();
//    const owl::common::vec3i primitiveIndices = self.index[optixGetPrimitiveIndex()];
//
//    SurfaceInteraction& si = owl::getPRD<SurfaceInteraction>();
//
//    // Exact hit point on the triangle
//    si.p = barycentricInterpolate(self.vertex, primitiveIndices);
//
//    // Out going direction pointing toward the pixel location
//    si.wo = owl::normalize(optixLaunchParams.camera.pos - si.p);
//
//    // UV coordinate of the hit point
//    si.uv = barycentricInterpolate(self.texCoord, primitiveIndices);
//
//    // geometric normal 
//    si.n_geom = normalize(barycentricInterpolate(self.normal, primitiveIndices));
//
//    // Initializes to_local from n_geo then obtains to_world by taking inverse of the to_local
//    orthonormalBasis(si.n_geom, si.to_local, si.to_world);
//
//    // obtain wo is in world space cam_pos - hit_loc_world get local frame of the wo as wo_local
//    si.wo_local = normalize(apply_mat(si.to_local, si.wo));
//
//    // axix independet prop
//    si.diffuse = self.diffuse;
//    if (self.hasDiffuseTexture)
//        si.diffuse = (owl::common::vec3f)tex2D<float4>(self.diffuse_texture, si.uv.x, si.uv.y);
//
//    si.alpha = self.alpha;
//    if (self.hasAlphaTexture)
//        si.alpha = tex2D<float4>(self.alpha_texture, si.uv.x, si.uv.y).x;
//    si.alpha = clamp(si.alpha, 0.01f, 1.f);
//
//    si.emit = self.emit;
//    si.isLight = self.isLight;
//
//    si.hit = true;
//}