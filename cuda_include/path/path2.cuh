//#pragma once
//
//#include <utils.cuh>
//#include <hit_miss.cuh>
//#include <ggx_vndf.cuh>
//#include <helper_math.cuh>
//
//__device__
//VEC3f sampleLight(int selectedLightIdx, VEC2f rand) {
//
//    TriLight triLight = optixLaunchParams.triLights[selectedLightIdx];
//    VEC3f lv1 = triLight.v1;
//    VEC3f lv2 = triLight.v2;
//    VEC3f lv3 = triLight.v3;
//    return samplePointOnTriangle(lv1, lv2, lv3, rand.x, rand.y);
//}
//
//__device__
//float sampleLightPdf(int selectedLightIdx)
//{
//    // Area calucation
//    TriLight triLight = optixLaunchParams.triLights[selectedLightIdx];
//    return 1 / (triLight.area * optixLaunchParams.numTriLights);
//}
//
//__device__
//// Convert from area measure to angle measure
//float pdfA2W(float pdf, float dist2, float cos_theta) {
//    float abs_cos_theta = abs(cos_theta);
//    if (abs_cos_theta < 1e-8) {
//        return 0.0;
//    }
//
//    return pdf * dist2 / abs_cos_theta;
//}
//
//
///*
//V/\    /\L
//  \    /
//   \  /
//    \/
//    x
//H = (V + L) / 2
//*/
//
//__device__
//VEC3f estimatePathTracing(SurfaceInteraction& si, LCGRand& rng, int max_ray_depth = 10)
//{
//    VEC3f color(0.f, 0.f, 0.f);
//    SurfaceInteraction current_si = si;
//    VEC3f tp(1.f, 1.f, 1.f);
//    current_si.wo *= -1.;
//    if (current_si.isLight)
//    {
//        color += tp * current_si.emit;
//    }
//
//    if (max_ray_depth > 1)
//        return VEC3f(1., 0., 0.);
//
//    for (int ray_depth = 0; ray_depth < max_ray_depth; ray_depth++)
//    {
//        if (!current_si.hit)
//            break;
//        if (current_si.isLight)
//        {
//            color += tp * current_si.emit;
//            break;
//        }
//
//        VEC2f rand1 = VEC2f(lcg_randomf(rng), lcg_randomf(rng));
//        VEC2f rand2 = VEC2f(lcg_randomf(rng), lcg_randomf(rng));
//        VEC3f V = owl::common::normalize(current_si.wo);
//        // going to the camera
//
//        int selectedLightIdx = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
//        VEC3f brdf(0.);
//        RadianceRay ray;
//        SurfaceInteraction brdf_si{ 0 }, light_si{ 0 };
//        // MIS
//        //Light sampling
//        {
//            float lightPdf = sampleLightPdf(selectedLightIdx);
//
//            VEC3f newPos = sampleLight(selectedLightIdx, rand1);
//            VEC3f L = owl::common::normalize(newPos - current_si.p);  // incoming from light
//
//            ray.origin = current_si.p + current_si.n_geom * EPS;
//            ray.direction = L;
//
//            owl::traceRay(optixLaunchParams.world, ray, light_si);
//
//            if (light_si.isLight) {
//                
//                float NdotL = owl::common::dot(L, current_si.n_geom);
//                float dist = owl::common::length(newPos - current_si.p);
//                dist = dist * dist;
//                
//                float lightPdfW = pdfA2W(lightPdf, dist, NdotL);
//                VEC3f H = normalize(L + V);
//
//                // Local Frame
//                VEC3f L_local, V_local, H_local;
//                orthonormalBasis(current_si.n_geom, current_si.to_local, current_si.to_world);
//                V_local = normalize(apply_mat(current_si.to_local, V));
//                H_local = normalize(apply_mat(current_si.to_local, H));
//                L_local = normalize(apply_mat(current_si.to_local, L));
//
//                float brdfPdf = get_brdf_pdf(current_si.alpha, current_si.alpha, V_local, normalize(L_local + V_local));
//                // brdf pdf of current point
//                //float metalness = 0.5f, reflectance = 0.5f;
//                //VEC3f f0 = 0.16f * reflectance * reflectance * (VEC3f(1.0f, 1.0f, 1.0f) - 
//                // metalness) + current_si.diffuse * metalness;
//                VEC3f brdf = evaluate_brdf(V_local, L_local, current_si.diffuse, current_si.alpha);
//
//                float misW = balanceHeuristic(1, lightPdfW, 1, brdfPdf);
//                
//                color += misW * light_si.emit * tp * brdf * NdotL / lightPdfW;
//                color.x = owl::common::max(color.x, 0.f);
//                color.y = owl::common::max(color.y, 0.f);
//                color.z = owl::common::max(color.z, 0.f);
//
//            }
//        }
//        //BRDF Sampling
//        {
//            VEC3f L_local, V_local, H_local;
//            orthonormalBasis(current_si.n_geom, current_si.to_local, current_si.to_world);
//            V_local = normalize(apply_mat(current_si.to_local, V));
//            H_local = sample_GGX(rand2, current_si.alpha, current_si.alpha, V_local);
//
//            VEC3f H = normalize(apply_mat(current_si.to_world, H_local));
//
//            VEC3f L = reflect(V, H);
//            L_local = normalize(apply_mat(current_si.to_local, L));
//
//            ray.origin = current_si.p + current_si.n_geom * EPS;
//            ray.direction = L;
//
//            owl::traceRay(optixLaunchParams.world, ray, brdf_si);
//
//            if (!brdf_si.hit)
//                return color;
//            float NdotL = clampDot(current_si.n_geom, L, false);
//            
//            VEC3f brdf = evaluate_brdf(V_local, L_local, current_si.diffuse, current_si.alpha);
//            float brdfPdf = get_brdf_pdf(current_si.alpha, current_si.alpha, V_local, H);
//            tp *= NdotL * brdf / brdfPdf;
//            if (brdf_si.isLight) {
//                // it has hit the light find which light is hit and calculate the pdf of light accordingly.
//                float lightPdf = 1 / (brdf_si.area * optixLaunchParams.numTriLights);
//                float dist = owl::common::length(brdf_si.p - current_si.p);
//                dist *= dist;
//                float lightPdfW = pdfA2W(lightPdf, dist, NdotL);
//                float misW = balanceHeuristic(1, brdfPdf, 1, lightPdfW);
//                // color from next hit _si.emit
//                // remove misW
//                color += misW * brdf_si.emit * tp;
//                color.x = owl::common::max(color.x, 0.f);
//                color.y = owl::common::max(color.y, 0.f);
//                color.z = owl::common::max(color.z, 0.f);
//                break;
//            }
//        }
//
//        current_si = brdf_si;
//        current_si.wo *= -1;
//    }
//    color.x = owl::common::max(color.x, 0.f);
//    color.y = owl::common::max(color.y, 0.f);
//    color.z = owl::common::max(color.z, 0.f);
//    return color;
//}