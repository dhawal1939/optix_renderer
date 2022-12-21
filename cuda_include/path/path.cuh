#pragma once
#include <hit_miss.cuh>
#include <frostbite.cuh>
#include <material.cuh>

__device__
owl::common::vec3f sampleLight(int selectedLightIdx, owl::common::vec2f rand) {

    TriLight triLight = optixLaunchParams.triLights[selectedLightIdx];
    owl::common::vec3f lv1 = triLight.v1;
    owl::common::vec3f lv2 = triLight.v2;
    owl::common::vec3f lv3 = triLight.v3;
    return samplePointOnTriangle(lv1, lv2, lv3, rand.x, rand.y);
}

__device__
float sampleLightPdf(int selectedLightIdx)
{
    // Area calucation
    TriLight triLight = optixLaunchParams.triLights[selectedLightIdx];
    return 1 / (triLight.area * optixLaunchParams.numTriLights);
}

__device__
// Convert from area measure to angle measure
float pdfA2W(float pdf, float dist2, float cos_theta) {
    float abs_cos_theta = abs(cos_theta);
    if (abs_cos_theta < 1e-8) {
        return 0.0;
    }

    return pdf * dist2 / abs_cos_theta;
}

/*
__device__
owl::common::vec3f estimatePathTracing(SurfaceInteraction& si, LCGRand& rng, RadianceRay  ray, int max_ray_depth = 10)
{
    owl::common::vec3f tp(1.f);
    owl::common::vec3f color(0.f);
    
    owl::common::vec3f mat[3];
    owl::common::vec3f invmat[3];
    
    SurfaceInteraction current_si = si;
    // actual ray is coming from pixel to scene to center it we reverse the direction
    owl::common::vec3f wo_global = normalize(-ray.direction);
    for (int i = 0; i < 5; i++)
    {
        if (!current_si.hit)
        {
            color = tp * owl::common::vec3f(0.);
            break;
        }
        if (current_si.isLight && i == 0)
        {
            color += tp * current_si.emit;
            break;
        }

        // calculated orthonornal
        orthonormalBasis(current_si.n_geom, current_si.to_world, current_si.to_local);
        owl::common::vec3f wo_local = normalize(apply_mat(current_si.to_local, wo_global));

        int selectedLightIdx = lcg_randomf(rng) * (optixLaunchParams.numTriLights);
        float lightProb = 1.0f / optixLaunchParams.numTriLights;

        owl::common::vec2f light_rand = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        owl::common::vec2f bsdf_rand = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));

        // // MIS light sampling
        float lightPdf;
        owl::common::vec3f pointOnLight = sampleLight(selectedLightIdx, light_rand);
        owl::common::vec3f light_direction = owl::common::normalize(pointOnLight - current_si.p);
        owl::common::vec3f wi_local = normalize(apply_mat(current_si.to_local, light_direction));
        float light_sampled_cosThetaI = abs(cosTheta(wi_local));
        
        if (light_sampled_cosThetaI <= 0.0f) {
            lightPdf = 0.0f;
        }
        else {
            // pdf = 1/A * |T|^-1 = 1/A * r^2/(cos(theta)*A) = r^2/cos(theta)
            lightPdf = owl::common::length(pointOnLight - current_si.p) / light_sampled_cosThetaI;
        }

        if (light_sampled_cosThetaI > 0.0f && lightPdf > 0.0f) {
            RadianceRay new_light_ray;
            new_light_ray.origin = current_si.p + current_si.n_geom * EPS;
            new_light_ray.direction = light_direction;

            SurfaceInteraction light_si;
            owl::traceRay(optixLaunchParams.world, new_light_ray, light_si);

            if (light_si.isLight == true)// && lightHit.shape != hit.shape) 
            {
                float bsdfPdf = pdf(wi_local, wo_local, current_si.diffuse, 1., current_si.alpha);
                if (bsdfPdf > 0.0f) {
                    float misWeight = balanceHeuristic(1, lightPdf, 1, bsdfPdf);
                    owl::common::vec3f bsdf = evaluate(wi_local, wo_local, current_si.diffuse, current_si.alpha, current_si.emit);
                    color += tp * light_si.emit * bsdf * light_sampled_cosThetaI * misWeight / (lightPdf * lightProb);
                }
            }
        }

         //MIS BSDF 

        float bsdfPdf;
        wi_local = sample_direction(wo_local, bsdf_rand.x, bsdf_rand.y, &bsdfPdf,
            current_si.diffuse, 1.f, current_si.alpha);
        float bsdf_sampled_cosThetaI = abs(cosTheta(wi_local));
        if (bsdf_sampled_cosThetaI <= 0.0f || bsdfPdf <= 0.0f) {
            break;
        }
        owl::common::vec3f bsdf = evaluate(wi_local, wo_local, current_si.diffuse, current_si.alpha, current_si.emit);

        RadianceRay new_bsdf_ray;
        owl::common::vec3f bsdf_ray_direciton = normalize(apply_mat(current_si.to_world, wi_local));
        new_bsdf_ray.origin = current_si.p + current_si.n_geom * EPS;
        new_bsdf_ray.direction = bsdf_ray_direciton;

        SurfaceInteraction bsdf_si;
        owl::traceRay(optixLaunchParams.world, new_bsdf_ray, bsdf_si);
        
        if (bsdf_si.isLight) 
        {
            lightPdf = owl::common::length(bsdf_si.p - current_si.p) / dot(-bsdf_ray_direciton, current_si.n_geom);
            if (lightPdf > 0.0f) {
                float misWeight = balanceHeuristic(1, bsdfPdf, 1, lightPdf);
                color += tp * bsdf_si.emit * bsdf * bsdf_sampled_cosThetaI * misWeight / (bsdfPdf * lightProb);
                //assert(isFinite(color) && color.r >= 0.0f && color.g >= 0.0f && color.b >= 0.0f);
            }
        }

        tp *= bsdf * bsdf_sampled_cosThetaI / bsdfPdf;


        wo_global = normalize(current_si.wo - bsdf_si.p);
        current_si = bsdf_si;
    }
    return color;
}
*/

__device__
owl::common::vec3f estimatePathTracing(SurfaceInteraction& si, LCGRand& rng, RadianceRay  ray, int max_ray_depth = 10)
{
    owl::common::vec3f color(0.f, 0.f, 0.f);
    SurfaceInteraction current_si = si;
    owl::common::vec3f tp(1.f, 1.f, 1.f);
    owl::common::vec3f V = current_si.wo; // here is going from x towards camera at the start
    for (int ray_depth = 0; ray_depth < max_ray_depth; ray_depth++)
    {
        if (!current_si.hit)
            break;

        if (current_si.isLight)
        {
            color += tp * current_si.emit;
            break;
        }

        orthonormalBasis(current_si.n_geom, current_si.to_world, current_si.to_local);
        owl::common::vec3f wo_local = normalize(apply_mat(current_si.to_local, V));

        owl::common::vec2f rand1 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        owl::common::vec2f rand2 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        // going to the camera

        int selectedLightIdx = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
        RadianceRay ray;
        SurfaceInteraction brdf_si{ 0 }, light_si{ 0 };
        float current_alpha = current_si.alpha;
        // MIS
        // Light sampling
        {

            float lightPdf = sampleLightPdf(selectedLightIdx);

            owl::common::vec3f newPos = sampleLight(selectedLightIdx, rand1);
            owl::common::vec3f L = owl::common::normalize(newPos - current_si.p);  // incoming from light
            float dist = owl::common::length(newPos - current_si.p);
            dist = dist * dist;

            ray.origin = current_si.p + current_si.n_geom * float(1e-3);
            ray.direction = L;

            owl::traceRay(optixLaunchParams.world, ray, light_si);
            float lightPdfW = pdfA2W(lightPdf, dist, dot(-L, light_si.n_geom)); // check if -L is required or just L works

            if (light_si.isLight) {
                owl::common::vec3f H = normalize(L + V);
                owl::vec3f wi_local = normalize(apply_mat(current_si.to_local, L));
                owl::vec3f wo_local = normalize(apply_mat(current_si.to_local, V));
                float brdfPdf = pdf(wi_local, wo_local, current_si.diffuse, 1., current_si.alpha); // brdf pdf of current point
                float metalness = 0.5f, reflectance = 0.5f;
                owl::common::vec3f f0 = 0.16f * reflectance * reflectance * (owl::common::vec3f(1.0f, 1.0f, 1.0f) - metalness) +
                    current_si.diffuse * metalness;
                owl::common::vec3f brdf = evaluate(wi_local, wo_local, current_si.diffuse, current_alpha, current_si.emit); // brdf of current point
                float misW = balanceHeuristic(1, lightPdfW, 1, brdfPdf);
                color += misW * light_si.emit * tp * brdf * clampDot(current_si.n_geom, L, false) / lightPdfW;
                color.x = owl::common::max(color.x, EPS);
                color.y = owl::common::max(color.y, EPS);
                color.z = owl::common::max(color.z, EPS);
            }
        }
        //BRDF Sampling
        {
            owl::vec3f wo_local = normalize(apply_mat(current_si.to_local, V));
            float bsdfPdf;
            owl::common::vec3f wi_local = sample_direction(wo_local, rand2.x, rand2.y, &bsdfPdf, current_si.diffuse, 1., current_alpha); // do all in global

            //owl::common::vec3f L = owl::common::normalize(2.f * owl::common::dot(V, H) * H - V);

            ray.origin = current_si.p + EPS * current_si.n_geom;
            ray.direction = normalize(apply_mat(current_si.to_world, wi_local));

            owl::traceRay(optixLaunchParams.world, ray, brdf_si);

            if (!brdf_si.hit)
                break;
            float metalness = 0.5f, reflectance = 0.5f;
            owl::common::vec3f f0 = 0.16f * reflectance * reflectance * (owl::common::vec3f(1.0f, 1.0f, 1.0f) - metalness) +
                current_si.diffuse * metalness;
            owl::common::vec3f brdf = evaluate(wi_local, wo_local, current_si.diffuse, current_alpha, current_si.emit);

            if (brdf_si.isLight) {
                // it has hit the light find which light is hit and calculate the pdf of light accordingly.
                float lightPdf = 1 / (brdf_si.area * optixLaunchParams.numTriLights);
                float dist = owl::common::length(brdf_si.p - current_si.p);
                dist *= dist;
                float lightPdfW = pdfA2W(lightPdf, dist, cosTheta(wi_local));
                float misW = balanceHeuristic(1, bsdfPdf, 1, lightPdfW);
                // color from next hit _si.emit
                // remove misW
                color += misW * brdf_si.emit * tp / bsdfPdf;
                break;
            }
            tp *= cosTheta(wi_local) * brdf / bsdfPdf;
            /*
            tp.x = owl::common::max(tp.x, EPS);
            tp.y = owl::common::max(tp.y, EPS);
            tp.z = owl::common::max(tp.z, EPS);*/
        }

        // wo calculation
       // New Out goind direction          
        V = owl::normalize(current_si.p - brdf_si.p);
        current_si = brdf_si;
        current_si.alpha = 0.;
        current_si.wo = V;
    }
    color.x = owl::common::max(color.x, EPS);
    color.y = owl::common::max(color.y, EPS);
    color.z = owl::common::max(color.z, EPS);
    return color;
}