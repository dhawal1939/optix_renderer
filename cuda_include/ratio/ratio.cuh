#pragma once
#include <hit_miss.cuh>
#include <frostbite.cuh>
#include <path/path.cuh>

struct triColor{
    owl::common::vec3f colors[2];
};

__device__
struct triColor ratio_based_shadow(SurfaceInteraction& si, LCGRand& rng, int max_ray_depth = 1)
{
    owl::common::vec3f color(0.f, 0.f, 0.f);
    owl::common::vec3f sto_illum(0.f);
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
            sto_illum = current_si.emit * tp;
            break;
        }
        owl::common::vec2f rand1 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        owl::common::vec2f rand2 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        // going to the camera

        int selectedLightIdx = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
        RadianceRay ray;
        SurfaceInteraction brdf_si{ 0 }, light_si{ 0 };
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

            owl::common::vec3f H = normalize(L + V);
            float brdfPdf = get_brdf_pdf(current_si.alpha, V, current_si.n_geom, H, L); // brdf pdf of current point
            float metalness = 0.5f, reflectance = 0.5f;
            owl::common::vec3f f0 = 0.16f * reflectance * reflectance * (owl::common::vec3f(1.0f, 1.0f, 1.0f) - metalness) +
                current_si.diffuse * metalness;
            owl::common::vec3f brdf = evaluate_brdf(V, current_si.n_geom, L, owl::common::vec3f(1.), current_si.alpha, owl::common::vec3f(1.)); // brdf of current point
            float misW = balanceHeuristic(1, lightPdfW, 1, brdfPdf);
            if (light_si.isLight) {
                color += misW * light_si.emit * tp * brdf * clampDot(current_si.n_geom, L, false) / lightPdfW;
                color = checkPositive(color);
            }
            TriLight triLight = optixLaunchParams.triLights[selectedLightIdx];
            sto_illum = misW * triLight.emit * tp * brdf * clampDot(current_si.n_geom, L, false) / lightPdfW;
            sto_illum = checkPositive(sto_illum);
        }
        //BRDF Sampling
        //{
        //    owl::common::vec3f H = sample_GGX(rand2, current_si.alpha, current_si.n_geom); // do all in global

        //    //owl::common::vec3f L = owl::common::normalize(2.f * owl::common::dot(V, H) * H - V);
        //    owl::common::vec3f L = owl::common::normalize(-V - 2.f * owl::common::dot(-V, H) * H);

        //    ray.origin = current_si.p + EPS * current_si.n_geom;
        //    ray.direction = L;

        //    owl::traceRay(optixLaunchParams.world, ray, brdf_si);

        //    float metalness = 0.5f, reflectance = 0.5f;
        //    owl::common::vec3f f0 = 0.16f * reflectance * reflectance * (owl::common::vec3f(1.0f, 1.0f, 1.0f) - metalness) +
        //        current_si.diffuse * metalness;
        //    owl::common::vec3f brdf = evaluate_brdf(V, current_si.n_geom, L, current_si.diffuse, current_si.alpha, f0);
        //    float brdfPdf = get_brdf_pdf(current_si.alpha, V, current_si.n_geom, H, L);


        //    float lightPdf = 1 / (brdf_si.area * optixLaunchParams.numTriLights);
        //    float dist = owl::common::length(brdf_si.p - current_si.p);
        //    dist *= dist;
        //    float lightPdfW = pdfA2W(lightPdf, dist, dot(-L, brdf_si.n_geom));
        //    float misW = balanceHeuristic(1, brdfPdf, 1, lightPdfW);
        //    if (!brdf_si.hit)
        //    {
        //        sto_illum = misW * brdf_si.emit * tp / brdfPdf;
        //        break;
        //    }
        //    if (brdf_si.isLight) {
        //        // it has hit the light find which light is hit and calculate the pdf of light accordingly.
        //        // color from next hit _si.emit
        //        // remove misW
        //        color += misW * brdf_si.emit * tp / brdfPdf;
        //    }

        //    tp *= clampDot(current_si.n_geom, L, false) * brdf / brdfPdf;
        //    /*
        //    tp.x = owl::common::max(tp.x, EPS);
        //    tp.y = owl::common::max(tp.y, EPS);
        //    tp.z = owl::common::max(tp.z, EPS);*/
        //}

        // wo calculation
       // New Out goind direction          
        //V = owl::normalize(current_si.p - brdf_si.p);
        //current_si = brdf_si;
        //current_si.alpha = sqrt(current_si.alpha);
        //current_si.wo = V;
    }
    color = checkPositive(color);
    sto_illum = checkPositive(sto_illum);

    struct triColor tri_color;
    tri_color.colors[0] = color;
    tri_color.colors[1] = sto_illum;

    return tri_color;
}