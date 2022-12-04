#pragma once
#include <hit_miss.cuh>
#include <frostbite.cuh>

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


__device__
owl::common::vec3f estimatePathTracing(SurfaceInteraction& si, LCGRand& rng, int max_ray_depth = 10)
{
    owl::common::vec3f color(0.f, 0.f, 0.f);
    SurfaceInteraction current_si = si;
    owl::common::vec3f tp(1.f, 1.f, 1.f);

    current_si.wo *= 1.;
    for (int ray_depth = 0; ray_depth < max_ray_depth; ray_depth++)
    {
        if (!current_si.hit)
            break;

        if (current_si.isLight)
        {
            color += tp * current_si.emit;
            break;
        }
        owl::common::vec2f rand1 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        owl::common::vec2f rand2 = owl::common::vec2f(lcg_randomf(rng), lcg_randomf(rng));
        owl::common::vec3f V = current_si.wo; // here it is incoming fix this
        // going to the camera

        int selectedLightIdx = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
        owl::common::vec3f brdf(0.);
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

            if (light_si.isLight) {
                owl::common::vec3f H = normalize(L + V);
                float brdfPdf = get_brdf_pdf(current_si.alpha, V, current_si.n_geom, H); // brdf pdf of current point
                float metalness = 0.5f, reflectance = 0.5f;
                owl::common::vec3f f0 = 0.16f * reflectance * reflectance * (owl::common::vec3f(1.0f, 1.0f, 1.0f) - metalness) +
                    current_si.diffuse * metalness;
                brdf = evaluate_brdf(V, current_si.n_geom, L, current_si.diffuse, current_si.alpha, owl::common::vec3f(1.f)); // brdf of current point
                float misW = lightPdfW / (lightPdfW + brdfPdf);
                color += misW * light_si.emit * tp * brdf * clampDot(current_si.n_geom, L, false) / lightPdfW;
                color.x = owl::common::max(color.x, 0.1f);
                color.y = owl::common::max(color.y, 0.1f);
                color.z = owl::common::max(color.z, 0.1f);
            }
        }
        //BRDF Sampling
        {
            owl::common::vec3f H = sample_GGX(rand2, current_si.alpha, current_si.n_geom); // do all in global

            owl::common::vec3f L = owl::common::normalize(2.f * owl::common::dot(V, H) * H - V);

            ray.origin = current_si.p + float(1e-3) * current_si.n_geom;
            ray.direction = L;

            owl::traceRay(optixLaunchParams.world, ray, brdf_si);

            if (!brdf_si.hit)
                return color;
            float metalness = 0.5f, reflectance = 0.5f;
            owl::common::vec3f f0 = 0.16f * reflectance * reflectance * (owl::common::vec3f(1.0f, 1.0f, 1.0f) - metalness) +
                current_si.diffuse * metalness;
            owl::common::vec3f brdf = evaluate_brdf(V, current_si.n_geom, L, current_si.diffuse, current_si.alpha, owl::common::vec3f(1.f));
            float brdfPdf = get_brdf_pdf(current_si.alpha, V, current_si.n_geom, H);

            tp *= clampDot(current_si.n_geom, L, false) * brdf / brdfPdf;

            if (brdf_si.isLight) {
                // it has hit the light find which light is hit and calculate the pdf of light accordingly.
                float lightPdf = 1 / (brdf_si.area * optixLaunchParams.numTriLights);
                float dist = owl::common::length(brdf_si.p - current_si.p);
                dist *= dist;
                float lightPdfW = pdfA2W(lightPdf, dist, dot(-L, brdf_si.n_geom));
                float misW = brdfPdf / (lightPdfW + brdfPdf);
                // color from next hit _si.emit
                // remove misW
                color += misW * brdf_si.emit * tp;
                color.x = owl::common::max(color.x, 0.1f);
                color.y = owl::common::max(color.y, 0.1f);
                color.z = owl::common::max(color.z, 0.1f);
                break;
            }
        }

        current_si = brdf_si;
        current_si.wo *= -1;

    }
    return color;
}