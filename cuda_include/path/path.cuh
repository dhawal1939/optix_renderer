#pragma once
#include <hit_miss.cuh>
#include <frostbite.cuh>
#include <material.cuh>
#include <common.h>

__device__
VEC3f sampleLight(int selectedLightIdx, VEC2f rand) {

    TriLight triLight = optixLaunchParams.triLights[selectedLightIdx];
    VEC3f lv1 = triLight.v1;
    VEC3f lv2 = triLight.v2;
    VEC3f lv3 = triLight.v3;
    return samplePointOnTriangle(lv1, lv2, lv3, rand.x, rand.y);
}

__device__
float sample_light_pdf(VEC3f current_location, SurfaceInteraction& si)
{
    float sinThetaMax2 = si.area / pow(owl::common::length(current_location - si.p), 2.);
    float cosThetaMax = sqrt(max(0., 1. - sinThetaMax2));
    return 1. / (TWO_PI * (1. - cosThetaMax));

    //return si.area / (pow(owl::common::length(current_location - si.p), 2) * optixLaunchParams.numTriLights);
}


#define TWO_PI 2*3.141

__device__
VEC3f polar_to_cartesian(float sinTheta,
    float cosTheta,
    float sinPhi,
    float cosPhi)
{
    return VEC3f(sinTheta * cosPhi,
        sinTheta * sinPhi,
        cosTheta);
}

__device__
VEC3f calc_binormals(VEC3f normal, VEC3f& tangent)
{
    if (abs(normal.x) > abs(normal.y))
    {
        tangent = normalize(VEC3f(-normal.z, 0., normal.x));
    }
    else
    {
        tangent = normalize(VEC3f(0., normal.z, -normal.y));
    }

    VEC3f binormal = cross(normal, tangent);
    return binormal;
}
__device__
VEC3 uniform_sample_cone(VEC2 u12,
    float cosThetaMax,
    VEC3 xbasis, VEC3 ybasis, VEC3 zbasis)
{
    float cosTheta = (1. - u12.x) + u12.x * cosThetaMax;
    float sinTheta = sqrt(1. - cosTheta * cosTheta);
    float phi = u12.y * TWO_PI;
    VEC3 samplev = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));
    return samplev.x * xbasis + samplev.y * ybasis + samplev.z * zbasis;
}

__device__
VEC3 sample_light(SurfaceInteraction& si, VEC3 light, float *pdf, VEC2 u12, float area)
{

    VEC3 tangent = VEC3(0.), binormal = VEC3(0.);
    VEC3 ldir = normalize(light - si.p);
    binormal = calc_binormals(ldir, tangent);

    float sinThetaMax2 = area / pow(owl::common::length(light - si.p), 2.f);
    float cosThetaMax = sqrt(max(0., 1. - sinThetaMax2));
    VEC3 light_sample = uniform_sample_cone(u12, cosThetaMax, tangent, binormal, ldir);

    *pdf = -1.f;
    if (dot(light_sample, si.n_geom) > 0.)
    {
        *pdf = 1.f / (TWO_PI * (1. - cosThetaMax));
    }

    //VEC3 ldir = normalize(light - si.p);
    //VEC3 light_sample = ldir;

    //*pdf = area / (pow(owl::common::length(light - si.p), 2) * optixLaunchParams.numTriLights);

    return light_sample;
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
VEC3f sample_brdf(SurfaceInteraction& si, float* pdf, VEC2f u12)
{

    float cosTheta = pow(max(0., u12.x), 1. / (si.spec_exponent + 1.));
    float sinTheta = sqrt(max(0., 1. - cosTheta * cosTheta));
    float phi = u12.y * TWO_PI;

    VEC3 whLocal = polar_to_cartesian(sinTheta, cosTheta, sin(phi), cos(phi));

    VEC3 tangent = VEC3(0.), binormal = VEC3(0.);
    binormal = calc_binormals(si.n_geom, tangent);

    VEC3 wh = whLocal.x * tangent + whLocal.y * binormal + whLocal.z * si.n_geom;

    VEC3 wo = -si.wo;
    if (dot(wo, wh) < 0.)
    {
        wh *= -1.;
    }

    VEC3 wi = reflect(si.wo, wh);

    *pdf = ((si.spec_exponent + 1.) * pow(clamp(abs(dot(wh, si.n_geom)), 0., 1.), si.spec_exponent)) / (TWO_PI * 4. * dot(wo, wh));
    return wi;
}

__device__
float brdf_pdf(VEC3 wi, VEC3 wo, SurfaceInteraction &si)
{
    VEC3 wh = normalize(wi + wo);
    float cosTheta = abs(dot(wh, si.n_geom));

    float pdf = -1.;
    if (dot(wo, wh) > 0.)
    {
        pdf = ((si.spec_exponent + 1.) * pow(max(0., cosTheta), si.spec_exponent)) / (TWO_PI * 4. * dot(wo, wh));
    }

    return pdf;
}

__device__
VEC3 brdf_value(VEC3 wi, VEC3 wo, VEC3 n, SurfaceInteraction& si)
{

    float cosThetaN_Wi = abs(dot(n, wi));
    float cosThetaN_Wo = abs(dot(n, wo));
    VEC3 wh = normalize(wi + wo);
    float cosThetaN_Wh = abs(dot(n, wh));

    // Compute geometric term of blinn microfacet      
    float cosThetaWo_Wh = abs(dot(wo, wh));
    float G = min(1., min((2. * cosThetaN_Wh * cosThetaN_Wo / cosThetaWo_Wh),
        (2. * cosThetaN_Wh * cosThetaN_Wi / cosThetaWo_Wh)));

    // Compute distribution term
    float D = (si.spec_exponent + 2.) * INV_TWO_PI * pow(max(0., cosThetaN_Wh), si.spec_exponent);

    // assume no fresnel
    float F = 1.;

    return si.diffuse * D * G * F / (4.f * cosThetaN_Wi * cosThetaN_Wo);
}

__device__
VEC3f integrate_lighting(SurfaceInteraction &si, LCGRand& rng, VEC3 wi)
{
    VEC3f lcol = VEC3f(0.);
    VEC2f rand1 = VEC2f(lcg_randomf(rng), lcg_randomf(rng));
    VEC2f rand2 = VEC2f(lcg_randomf(rng), lcg_randomf(rng));
    unsigned int selectedLightIndx = lcg_randomf(rng) * (optixLaunchParams.numTriLights - 1);
    RadianceRay ray;
    // // LIGHT SAMPLE
    {
        VEC3 light = sampleLight(selectedLightIndx, rand1);
        float light_pdf = -1.;
        float area = optixLaunchParams.triLights[selectedLightIndx].area;
        VEC3 light_sample = sample_light(si, light, &light_pdf, rand1, area);

        if (light_pdf > 0.f)
        {
            float brdf_pdf_val = 0.f;
            float misWeight = 0.f;
            float visibility = 0.f;
            VEC3 le = 0.f;
            VEC3 brdf_val;
            float cosine_term;

            SurfaceInteraction light_si = { 0 };
            ray.direction = light_sample; // for some reason it wants a - here.
            ray.origin = si.p + si.n_geom* EPS;
            owl::traceRay(optixLaunchParams.world, ray, light_si);
            if(light_si.isLight)
            {
                visibility = 1.f;
                brdf_pdf_val = brdf_pdf(wi, light_sample, si);
                misWeight = PowerHeuristic(1., light_pdf, 1., brdf_pdf_val);
                brdf_val = brdf_value(wi, light_sample, si.n_geom, si);
                cosine_term = abs(dot(light_sample, si.n_geom));
                float pdf_term = misWeight / light_pdf;
                le = light_si.emit;

                lcol += le * visibility * (si.spec_intensity * brdf_val) * cosine_term * pdf_term;
                // diffuse
                lcol += le * visibility * (si.diffuse * INV_TWO_PI) * cosine_term * pdf_term;
            }

        }
    }

    ////BRDF
    {
        SurfaceInteraction brdf_si = { 0 };
        float brdf_pdf = -1.;
        VEC3f brdfSample = -sample_brdf(si, &brdf_pdf, rand2);
        if (brdf_pdf > 0.)
        {
            ray.direction = brdfSample; // for some reason it wants a - here.
            ray.origin = si.p + si.n_geom * EPS;
            owl::traceRay(optixLaunchParams.world, ray, brdf_si);

            float visibility = 0.;
            float light_pdf = 0.;
            float misWeight = 0.;
            float cosine_term = 0;
            VEC3 brdf_val = 0.;
            VEC3 le = 0;
            brdf_val = brdf_value(wi, brdfSample, si.n_geom, si);

            if(brdf_si.isLight)
            {
                // we need to see current position(si.p) to light distance
                visibility = 1.f;
                light_pdf = sample_light_pdf(si.p, brdf_si);
                misWeight = PowerHeuristic(1., brdf_pdf, 1., light_pdf);
                cosine_term = abs(dot(brdfSample, si.n_geom));
                float pdf_term = misWeight / brdf_pdf;
                le = brdf_si.emit;
                // specular
                // Le V Brdf Cosine / pdf
                lcol += le * visibility * (si.spec_intensity * brdf_val) * cosine_term * pdf_term;
                // diffuse
                lcol += le * visibility * (si.diffuse * INV_TWO_PI) * cosine_term * pdf_term;
            }
        }

    }
    return lcol;
}


__device__
VEC3f estimatePathTracing(SurfaceInteraction& si, LCGRand& rng, RadianceRay  ray, int max_ray_depth = 10)
{
    VEC3f color(0.f, 0.f, 0.f);
    VEC3f tp(1.f, 1.f, 1.f);
    for (int ray_depth = 0; ray_depth < 1; ray_depth++)
    {
        VEC3f V = si.wo; 
        if (si.hit)
        {
            si.spec_exponent = floor(max(1., (1. - pow(si.alpha, .15)) * 40000.));
            si.spec_intensity = 1.;

            color += integrate_lighting(si, rng, V);
            ray.direction = reflect(si.wo, si.n_geom);
            ray.origin = si.p + EPS * si.wo;
        }
        else
            break;
    }

    color = owl::common::max(color, VEC3(0));

    return color;
}