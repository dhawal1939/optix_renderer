#pragma once

#include "Model.h"
#include "json.h"

using namespace osc;

struct SceneCamera
{
	VEC3f from;
	VEC3f at;
	VEC3f up;
	float cosFovy;
};

struct Scene
{
	nlohmann::json json;
	std::string jsonFilePath;

	// Scene contents
	Model* model;
	Model* triLights;
	std::vector<int> renderers;
	std::vector<SceneCamera> cameras;

	// Other information
	int spp;
	int imgWidth, imgHeight;
	std::string render_output;
	std::string render_stats_output;

	void syncLights();
};

bool parseScene(std::string scene_file, Scene &scene);
