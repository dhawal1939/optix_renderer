#pragma once

#include "Model.h"
#include "json.h"

using namespace osc;

struct SceneCamera
{
	owl::common::vec3f from;
	owl::common::vec3f at;
	owl::common::vec3f up;
	float cosFovy;
};

struct Scene
{
	nlohmann::json json;
	std::string jsonFilePath;

	// Scene contents
	Model* model;
	Model* tri_lights;
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
