#pragma once

#include "Model.h"
#include "json.h"

struct SceneCamera {
	vec3f from;
	vec3f at;
	vec3f up;
	float cosFovy;
};

struct Scene {
	nlohmann::json json;
	std::string jsonFilePath;

	// Scene contents
	Model* model;
	Model* tri_lights;
	std::vector<int> renderers;
	std::vector<SceneCamera> cameras;

	// Other information
	int spp;
	int img_width, img_height;
	std::string render_output;
	std::string render_stats_output;

	void syncLights();
};

bool parseScene(std::string scene_file, Scene &scene);
