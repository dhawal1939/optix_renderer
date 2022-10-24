#include <chrono>
#include <ostream>
#include "viewer.hpp"


int main(int argc, char **argv)
{
    std::string savePath;
    bool isInteractive = false;

    std::string currentScene;
    std::string defaultScene = "../scenes/scene_configs/test_scene.json";

    if (argc == 2)
        currentScene = std::string(argv[1]);
    else
        currentScene = defaultScene;

    if (argc >= 3)
    {
        isInteractive = atoi(argv[2]);
    }

    LOG("Loading scene " + currentScene);

    Scene scene;
    if (!parseScene(currentScene, scene))
    {
        LOG("Error loading scene");
        return -1;
    }

    vec2i resolution(scene.imgWidth, scene.imgHeight);
    Viewer win(scene, resolution, isInteractive);

    if (isInteractive)
    {
        win.camera.setOrientation(scene.cameras[0].from,
                                  scene.cameras[0].at,
                                  scene.cameras[0].up,
                                  owl::viewer::toDegrees(acosf(scene.cameras[0].cosFovy)));
        win.enableFlyMode();
        win.enableInspectMode(owl::box3f(scene.model->bounds.lower, scene.model->bounds.upper));
        win.setWorldScale(length(scene.model->bounds.span()));
        win.setRendererType(static_cast<RendererType>(14));

        // ##################################################################
        // now that everything is ready: launch it ....
        // ##################################################################
        win.showAndRun();
    }

    return 0;
}
