#include <chrono>
#include <ostream>
#include "viewer.hpp"

#define __TBB_NO_IMPLICIT_LINKAGE 1
#define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1

int main(int argc, char** argv)
{
    bool isInteractive = true;

    std::string currentScene;
    std::string defaultScene = "C:/Users/dhawals/repos/optix_renderer/scenes/scene_configs/bistro.json";

    /*if (argc == 2)
        currentScene = std::string(argv[1]);
    else*/
        currentScene = defaultScene;

    /*if (argc >= 3)
    {
        isInteractive = atoi(argv[2]);
    }*/

    LOG("Loading scene " + currentScene);

    Scene scene;
    if (!parseScene(currentScene, scene))
    {
        LOG("Error loading scene");
        return -1;
    }

    owl::common::vec2i resolution(scene.imgWidth, scene.imgHeight);
    printf("%d %d reso\n", resolution.x, resolution.y);
    std::string savePath = "C:/Users/dhawals/repos/optix_renderer/mask.png";
    Viewer win(scene, resolution, DIFFUSE);

    win.to_save_file = savePath;

    if (isInteractive)
    {
        win.camera.setOrientation(scene.cameras[0].from,
            scene.cameras[0].at,
            scene.cameras[0].up,
            owl::viewer::toDegrees(acosf(scene.cameras[0].cosFovy)));
        win.enableFlyMode();
        win.enableInspectMode(owl::box3f(scene.model->bounds.lower, scene.model->bounds.upper));
        win.setWorldScale(length(scene.model->bounds.span()));

        // ##################################################################
        // now that everything is ready: launch it ....
        // ##################################################################
        win.showAndRun();


    }

    return 0;
}
