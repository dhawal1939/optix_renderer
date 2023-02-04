#include <chrono>
#include <ostream>
#include "viewer.hpp"

#define __TBB_NO_IMPLICIT_LINKAGE 1
#define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1

int main(int argc, char** argv)
{
    bool isInteractive = true;

    std::string currentScene;

    std::string defaultScene = "C:/Users/dhawals/repos/scenes/scene_config/rgb_test_scene.json";
    //std::string defaultScene = "C:/Users/dhawals/repos/scenes/scene_config/cornell_box.json";


    currentScene = defaultScene;

    LOG("Loading scene " + currentScene);

    Scene scene;
    if (!parseScene(currentScene, scene))
    {
        LOG("Error loading scene");
        return -1;
    }

    VEC2i resolution(1024, 1024);
    printf("%d %d reso\n", resolution.x, resolution.y);

    Viewer win(scene, resolution, PATH, false);
    printf("framebuffer reso %d %d\n", win.getScreenSize().x, win.getScreenSize().y);

    //win.to_save_file = savePath;


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
