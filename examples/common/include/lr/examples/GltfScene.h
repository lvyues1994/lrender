#pragma once

#include <lr/asset/AssetLibrary.h>
#include <lr/asset/Scene.h>
#include <lr/asset/SceneDelta.h>
#include <lr/core/RenderWorldDelta.h>
#include <lr/core/Scene.h>

#include <string>

namespace lr::examples {

struct ViewableScene {
    core::RenderWorldDelta delta; // adds the scene, a sky, and a sun if the file has no lights
    core::Removals removal;       // removes what `delta` added
    core::CameraDesc camera;      // the file's first camera, or one that frames the scene
};

ViewableScene viewableScene(asset::SceneData const &scene, asset::IdAllocator &ids);

// Looking at the box from above and in front, all of it in view.
core::CameraDesc framingCamera(core::Aabb const &bounds);

// Waits for the load on this thread; throws std::runtime_error with the AssetError.
asset::SceneHandle loadSceneNow(asset::AssetLibrary &assets, std::string const &path);

} // namespace lr::examples
