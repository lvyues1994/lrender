# lrender

一个 C++17 渲染引擎库，GPU 上的工作全部是用 C++ 写成的 compute kernel。设计见 [docs/architecture.md](docs/architecture.md)。

## 构建

需要 Linux、CMake 3.25+、Ninja、GCC 10+ 或 Clang 12+。Ubuntu 24.04 上：

```sh
sudo apt install ninja-build libclang-18-dev glslang-tools libvulkan-dev mesa-vulkan-drivers \
                 libegl-dev libgles-dev libsdl2-dev
cmake --workflow --preset debug        # 配置、构建、测试
```

预设有 `debug`、`release`、`clang-debug` 和 `asan`。依赖（lexec、co2、stackfull、compute_shaders，以及 asset 用的 tl-expected、cgltf、stb）按固定提交拉取。SDL2 用系统里安装的（`libsdl2-dev`），只有窗口上屏（`lr::present_sdl`）和 `lrender_view` 需要它；没装时这两个不构建，其余照常。要改用依赖库的本地副本，在不纳入版本管理的 `CMakeUserPresets.json` 里设置 `FETCHCONTENT_SOURCE_DIR_<NAME>`：

```json
{
  "version": 6,
  "configurePresets": [
    {
      "name": "local",
      "inherits": "debug",
      "cacheVariables": {"FETCHCONTENT_SOURCE_DIR_COMPUTE_SHADERS": "/path/to/compute_shaders"}
    }
  ]
}
```

离线渲染演示场景：

```sh
./build/debug/examples/lrender_cli --backend vulkan --size 960x540 --out demo.ppm
./build/debug/examples/lrender_cli --backend vulkan --taa --out demo-taa.ppm             # TAA，16 帧
./build/debug/examples/lrender_cli --pipeline pathtrace --samples 256 --timers on --out reference.ppm
./build/debug/examples/lrender_cli --pipeline pathtrace --samples 16 --sequence frames/   # frames/frame0.ppm …
```

演示场景的纹理都是程序生成的：棋盘格地面、带法线贴图的木箱、用 alpha 遮罩镂空的格栅，还有一个半透明的球，太阳投下级联阴影。`--pipeline pathtrace` 用参考路径追踪器渲染 `--samples` 帧、每帧每像素一个采样，写出最后一帧；`--sequence <目录>` 改为把每一帧写成 `<目录>/frame<帧号>.ppm`（目录要先存在）；`--timers on` 打印每个 pass 的设备耗时；`--graph <文件>` 把最后一帧的帧图写成 Graphviz 的 DOT（`dot -Tsvg 文件 > graph.svg` 画出来），程序也会打印帧图资源共用内存前后各要多少。`--taa` 打开可见性缓冲管线的时间抗锯齿，渲染 `--samples` 帧（默认 16）静止的场景再写出最后一帧。程序启动时打印设备选中的 kernel 变体（子组、一遍写入的可见性缓冲、路径追踪用不用 ray query）。`./build/debug/probes/layout_probe --backend vulkan` 在设备上给场景表的各种布局计时。

在窗口里看演示场景：

```sh
./build/debug/examples/lrender_view --backend vulkan --size 1280x720
SDL_VIDEODRIVER=offscreen ./build/debug/examples/lrender_view --frames 30   # 不开窗口，跑 30 帧就退出
```

场景放在一个 `lr::scene::World` 里，由脚本每帧推动：相机慢慢绕场景转，发光的球带着上方的点光源上下浮动。左键拖动改变相机的方向，滚轮缩放；Tab 在可见性缓冲和路径追踪两条管线之间切换（1、2 直接选），T 开关时间抗锯齿（默认开），Esc 或 Q 退出。`--backend`、`--device`、`--pipeline` 与 `lrender_cli` 相同，`--frames <n>` 渲染 n 帧后退出，这时每帧按 1/60 秒模拟，退出前打印 tick 数和相机从哪里转到了哪里。没有显示器时 SDL 可能选不到加速渲染器，会自动退回软件渲染器，也可以用 `SDL_RENDER_DRIVER=software` 指定。

两个程序都可以用 `--gltf` 换成 glTF 场景（`.gltf` 带外部文件或 data URI，或 `.glb`；图片是 PNG、JPEG 或 KTX2）：

```sh
./build/debug/examples/lrender_cli --backend vulkan --gltf models/scene.gltf --out scene.ppm
./build/debug/examples/lrender_view --backend vulkan --gltf models/scene.glb
```

有相机就用文件里的第一个，否则按场景的包围盒取景；另外加一个天空，文件里没有灯时再加一个太阳。`lrender_cli` 会打印加载的网格、材质、纹理、灯光和相机数，以及没能加载的内容。`lrender_view` 由一个脚本加载场景，并监听场景用到的文件，改了就重新加载，在 World 里换掉旧场景；debug 构建在 GPU 后端上还监听 lrender 自己的 kernel 目录，改了 kernel 再构建一次，运行中的窗口就换上新程序（release 预设关掉了这个功能）。`lrender_view --mesh <文件>` 把 glTF 合并成一个网格，运行 `docs/architecture.md` 里的 `orbitCamera` 脚本。

性能用 release 构建的 `lrender_bench` 测：先预热到 GPU 频率稳定，再让几种配置在同一个进程里一帧一帧交替渲染，输出每个 pass 的设备时间、主机上各线程的时间和帧图显存，可以写成 JSON。场景、方法和基线见 `docs/benchmarks.md`。

```sh
tools/fetch-scenes.sh          # Sponza 和 ABeautifulGame，下载到 build/scenes
./build/release/probes/lrender_bench --device NVIDIA --scene sponza \
    --config shared:aliasing=on --config separate:aliasing=off --json sponza.json
./build/release/probes/lrender_bench --device NVIDIA --scene scale --instances 1000000
```

在自己的程序里加载 glTF：

```cpp
auto source = lr::asset::makeFileSource({});
auto config = lr::asset::AssetLibraryConfig{};
config.source = source.get();
config.runtime = runtime.get();
config.compressedFormats = renderer->deviceInfo().limits.compressedArrayFormats;
auto assets = lr::asset::makeAssetLibrary(config);

auto loaded = lexec::sync_wait(assets->loadScene("models/scene.gltf"));   // 或者组合进别的 sender
auto const &scene = std::get<0>(*loaded);                                 // AssetResult<SceneHandle>
if (not scene) { /* scene.error().message */ }
auto ids = lr::asset::makeSequentialIds();
auto added = lr::asset::sceneDelta(**scene, *ids);          // added.removal 以后用来移除它
auto view = lr::render::ViewDesc{};
view.target = renderer->targets().createImage({lr::core::ivec2{960, 540}});
view.camera = (*scene)->cameras.at(0).camera;
lexec::sync_wait(renderer->submit({std::move(added.delta), {view}}));
```

用 World 和脚本管理场景（`lr::scene`）：World 记着实体、层级和资源，每帧只交出变化的部分；脚本是 co2 协程，只在 `tick()` 里、在调用它的线程上恢复，所以可以直接改 World。

```cpp
lr::core::Transform raisedTo(float const height) {
    auto transform = lr::core::Transform{};
    transform.translation.y = height;
    return transform;
}

// 跨挂起点的局部写在 CO2_BEG 里；被等待的表达式里不能有 lambda。
auto bob(lr::scene::World &world, lr::scene::EntityId const entity)
    CO2_BEG(lr::scene::ScriptTask, (world, entity), lr::scene::SimDuration elapsed{}; float seconds{};) {
    while (world.isAlive(entity)) {
        world.setTransform(entity, raisedTo(0.5f + 0.2f * std::sin(seconds)));
        CO2_AWAIT_SET(elapsed, lr::scene::nextFrame());   // 也可以等 waitFor(时长)，或者任何 lexec sender
        seconds += std::chrono::duration<float>{elapsed}.count();
    }
}
CO2_END

auto world = lr::scene::makeWorld({});
auto scripts = lr::scene::makeScriptHost({});              // 析构时停止所有脚本
auto const ball = world->spawnInstance(mesh, lr::core::Transform::identity());
scripts->start(bob(*world, ball));
for (;;) {
    scripts->tick(std::chrono::milliseconds{16});
    auto frame = lr::render::FrameDesc{world->extractDelta(), {view}};
    lexec::sync_wait(renderer->submit(std::move(frame)));
}
```

脚本等待的 sender 不管在哪个线程上完成，脚本都在之后的某次 tick 里恢复。`ScriptHostConfig::release` 设成 `Deterministic` 后，第 t 次 tick 里等待的操作恰好在第 t + 1 次 tick 恢复（没完成就等它），加载快慢不影响脚本的时间线。

自定义材质模型的写法见 `render/include/lr/render/MaterialModel.h` 开头的例子，测试里的 `tests/render/TestMaterialModels.h` 是一个完整的例子。

GPU 后端的测试跑在 Mesa 的软件设备（llvmpipe、lavapipe）上，没有这些设备时跳过；设置 `LRENDER_TEST_REQUIRE_GPU=1` 后改为失败，CI 就是这样运行的。设置 `LRENDER_TEST_DEVICE=NVIDIA`（设备名的一部分）可以改在硬件设备上跑，验证层的用法见 `docs/architecture.md` 的“验证与里程碑”。SDL 的测试用 offscreen（或 dummy）视频驱动，不打开可见的窗口，SDL 启动不了时跳过。
