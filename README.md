# lrender

一个 C++17 渲染引擎库，GPU 上的工作全部是用 C++ 写成的 compute kernel。设计见 [docs/architecture.md](docs/architecture.md)。

## 构建

需要 Linux、CMake 3.25+、Ninja、GCC 10+ 或 Clang 12+。Ubuntu 24.04 上：

```sh
sudo apt install ninja-build libclang-18-dev glslang-tools libvulkan-dev mesa-vulkan-drivers \
                 libegl-dev libgles-dev libsdl2-dev
cmake --workflow --preset debug        # 配置、构建、测试
```

预设有 `debug`、`release`、`clang-debug` 和 `asan`。依赖（lexec、co2、stackfull、compute_shaders）按固定提交拉取。SDL2 用系统里安装的（`libsdl2-dev`），只有窗口上屏（`lr::present_sdl`）和 `lrender_view` 需要它；没装时这两个不构建，其余照常。要改用依赖库的本地副本，在不纳入版本管理的 `CMakeUserPresets.json` 里设置 `FETCHCONTENT_SOURCE_DIR_<NAME>`：

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

演示场景的纹理都是程序生成的：棋盘格地面、带法线贴图的木箱、用 alpha 遮罩镂空的格栅，还有一个半透明的球，太阳投下级联阴影。`--pipeline pathtrace` 用参考路径追踪器渲染 `--samples` 帧、每帧每像素一个采样，写出最后一帧；`--sequence <目录>` 改为把每一帧写成 `<目录>/frame<帧号>.ppm`（目录要先存在）；`--timers on` 打印每个 pass 的设备耗时。`--taa` 打开可见性缓冲管线的时间抗锯齿，渲染 `--samples` 帧（默认 16）静止的场景再写出最后一帧。程序启动时打印设备选中的 kernel 变体（子组、一遍写入的可见性缓冲）。`./build/debug/probes/layout_probe --backend vulkan` 在设备上给场景表的各种布局计时。

在窗口里看演示场景：

```sh
./build/debug/examples/lrender_view --backend vulkan --size 1280x720
SDL_VIDEODRIVER=offscreen ./build/debug/examples/lrender_view --frames 30   # 不开窗口，跑 30 帧就退出
```

左键拖动绕场景旋转，滚轮缩放；Tab 在可见性缓冲和路径追踪两条管线之间切换（1、2 直接选），T 开关时间抗锯齿（默认开），Esc 或 Q 退出。`--backend`、`--device`、`--pipeline` 与 `lrender_cli` 相同，`--frames <n>` 渲染 n 帧后退出。没有显示器时 SDL 可能选不到加速渲染器，会自动退回软件渲染器，也可以用 `SDL_RENDER_DRIVER=software` 指定。

自定义材质模型的写法见 `render/include/lr/render/MaterialModel.h` 开头的例子，测试里的 `tests/render/TestMaterialModels.h` 是一个完整的例子。

GPU 后端的测试跑在 Mesa 的软件设备（llvmpipe、lavapipe）上，没有这些设备时跳过；设置 `LRENDER_TEST_REQUIRE_GPU=1` 后改为失败，CI 就是这样运行的。SDL 的测试用 offscreen（或 dummy）视频驱动，不打开可见的窗口，SDL 启动不了时跳过。
