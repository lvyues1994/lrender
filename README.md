# lrender

一个 C++17 渲染引擎库，GPU 上的工作全部是用 C++ 写成的 compute kernel。设计见 [docs/architecture.md](docs/architecture.md)。

## 构建

需要 Linux、CMake 3.25+、Ninja、GCC 10+ 或 Clang 12+。Ubuntu 24.04 上：

```sh
sudo apt install ninja-build libclang-18-dev glslang-tools libvulkan-dev mesa-vulkan-drivers \
                 libegl-dev libgles-dev
cmake --workflow --preset debug        # 配置、构建、测试
```

预设有 `debug`、`release`、`clang-debug` 和 `asan`。依赖（lexec、co2、stackfull、compute_shaders）按固定提交拉取。要改用本地的副本，在不纳入版本管理的 `CMakeUserPresets.json` 里设置 `FETCHCONTENT_SOURCE_DIR_<NAME>`：

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
```

GPU 后端的测试跑在 Mesa 的软件设备（llvmpipe、lavapipe）上，没有这些设备时跳过；设置 `LRENDER_TEST_REQUIRE_GPU=1` 后改为失败，CI 就是这样运行的。
