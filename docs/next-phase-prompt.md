# lrender 下一阶段：先做架构与模块设计，再实现

你接手 lrender（`/home/lvyues/code/render/lrender`），一个 C++17 的渲染引擎库。这一轮做 `docs/roadmap.md` 的第 0 阶段（B1–B4，基准与测量）和第 1 阶段（P1–P7，大场景下的性能结构）。后面几个阶段这一轮不实现，但设计不能堵住它们的路。另一个 agent 会审阅你的设计和每个里程碑。

## 流程

1. 读 `docs/architecture.md`（全文）、`docs/roadmap.md`、`docs/library-requirements.md`、`docs/hardware-raster.md`，再读工作项涉及的代码。
2. 写设计文档 `docs/phase-1-design.md`（要求见下文）。写完停下，按“汇报”一节交审阅，等意见回来再动代码。
3. 按设计里定的里程碑顺序实现。每个里程碑做完、验证通过，就在本地提交，写里程碑汇报，然后停下等审阅。
4. 实现中发现设计要改，先改设计文档并在汇报里说明原因。

## 项目要点

这些决定了大部分设计取舍，细节见 `architecture.md`：

- 渲染全部是 compute：kernel 用 compute_shaders（下称 tc）的 C++ 子集写（`TC_KERNEL`），构建时转译成 GLSL/SPIR-V。同一个 kernel 也在 tc 的 CPU 后端上运行，它是测试参照。后端有 `cpu`、`opengl`、`gles`、`vulkan`，GLES 3.1 是能力底线。
- 两条状态时间线：渲染线程（`RenderWorld`、帧图的编译）和 GPU 线程（`FrameQueue`、`DeviceScene`、`TexturePool`、`AccelerationScene`，只有它碰 `tc::Device`）。模块之间的异步接口是 lexec sender；帧池跑帧内的并行工作，后台池跑长任务。
- 帧图：feature 在渲染线程上声明 pass 和资源，编译时算生命周期，GPU 线程把生命周期不重叠的设备资源放进同一个 tc 堆。
- 设备能力不同的做法写成 kernel 变体（`RendererCapabilities`，`VariantUse` 的 `Auto` 和 `Off`），两种变体在同一个设备上互相比对。
- 已有、可以直接用的：tc 的 ray query、只在设备上的 buffer、内存堆与别名、传输队列、`importBuffer` 和 GPU UUID、sync file。光栅化已经按三角形大小分块（`render/kernels/Visibility.h`）。
- 已经实测过、不要重做的：桌面硬件光栅暂缓，原因是两个设备之间在一帧中间来回一次要 0.92 ms，见 `hardware-raster.md`。P1 的交接在帧末，可以和下一帧流水重叠，所以结论不同，但同样要实测。

## 约束

- 依赖库（tc、co2、stackfull、lexec）的用户原话：“这必须是库所支持得能力范围，不能超出库的能力，也就是越界的能力”。tc 只做 compute：交换链、窗口、图形管线都不向它要。需要依赖库新增能力时，先按 `library-requirements.md` 的格式写进去，交审阅，经用户同意后才在库自己的仓库里实现（tc 在 `/home/lvyues/code/compute_sharders/compute_shaders`，带上它自己的测试），再更新 `cmake/LrDependencies.cmake` 的固定提交。debug、clang-debug、asan 三个预设的缓存把 tc 指向这个本地仓库。
- 只在本地提交，不 push，推送由用户做。
- 需要协程时用 co2。C++ 风格用 cpp-oop-style 技能（没有的话照现有代码）：接口加工厂、数据类、依赖注入、RAII、east const、camelCase、120 列。注释只写代码本身表达不了的约束。
- 文档用中文，照 `architecture.md` 的写法：陈述式，给出实测数据，不写宣传语。
- 只改性能、不改语义的工作（P2、P4、P5、P6），要让图像逐位不变；做不到时在设计里说明原因，并给出容差和理由。测试必须确定：harness 用 `ConversionDelivery::Deterministic`，纹理比上传晚一帧出现，测试按帧数写。

## 验证

- 每个里程碑都要过三套预设：`LRENDER_TEST_REQUIRE_GPU=1 cmake --workflow --preset <debug|clang-debug|asan>`。GPU 后端的测试默认跑在 llvmpipe 和 lavapipe 上。
- 硬件上：`LRENDER_TEST_DEVICE=NVIDIA` 让 GPU 后端的测试改在 RTX 5070 Ti 上跑。Vulkan 的同步检查用 `build/vklayers-new/1.4.363.0/x86_64/share/vulkan/explicit_layer.d` 下的验证层（设 `VK_LAYER_PATH`、`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`、`VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`），NVIDIA 和 lavapipe 上都不能有消息。系统自带的 1.3.275 版不认 timeline semaphore 上的等待，会误报。
- 新测试要证明自己有用：对关键的正确性性质，故意改坏实现，确认测试会失败，再改回来。
- 性能结论只来自测量：release 构建；GPU 先持续加载到频率稳定；A/B 两种配置在同一个进程里隔帧交替；汇报具体数字。不要把 `ctest -j4` 下的时间当作结论，并行测试会互相拖慢。

## 已知的坑

- CPU 后端会把声明了 `usesBarrier` 的 kernel 的每个调用都放进纤程，不需要屏障的部分要拆成另一个 kernel（光栅化的三角形阶段和块阶段就是这样拆的）。CPU 后端的 kernel 在 debug 构建里不优化，测试图像要小；必须用大图的测试只在 GPU 后端上跑，并在注释里写明原因。
- tc 转译器：宏里的注释用 `/* */`；成员函数不要按值传 `std::array`；kernel 的成员函数可以调用源码里在它后面定义的成员函数（光栅化宏就这样调用 kernel 自己的 `coverPixel`）。
- doctest 的 `MESSAGE` 把 `char const *` 打成 `1`，用 `std::string`。
- lavapipe（Mesa 25.2.8）上，排在有效实例前面的无效实例（底层引用为 0 或掩码为 0）会让后面的实例都不被命中；要留空位，就引用有效的底层，并用射线不追踪的掩码位。
- `KernelReloadTest` 只统计有 OpenGL GLSL 的程序，只有 Vulkan 版本的 kernel 不计入。

## 设计文档要求

`docs/phase-1-design.md` 要让审阅者不读代码也能判断方案是否成立：

- 总体架构：第 1 阶段之后一帧的数据流，包括簇数据放在哪、剔除和光栅化在帧图里的 pass 与资源、上屏经过 `FrameQueue` 和 `Presenter` 的路径；每一项落在哪个时间线上。
- 每个模块和工作项：
  - 职责与边界，公开接口用现有风格写的 C++ 草案；
  - 内部设计和数据格式（`GpuTypes.h` 里新增或改动的类型）；
  - 在各后端和变体下怎么走，包括 GL 不能放置图像、GLES 的绑定数限制、CPU 后端；
  - 设备丢失后怎么恢复；
  - 依赖库边界的检查，需要库新增的能力单独列出。
- 考虑过的其他方案，以及选定方案的理由。尤其是 P4 的两遍遮挡剔除怎么保证结果与不剔除时逐位相同，P2 的并行前缀和怎么保持实例顺序。
- 测试计划：哪些测试新增、哪些要改、哪些比对证明语义不变。
- 测量计划：用 B2 的哪些指标、哪些场景证明每一项的收益，以及预期的量级。
- 风险，以及里程碑顺序和每个里程碑的验收标准。B1–B4 排在最前，其余每一项做完都要拿基线比对。

## 汇报

- 设计完成时：设计文档的路径，一段概括，以及需要用户决定的问题（每个问题带上你推荐的选项和理由）。
- 每个里程碑完成时：
  - 本地提交的列表；
  - 改了什么；
  - 跑了哪些验证命令、结果如何；
  - B2 测出的前后对比（场景、配置、数字）；
  - 与设计的出入和原因；
  - 遗留的问题。
