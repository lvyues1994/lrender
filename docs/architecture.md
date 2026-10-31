# lrender 架构

## 问题

lrender 是一个 C++17 渲染引擎库。宿主程序把它嵌进自己的主循环，每帧提交场景的变化和若干视图；引擎把每个视图渲染到宿主指定的目标上，可以是窗口、离屏图像或图片序列。所有 GPU 工作都是同一份 C++ 源码写成的 compute kernel，既能在 CPU 上直接运行和调试，也能跑在 Vulkan、OpenGL 和 OpenGL ES 上。目标平台是 Linux 与 Android。

## 五个决定

1. **渲染全部用 compute。** compute_shaders（下称 tc）只有 compute，没有光栅化管线，也没有交换链。可见性缓冲、阴影、着色、后处理、参考路径追踪都写成 `TC_KERNEL`。这样整个渲染核心都是确定性模块：同一个 kernel 在 CPU 后端上的输出，就是 GPU 后端的测试参照。
2. **sender 是模块之间唯一的异步契约，每条状态时间线是一个串行执行器。** 每个模块的异步接口都返回 lexec sender；co2 和 stackfull 都已经有到 lexec 的桥（`lexec/coro/co2.hpp`、`stackfull::exec`）。引擎内部的状态分两条时间线：渲染线程上的准备侧状态和 GPU 线程上的设备侧状态，各自是一个 `lexec::run_loop`。状态只在自己的时间线上修改，串行由结构保证，不靠调用方遵守前置条件。
3. **两种协程按“谁的代码要挂起”来分工。** 我们自己写、按帧推进的逻辑（脚本、动画时间线）用 co2 无栈协程，由宿主在模拟阶段逐帧驱动。需要在第三方代码深处或阻塞调用里挂起的工作（读文件、解码、监听文件变化、等待 GPU 完成）用 stackfull 有栈任务。
4. **主循环和窗口归宿主所有。** 引擎不阻塞宿主线程，不读输入。`render` 是库的核心；`scene`（世界模型与脚本）和 `asset`（资产加载）是建在它之上的可选层，宿主也可以用自己的 ECS 和资产系统直接驱动 `Renderer`。
5. **只用依赖库职责之内的能力。** 设计用到的能力四个库都已经提供；超出库职责的能力由引擎自己负责，见“依赖库的边界”。

## 四个库各自的位置

### lexec：异步契约、计算池与串行时间线

- 所有跨模块的异步接口都返回 sender。虚接口边界上用 `any_sender_of` 擦除类型；每次 connect 分配一次，每帧只发生几次。
- 两个 `static_thread_pool`：帧池跑帧内关键工作，后台池跑长任务（解码）。长任务因此挡不住帧内工作；lexec 的池没有优先级，分成两个池之后也不需要优先级。co2 的 `ThreadPool` 和 stackfull 的 worker 都不跑计算。帧池对 `bulk` 有并行定制，而且不分配内存。
- tc CPU 后端的 `ParallelExecutor` 由帧池上的 `bulk` 实现，CPU 后端的 kernel 和引擎共用线程。
- 两个 `run_loop`：渲染线程和 GPU 线程。`continues_on(gpu)` 之后的代码就运行在独占设备的那个线程上。
- `counting_scope` 加 `spawn_future` 管理在途帧和后台工作的生命周期，关闭时统一 join。
- `when_all` 用来组织一帧之内互相独立的编码工作。

### stackfull：IO 与等待

- 资产加载器是按同步写法写的普通函数，运行在 stackfull 任务里。读文件交给 `stackfull::blocking()`，因为普通文件不支持 epoll。默认先把文件整个读进来，再用 `exec::await(schedule(background) | then(decode))` 转到后台池解码，做完再回来；任务 park 期间，worker 去跑别的加载任务。
- 需要边读边解的解码器（通过回调取数据，比如 stb_image 的 IO 回调）直接在任务里运行，回调里读文件会让任务 park。挂起点位于第三方代码的调用栈深处，而这些代码不需要任何改造；无栈协程和 sender 都做不到这一点。
- 加载任务在步骤之间检查 `exec::stop_requested()`，被取消的加载以 `set_stopped` 结束，脚本的销毁依赖这一点。
- 等待 GPU 完成：tc 导出的 `SyncFence` 是一个可 poll 的 sync file，用 `io::Registration` 注册进 Poller，由一个任务等它变成可读；详见 gpu 模块。
- 文件监听：inotify fd 同样注册进 Poller，由一个任务循环读取事件，不额外占用线程。资产热重载和开发期的 kernel 重新加载共用这个监听任务。
- 同一资产的并发加载会去重：在途表里存放 `sync::Completion`，后来的请求方用 `exec::async_get` 直接拿到 sender，不再占用任务。
- 加载接口对外暴露为 `exec::spawn(io, fn)`，它本身就是 sender。

### co2：逐帧推进的逻辑

- 脚本返回 `ScriptTask`，这是按 co2 promise 协议写的自定义返回类型，用 `CO2_*` 宏编写。脚本只会在宿主调用 `ScriptHost::tick()` 时、由其中的 `co2::ManualExecutor::run()` 恢复，所以可以无锁读写场景。
- `ScriptTask` 的 promise 用 `await_transform` 把每个被等待的 sender 包成 `continues_on(sim)`。例如 `CO2_AWAIT_SET(mesh, assets.loadMesh(path))` 完成在 IO 线程上，但脚本一定在下一次 tick 时回到宿主线程继续执行。“脚本在别的线程上访问场景”这种状态，在类型层面就构造不出来。
- 每个脚本只是一个堆上的帧（通常几百字节），不需要独立的栈，上万个实体脚本也负担得起。
- co2 v2 还没有定时器和 `whenAll`：模拟时间上的等待（`nextFrame()`、`waitFor(duration)`）由 ScriptHost 提供，组合则交给 lexec。

### compute_shaders：同一份 kernel 跑在 GPU 和 CPU 上

- `render/kernels/` 下的 kernel 在构建时由 `tc_add_kernels` 转译成 GLSL/SPIR-V；用户自己的 kernel 在用户的目标里用同一个函数转译。材质模型这类变体写成 kernel 模板，用 `TC_KERNEL_INSTANCE` 实例化。用户 kernel 要用到的部分（着色 kernel 模板、材质采样、光照）是公开头文件，在 `render/include/lr/render/kernels/`。tc 只为被转译的那个头文件里的 kernel 和 `TC_KERNEL_INSTANCE` 生成程序，所以用户头文件包含这些公开头文件不会重复生成 lrender 自己的程序。
- 与 kernel 共享的数据（顶点、三角形、BVH 节点、实例、材质、灯光）存成表（`TableBinding`）。元素是按自然写法定义的 C++17 struct，主机端和 kernel 用同一份类型定义；布局（`Aos`、`Soa<P>`）和字段编码是类型参数。
- 提交是异步的：`submit()` 返回 `Submission`；支持 sync file 的设备用 `submitWithFence()`，返回的 `FencedSubmission` 里同时有 `.submission` 和 `.fence`。后端只在真正有读写冲突的 dispatch 之间插屏障，互不相关的 pass 可以在设备上重叠。
- 纹理池是只驻留在设备上的图像（`createImage` 返回的 `DeviceImage` / `DeviceCompressedImage`），没有主机副本；内容由 kernel 写入，或者用 `copyBufferToImage` 在设备时间线上从 buffer 拷贝。
- 间接 dispatch 让剔除、材质分类、压缩之后的工作量留在 GPU 上，不必回读。
- 设备能力在 `DeviceInfo::limits` 里：数量上限、子组、64 位整数与原子、支持的压缩格式。引擎启动时用 `checkKernel` 检查所有程序，并按能力选择 kernel 变体。
- 原子操作只能作用于 buffer 元素和 `Shared` 变量，所以可见性缓冲和阴影深度存在 `BufferBinding` 里。
- 同一张图像可以先作为存储图像写入，再在后面的 pass 里作为采样纹理读取（两个后端都按资源身份只保存一份图像），TAA 历史、阴影和 mip 生成都依赖这一点。
- `tc::Device` 只在 GPU 线程上调用；OpenGL 设备本来就只能在创建它的线程上使用。
- CPU 后端既是测试参照，也是调试后端，tc 的每项能力在它上面都有实现（子组用协作纤程模拟，压缩纹理在上传时解码）。

## 执行拓扑

| 执行资源 | 所属模块 | 线程 | 运行什么 |
|---|---|---|---|
| 宿主线程 | 宿主 | — | 输入、模拟、`ScriptHost::tick`、`Renderer::submit` |
| 模拟执行器 `co2::ManualExecutor` | scene | 调用 tick 的宿主线程 | 脚本 |
| 渲染线程 `lexec::run_loop` | render | 1，大部分时间空闲 | 应用增量、构建 BVH、编译帧图、编排编码 |
| GPU 线程 `lexec::run_loop` | gpu | 1 | 所有 `tc::Device` 调用：录制、提交、交给 Presenter；从不等待 GPU |
| 帧池 `lexec::static_thread_pool` | runtime | 默认为硬件线程数 − 3 | 编码、变换传播、tc CPU 后端 |
| 后台池 `lexec::static_thread_pool` | runtime | 默认 1 | 解码 |
| IO 调度器 `stackfull::sched` | runtime | 2 个 worker 加阻塞线程池，大部分时间 park | 读文件、文件监听、等待 GPU 完成 |

线程数都可以在 `RuntimeConfig` 里配置。四条线程规则：

- 只有 GPU 线程能访问 `tc::Device`。`GpuContext` 是一个能力令牌，只会作为 `GpuQueue::record(fn)` 回调的参数出现。GPU 侧状态（常驻表、资源池、历史图像、在途帧）都在 render 的 `FrameQueue` 里，它只在 GPU 线程上运行；GPU 线程丢掉的东西进当前录制那一帧的退役列表，等这一帧完成（也就是它之前的帧都完成）之后才释放。
- 准备侧状态（`RenderWorld` 镜像、BLAS 缓存、帧图的资源池、历史资源记录）只在渲染线程上修改。
- 场景只在调用 tick 的那个宿主线程上访问。
- 跨线程只传递不可变值（`RenderWorldDelta`、`shared_ptr<T const>` 形式的资产），或者转移所有权（准备好的帧、回读的图像）。

## 模块

```
examples → present, scene, asset, render, runtime
present  → render, runtime, core   实现 render 定义的 Presenter 接口
render   → gpu, runtime, core
gpu      → runtime, core
asset    → runtime, core
scene    → core
runtime  → 只依赖 lexec 与 stackfull
```

库由 7 个模块组成：core 和 runtime 是基础；render 是核心，gpu 是它面对驱动和线程的那一半；present、asset、scene 是可选层；examples 里是组合根。scene、render、asset 互不依赖，只通过 core 里的数据词汇交流。

### core：值类型与数据词汇

只有数据，没有行为，作为一个 INTERFACE 目标。

- 数学类型直接使用 tc 的 `vec`/`mat`，因为 kernel 只认得它们。`Transform`、`Aabb`、四元数也建立在它们之上，并且没有基类、运算符都按值返回。tc 转译器会把用户类型的运算符和函数模板转译成 GLSL，所以这些类型在 kernel 里同样能用。
- 强类型 ID：`MeshId`、`TextureId`、`MaterialId`、`InstanceId`、`LightId`、`ViewId`、`RenderTargetId`，以及 scene 用的、带代数的 `EntityId { index, generation }`。
- 数据词汇：`MeshData`、`ImageData`、`MaterialDesc`、`LightDesc`、`CameraDesc`、`RenderWorldDelta`。资产以 `MeshHandle`（即 `shared_ptr<MeshData const>`）和 `ImageHandle` 的形式流转。
- `Result<T, E>` 使用 `tl::expected`。

```cpp
struct RenderWorldDelta {                    // 自上一帧以来的变化；没变的东西不出现
    std::vector<MeshUpsert> meshes;          // 新增，或热重载替换
    std::vector<TextureUpsert> textures;
    std::vector<MaterialUpsert> materials;
    std::vector<InstanceUpsert> instances;   // 新增，或世界矩阵、网格、材质有变化
    std::vector<LightUpsert> lights;
    Removals removed;                        // 按种类分开的 ID 列表
};
```

### runtime：进程级执行资源

拥有帧池、后台池和 IO 调度器，因为它们被多个模块共享，并且关闭顺序有讲究；同时定义 `ComputeScheduler`、`IoScheduler` 这两个调度器类型。渲染线程归 render，GPU 线程归 gpu，模拟执行器归 scene，它们分别和准备侧状态、设备、场景绑在一起。

```cpp
struct Runtime {
    virtual ~Runtime() = default;
    virtual ComputeScheduler framePool() noexcept = 0;
    virtual ComputeScheduler backgroundPool() noexcept = 0;
    virtual IoScheduler io() noexcept = 0;                  // stackfull::exec::scheduler
    virtual io::Poller &poller() noexcept = 0;              // IO 调度器轮询的那个 Poller
};
std::unique_ptr<Runtime> makeRuntime(RuntimeConfig const &config);   // 析构：先停 IO 调度器，再 join 两个池
```

### gpu：独占设备的线程

```cpp
struct GpuContext {                 // 能力令牌：只在 GPU 线程上的回调里拿得到
    tc::Device *device = nullptr;
};

using CompletionSender = lexec::any_sender_of<lexec::set_value_t(), lexec::set_error_t(std::exception_ptr),
                                              lexec::set_stopped_t()>;

struct GpuQueue {
    virtual ~GpuQueue() = default;
    virtual tc::DeviceInfo const &info() const noexcept = 0;   // 含 DeviceLimits 与 syncFiles
    virtual runtime::SerialScheduler scheduler() noexcept = 0;

    // 在 GPU 线程上运行 fn(GpuContext &)，以它的返回值完成；抛出的异常走 set_error。
    template <class Fn>
    auto record(Fn fn) {
        return lexec::schedule(scheduler())
             | lexec::then([this, fn = std::move(fn)]() mutable { return fn(contextOnGpuThread()); });
    }

    // 在 GPU 线程上调用：提交到目前为止录制的工作，不等待。
    virtual tc::FencedSubmission submit(GpuContext &context) = 0;
    // 任意线程：一个 sender，设备做完这次提交时，在 IO 调度器上完成。
    virtual CompletionSender whenComplete(tc::FencedSubmission submitted) = 0;
    // 在 GPU 线程上调用：销毁设备，再按同样的配置建一个；建不出来时抛 tc::DeviceError。
    virtual void recreateDevice(GpuContext &context) = 0;

protected:
    virtual GpuContext &contextOnGpuThread() noexcept = 0;   // 在其他线程调用属于契约违规
};
std::unique_ptr<GpuQueue> makeGpuQueue(GpuQueueConfig const &config);   // 后端名、设备名过滤、Runtime（必需）
```

- **启动**：GPU 线程先调用 `tc::createDevice()`，CPU 后端的 `CpuDeviceConfig::executor` 设成帧池上的 `bulk`，然后进入 `run_loop::run()`。设备也在这个线程上销毁。检查 kernel 能不能运行、选择变体归 render，见“启动检查与 kernel 变体”。
- **提交**：`info().syncFiles` 为真时用 `submitWithFence()`；否则用 `submit()`，包成一个 `.fence` 为空的 `FencedSubmission`。两种调用都不等待。`syncFiles` 为假时 `submitWithFence()` 会先等工作完成再返回空 fence，所以这时只用 `submit()`。
- **等待完成**：`whenComplete` 是 IO 调度器上的一个 stackfull 任务（`exec::spawn(io, …)`），GPU 线程因此从不等待 GPU，可以接着录制下一帧：
  - `.fence` 不为空：先给 fd 设上 `O_NONBLOCK`，再用 `io::Registration` 注册进 Poller。任务先读 `readEvents()`，`isSignaled()` 为假时 `waitReadable(seen)`，然后重新检查。任务 park 期间不占线程，完成由 epoll 报告；`Registration` 在 `SyncFence` 之前析构。
  - `.fence` 为空：`.submission.isComplete()` 已为真就立即结束（CPU 后端总是如此）；否则在任务里 `stackfull::blocking([&] { submission.wait(); })`，等待落在 IO 调度器的阻塞线程池上。llvmpipe 和 lavapipe 没有 sync file，走的就是这条路径。
  - fd 设不上非阻塞或注册失败时，改在阻塞池上 `fence.wait()`。fence 可读之后，`Submission` 自己的完成（Vulkan 的 `VkFence`）可能晚一点才报告，这时也在阻塞池上等它，这样回到 GPU 线程之后 `takeTimings` 一定不会等待。
  - 等待抛出的异常（比如 `tc::DeviceError`）走 `set_error`；任务没能启动时 stackfull 给出的 `error_code` 包成 `std::system_error`。
  - 测试里用 eventfd 代替 sync file：两者都是变成可读就算完成，所以没有 sync file 的机器上也能测到 Poller 这条路径。
- **重建设备**：`recreateDevice` 在 GPU 线程上先销毁旧设备（tc 的 Vulkan 设备析构时会等它排队的工作，真正丢失的设备会立即返回），再按原来的配置创建新设备。`info()` 一直是第一个设备的信息，配置相同，设备也相同。
- **同一个 fence 有两个使用者**：Android 零拷贝上屏时，`ASurfaceTransaction_setBuffer` 也要这个 fence，此时 `dup` 一份 fd 给 Presenter。
- **计时**：完成之后回到 GPU 线程，用 `takeTimings(submitted.submission)` 取这一帧的设备耗时。这次提交已经完成，调用不会等待，也不会碰正在录制或执行的下一帧。
- **开发期重新加载**：文件监听任务发现 `tc_generated/<target>/glsl` 有变化时，通过 `record` 在 GPU 线程上调用 `ProgramRegistry::instance().reloadFrom(dir)`，结果写进日志；下一次 `useKernel` 就换成新程序。发布构建关闭 `TC_PROGRAM_RELOAD`。

### render：渲染核心

#### Renderer 接口

```cpp
struct ViewDesc {
    ViewId id;                  // 跨帧稳定；历史资源（TAA、累积）按它保存
    RenderTargetId target;
    PipelineId pipeline;
    CameraDesc camera;
    ViewSettings settings;
};

struct FrameDesc {
    RenderWorldDelta delta;
    std::vector<ViewDesc> views;
};

using FrameSender = lexec::any_sender_of<lexec::set_value_t(FrameResult),
                                         lexec::set_error_t(std::exception_ptr), lexec::set_stopped_t()>;

struct Renderer {
    virtual ~Renderer() = default;
    virtual tc::DeviceInfo const &deviceInfo() const noexcept = 0;
    virtual RendererCapabilities const &capabilities() const noexcept = 0;   // 为设备选中的 kernel 变体
    virtual RenderTargets &targets() noexcept = 0;
    virtual PipelineRegistry &pipelines() noexcept = 0;
    virtual MaterialModels &materialModels() noexcept = 0;                  // 见“材质”
    // 立即返回，任意线程可调用。帧按调用顺序编号、依次处理，不管它的 sender 有没有启动；
    // sender 按同样的顺序在设备做完该帧之后完成，FrameResult 里带统计、墙钟时间和离屏目标的图像。
    virtual FrameSender submit(FrameDesc frame) = 0;
};
std::unique_ptr<Renderer> makeRenderer(RendererConfig const &config);   // 后端、Runtime、framesInFlight（默认 2）、变体开关

struct RenderTargets {                                                       // 线程安全，改动从之后提交的帧起生效
    virtual ~RenderTargets() = default;
    virtual RenderTargetId createWindow(WindowTargetDesc const &desc) = 0;   // {extent, shared_ptr<Presenter>}
    virtual RenderTargetId createImage(ImageTargetDesc const &desc) = 0;     // 结果在 ViewResult::image 里
    virtual void resize(RenderTargetId id, core::ivec2 extent) = 0;
    virtual void destroy(RenderTargetId id) = 0;
};

struct PresentFrame {               // Readback 路径：设备做完之后，映射内存里的像素
    std::uint64_t frameIndex = 0;
    ViewId view;
    RenderTargetId target;
    ImageRgba8View pixels;          // 只在 present() 调用期间有效
};

struct Presenter {                  // lr::present 提供实现，宿主也可以自己写
    virtual ~Presenter() = default;
    virtual void present(PresentFrame const &frame) = 0;   // GPU 线程上按帧序调用，不得阻塞
};
```

- 窗口目标的像素交给它的 Presenter，`ViewResult::image` 留空；离屏目标的像素拷进 `ViewResult::image`。同一帧可以两种都有。
- 目前只有 Readback 路径。SharedBuffer 路径（引擎直接渲染进 `AHardwareBuffer`，提交之后立即把 buffer 和 acquire fence 交给 Presenter）留给 Android：到时候 `Presenter` 加一个默认是 Readback 的 `path()`，再加一个接收 `SharedBufferFrame` 的入口，现有的 Presenter 不用改。
- 多视图：一帧可以有任意个视图，渲染到不同目标，比如编辑器的多个视口、分屏、离屏缩略图。阴影贴图是管线内部的 pass，不算视图。
- 缩放：帧图里的资源尺寸相对目标尺寸声明，`resize` 之后按新尺寸重建；历史资源按尺寸记录，尺寸变了就换一对新的，`valid` 为假，累积从头开始。
- 销毁：已经提交的帧照样交给原来的 Presenter（帧图里持有它的 `shared_ptr`），之后再提交这个目标就抛 `std::invalid_argument`。
- Android 生命周期：surface 销毁时宿主 `destroy` 对应的目标，重建时再 `createWindow`。tc 的设备不依赖窗口，所以这期间设备和常驻资源都留着。
- 在途帧：录制了、但还没收尾的帧最多 `framesInFlight` 个，见“一帧”。宿主通过等待 `FrameSender` 自己决定提交节奏；超过上限的帧准备好之后在 GPU 线程上排队，不会被丢掉。

#### RenderWorld：渲染侧的常驻镜像

渲染线程上的一份场景镜像，按种类以 SoA 存放，槽位稳定。每帧只应用 `RenderWorldDelta`，并记下每张表的脏区间；编码只处理脏的部分，在帧池上并行完成。每个实例还记着上一帧的世界矩阵（`GpuInstance::previousObjectToWorld`），用来算 TAA 的运动矢量：增量移动了实例，这一帧写的是移动前的矩阵，下一帧再写一次这个实例，把它改成当前的矩阵；新实例两者相同。所以每帧必须恰好应用一次增量。

镜像持有网格的不可变句柄，设备丢失之后可以重新上传几何；纹理的像素数据不保留（见“纹理”）。

#### 帧图与扩展

```cpp
struct RenderFeature {                      // 扩展点：往一个视图的帧图里加 pass
    virtual ~RenderFeature() = default;
    virtual FeatureStage stage() const = 0;  // Visibility、Shadow、Shading、Transparent、PostProcess、Output
    virtual void setup(FrameGraphBuilder &graph, ViewContext const &view) = 0;   // 渲染线程
};

void TonemapFeature::setup(FrameGraphBuilder &graph, ViewContext const &view) {
    auto pass = graph.addPass("tonemap");
    auto const hdr = pass.read(view.sceneColor());
    auto const ldr = pass.write(view.output());
    pass.record([hdr, ldr](PassContext &context) {          // GPU 线程
        auto kernel = TonemapKernel{};
        context.attach(kernel.input, hdr);
        context.attach(kernel.output, ldr);
        context.dispatch(kernel, context.groupsCovering(ldr, kernel.local_size));
    });
}
```

- 一条管线就是按阶段排好的一组 feature。`PipelineRegistry` 提供两条内置管线，用户可以在它们的基础上按阶段插入自己的 feature，组成新的管线。
- 帧图在渲染线程上编译：检查读写关系，剔除没有被用到的 pass，分配临时资源（按描述符在帧之间复用），维护按 `ViewId` 保存的历史资源。
- pass 按编译后的顺序在 GPU 线程上录制。屏障由 tc 按实际的读写冲突插入，互不相关的 pass 可以在设备上重叠，帧图不需要自己计算屏障。
- `PassContext` 把帧图的资源句柄解析成 tc 资源并绑定，`dispatch` 和 `dispatchIndirect` 分别对应 `execute` 和 `executeIndirect`（可以指定读参数 buffer 的第几项），同时记下这次 dispatch 属于哪个 pass；用户 kernel 和内置 kernel 走同一条路径。
- 历史资源：`graph.history<G>(name, extent)` 返回一对图像 `{previous, current, valid}`，同一个 `ViewId` 和名字在每个用到它的帧里拿到同一对，并且互换一次：上一帧写进 `current` 的内容，这一帧从 `previous` 读。互换而不是原地读写，是因为 GLES 3.1 不允许 kernel 同时读写 r32 以外格式的存储图像。两张图像在帧开始时都算已写入，写 `current` 的 pass 总会保留。第一次用到、尺寸或格式变了、或者被释放过之后，`valid` 为假。记录在渲染线程上（只是 id），图像在 GPU 线程上按 id 创建；连续 60 帧没用到、或者视图的目标已经销毁，就释放。TAA 也用它。
- feature 用 `requirements()`（`FeatureRequirements`）声明额外的需求：要追踪光线的 feature 让 `RenderWorld` 从这一帧起维护 BVH（见“参考路径追踪”）；要抖动投影的 feature（TAA）让设置了 `AntiAliasing::Taa` 的视图每帧换一个子像素偏移，`ViewContext::temporal` 里有偏移、不带偏移的这一帧和上一帧的视图投影矩阵，以及历史能不能接着用。`graph.onFinish(fn)` 在帧完成后填写 `ViewResult`，比如累积的采样数。

#### 两条内置管线

- **可见性缓冲管线（实时主管线）**：可见性、阴影、着色、透明、时间抗锯齿、色调映射六个 feature。
  1. GPU 实例剔除：写出可见实例列表（实例、它的第一个可见三角形、网格的第一个索引和第一个顶点，每项四个字，后面的 pass 因此不必再读网格表），以及后续光栅化的 `DispatchIndirectCommand`。材质是 `Blend` 的实例不进这张列表，留给透明 pass。
  2. 计算光栅化写可见性缓冲：每个像素留下最近的深度，以及这个深度上编号最小的可见三角形。设备有 64 位原子（`KernelFeature::Int64Atomics`）时一遍完成：每个覆盖的像素对一个 `tc::u64` 做 `atomicMin(floatBitsToUint(depth) << 32 | 三角形)`，最小值正好是这两样，再由一个逐像素的 pass 拆进两个 32 位 buffer；其他设备分两遍，先 `atomicMin` 深度，再在深度相等时 `atomicMin` 三角形。两个光栅化 kernel 展开同一个宏 `LR_TRIANGLE_RASTERIZER`，只有写像素的那一步不同，算出的深度逐位相同，所以两种做法的三角形和深度也逐位相同（测试在 CPU 后端和 lavapipe 上比对）。后面的 pass（着色、透明的深度测试、TAA）只读拆开后的 `VisibilityOutputs`，不关心是哪一种。`Mask` 材质在光栅化时按透视校正的 UV 采基础色纹理的第 0 层，alpha 低于阈值的像素不覆盖。阴影贴图本来就只写深度，总是用两遍的那个 kernel 的深度模式。
  3. 阴影：场景里第一个 `castsShadows` 的方向光有级联阴影贴图，一次只有这一盏灯有阴影。视锥在 `ViewSettings::shadows.distance` 之内按对数与均匀 3:1 混合的分割点切成 1 到 4 段，每段取包住这段视锥的球，球的大小只由这段视锥的形状决定，球心在光空间里对齐到整数个纹素，所以相机移动时级联只按整纹素移动，阴影边缘不闪。深度范围取整个场景包围盒沿光方向的投影，视锥外的遮挡物也能投下阴影。每个级联用同一套剔除、压缩和光栅化 kernel（只写深度）画进一个 `BufferBinding<uint>` 的一段，深度是 `floatBitsToUint` 后 `atomicMin`；级联的矩阵放在这个 buffer 开头的 128 个字里，由清除 pass 从主机上传。着色时选最细的、能容下 PCF 核的级联，接收点先沿几何法线偏移约一个纹素（按坡度增大），深度再减去随坡度增大的偏置，然后做 3×3 的双线性 PCF（16 次手动比较，因为深度在 buffer 里）。透明物体不投阴影，但接收阴影。
  4. 着色：先分类，再按材质模型间接着色。分类 pass 逐像素从三角形找到实例和材质，读出材质模型号；没有三角形的像素当场写环境光。每个工作组先在内部给每个模型的像素排好名次（共享内存的 `atomicAdd`，或者子组变体里每个子组对每个模型做一次 `subgroupExclusiveAdd`），再给每个模型做一次全局 `atomicAdd`，取得这个工作组在该模型里的位置，写进逐像素的“模型、名次”。一个工作组的小 pass 按模型号求前缀和，得到每个模型列表的起点，同时写好每个模型的 `DispatchIndirectCommand`（每 64 个像素一个工作组）；再一个逐像素的 pass 把（像素、三角形）放进各自模型的列表。最后每个模型一个 pass，用 `executeIndirect` 读自己那一项参数，只跑自己的像素。列表里像素的顺序随原子操作而变，但每个像素的着色只取决于它自己，所以图像是确定的。
  5. 透明：`Blend` 实例单独剔除、压缩，再用同一个光栅化 kernel 的第三种模式把片元插进逐像素链表：一个 `uint` buffer 里依次是每个像素的表头、片元计数、每个片元三个字（深度、下一个、可见三角形），`atomicAdd` 分配、`atomicExchange` 挂表头，容量是每像素平均 2 个，超出的片元丢弃。只插入比不透明深度更近的片元。这一遍按 top-left 规则覆盖像素，边函数按端点的固定顺序计算，共享一条边的两个三角形算出的值严格相反，所以落在边上的像素只归其中一个，不会混合两次。之后把场景颜色拷一份，透明混合 kernel 对每个像素的链表按（深度、三角形）排序，保留最近的 8 个，用内置的 metallic-roughness 模型、相同的灯光和阴影着色（材质指定了别的模型也一样），从前往后混合到不透明颜色上。排序键里有三角形，所以结果与片元插入的顺序无关。
  6. 时间抗锯齿（`ViewSettings::antiAliasing` 为 `Taa` 时，在色调映射之前）。视图的投影每帧按 Halton(2, 3) 的第 1 到 16 个点（减 0.5）平移不到一个像素，渲染线程按视图记住抖动序号和上一帧不带抖动的视图投影矩阵；上一帧没有渲染这个视图、尺寸变了、`ViewDesc::cameraCut` 为真或设备重建过，序号回到 0，历史作废。运动 pass 逐像素在实例的物体空间里求出可见点，分别用这一帧和上一帧的世界矩阵、视图投影矩阵（都不带抖动）投到屏幕上，差就是运动矢量；静止的物体在静止的相机下运动恰好是 0，历史因此在纹素中心采样，不会变模糊。没有三角形的像素跟着远平面走。解析 pass 在 `p + 0.5 - 运动` 处双线性采样 RGBA16F 的历史（`graph.history`），把它夹在这一帧 3×3 邻域颜色在 YCoCg 里的包围盒中，再与这一帧混合：历史的 alpha 里存平均了多少帧，前 10 帧取平均，之后新一帧的权重固定为 0.1。历史作废、或者点上一帧在屏幕外的像素从这一帧重新开始。结果写进历史，再拷回场景颜色。透明像素用的是它后面不透明表面的运动。路径追踪管线没有这个 feature，不抖动。
     收敛与拖影的测量（32×32、斜 0.3 弧度的直边，64×64、每帧移动 2 像素的白块）：32 帧后边缘像素的值与像素被覆盖面积的平均差 0.036（不开 TAA 时 0.197），这 16 个点本身能做到的最大差是 0.12；GPU 的值与在主机上按同样的抖动和混合模拟的结果相差不到 4e-4（半精度的舍入）。白块移走后留下的轨迹与背景的差在 5e-5 以内（就是 0.2 存成半精度的误差），白块内部仍然是 1。
  7. 色调映射；后处理 kernel 声明中等精度（`Precision::Medium`）。
- **参考路径追踪**：两级 BVH，用固定大小的 `std::array` 栈做迭代遍历（tc 不允许递归），渐进累积。它是画质参照，也是可见性缓冲管线着色结果的测试参照。
  - BVH 只有它用。第一次有视图需要时，渲染线程给每个网格建一棵 BLAS（分箱 SAH，叶子最多 4 个三角形），再在实例上建 TLAS（叶子一个实例）；之后增量里的网格重建自己的 BLAS，实例或网格有变化就重建 TLAS。深度不超过 31 层（快到上限时改按中位数对半分），遍历栈是 32 个元素。节点、三角形（三个角在顶点表里的位置）、每个网格的根、TLAS 节点经 `SceneUpdate` 进 `DeviceScene` 的表，可见性缓冲管线不读这些表。构建是同步的，大场景第一次用到时那一帧的准备会变长。
  - kernel 每帧每像素一条路径：先 TLAS，再用 `worldToObject` 把射线变换到实例的物体空间遍历 BLAS（方向不归一化，两个空间里的距离相同）。着色和光源用与可见性缓冲管线同一份代码（`kernels/Lighting.h`、`kernels/Material.h`），双面材质的处理也相同：单面材质的背面对所有射线都不可见，双面材质从背面看时法线翻转。纹理固定采第 0 层（多个采样平均下来就是像素足迹上的平均）。`Mask` 材质在遍历时按 alpha 剔掉命中；`Blend` 材质是随机不透明：射线（包括阴影射线）以 1 - alpha 的概率穿过它，判定用射线的种子和三角形号散列，期望正好是按 alpha 混合，所以简单场景两条管线一致。每个顶点对方向光和点光源做下一事件估计并追踪阴影射线；加上自发光；逃逸的射线取均匀环境光；继续的方向按漫反射和 GGX 两个波瓣的混合采样，按混合密度加权；第三次弹射起用俄罗斯轮盘。随机数是像素、采样序号、维度的 PCG 散列，所以每个设备抽到的随机数相同。
  - 累积放在 RGBA32F 的历史资源里，存运行均值，均值写进 `sceneColor`，后面照常色调映射。相机、目标尺寸或场景（任何改变网格、纹理、材质、实例、灯光、环境的增量）变了就从头开始，这由渲染线程判断；纹理流式写完或消失也要从头开始，而这只有 GPU 线程知道，所以采样数在 GPU 线程上按视图记：录制时看渲染线程给的“可以继续”和纹理驻留的代数，`ViewStats::accumulatedSamples` 给出当前的采样数。
- 可见实例的压缩和像素分类各有两个 kernel：用子组的（`subgroupExclusiveAdd`、`subgroupAdd`，`Shared` 只存每个子组的总数，子组大小取 `gl_SubgroupSize`），和只用 `Shared` 的。两者的输出相同（分类的列表顺序除外，图像相同），启动时选择，见下一节。

#### 启动检查与 kernel 变体

- render 自己维护一张程序表：任何一帧都可能用到的程序，和每种可选变体的程序（子组：压缩和分类的子组版本；一遍写入的可见性缓冲：清除、光栅化、拆分三个用 64 位整数的 kernel）。测试检查注册表里每个 `lr_` 开头的程序都在表里。
- `makeRenderer` 建好设备后，在 GPU 线程上对表里的程序逐个 `checkKernel`。必需的程序有不能运行的，就抛 `tc::DeviceError`，消息里列出每个程序和原因，比如“`lr_shade_metallic_roughness` 用了 8 个 storage buffer，设备只允许 4 个”。可选变体的程序全部能运行、并且 `RendererConfig::subgroups` / `singlePassVisibility` 不是 `VariantUse::Off`，才用这个变体。选择结果是 `Renderer::capabilities()`（`RendererCapabilities`：是否用子组、是否一遍写可见性缓冲、子组大小），feature 从 `ViewContext::capabilities` 读到，按它在帧图里放 pass。`Off` 让测试能在同一个设备上比较两种变体。
- 设备丢失后重建的设备用同样的配置创建，但仍然检查一遍：已选变体和所有材质模型的程序都要能运行，否则按建不出设备处理。
- 测试用的设备选到的变体：CPU 后端两者都用（子组大小 32）；lavapipe 的 Vulkan 两者都用（子组大小 8）；llvmpipe 的 OpenGL 和 GLES 都不用（没有子组，也没有 64 位原子）。GLES 本来就没有 64 位整数。

#### 材质

- 内置一个 uber 材质模型，覆盖 glTF 的 metallic-roughness：基础色、metallic-roughness（G 是粗糙度、B 是金属度）、自发光、法线四张可选纹理，各自乘对应的因子；法线强度 `normalScale`；一个对所有纹理生效的 UV 变换（KHR_texture_transform：先缩放、再逆时针旋转、再平移）；alpha 模式 `Opaque`、`Mask`、`Blend`。材质引用的纹理不存在或还没流式写完时，按缺席处理，只用因子。
- 没有存切线：法线贴图的切线空间由三角形三个角的位置差和 UV 差求出 U、V 增大的方向，切线沿 U 方向对着色法线正交化；按 glTF 的约定，切线空间的 +Y 指向图像的上方（V 减小的方向），副切线按这个方向定号，所以 UV 镜像时也对。双面材质从背面看时，整个扰动后的法线翻转。
- 均匀环境光：漫反射部分是 `baseColor·(1 - metallic)·L`；镜面部分是 `L·(F0·A + B)`，(A, B) 是这套 GGX（高度相关的 Smith 可见性、Schlick 菲涅尔）在 (n·v, 粗糙度) 上的方向反照率，进程里第一次用到时在主机上用重要性采样（每格 2048 个 Hammersley 样本）算出 32×32 的表，放进一张 RG16F 纹理，着色时双线性采样。与暴力积分相比，表的误差不到 0.25%；在白炉测试里，金属球和电介质球的光栅结果与路径追踪的平均差在 0.1% 以内。
- 材质模型：`MaterialDesc::model` 指定用哪个模型着色，0 是内置的 metallic-roughness；`MaterialDesc::modelParameters`（一个 `vec4`）留给其他模型用。每个模型是着色 kernel 模板 `template <class Model> struct TC_KERNEL ShadeMaterialModel` 的一个实例，一个模型一个程序。kernel 在像素看到的点上插值出表面、采好材质的四张纹理，交给 `Model::evaluate(ShadingInputs)`（位置、着色法线、几何法线、视线方向、UV、`GpuMaterial`、纹理乘因子后的 `MaterialSample`、实例号）；模型返回 `ShadedSurface`（自发光、基础色、金属度、粗糙度、法线、覆盖率、是否受光），kernel 再用场景的灯光、阴影和环境光照亮它，每个模型都一样。所以模型决定的是“这一点是什么样的表面”，光照模型（Lambert 加 GGX）是公共的；`emissionOnly(radiance)` 是只发光的表面。这对应可见性缓冲引擎里“材质 pass 求表面属性、光照统一”的分法。
- 应用自己的模型：在自己用 `tc_add_kernels` 转译的头文件里写模型类型和一行 `TC_KERNEL_INSTANCE("app_xxx", ShadeMaterialModel<app::Xxx>)`（写在 `lr::render::kernels` 命名空间里），再 `renderer->materialModels().add(makeMaterialModel<app::Xxx>())` 得到模型号。`MaterialModel` 接口有两个函数：`program()` 给出 kernel 的程序名，`add` 在 GPU 线程上 `checkKernel`，不能运行就抛 `tc::DeviceError`；`shade(ModelShading &)` 在 GPU 线程上录制，`makeMaterialModel` 给出的实现调用 `ModelShading::dispatch(kernel)`，它绑定所有实例共有的绑定，按模型号 `executeIndirect`。最多 16 个模型（`maxMaterialModels()`），因为分类时每个工作组在共享内存里给每个模型计数。材质用了渲染器没有的模型号，增量在渲染线程上被拒绝（`std::invalid_argument`），世界不变。
- 透明片元和路径追踪器总是用内置模型着色，材质指定了别的模型也一样。
- tc 只认 kernel 自己的成员为绑定，没有办法把一组绑定放进别的结构体，所以采样纹理池的 kernel 都展开同一个宏 `LR_MATERIAL_TEXTURES(first)`，它声明 12 个池绑定和采样它们的成员函数。同样的办法：所有模型的着色 kernel 和透明混合 kernel 展开 `LR_SURFACE_SHADING(first)`（材质纹理、环境表、场景的表、可见实例、阴影贴图，以及求表面、阴影和光照的成员函数），两种可见性光栅化展开 `LR_TRIANGLE_RASTERIZER`，用到可见实例列表的 kernel 展开 `LR_FIND_VISIBLE_SLOT`。

#### 纹理

- tc 没有无绑定资源（它以 GLES 3.1 为底线），所以纹理按“格式 + 尺寸档”分池：每个池是一张只在设备上的 `Dim::D2Array` 图像，正方形、边长是 2 的幂、带完整的 mip 链。kernel 里有固定的 12 个池槽：8 个 RGBA8 池（`DeviceImage`，第 k 个放边长 16·2^k 的纹理，16 到 2048），4 个压缩池（`DeviceCompressedImage`，先到先得，每个池一种格式一种尺寸），分别通过 `TextureBinding` 和 `CompressedTextureBinding<…, Dim::D2Array>` 采样；没有池的槽绑一张 1×1 的空图像。池的尺寸和层数受 `DeviceLimits::imageDimension2D` 与 `imageArrayLayers` 限制（规范下限是 2048 和 256），也受 `RendererConfig::maxTextureSize` 限制。
- 纹理在材质里是一个 32 位的引用：槽、层、log2 边长、是否 sRGB、U 和 V 的寻址方式。池的采样器总是 Repeat、线性、三线性 mip；Clamp 和 Mirror 在 kernel 里折叠坐标，并让坐标离边缘半个纹素（取三线性混合的两层里较粗的那层），这样不会混进对边的纹素。
- 渲染线程上的 `TextureLayout` 决定纹理放进哪个池的哪一层：压缩图像在设备有这种格式的数组、是正方形 2 的幂、不超过上限、带完整 mip 链、还有空的压缩槽时，原样进压缩池，所有级别都从图像拷贝；其余的图像（包括设备不支持的压缩格式）在渲染线程上用 `tc::decodeCompressed` 解码成 RGBA8，在线性颜色里重采样成下一个 2 的幂的正方形（缩小用面积平均，放大用按寻址方式环绕的双线性；超过上限的缩小到上限），进 RGBA8 池，只上传第 0 层。RGBA8 池满了就把层数翻倍：GPU 线程建一张更大的图像，用 kernel 逐层逐级拷过去，旧图像进退役列表。压缩池不能被 kernel 写，也没有图像到图像的拷贝，所以一开始就按约 32 MiB 分配层数，满了的新纹理改走解码。移除纹理时它的层回到空闲列表。
- RGBA8 池存纹理自己的编码（sRGB 纹理存 sRGB 值），kernel 采样之后再解码。下采样 kernel 在线性颜色里求 2×2 平均（黑白相间的棋盘格在第 1 层是线性 0.5，而不是 sRGB 平均出来的 0.21），所以 mip 级别是对的；代价是纹素之间的双线性插值发生在 sRGB 值上，高对比边缘的插值偏暗。tc 没有 sRGB 格式，要完全正确只能存 RGBA16F，内存翻倍，现在没有这样做。
- compute 没有导数，`texture()` 只采第 0 层，所以着色时用射线微分算 LOD：像素右边和下边相邻像素中心的相机射线与三角形所在平面相交，求出那里的 UV，与本像素 UV 的距离取较大者就是足迹，LOD = log2(足迹 × 边长)，再用 `textureLod`（`Mip::Linear`）三线性采样。各向同性，斜看时偏模糊。
- 新纹理流入池时，GPU 线程把数据拷进当前帧的暂存段（`createBuffer` 分配的映射内存，每个在途帧一段，第一次用到时才分配），再 `copyBufferToImage` 写进对应的层和 mip 级别；拷贝和前后的 dispatch 一样在设备时间线上排序。每段从录制第 k 帧时用到的是第 k - framesInFlight 帧用过的那一段，那一帧已经做完，主机写它不会和设备读冲突。每帧最多拷 `RendererConfig::textureStreamingBytes`（默认 8 MiB），按 mip 级别排队，一帧放不下的留到下一帧；`copyBufferToImage` 只能整级拷贝，所以比预算还大的单个级别独占一帧，暂存段为它临时变大。RGBA8 纹理的第 0 层到了，同一帧里接着用下采样 kernel 逐级生成其余各级（每次 dispatch 一层，因为 llvmpipe 的 `texelFetch` 在一次 dispatch 里层号不同时会读错层）。最后一级写完，纹理才算驻留，材质才引用它；之前材质只用因子。
- 材质里的引用由 GPU 线程填：`SceneUpdate` 里的材质带着纹理 ID，`DeviceScene` 按 `TexturePool` 的驻留情况把它们换成引用，驻留情况一变就重写所有材质。替换一个纹理（同一个 ID 再 upsert）时，旧数据马上不再被引用，新数据写完之前只用因子。
- 压缩格式按 `DeviceLimits::compressedArrayFormats` 选，因为要放进数组（Adreno 的 GLES 上 BC4、BC5 只能是 2D）。资产离线烘焙成 KTX2，桌面一份 BC、移动端一份 ASTC 或 ETC2；KTX2 的块顺序就是 `copyBufferToImage` 要求的顺序，可以直接拷贝。lavapipe 没有 ETC2，测试里这条路走解码。解码和重采样现在在渲染线程上做，大纹理多的场景应该挪到后台池。
- `RenderWorld` 只把图像句柄交给 GPU 线程，自己不留；`TexturePool` 写完最后一级就放掉它，纹理在整个系统里只有设备上这一份（宿主自己留的句柄除外）。设备丢失后，渲染线程在下一帧把放置过的所有纹理记为丢失，`FrameResult::lostTextures` 列出需要重新提交的 `TextureId`，在此之前材质只用因子；那一帧失败的话，由下一个成功的帧报告。用了 scene 和 asset 时，由它们自动重新加载并提交。

#### 与 kernel 共享的数据

```cpp
struct GpuVertex {                                  // 5 个字；std430 下要 12 个字
    vec3 position;
    Encoded<vec3, OctahedralNormal> normal;
    Encoded<vec2, Half2> uv;
};
struct GpuInstance {
    Encoded<mat4, Affine3x4> objectToWorld;         // 12 个字，原本 16 个
    Encoded<mat4, Affine3x4> worldToObject;
    uint meshIndex;
    uint materialIndex;
    Encoded<mat4, Affine3x4> previousObjectToWorld; // 上一帧的，算运动矢量用
};
using VertexTable   = TableBinding<GpuVertex, 0, Soa<>>;    // 光栅化和遍历只读位置
using BvhNodeTable  = TableBinding<GpuBvhNode, 1>;          // 8 个字，按 uvec4 宽读
using InstanceTable = TableBinding<GpuInstance, 2, Soa<>>;  // 剔除只读部分字段
```

- 布局是用 `probes/layout_probe` 的设备计时定的：各布局轮流 dispatch，设备的时钟变化对它们影响相同。RTX 5070 Ti 的 Vulkan 上，表远大于缓存时（400 万个实例、840 万个顶点）：剔除 `Aos` 0.49 ms、`Soa<>` 0.27 ms；只读三个角的位置，`Aos`（5 个字）0.54 ms、`Soa<>` 0.47 ms、补齐到 8 个字的 `PaddedAos` 0.66 ms；读整个顶点三者是 0.53、0.53、0.66 ms。补齐带来的宽读写抵不上多出来的字节。lavapipe 上 `Aos` 读位置快一成多，CPU 后端上布局没有影响；按硬件 GPU 选。
- 表在 CPU 后端上读写比 std430 慢约 3 倍（要逐字打包，还要经过元素缓存），所以金标准测试的场景要保持很小。
- 常驻表放在设备内存里。帧池把脏元素编码好，GPU 线程把它们写进自己持有的主机端表，再 `uploadTable(table, first, count)` 只传脏区间。传输在设备时间线上排队，与在途帧的读写冲突由 tc 的屏障处理，所以常驻表不需要为每个在途帧各备一份。
- 上传在调用时就把主机端的数据拷走：Vulkan 后端 `memcpy` 进这次提交自己的暂存内存（提交完成之前不复用），再录一条 `vkCmdCopyBuffer`，屏障跟踪跨提交保留；GL 后端用 `glBufferData` / `glBufferSubData`，它们本身就是拷贝；CPU 后端的 kernel 在 `execute()` 时已经算完。所以帧 k 的场景更新可以在帧 k-1 还在设备上时录制，主机端表之后随便改。例外是 `createBuffer` 分配的映射内存，设备原地读写，回读 buffer 就是这种，它们由帧图在帧完成之前一直持有。
- 表变大时换上一张新表，旧表进这一帧的退役列表，帧完成之后才释放。tc 自己也会把设备端的副本推迟到用过它的提交完成之后再销毁（Vulkan 的 graveyard，GL 删除对象时驱动本来就推迟），退役列表让主机端对象的寿命也不依赖这一点。
- 每个内置 kernel 最多用 8 个 storage buffer、4 个存储图像、16 个采样纹理，push constant 不超过 128 字节；8 个 storage buffer 高于规范下限的 4 个，不满足的设备在启动时由 `checkKernel` 报告出来。着色 kernel 正好用满 8 个（实例、顶点、索引、材质、灯光、可见实例、像素列表、阴影贴图），所以可见实例列表带上网格的首索引和首顶点，省掉网格表；阴影级联的矩阵放在阴影贴图 buffer 的开头，省掉一个参数 buffer；模型的像素列表每项是（像素、三角形），着色 kernel 因此不必再读逐像素的三角形 buffer，列表的表头（每个模型的长度和起点）放在同一个 buffer 的开头。透明混合 kernel 的第 8 个是片元链表，片元和它们的表头放在同一个 buffer 里。TAA 的运动 pass 要三个矩阵，超过 128 字节，所以放进一个小 buffer，由 pass 每帧上传。
- 每帧变化的小块数据（阴影级联的矩阵）由 pass 写进帧图 buffer 的主机端副本，再 `uploadBuffer` 一段：帧图 buffer 按在途帧各取一份，上传在调用时拷走，所以不会和在途帧冲突。
- HDR 中间目标用 `RGBA16F`，它在 GLES 3.1 和 Vulkan 上都是强制支持的存储格式，也能线性过滤。

#### 回读

窗口目标的 Readback 路径和离屏目标都不调用 `downloadImage`，因为它总会等待。最后一个 pass 把打包的 RGBA8 直接写进 `ViewContext::output`：渲染线程给每个视图声明的一个 `createReadbackBuffer`，也就是 `createBuffer` 分配的映射内存。它和别的帧图资源一样从资源池里按描述取，在途的每一帧各持有一份，帧收尾之后才回到池里，所以池里自然有 `framesInFlight` 份，不需要再按偏移分段。这个回读由 Renderer 自己登记，不归某个 feature：`whenComplete` 完成之后回到 GPU 线程，窗口目标把这段内存直接交给 Presenter，离屏目标拷进 `ViewResult::image`。

#### 计时

- `FrameResult::wallTime` 总是带从 `submit()` 到这一帧收尾的墙钟时间。
- `RendererConfig::gpuTimers` 打开且 `DeviceInfo::timers` 为真时，`ViewResult::passTimings` 还带每个 pass 的设备耗时。`takeTimings(submission)` 按执行顺序返回这次提交里的每次 dispatch（帧按顺序收尾，失败的帧也取一次再丢掉，所以取到的正好是这一帧的），帧图按同样的顺序记下每次经 `PassContext` 的 dispatch 属于哪个 pass，两者逐个对应。不能按程序名对应，因为同一个 kernel 会出现在多个 pass 里（比如可见性缓冲和阴影的光栅化）；程序名只用来核对：有 pass 绕过 `PassContext` 直接调用设备时，两者对不上，这一帧就不给 pass 耗时。
- 开着计时器时，tc 会在每个计时的 dispatch 前插入完整屏障，dispatch 之间不再重叠，所以 `gpuTimers` 默认关闭，只在开发、剖析和挑选表布局时打开。帧与帧之间的重叠不受影响。
- Adreno 上 GLES 的计时查询几乎总是返回接近 0 的值，又不报告时钟不连续，所以这类设备上只看墙钟时间。llvmpipe 的 OpenGL 和 GLES 也一样。

### present：Presenter 的实现

每种实现是一个单独的 CMake 目标，宿主只链接用得到的。需要特定线程或者可能阻塞的上屏，由 Presenter 自己转交，因为 `present()` 是在 GPU 线程上调用的。

- `SdlPresenter`（Readback，`lr::present_sdl`，只在找到 SDL2 时构建）：`present()` 把像素拷进三缓冲信箱，宿主主线程调用 `blitLatest()` 上屏。SDL 要求在主线程上屏。信箱是三个槽加一个原子交换的“共享槽下标 + 新帧标记”：GPU 线程写自己的槽再换进共享槽，主线程看到新帧标记就换出来，双方都不等对方，主线程总是拿到最新的一帧，中间的帧直接被覆盖。`blitLatest()` 把它传进一张流式纹理（`SDL_PIXELFORMAT_ABGR8888`，就是低字节为红的打包 RGBA8），拉伸到整个输出再 `SDL_RenderPresent`，返回帧号；没有新帧时什么都不画。纹理随 Presenter 销毁，所以最后一个引用要在主线程上、Renderer 之后、`SDL_Renderer` 之前放掉。1080p 的 RGBA8 约 8 MB 一帧，桌面的带宽足够，一直走这条路径。
- `SurfaceControlPresenter`（Android API 29+，SharedBuffer）：引擎直接渲染进一组 RGBA8 的 AHardwareBuffer（通过 `ExternalImageBinding` 写入），这个 Presenter 用 `ASurfaceTransaction_setBuffer` 提交，acquire fence 是 `submitWithFence()` 返回的 `.fence`（`dup` 出来的那一份）。合成器在回调里交还的 release fence，在 GPU 线程下次写这块 buffer 之前交给 `waitFor()`，由设备等待。没有回读，也没有拷贝。Vulkan 第一次以存储图像使用新导入的帧时可能丢弃原有内容，而输出 pass 本来就写满整帧，不受影响。
- `NativeWindowPresenter`（Android API 29 以下，Readback）：在自己的线程上 `ANativeWindow_lock` 并拷贝。
- `ImageSequencePresenter`（Readback，`lr::present_image`）：离线输出。`present()` 拷一份像素，用 `exec::spawn(io, …)` 起一个任务，在 `stackfull::blocking()` 里写 `<directory>/<prefix><帧号>.ppm`，不阻塞 GPU 线程。`flush()` 等到目前交给它的帧都写完（或失败），返回写成和失败的帧数；析构时等还在写的帧，所以宿主要在 Renderer 之后、不在 GPU 线程上放掉最后一个引用。`lrender_cli --sequence <目录>` 用它输出每一帧。

### asset：资产 IO 与解码（可选）

```cpp
struct AssetSource {                    // 同步写法，只在 stackfull 任务里调用
    virtual ~AssetSource() = default;
    virtual AssetResult<Bytes> read(AssetPath const &path) = 0;
};
std::unique_ptr<AssetSource> makeFileSource(FileSourceConfig const &config);      // 内部 stackfull::blocking()
std::unique_ptr<AssetSource> makeAndroidAssetSource(AAssetManager *manager);
std::unique_ptr<AssetSource> makeMemorySource(MemorySourceConfig const &config);  // 测试用

template <class T>
using LoadSender = lexec::any_sender_of<lexec::set_value_t(AssetResult<T>), lexec::set_stopped_t()>;

struct AssetLibrary {
    virtual ~AssetLibrary() = default;
    virtual LoadSender<MeshHandle> loadMesh(AssetPath const &path) = 0;
    virtual LoadSender<ImageHandle> loadImage(AssetPath const &path) = 0;
    virtual void drainChanges(std::vector<AssetChange> &out) = 0;   // 宿主每帧调用，用于热重载
};
std::unique_ptr<AssetLibrary> makeAssetLibrary(AssetLibraryConfig const &config);  // source、io、后台池、poller
```

加载函数运行在 stackfull 任务里：

```cpp
AssetResult<MeshHandle> loadMeshInTask(LoadContext const &context, AssetPath const &path) {
    auto const bytes = context.source->read(path);
    if (not bytes) return tl::make_unexpected(bytes.error());
    if (stackfull::exec::stop_requested()) throw stackfull::exec::stopped_error{};
    return stackfull::exec::await(lexec::schedule(context.background)
                                  | lexec::then([&] { return decodeGltf(*bytes); }));
}
```

- 监听任务把 `AssetChange` 写入 `sync::Channel`；`drainChanges` 在宿主线程上用 `tryRecv` 取出，不会阻塞。
- 资产只产出句柄，把它们变成 `RenderWorldDelta` 里的条目是 scene 或宿主的事。

### scene：世界模型与脚本（可选）

- `World`：组件按 SoA 存储（变换、层级、网格实例、灯光、相机），只在宿主线程上使用，每个组件都带变化标记。层级数量大时，变换传播用 `bulk` 分发到帧池，再 `sync_wait`。
- `World::extractDelta() -> RenderWorldDelta`：只收集上次提取以来变化的部分，这是模拟与渲染之间唯一的交接点。
- `ScriptHost`：拥有 `co2::ManualExecutor` 和一个按模拟时间排序的等待堆。

```cpp
struct ScriptHost {
    virtual ~ScriptHost() = default;
    virtual ScriptId start(ScriptTask script) = 0;
    virtual void requestStop(ScriptId const id) = 0;
    virtual void tick(SimDuration const dt) = 0;   // 推进模拟时间，放行到期的等待和完成的操作，把执行器跑空
    virtual bool isIdle() const = 0;               // 没有脚本还在等操作完成
};
```

脚本的生命周期遵守 co2 的销毁契约：挂起在某个操作上的协程不能销毁。

1. `requestStop` 请求脚本的 co2 stop token。
2. lexec 的 co2 桥把这个 token 放进被等待 sender 的环境里，sender 以 `set_stopped` 完成。
3. 完成照样经过 `continues_on(sim)`，所以下一次 tick 时，脚本在宿主线程上收到 `stopped_error`，结束，帧被销毁。
4. 不响应停止的操作会让脚本停在“停止中”，直到操作完成。实体先被销毁也没关系，脚本之后通过带代数的 `EntityId` 访问时能发现实体已经失效。
5. 关闭时对所有脚本 `requestStop`，然后反复 tick 直到 `isIdle()`。

`ScriptHostConfig::release` 有两种取值：

- `AsSoonAsReady`（默认）：操作一完成，就在下一次 tick 放行。
- `Deterministic`：tick t 发起的操作恰好在 tick t + k 放行，没完成就在那次 tick 里等它。这样，资产在 IO 上快一点还是慢一点，都不会改变脚本的时间线，测试和重放才真正可重复。

脚本示例：

```cpp
auto orbitCamera(World &world, AssetLibrary &assets, EntityId const camera)
    CO2_BEG(ScriptTask, (world, assets, camera), AssetResult<MeshHandle> mesh; float angle{};) {
    CO2_AWAIT_SET(mesh, assets.loadMesh("models/bunny.glb"));   // 解码在别处，恢复时已在宿主线程
    if (mesh) world.spawnInstance(*mesh, Transform::identity());
    for (angle = 0.0f;; angle += 0.01f) {
        world.setTransform(camera, orbitAround(angle));
        CO2_AWAIT(nextFrame());
    }
}
CO2_END
```

### examples：组合根

`lrender_view`（SDL 窗口与输入，加上 scene、asset、Renderer；scene 和 asset 还没有，目前显示演示场景；TAA 默认打开，T 键开关）和 `lrender_cli`（离线，按指定的帧数或采样数输出图片，也是金标准测试的入口；`--taa` 打开 TAA，默认关）。`lrender_view` 的主循环从不等帧：帧的 sender 用 `lexec::spawn` 放进一个 `counting_scope`，完成时给一个原子计数加一，循环只在未完成的帧少于 `framesInFlight` 时提交新帧，每一轮都调 `blitLatest()`。宿主循环也可以写成下面这样，阻塞与否完全由宿主决定：

```cpp
for (;;) {
    if (not pumpSdlEvents(input)) break;
    assets->drainChanges(changes);
    world->applyAssetChanges(changes);
    scripts->tick(clock.step());
    auto frame = FrameDesc{};
    frame.delta = world->extractDelta();
    frame.views.push_back(mainView(*world, window));
    pending.push_back(lexec::spawn_future(renderer->submit(std::move(frame)), scope.get_token()));
    if (pending.size() > kFramesInFlight) {
        lexec::sync_wait(std::move(pending.front()));
        pending.pop_front();
    }
    sdlPresenter->blitLatest();
}
```

## 一帧

```
宿主线程 │ 输入 │ tick │ extractDelta │ submit N │ ……（不等引擎）
渲染线程 │ 应用增量 N │ 编译帧图 N │ 交给 GPU 线程 │ 应用增量 N+1 │ ……
GPU 线程 │ 录制 N │ 提交 N │ 录制 N+1 │ 提交 N+1 │（N+2 排队）│ 收尾 N │ 录制 N+2 │ ……
设备     │          ████████ N ████████ ████████ N+1 ████████
IO       │          等 N 完成 ────────────────────────┘
```

- `submit()` 在调用线程上取下目标和管线的当前状态，然后把这一帧的工作用 `lexec::spawn` 放进 Renderer 的 `counting_scope`：`schedule(render) | then(prepare) | let_value(record(accept))`。这条链不管宿主有没有启动 `FrameSender` 都会跑完，所以丢掉 sender 也不会让某一帧的增量只到了一半。渲染线程从不阻塞。
- `FrameSender` 是 `exec::async_get(completion) | let_error(…)`，等的是这一帧自己的 `sync::Completion<FrameResult>`；`let_error` 把 `error_code` 和 `exception_ptr` 都变成 `set_error(std::exception_ptr)`。sender 在 GPU 线程上完成（启动时结果已经在了就当场完成），宿主要等别的东西之前先 `continues_on` 离开。用 `Completion` 做桥，是因为结果由 GPU 线程的状态机在帧收尾时给出，不是启动这一帧的那条链的延续：需要一个任意线程可写、单个消费者、结果先到或 sender 先启动都行、还能响应停止请求的一次性交接点，这正是 `Completion` 加 `async_get`；自己写一个 lexec sender 也要重做同样的原子状态和取消协议。
- GPU 线程上是 `FrameQueue`，一个只在这个线程上运行的状态机，不用锁。准备好的帧先进等待队列；在途（录制了、还没收尾）的帧少于 `framesInFlight` 时取出来录制并提交，然后 `lexec::spawn(whenComplete(…) | … | let_value(record(deviceDone)))`，在 IO 调度器上等设备做完再回到 GPU 线程。设备完成的通知可能乱序到达，收尾总是从在途队列的头上按顺序做：取计时（打开时）、调回读（交给 Presenter 或拷进 `ViewResult`）和 `onFinish`、释放这一帧的帧图资源和退役列表，然后再录制等着的帧。所有收尾完的帧的 `Completion` 在这一步的最后才设置，宿主的接收者即使在 GPU 线程上同步运行，也看到一个已经整理好的队列。
- 渲染线程准备失败的帧（增量无效、帧图建不起来）也走同一个队列，按顺序以 `set_error` 完成；视图的帧图失败时，场景更新照样先在 GPU 线程上应用，两份场景保持一致。录制中途抛出的非设备错误，已经录下的工作照样提交，这一帧的资源留到设备做完再释放，然后以那个错误完成。
- 准备侧状态只在渲染线程上修改，设备侧状态只在 GPU 线程上修改，两个 `run_loop` 都是先进先出的，帧天然按序。

## 数据与状态

| 状态 | 所有者 | 时间线 | 一致性 |
|---|---|---|---|
| 场景组件 | `World` | 宿主线程 | 单线程，无锁 |
| 脚本帧 | `ScriptHost` | 宿主线程 | 只在 tick 内恢复 |
| `RenderWorldDelta` | 按帧移动的值 | 宿主 → 渲染线程 | 不可变 |
| `MeshData` / `ImageData` | `shared_ptr<T const>` | 任意线程只读 | 不可变；热重载产生新对象 |
| `RenderWorld` 镜像、BLAS 缓存、帧图资源池 | render | 渲染线程 | 单一序列 |
| 常驻表、纹理池、临时资源、历史图像 | `FrameQueue` | GPU 线程 | 单一序列；丢掉的进当前帧的退役列表，帧完成之后释放 |
| 回读 buffer、暂存段 | 帧图资源池 | GPU 线程写暂存、设备写回读 | 每个在途帧各持有一份，帧收尾之前不回池 |
| `FrameResult`（统计、回读图像） | 按帧移动的值 | GPU 线程 → 宿主 | 通过 sender 交接 |

## 跨模块约定

- 错误：
  - 可恢复的情况（资产缺失、解码失败）作为值返回（`AssetResult<T>`），走 `set_value`。
  - 设备错误（`tc::DeviceError`）是异常，可能出现在录制、提交、等待完成或收尾时。GPU 线程收到后让出错的帧和所有在途帧以这个错误完成，先放掉旧设备上建的一切（帧图资源、资源池、历史图像、`DeviceScene`、退役列表），再 `recreateDevice`，最后把设备代数加一。等待完成失败也按设备丢失处理，因为那时不知道设备还用不用这一帧的资源。
  - 每一帧在渲染线程上准备时记下当时的设备代数。代数已经过时的帧（设备重建之前准备好的）不再录制，同样以那个错误完成；渲染线程发现代数变了，下一帧用 `RenderWorld::fullUpdate()` 从网格句柄重新编码整个场景（几何、各个槽、要追踪光线时重建 BVH），并让所有历史资源从头开始。这一帧之后的帧照常渲染。纹理的像素不在渲染线程上，没法重新上传：这一帧把放置过的纹理记为丢失，在 `FrameResult::lostTextures` 里列出，宿主重新 upsert 之前材质只用因子；宿主也可以选择换成 `cpu` 后端。新设备建不出来时，下一帧再试一次，还不行就以那个原因失败。
  - 测试里用一个录制时抛 `tc::DeviceError` 的 feature 模拟设备丢失。
  - 契约违规交给进程级处理器；co2 和 stackfull 都有这样的处理器，lrender 安装一个统一的。
- 开启异常编译：co2、lexec 的 co2 桥和 tc 的错误报告都依赖异常。
- 风格与 co2、stackfull、tc 一致：`struct`、east const、接口加工厂、数据类、camelCase；lexec 的 snake_case 只出现在调用处。C++17 没有指定初始化器，所以配置类型用默认成员初始化，再逐个字段赋值。

## 依赖库的边界

下面这些能力超出了四个库各自的职责，不向库提需求，由引擎自己负责：

| 能力 | 为什么不归依赖库 | 引擎的做法 |
|---|---|---|
| 交换链、窗口上屏、vsync | tc 是无窗口的 compute 框架 | present 模块；Android 上用 `ASurfaceControl` 零拷贝上屏 |
| 无绑定资源 | tc 以 GLES 3.1 为底线，它没有 descriptor indexing | 按尺寸档分组的纹理数组 |
| 任务优先级 | std::execution 没有这个概念 | 帧池与后台池分开 |
| 光栅化管线、硬件光线追踪 | 超出 compute | 计算光栅化、软件 BVH |
| 无异常构建 | 会改变 co2 和 tc 的错误模型 | 引擎要求开启异常 |
| Windows、macOS | 超出 stackfull 声明的目标平台 | 引擎只支持 Linux 与 Android |
| CPU 后端的 kernel 热重载 | CPU 后端运行的是编译好的 C++ | 重新加载只在 GPU 后端提供 |
| 等待 sync file | tc 只提供阻塞的 `wait()` 和非阻塞的 `isSignaled()`，与事件循环集成不是它的事 | gpu 模块在 stackfull 的 Poller 上等待 |

## 放弃的方案

- **全部跑在 stackfull 上（计算、IO、脚本共用 M:N 调度器）。** 它的 lexec 调度器没有对 `bulk` 做并行定制，`bulk` 会在一个任务里串行执行；长计算会占住 worker；宿主无法在帧的固定阶段驱动它，而脚本恰恰需要这一点；每个脚本还要一个独立的栈。
- **全部用 co2。** 无栈协程不能在第三方代码内部挂起，加载器就写不成同步风格。
- **加载器写成 sender 管线（`let_value` 链）。** 读文件、解析、再按引用去读 buffer 和图片、解码，是一个带分支的顺序流程，第三方解码器又都是同步代码。在有栈任务里按同步写法写最直接。
- **在 tc 之外接一套 Vulkan 图形管线。** 那样会失去单一源码、CPU 调试和跨后端等价测试，而这正是这个引擎最有价值的地方。代价是用不上硬件光栅化和 RT core，性能比传统引擎低，这个项目接受这个代价。
- **每帧提取完整快照，或者渲染线程加锁访问共享场景。** 前者每帧的拷贝量随场景规模增长，后者让模拟和渲染互相等待。改为增量加渲染侧常驻镜像。
- **让宿主线程 `sync_wait` 准备工作来保证串行。** 库会因此阻塞宿主，串行也要靠调用方遵守前置条件。改为渲染线程上的 `run_loop`。
- **一个计算池加任务优先级。** lexec 不提供优先级，自己实现一个带优先级的池又会和 `bulk` 的池定制脱节。两个池最简单。
- **GPU 线程自己等 GPU 完成（`Submission::wait()`，或者轮询 `isComplete()`）。** 前者让 GPU 线程在设备执行期间无法录制下一帧；后者需要定时唤醒，而 `run_loop` 没有定时器。等待交给 IO 调度器，GPU 线程只在完成之后被调度回来收尾。

## 工程结构

```
lrender/
  CMakeLists.txt  CMakePresets.json
  cmake/LrDependencies.cmake FetchContent，固定提交：lexec、co2、stackfull、compute_shaders
                             （以后还有 cgltf、stb、tl-expected）；doctest 在 tests/ 里拉取；
                             SDL2 用系统里的（find_package(SDL2 CONFIG)），找不到就不构建
                             lr::present_sdl 和 lrender_view
  core/      include/lr/core/                     lr::core     INTERFACE
  runtime/   include/lr/runtime/   src/           lr::runtime  lexec::runtime、stackfull
  gpu/       include/lr/gpu/       src/           lr::gpu      tc、tc_devices、stackfull::exec、stackfull::runtime
  render/    include/lr/render/    src/  kernels/ lr::render   tc_add_kernels(...)；include/lr/render/kernels/ 是用户 kernel 也要用的公开 kernel 头文件
  present/   image_sequence/  sdl/  （android/）  lr::present_image  lr::present_sdl  （lr::present_android）
  asset/     include/lr/asset/     src/           lr::asset    stackfull::exec、cgltf、stb
  scene/     include/lr/scene/     src/           lr::scene    co2、lexec 的 co2 桥
  examples/  view/  cli/                          lrender_view  lrender_cli
  probes/    gpu_probe  present_probe  asset_probe  layout_probe
  tests/
```

- 命名空间为 `lr::<module>`，每个模块一个 CMake 目标，目标图与模块图一致。
- 作为库交付：提供 install 规则和 `find_package(lrender)` 用的包配置，每个可选层单独导出为一个组件。
- compute_shaders 至少要固定到 `915657a`：按提交取计时、只在设备上的图像、压缩纹理数组和 `FencedSubmission` 从这个提交开始才有。
- 平台取各库支持范围的交集：Linux x86_64/aarch64 和 Android arm64。
- 主机构建依赖 libclang-18-dev 和 glslang-tools（tc 转译器）；Android 交叉编译时使用主机构建出的转译器。发布构建加 `-DTC_PROGRAM_RELOAD=OFF`。

## 验证与里程碑

| 模块 | 类型 | 验证方式 |
|---|---|---|
| core、RenderWorld、帧图编译 | 确定 | 单元测试 |
| render kernel | 确定 | CPU 后端的金标准图像，加上跨后端等价测试；子组和 64 位原子的变体与回退变体在同一个设备上互相比对（三角形、深度、可见实例列表、图像逐位相同）；自定义材质模型只改变自己的像素；TAA 与主机上的模拟和解析覆盖率比对，并测拖影、运动矢量和历史的重置；设备图像没有读回接口，纹理池的内容由着色 kernel 采样（自发光纹理贴满一个四边形，像素中心对准纹素中心），再用读 HDR 颜色的 feature 读进 buffer 比对；阴影、透明、环境光的镜面部分、法线贴图与路径追踪器或解析结果比对 |
| Renderer 整体 | 确定 | cpu 后端，以及 llvmpipe、lavapipe 上的 GPU 后端（没有 sync file，走 `Submission` 的等待路径）：连续提交的帧按序完成、累积结果与逐帧提交相同、在途帧不超过上限（用一个数录制而未收尾帧数的 feature）、多视图、窗口目标交给测试用的 Presenter、缩放、设备丢失恢复（用一个录制时抛 `tc::DeviceError` 的 feature）；`ImageSequencePresenter` 写出的文件与离屏目标的图像逐像素相同 |
| scene、ScriptHost | 确定 | 手动 tick 的单元测试；`Deterministic` 模式下的时间线逐帧比对 |
| asset | 半确定 | 用内存源做单元测试；`asset_probe` 读真实文件、测热重载与取消 |
| gpu | 边界 | 单元测试用 eventfd 代替 sync file、用自定义的 `Submission::Completion` 代替还没完成的提交，分别覆盖两条等待路径；`gpu_probe`：选择后端、打印 `checkKernel` 的结果，在真实设备上提交并走 sync file 路径，打开计时器时按提交打印每次 dispatch 的耗时 |
| 表布局 | 边界 | `layout_probe`：在选定的后端和设备上，给剔除和顶点读取的访问模式按各候选布局计时 |
| present | 边界 | `SdlPresenter` 在 SDL 的 offscreen（或 dummy）驱动和软件渲染器上测，读回像素比对；`present_probe`：每种 Presenter 显示测试图案 |

1. **M0** 工程骨架：CMake、presets、固定版本的依赖、install 与包配置、CI（llvmpipe 与 lavapipe 上跑 GPU 后端，覆盖没有 sync file 的等待路径）。
2. **M1** 渲染核心：`RenderWorld` 增量、帧图、可见性缓冲管线的最小版本（GPU 剔除与间接 dispatch、两遍可见性缓冲、基础 PBR 着色、色调映射），在 cpu 后端上配合 `lrender_cli` 建立金标准测试。
3. **M2** 参考路径追踪，作为画质参照；Vulkan、OpenGL、GLES 与 CPU 的等价测试；打开设备计时器，定下每张表的布局和编码。
4. **M3** 完整的 Renderer 接口：渲染线程、GPU 线程、异步提交与 IO 上的完成等待、回读 buffer、在途帧、多视图、缩放、设备丢失恢复、SDL 窗口目标与图片序列；`lrender_view`。`gpu_probe` 和 `present_probe` 还没有写。
5. **M4** 画质。M4a：设备端纹理池（mip、流式写入、压缩格式与解码回退、设备丢失后列出丢失的纹理）、带纹理的材质（法线贴图、UV 变换、alpha 遮罩）、级联阴影、逐像素链表的顺序无关透明、均匀环境光的镜面部分；路径追踪器同样采样纹理，混合材质按随机不透明处理。M4b：启动时的 kernel 检查与变体选择、一遍写入的可见性缓冲和子组变体、材质分类与间接着色（应用可以加自己的材质模型）、TAA。
6. **M5** asset：流式加载、取消、热重载、KTX2 与按设备选择压缩格式，以及开发期的 kernel 重新加载。
7. **M6** scene 与脚本：生命周期、确定性模式。
8. **M7** Android：`SurfaceControlPresenter` 零拷贝上屏、API 29 以下的回读上屏、生命周期。
