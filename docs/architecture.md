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

- 资产加载器是按同步写法写的普通函数，运行在 stackfull 任务里。读文件交给 `stackfull::blocking()`，因为普通文件不支持 epoll。先把文件整个读进来，再用 `exec::await(schedule(background) | then(decode))` 转到后台池解码，做完再回来；任务 park 期间，worker 去跑别的加载任务。
- cgltf 和 stb_image 都从内存解码，没有用它们读文件的回调。需要边读边解时，可以让解码器直接在任务里运行，回调里读文件会让任务 park；挂起点在第三方代码的调用栈深处，这些代码也不需要改造，无栈协程和 sender 都做不到这一点。
- 加载任务在步骤之间检查 `exec::stop_requested()`，被取消的加载以 `set_stopped` 结束，脚本的销毁依赖这一点。
- 等待 GPU 完成：tc 导出的 `SyncFence` 是一个可 poll 的 sync file，用 `io::Registration` 注册进 Poller，由一个任务等它变成可读；详见 gpu 模块。
- 文件监听：inotify fd 同样注册进 Poller，由一个任务用 `exec::wait(exec::async_read(…))` 循环读取事件，不额外占用线程。资产热重载和 `lrender_view` 的 kernel 重新加载各用一个 `FileWatcher`。
- 同一资产的并发加载会去重。`sync::Completion` 只有一个消费者（只记一个 `Waker`，`get()` 把结果移走），所以在途表里每个请求方有自己的 `Completion`，加载结束时每个都 `set` 一份结果；请求方的 sender 是 `exec::async_get(自己的 Completion)`，不占任务。
- 加载任务是 `exec::spawn(io, fn)`，由 `AssetLibrary` 用 `lexec::spawn` 放进自己的 scope，它的停止令牌来自这次加载，而不是某个请求方；见 asset 模块。

### co2：逐帧推进的逻辑

- 脚本返回 `ScriptTask`，这是按 co2 promise 协议写的自定义返回类型，用 `CO2_*` 宏编写。脚本只会在宿主调用 `ScriptHost::tick()` 时、由其中的 `co2::ManualExecutor::run()` 恢复，所以可以无锁读写场景。
- `ScriptTask` 的 promise 用 `await_transform` 把每个被等待的 sender 包成 `continues_on(sender, 闸门)`，再交给 `lexec::coro::as_awaitable`。闸门是每次等待各有一个的 co2 `Scheduler`，经 `lexec::coro::scheduler` 当作 lexec 调度器：sender 在哪个线程上完成都行，`continues_on` 把结果存进操作状态，把续体交给闸门，闸门只把它记在 ScriptHost 那里，等某次 tick 放行时才排进 `ManualExecutor`。例如 `CO2_AWAIT_SET(mesh, assets.loadMesh(path))` 完成在 IO 线程上，但脚本一定在之后的某次 tick 里回到宿主线程继续执行。“脚本在别的线程上访问场景”这种状态，在类型层面就构造不出来；不是 sender 的 awaitable（`co2::Task`、`scheduleOn` 之类）在脚本里编译不过。
- 每个脚本只是一个堆上的帧（通常几百字节），不需要独立的栈，上万个实体脚本也负担得起。
- co2 v2 还没有定时器和 `whenAll`：模拟时间上的等待（`nextFrame()`、`waitFor(duration)`）由 ScriptHost 提供，组合则交给 lexec。
- 宏 DSL 的限制写脚本时会碰到：跨挂起点的局部要写在 `CO2_BEG` 里；被等待的表达式里不能有 lambda，要先包成返回 sender 的函数。

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
| 帧池 `lexec::static_thread_pool` | runtime | 默认为硬件线程数 − 3 | 网格的编码和 BVH、变换传播、tc CPU 后端 |
| 后台池 `lexec::static_thread_pool` | runtime | 默认 1 | 解码、纹理的转换 |
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
scene    → runtime, core           以及 co2 和 lexec 的 co2 桥
runtime  → 只依赖 lexec 与 stackfull
```

库由 7 个模块组成：core 和 runtime 是基础；render 是核心，gpu 是它面对驱动和线程的那一半；present、asset、scene 是可选层；examples 里是组合根。scene、render、asset 互不依赖，只通过 core 里的数据词汇交流。

### core：值类型与数据词汇

只有数据，没有行为，作为一个 INTERFACE 目标。

- 数学类型直接使用 tc 的 `vec`/`mat`，因为 kernel 只认得它们。`Transform`、`Aabb`、四元数也建立在它们之上，并且没有基类、运算符都按值返回。tc 转译器会把用户类型的运算符和函数模板转译成 GLSL，所以这些类型在 kernel 里同样能用。
- 强类型 ID：`MeshId`、`TextureId`、`MaterialId`、`InstanceId`、`LightId`、`ViewId`、`RenderTargetId`，以及 scene 用的、带代数的 `EntityId { index, generation }`。
- 数据词汇：`MeshData`、`ImageData`、`MaterialDesc`、`LightDesc`、`CameraDesc`、`RenderWorldDelta`。资产以 `MeshHandle`（即 `shared_ptr<MeshData const>`）和 `ImageHandle` 的形式流转。
- 可恢复错误的结果用 `tl::expected`（asset 的 `AssetResult<T>`），core 自己不依赖它。

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
- **同一个 fence 有两个使用者**：SharedBuffer 路径上 Presenter 也要这个 fence（桌面的 `VulkanPresenter` 让自己的队列等它，Android 的 `ASurfaceTransaction_setBuffer` 交给合成器），此时 `dup` 一份 fd 给 Presenter。
- **计时**：完成之后回到 GPU 线程，用 `takeTimings(submitted.submission)` 取这一帧的设备耗时。这次提交已经完成，调用不会等待，也不会碰正在录制或执行的下一帧。
- **开发期重新加载**：`Renderer::reloadKernels(dir)` 通过 `record` 在 GPU 线程上调用 `ProgramRegistry::instance().reloadFrom(dir)`，在它之前和之后提交的帧之间生效，以 tc 的报告完成（换掉的程序，被拒绝的程序和原因）；GPU 设备下一次 `useKernel` 就用新程序，接口变了的程序被拒绝，CPU 后端照旧运行编译进来的 C++。`builtInKernelDirectory()` 是 CMake 写进来的 lrender 自己的 `tc_generated/lr_render/glsl`，只在打开 `LRENDER_KERNEL_RELOAD`（也就是 `TC_PROGRAM_RELOAD`）的构建里有；发布预设关掉它，`reloadFrom` 不读磁盘，拒绝一切。`lrender_view` 在 GPU 后端上用 asset 的 `FileWatcher` 监听这个目录，变化停下 300 ms 之后重新加载一次，因为构建会给每个 kernel 写好几个文件。转译器不删掉改名之前的 kernel 留下的文件，`reloadFrom` 找不到对应的程序，就跳过它们。

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
    // 开发期：在 GPU 线程上从 tc_add_kernels 的 glsl 目录换上新程序，以 tc 的报告完成，见 gpu 的“开发期重新加载”。
    virtual KernelReloadSender reloadKernels(std::string directory) = 0;
};
std::unique_ptr<Renderer> makeRenderer(RendererConfig const &config);   // 后端、Runtime、framesInFlight（默认 2）、变体开关
std::optional<std::string> builtInKernelDirectory();                    // lrender 自己的 kernel 目录，只在开发构建里有

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

struct Presenter {                  // lr::present 提供实现，宿主也可以自己写；调用都在 GPU 线程上、按帧序，不得阻塞
    virtual ~Presenter() = default;
    virtual void present(PresentFrame const &frame) = 0;                   // Readback 路径
    // SharedBuffer 路径，默认不用
    virtual bool sharesBuffersWith(tc::DeviceInfo const &device) const;    // 建目标时问一次
    virtual SharedBufferSlot acquireShared(core::ivec2 extent);            // {slot, generation, release fence}
    virtual tc::ExternalBufferMemory exportShared(std::uint32_t slot);     // 每次一个新的 fd
    virtual void presentShared(SharedBufferFrame frame);                   // 提交之后立即交出，带 acquire fence
    virtual void discardShared(std::uint32_t slot);                        // 这一帧失败了，槽原样还回去
};
```

- 窗口目标的像素交给它的 Presenter，`ViewResult::image` 留空；离屏目标的像素拷进 `ViewResult::image`。同一帧可以两种都有。
- 两条路径。Readback：最后一个 pass 写进主机映射的内存，设备做完之后 `present()` 拿到像素的指针。SharedBuffer：最后一个 pass 直接写进 Presenter 导出的、同一块 GPU 上的内存，帧一提交就连同它的 fence 交给 `presentShared()`，没有回读。建窗口目标时，Presenter 的 `sharesBuffersWith()` 说可以、设备又能导出 sync file 和导入内存（`syncFiles`、`importsMemory`），这个目标就走 SharedBuffer，否则走 Readback；SharedBuffer 的几个入口都有默认实现，只做 Readback 的 Presenter 不用改。
- SharedBuffer 的一帧：渲染线程把视图的输出声明成帧图的外部 buffer（见“帧图与扩展”）。GPU 线程录制这个视图之前向 Presenter 要一个槽：没导入过的槽（或者 `generation` 变了，或者尺寸变了）用 `exportShared()` 的 fd 经 `importBuffer` 导入，导入的 buffer 按目标和槽留着，60 帧没用就放掉；槽带的 release fence 交给 `waitFor`，设备等 Presenter 上一次读完这个槽再写。帧提交之后，把 `submitWithFence()` 的 fence `dup` 一份连同槽交给 `presentShared()`；原来那份照旧给 IO 调度器等帧完成。帧失败了（视图录制时抛了异常，或者设备丢了）就 `discardShared()`，什么也不显示。设备丢失时导入的 buffer 一起丢掉，下一次重新导入。
- 多视图：一帧可以有任意个视图，渲染到不同目标，比如编辑器的多个视口、分屏、离屏缩略图。阴影贴图是管线内部的 pass，不算视图。
- 缩放：帧图里的资源尺寸相对目标尺寸声明，`resize` 之后按新尺寸重建；历史资源按尺寸记录，尺寸变了就换一对新的，`valid` 为假，累积从头开始。
- 销毁：已经提交的帧照样交给原来的 Presenter（帧图里持有它的 `shared_ptr`），之后再提交这个目标就抛 `std::invalid_argument`。
- Android 生命周期：surface 销毁时宿主 `destroy` 对应的目标，重建时再 `createWindow`。tc 的设备不依赖窗口，所以这期间设备和常驻资源都留着。
- 在途帧：录制了、但还没收尾的帧最多 `framesInFlight` 个，见“一帧”。宿主通过等待 `FrameSender` 自己决定提交节奏；超过上限的帧准备好之后在 GPU 线程上排队，不会被丢掉。

#### RenderWorld：渲染侧的常驻镜像

渲染线程上的一份场景镜像，按种类以 SoA 存放，槽位稳定。每帧只应用 `RenderWorldDelta`，并记下每张表的脏区间；编码只处理脏的部分。一个增量里的网格，每个的编码（顶点和包围盒）在帧池上各是一项并行完成，槽位和顶点、索引的区间仍按增量里的顺序在渲染线程上分，所以结果与逐个编码逐位相同；实例、材质和灯光的编码很小，在渲染线程上做。每个实例还记着上一帧的世界矩阵（`GpuInstance::previousObjectToWorld`），用来算 TAA 的运动矢量：增量移动了实例，这一帧写的是移动前的矩阵，下一帧再写一次这个实例，把它改成当前的矩阵；新实例两者相同。所以每帧必须恰好应用一次增量。

镜像持有网格的不可变句柄，设备丢失之后可以重新上传几何；纹理的像素数据不保留（见“纹理”）。

几何表按区间分配。每个网格的顶点、索引、簇，以及保留层次结构时它的 BLAS 节点和三角形，各占所在表的一个连续区间（`RangeAllocator`：放进放得下的最小空闲区间，同样大的取最低的，都放不下才接在末尾；归还的区间与相邻的空闲区间合并，挨着末尾的直接缩短表）。网格被替换或删除时归还它的区间，同一帧里的新网格就能用上：上传在 GPU 的命令流里排在之前的帧之后，还在飞行的帧读到的仍是旧内容。同一个增量里重复的网格以最后一个为准。某张表末尾之前的空闲元素多于占用的元素时，下一次 `apply()` 先把所有网格按现在的先后在各自的表里紧挨着排到表头（`SceneUpdate::moves`）。移动都往低处，按位置顺序逐个元素地拷，不会覆盖还没移动的元素；GPU 线程在表的主机副本里搬，再上传搬到的区间。索引相对于网格的第一个顶点，簇相对于网格，所以搬几何只是拷贝；层次结构里记的是绝对位置，搬的时候节点的子节点、叶子的三角形和三角形的三个角各按自己的移动量修正，不必重建（ray query 的底层结构对动过的网格重建）。设备上的表比需要的大四倍以上时缩到需要的大小。随机增删、替换网格 1000 轮，每轮之后在主机模型和 CPU 设备的 `DeviceScene` 里检查，每个网格的顶点、索引、簇和层次结构都在它的 `GpuMesh` 说的地方：同时最多 8101 个顶点，表最大 16384 个，整理了 24 次。另一个测试在四个后端上让大网格来来去去 26 帧，之后可见性缓冲管线和路径追踪（软件遍历和 ray query）的图像与只见过最终场景的渲染器逐位相同，几何表从高峰的 832 KiB 缩到 42 KiB。

簇：编码网格的同一项里，网格的三角形按索引顺序切成簇，每个簇取接下来的三角形，直到超过 124 个三角形或 64 个不同的顶点。簇只是三角形的连续区间，所以三角形的编号不变，可见性缓冲、路径追踪的层次结构和 ray query 的底层结构都不受影响。每个簇（`GpuCluster`，16 个字）在网格空间里有包围盒，还有给背面剔除用的包围球和法线锥（meshoptimizer 的判据：观察点 p 满足 `dot(center - p, axis) >= cutoff * |center - p| + radius` 时，簇里每个三角形都背对它；法线张开太大时 cutoff 取 1，永远不剔除）。`GpuMesh` 记着自己的第一个簇和簇数，簇表和顶点、索引一样按区间分配，网格的簇也留在镜像里，供设备丢失后的全量更新。簇的质量取决于网格的索引顺序：多数网格按顶点缓存排过序，在空间上是局部的；顺序乱的网格，簇的包围盒会偏大，剔除效果变差但结果仍然正确。Sponza 的 26 万个三角形是 0.3 MiB 的簇表，ABeautifulGame 的 150 万个是 1.0 MiB。

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
- 帧图在渲染线程上编译：检查读写关系，剔除没有被用到的 pass，算出每个资源的生命周期（第一个和最后一个用到它的、会运行的 pass；回读 buffer 活到帧结束），维护按 `ViewId` 保存的历史资源。没有会运行的 pass 用到的资源不分配。
- 内存别名：GPU 线程录制一个视图的帧图时，把只在设备上的 buffer 和图像（设备能放置图像时，`DeviceInfo::placedImages`；GL 不能，图像各占一份）放进一个 tc 的堆，生命周期重叠的资源字节不重叠，不重叠的共用同一段内存。放法是从大到小，每个放在它的对齐允许、又不和同时活着的资源重叠的最低偏移上，堆的大小就是同一时刻最多要的那么多。堆和放好的资源按“放法”（每个资源的类型、大小、偏移）在帧之间复用，帧图不变时每一帧拿到同一个堆和同一批资源，每个在途帧一份；60 帧没用到的放法和单独的资源都释放。资源在它的第一个 pass 开始时内容未定义，这本来就是 `createBuffer` 的约定；tc 在别名切换处插屏障，Vulkan 的图像从 UNDEFINED 转过去，CPU 后端用 0xCD 填满接手的内存，所以哪个 pass 读了自己没写的东西，CPU 后端上的测试会看出来。上传和回读 buffer 有主机端的内存，不进堆。`RendererConfig::aliasGraphResources` 关掉它，`ViewStats::graphBytes` 和 `unaliasedBytes` 给出共用和不共用时各要多少设备内存。1920×1080、开 TAA 的演示场景，一个视图从 173.6 MiB 降到 109.0 MiB，在途两帧共省约 130 MiB。
- `ViewDesc::describeGraph` 让 `ViewResult::graph` 带上这一帧的帧图，用 Graphviz 的 DOT 语言：会运行的 pass 按顺序（开了计时器时带上 dispatch 数和设备耗时），它们读写的资源（大小、生命周期、在堆里的偏移，或者是历史资源），以及读写的边。`lrender_cli --graph <文件>` 写出最后一帧的。
- 外部 buffer：`GraphBuilder::createExternalBuffer` 声明一个由 GPU 线程在录制前交给 `GraphRun` 的 buffer，不从资源池里取，也不还回去；它算作帧图的输出，写它的 pass 总会运行。SharedBuffer 路径的视图输出就是它。这是 Renderer 内部用的，不在 `FrameGraphBuilder` 的公开接口上。
- 帧图 buffer 按除了设备还有谁碰它分三种（`BufferAccess`）：`createBuffer` 是只在设备上的 `tc::DeviceBuffer`，没有主机端副本，只能由 kernel 和 `copyBuffer` 读写，帧开始时内容未定义；`createUploadBuffer` 带主机端副本，pass 录制时写好再 `uploadBuffer`；`createReadbackBuffer` 是 `createBuffer(n, tc::HostAccess::Reads)` 分配的映射内存，在主机缓存的系统内存里，设备原地写，帧完成之后由主机读（RTX 5070 Ti 上主机读 8 MB 不到 1 ms；GPU 经 PCIe 写它，1080p 的最后一个 pass 要 0.74 ms）。中间结果都是第一种，GPU 后端不为它们分配主机内存。三种描述不同，资源池不会把一种当成另一种复用。
- pass 按编译后的顺序在 GPU 线程上录制。屏障由 tc 按实际的读写冲突插入，互不相关的 pass 可以在设备上重叠，帧图不需要自己计算屏障。
- `PassContext` 把帧图的资源句柄解析成 tc 资源并绑定，`dispatch` 和 `dispatchIndirect` 分别对应 `execute` 和 `executeIndirect`（可以指定读参数 buffer 的第几项），同时记下这次 dispatch 属于哪个 pass；用户 kernel 和内置 kernel 走同一条路径。
- 历史资源：`graph.history<G>(name, extent)` 返回一对图像 `{previous, current, valid}`，同一个 `ViewId` 和名字在每个用到它的帧里拿到同一对，并且互换一次：上一帧写进 `current` 的内容，这一帧从 `previous` 读。互换而不是原地读写，是因为 GLES 3.1 不允许 kernel 同时读写 r32 以外格式的存储图像。两张图像在帧开始时都算已写入，写 `current` 的 pass 总会保留。第一次用到、尺寸或格式变了、或者被释放过之后，`valid` 为假。记录在渲染线程上（只是 id），图像在 GPU 线程上按 id 创建；连续 60 帧没用到、或者视图的目标已经销毁，就释放。TAA 也用它。
- feature 用 `requirements()`（`FeatureRequirements`）声明额外的需求：要追踪光线的 feature 让 `RenderWorld` 从这一帧起维护 BVH（见“参考路径追踪”）；要抖动投影的 feature（TAA）让设置了 `AntiAliasing::Taa` 的视图每帧换一个子像素偏移，`ViewContext::temporal` 里有偏移、不带偏移的这一帧和上一帧的视图投影矩阵，以及历史能不能接着用。`graph.onFinish(fn)` 在帧完成后填写 `ViewResult`，比如累积的采样数。

#### 两条内置管线

- **可见性缓冲管线（实时主管线）**：可见性、阴影、着色、透明、时间抗锯齿、色调映射六个 feature。
  1. GPU 实例剔除：写出可见实例列表（实例、它的第一个可见三角形、网格的第一个索引和第一个顶点，每项四个字，后面的 pass 因此不必再读网格表），以及后续光栅化的 `DispatchIndirectCommand`。材质是 `Blend` 的实例不进这张列表，留给透明 pass。
     剔除分两级。实例级（`CullInstances`）对网格的包围盒做视锥测试，可见实例按实例顺序压缩成一张表，表项记着它的簇在“候选簇”里从哪里开始。簇级（`CullClusters`，每个候选簇一个调用，间接 dispatch）在网格空间里测试簇：包围盒完全在屏幕某一边之外或近平面之前的去掉（不测远平面，因为光栅化不在远平面裁剪），单面材质的簇用法线锥做背面测试（观察点变换到网格空间里测，对任何仿射变换都精确，镜像的实例也不用特殊处理；阴影级联的观察者是光的方向；判据留了 0.001 的余量，几乎侧对的簇交给光栅化自己判断）。留下的簇再按候选顺序压缩成光栅化读的列表，表项是（实例、在这张表里的第一个三角形、这个簇的第一个索引、网格的第一个顶点），光栅化按它把调用对应到三角形。写进可见性缓冲的编号却不是这张表里的序号，而是三角形在可见实例里的编号：可见实例那张表的表项还是原来的格式（实例、它的第一个三角形、网格的第一个索引和第一个顶点），簇在其中的编号是实例的第一个三角形加上簇的第一个三角形，记在簇表项的“尾部”（列表在 `visible[0].w` 之后，每个表项再占一个 uvec4，避免几个调用写同一个 uvec4 的不同分量）。所以可见性缓冲里的值不随哪些簇被剔除而变，同一深度上取编号最小的三角形也和不剔除时一样；着色、透明和 TAA 照旧通过可见实例那张表把编号还原成实例和三角形。光栅化的存储 buffer 不能再多：GLES 3.1 只保证 8 个，它已经用满，这也是编号放在列表尾部而不是另开 buffer 的原因。簇级的压缩只知道候选簇至多有多少（`RenderWorld` 记着所有实例的簇数之和），按这个上限 dispatch，超出实际候选数的工作组什么也不做。`RendererConfig::cullClusters` 关掉簇级的测试；开不开，可见性缓冲、深度和图像都逐位相同（测试在四个后端上比对），`ViewStats::visibleClusters` 给出留下的簇数。RTX 5070 Ti、1080p：ABeautifulGame 去掉 24% 的簇，GPU 帧从 1.580 ms 降到 1.287 ms；Sponza 只去掉 8%，看不出差别。按索引顺序切的簇在 UV 球这类按环排列的网格上是整圈的带子，法线朝向各个方向，锥测试剔不掉它们。
     列表按实例顺序编号，因为光栅化在同一深度上取编号最小的三角形。压缩分三次 dispatch：每个工作组把 1024 个实例（每个调用 4 个）的可见数和三角形数加起来；一个工作组对这些块的和做前缀和，每次 256 块，写出总数和光栅化的间接参数；每个工作组再扫一次自己的块，从块的起点写出列表项。块内的前缀和有共享内存和子组两种写法。不用 decoupled look-back，因为它依赖工作组之间的前向进度保证，不是每个设备都有。原来一个工作组串行扫完所有实例，100 万个实例时每次压缩要 3.5 到 4.6 ms，现在是 0.017 到 0.033 ms。
  2. 计算光栅化写可见性缓冲：每个像素留下最近的深度，以及这个深度上编号最小的可见三角形。设备有 64 位原子（`KernelFeature::Int64Atomics`）时一遍完成：每个覆盖的像素对一个 `tc::u64` 做 `atomicMin(floatBitsToUint(depth) << 32 | 三角形)`，最小值正好是这两样，再由一个逐像素的 pass 拆进两个 32 位 buffer；其他设备分两遍，先 `atomicMin` 深度，再在深度相等时 `atomicMin` 三角形。两个光栅化 kernel 展开同一个宏 `LR_TRIANGLE_RASTERIZER`，只有写像素的那一步不同，算出的深度逐位相同，所以两种做法的三角形和深度也逐位相同（测试在 CPU 后端和 lavapipe 上比对）。后面的 pass（着色、透明的深度测试、TAA）只读拆开后的 `VisibilityOutputs`，不关心是哪一种。`Mask` 材质在光栅化时按透视校正的 UV 采基础色纹理的第 0 层，alpha 低于阈值的像素不覆盖。阴影贴图本来就只写深度，总是用两遍的那个 kernel 的深度模式。
     遮挡剔除（`RendererConfig::cullOccluded`，默认开，`cullClusters` 也开时才生效）只用于主视图的不透明几何，阴影级联和透明不做。它分两个阶段，中间隔着一次光栅化。每个视图有一个历史 buffer（帧图的 `historyBuffer`），按簇在整个场景里的编号记着上一帧哪些簇可能看得见；编号是实例按顺序、簇在实例里按顺序数的，实例压缩的第四个分量是每个实例的簇数（剔除掉的实例也算），所以实例的起点记在可见实例表项尾部的 y，不随视图而变。第一阶段 `CullClusters` 只把通过视锥和背面测试、上一帧又看得见的簇列进光栅化的列表，并给每个候选簇记下它是否留下、是否已经列出；这张列表照常光栅化。然后从深度建最远深度金字塔：第 0 级是视图的一半大小（向上取整），每个纹素取下面至多 2×2 个像素里最大的 `floatBitsToUint(depth)`，之后每级再减半，直到 1×1，各级接连放在一个 buffer 里，每级一个 dispatch；没有覆盖的像素是清除值 0xFFFFFFFF，什么也挡不住。第二阶段 `CullOccludedClusters` 测试每个留下的候选簇：网格空间包围盒的八个角投到屏幕上（有一个角在近平面之前就算看得见），取能用 2×2 个纹素盖住这片像素的那一级，只有簇最近的深度减去 1e-5 的余量仍比这些纹素里最远的深度还远，才算被挡住：盒里的东西不可能覆盖任何像素，也不会和已经画的打平。没被挡住的簇写进本帧的历史，其中第一阶段没列出的再压缩成第二张列表，画进同一个可见性缓冲（两遍光栅化时，先画两张列表的深度，再画两张列表的三角形）。所以历史只决定先画哪些、哪些要测试，不影响结果：可见性缓冲、深度和图像与不做遮挡剔除时逐位相同，测试用移动的相机连续比对 4 帧，覆盖四个后端和一遍、两遍两种光栅化。没有历史时（第一帧、视图尺寸变了、场景的簇数跨过 2 的幂使历史 buffer 重建），第一阶段什么也不画，金字塔全是远处，所有候选都留到第二阶段，结果和不做遮挡剔除一样；再下一帧起，被挡住的簇就不画了。`ViewStats` 的簇数和三角形数是两张列表的和。RTX 5070 Ti、1080p，100 万个实例的规模场景里画的三角形从 7366 万个降到 218 万个，GPU 帧从 17.85 ms 降到 3.12 ms；遮挡少的场景每帧多出约 0.07 ms 的固定开销（`benchmarks.md`）。
     光栅化按三角形大小分流，每个光栅化 pass 有两个阶段。三角形阶段一个可见三角形一个调用：变换、近平面裁剪、背面剔除之后，像素包围盒不超过 256 个像素的就地光栅化，更大的追加进这个 pass 的大三角形列表，记下它的包围盒碰到哪些 16×16 的块。一个工作组的 kernel 对列表求前缀和，给每个大三角形一段连续的块号，并写出块阶段的间接 dispatch 参数。块阶段一个块一个工作组：二分查找块属于哪个三角形，第一个调用把这个三角形重新建好放进共享内存（CPU 后端上读表很贵，不能每个调用各建一次），64 个调用每个覆盖块里的一列、隔四行一个像素。原来一个调用要扫完整个三角形，铺满屏幕的地面只有两个三角形，就落在两个调用上串行扫完整个屏幕。两个阶段的每个像素走同一段算术，所以结果和分流前逐位相同。列表最多 16384 个三角形，满了的由三角形阶段自己光栅化。三角形阶段不用屏障，块阶段用，所以各是一个 kernel（`lr_raster_visibility` 和 `lr_raster_visibility_tiles`，64 位的那一对同样），因为 CPU 后端会把用屏障的 kernel 的每个调用都放进纤程。
     1920×1080、一个铺满画面的地面加 400 个小球、三级阴影级联，RTX 5070 Ti 上光栅化可见性缓冲从 156 ms 降到 0.31 ms，每级阴影从 100 到 190 ms 降到 0.1 ms 左右，所有 pass 合计从 603 ms 降到 0.91 ms，图像逐位相同；lavapipe 上（480×270）所有 pass 从 89 ms 降到 30 ms，CPU 后端不变。
  3. 阴影：场景里第一个 `castsShadows` 的方向光有级联阴影贴图，一次只有这一盏灯有阴影。视锥在 `ViewSettings::shadows.distance` 之内按对数与均匀 3:1 混合的分割点切成 1 到 4 段，每段取包住这段视锥的球，球的大小只由这段视锥的形状决定，球心在光空间里对齐到整数个纹素，所以相机移动时级联只按整纹素移动，阴影边缘不闪。深度范围取整个场景包围盒沿光方向的投影，视锥外的遮挡物也能投下阴影。每个级联用同一套剔除、压缩和光栅化 kernel（只写深度）画进一个 `BufferBinding<uint>` 的一段，深度是 `floatBitsToUint` 后 `atomicMin`；级联的矩阵放在这个 buffer 开头的 128 个字里，由清除 pass 从主机上传。着色时选最细的、能容下 PCF 核的级联，接收点先沿几何法线偏移约一个纹素（按坡度增大），深度再减去随坡度增大的偏置，然后做 3×3 的双线性 PCF（16 次手动比较，因为深度在 buffer 里）。透明物体不投阴影，但接收阴影。
  4. 着色：先分类，再按材质模型间接着色。分类 pass 逐像素从三角形找到实例和材质，读出材质模型号；没有三角形的像素当场写环境光。每个工作组先在内部给每个模型的像素排好名次（共享内存的 `atomicAdd`，或者子组变体里每个子组对每个模型做一次 `subgroupExclusiveAdd`），再给每个模型做一次全局 `atomicAdd`，取得这个工作组在该模型里的位置，写进逐像素的“模型、名次”。一个工作组的小 pass 按模型号求前缀和，得到每个模型列表的起点，同时写好每个模型的 `DispatchIndirectCommand`（每 64 个像素一个工作组）；再一个逐像素的 pass 把（像素、三角形）放进各自模型的列表。最后每个模型一个 pass，用 `executeIndirect` 读自己那一项参数，只跑自己的像素。列表里像素的顺序随原子操作而变，但每个像素的着色只取决于它自己，所以图像是确定的。
     灯光按网格分配（`RendererConfig::clusterLights`，默认开）。第一个着色的 feature 加一个 `assign lights` pass：视图分成 64×64 像素的块，深度上分成 32 片，从近平面按几何级数增长到远平面（第一片和最后一片一直延伸到相机和无穷远），每个格子（一块的一片）有一个位掩码，第 i 位是第 i 盏灯。没有范围的灯（方向光和 `range` 为 0 的点光源）在每个格子里；有范围的点光源，范围的球够得着格子才在。格子的四个侧面是过相机、过块的边各向外 1 个像素的平面，深度上前后各放宽 0.1%，球的半径也放宽 0.1%，平面取自这一帧（抖动后）的视图投影。一个调用算一个格子的一个掩码字（32 盏灯），相邻的调用算不同格子的同一个字，读的是同样的灯；掩码按格子连续存放，着色时一个格子的字挨在一起。着色由着色点自己的位置找格子（投影到屏幕上取块，由深度取片），只遍历这个格子掩码里置位的灯；点投影到屏幕外时遍历所有灯。灯光表的副本和掩码放在同一个 buffer 里，着色 kernel 读它而不读场景的灯光表，storage buffer 的数目不变。图像与逐灯循环逐位相同，理由有三：范围外的点光源对一个点的贡献正好是 0（`rangeWindow` 为 0），加 0 不改变和；掩码从低位到高位遍历，留下的灯按灯光表的顺序相加，与逐灯循环的加法顺序相同；格子和半径的放宽远大于浮点的舍入，贡献不为 0 的灯不会漏掉。测试用 301 盏灯（散在画面里、相机背后、跨过近平面、画面两侧之外，其中一盏没有范围）、阴影、金属和透明的球，在四个后端上比较分格子和关掉 `clusterLights`（一个格子装下所有灯）的 HDR 颜色，连续 4 帧（后两帧有 TAA 的抖动）逐位相同，平均每个格子 10.5 盏灯；把半径的放宽改成缩小 2%，测试就失败。RTX 5070 Ti、1080p、10 万个实例的规模场景里，1000 盏灯时着色从 0.680 ms 降到 0.063 ms，分配要 0.037 ms（`benchmarks.md`）。
  5. 透明：`Blend` 实例单独剔除、压缩，再用同一个光栅化 kernel 的第三种模式把片元插进逐像素链表：一个 `uint` buffer 里依次是每个像素的表头、片元计数、每个片元三个字（深度、下一个、可见三角形），`atomicAdd` 分配、`atomicExchange` 挂表头，容量是每像素平均 2 个，超出的片元丢弃。只插入比不透明深度更近的片元。这一遍按 top-left 规则覆盖像素，边函数按端点的固定顺序计算，共享一条边的两个三角形算出的值严格相反，所以落在边上的像素只归其中一个，不会混合两次。之后把场景颜色拷一份，透明混合 kernel 对每个像素的链表按（深度、三角形）排序，保留最近的 8 个，用内置的 metallic-roughness 模型、相同的灯光和阴影着色（材质指定了别的模型也一样），从前往后混合到不透明颜色上。排序键里有三角形，所以结果与片元插入的顺序无关。
  6. 时间抗锯齿（`ViewSettings::antiAliasing` 为 `Taa` 时，在色调映射之前）。视图的投影每帧按 Halton(2, 3) 的第 1 到 16 个点（减 0.5）平移不到一个像素，渲染线程按视图记住抖动序号和上一帧不带抖动的视图投影矩阵；上一帧没有渲染这个视图、尺寸变了、`ViewDesc::cameraCut` 为真或设备重建过，序号回到 0，历史作废。运动 pass 逐像素在实例的物体空间里求出可见点，分别用这一帧和上一帧的世界矩阵、视图投影矩阵（都不带抖动）投到屏幕上，差就是运动矢量；静止的物体在静止的相机下运动恰好是 0，历史因此在纹素中心采样，不会变模糊。没有三角形的像素跟着远平面走。解析 pass 在 `p + 0.5 - 运动` 处双线性采样 RGBA16F 的历史（`graph.history`），把它夹在这一帧 3×3 邻域颜色在 YCoCg 里的包围盒中，再与这一帧混合：历史的 alpha 里存平均了多少帧，前 10 帧取平均，之后新一帧的权重固定为 0.1。历史作废、或者点上一帧在屏幕外的像素从这一帧重新开始。结果写进历史，再拷回场景颜色。透明像素用的是它后面不透明表面的运动。路径追踪管线没有这个 feature，不抖动。
     收敛与拖影的测量（32×32、斜 0.3 弧度的直边，64×64、每帧移动 2 像素的白块）：32 帧后边缘像素的值与像素被覆盖面积的平均差 0.036（不开 TAA 时 0.197），这 16 个点本身能做到的最大差是 0.12；GPU 的值与在主机上按同样的抖动和混合模拟的结果相差不到 4e-4（半精度的舍入）。白块移走后留下的轨迹与背景的差在 5e-5 以内（就是 0.2 存成半精度的误差），白块内部仍然是 1。
  7. 色调映射；后处理 kernel 声明中等精度（`Precision::Medium`）。
- **参考路径追踪**：渐进累积，是画质参照，也是可见性缓冲管线着色结果的测试参照。追踪射线有两种做法，是一种 kernel 变体（见下一节）：设备有 ray query 时追踪设备自己建的加速结构（`lr_path_trace_ray_query`），否则遍历渲染线程建的两级 BVH（`lr_path_trace`）。两个 kernel 展开同两个宏，`LR_PATH_TRACER_INPUTS` 是共同的绑定和 alpha 判定，`LR_PATH_TRACER_PATHS` 是路径本身，各自只写 `trace()`，所以射线在同一个点打到同一个三角形，之后的路径就完全相同。
  - 两级 BVH 只有软件遍历用。第一次有视图需要时，渲染线程给每个网格建一棵 BLAS（分箱 SAH，叶子最多 4 个三角形），再在实例上建 TLAS（叶子一个实例）；之后增量里的网格重建自己的 BLAS，实例或网格有变化就重建 TLAS。深度不超过 31 层（快到上限时改按中位数对半分），遍历用固定大小的 `std::array` 栈迭代（tc 不允许递归），栈是 32 个元素。节点、三角形（三个角在顶点表里的位置，和它是网格的第几个三角形）、每个网格的根、TLAS 节点经 `SceneUpdate` 进 `DeviceScene` 的表，可见性缓冲管线不读这些表。每个网格的 BLAS 在帧池上各是一项并行建，节点的位置再按顺序在渲染线程上分；TLAS 在渲染线程上建。16 个约 6.5 万个三角形的球第一次建层次，debug 构建里从 11.0 s 降到 1.05 s（帧池 21 个线程）。遍历先 TLAS，再用 `worldToObject` 把射线变换到实例的物体空间遍历 BLAS（方向不归一化，两个空间里的距离相同）。
  - 用 ray query 时渲染线程不建 BVH，只告诉 GPU 线程这一帧要追踪射线。GPU 线程的 `AccelerationScene` 跟着 `DeviceScene` 应用的每个 `SceneUpdate` 记下哪些网格槽变了、哪些网格没了、实例和材质变没变，只在要追踪射线的帧里建：变了的网格先由 `lr_copy_positions` 把顶点表里的位置拷进一个临时的 `DeviceBuffer<vec4>`（顶点表是 SoA，位置只在每页 256 个顶点里连续，不能直接给加速结构用），再按它和索引表各建一个 BLAS；TLAS 的实例由 `lr_write_top_level_instances` 在设备上从实例表写出，自定义索引是实例在表里的位置，再 `buildTopLevel`。被换掉的 BLAS、TLAS 和临时 buffer 进这一帧的退役列表，因为 TLAS 不持有它引用的 BLAS，在途帧还可能在追踪旧的。没有三角形的网格的实例引用一个只有一个三角形的占位 BLAS，掩码是射线不追踪的那一位：底层引用为 0 的实例按 Vulkan 的规定本该不参与追踪，lavapipe 上它会让后面的实例也都不被命中。
  - 背面剔除和 alpha 交给实例的标志：射线带 `CullBackFacingTriangles`，双面材质的实例关掉剔除，镜像的实例翻转正反面（ray query 在物体空间判定正反面，软件遍历在世界空间判定），只有 `Mask` 和 `Blend` 材质的实例是不透明以外的，`rayQueryProceedEXT` 在它们的三角形上停下，由和软件遍历同一个函数决定穿不穿过。
  - kernel 每帧每像素一条路径。着色和光源用与可见性缓冲管线同一份代码（`kernels/Lighting.h`、`kernels/Material.h`），双面材质的处理也相同：单面材质的背面对所有射线都不可见，双面材质从背面看时法线翻转。纹理固定采第 0 层（多个采样平均下来就是像素足迹上的平均）。`Mask` 材质在遍历时按 alpha 剔掉命中；`Blend` 材质是随机不透明：射线（包括阴影射线）以 1 - alpha 的概率穿过它，判定用射线的种子、实例和网格里的三角形号散列，期望正好是按 alpha 混合，所以简单场景两条管线一致。每个顶点对方向光和点光源做下一事件估计并追踪阴影射线；加上自发光；逃逸的射线取均匀环境光；继续的方向按漫反射和 GGX 两个波瓣的混合采样，按混合密度加权；第三次弹射起用俄罗斯轮盘。随机数是像素、采样序号、维度的 PCG 散列，所以每个设备抽到的随机数相同。
  - 两种做法在 CPU 后端、lavapipe 和 RTX 5070 Ti 上逐像素相同（48×48、4 个采样，场景里有没有三角形的网格、两种 `Mask`、`Blend`、两个方向的镜像实例、从背面看的双面材质，换一次网格、增删实例和网格之后再比一次）。640×480、400 个小球的场景，RTX 5070 Ti 上每帧的路径追踪从 1.38 ms 降到 0.11 ms，两幅图平均差 0.0075 级。
  - 累积放在 RGBA32F 的历史资源里，存运行均值，均值写进 `sceneColor`，后面照常色调映射。相机、目标尺寸或场景（任何改变网格、纹理、材质、实例、灯光、环境的增量）变了就从头开始，这由渲染线程判断；纹理流式写完或消失也要从头开始，而这只有 GPU 线程知道，所以采样数在 GPU 线程上按视图记：录制时看渲染线程给的“可以继续”和纹理驻留的代数，`ViewStats::accumulatedSamples` 给出当前的采样数。
- 可见实例的压缩和像素分类各有两个 kernel：用子组的（`subgroupExclusiveAdd`、`subgroupAdd`，`Shared` 只存每个子组的总数，子组大小取 `gl_SubgroupSize`），和只用 `Shared` 的。两者的输出相同（分类的列表顺序除外，图像相同），启动时选择，见下一节。

#### 启动检查与 kernel 变体

- render 自己维护一张程序表：任何一帧都可能用到的程序，和每种可选变体的程序（子组：压缩和分类的子组版本；一遍写入的可见性缓冲：清除、光栅化、拆分，和从打包的可见性缓冲建金字塔第 0 级，四个用 64 位整数的 kernel；ray query：路径追踪、拷位置、写 TLAS 实例三个 kernel，其中路径追踪只有 Vulkan 版本）。测试检查注册表里每个 `lr_` 开头的程序都在表里。
- `makeRenderer` 建好设备后，在 GPU 线程上对表里的程序逐个 `checkKernel`。必需的程序有不能运行的，就抛 `tc::DeviceError`，消息里列出每个程序和原因，比如“`lr_shade_metallic_roughness` 用了 8 个 storage buffer，设备只允许 4 个”。可选变体的程序全部能运行、并且 `RendererConfig::subgroups` / `singlePassVisibility` / `rayQueries` 不是 `VariantUse::Off`，才用这个变体。选择结果是 `Renderer::capabilities()`（`RendererCapabilities`：是否用子组、是否一遍写可见性缓冲、子组大小、是否用 ray query），feature 从 `ViewContext::capabilities` 读到，按它在帧图里放 pass 或选 kernel。`Off` 让测试能在同一个设备上比较两种变体。
- 设备丢失后重建的设备用同样的配置创建，但仍然检查一遍：已选变体和所有材质模型的程序都要能运行，否则按建不出设备处理。
- 测试用的设备选到的变体：CPU 后端三种都用（子组大小 32，ray query 由 tc 在主机上建 BVH 遍历）；lavapipe 的 Vulkan 三种都用（子组大小 8）；llvmpipe 的 OpenGL 和 GLES 都不用（没有子组，没有 64 位原子，也没有 ray query）。GLES 本来就没有 64 位整数。

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
- 渲染线程上的 `TextureLayout` 决定纹理放进哪个池的哪一层：压缩图像在设备有这种格式的数组、是正方形 2 的幂、不超过上限、带完整 mip 链、还有空的压缩槽时，原样进压缩池，所有级别都从图像拷贝；其余的图像（包括设备不支持的压缩格式）用 `tc::decodeCompressed` 解码成 RGBA8，在线性颜色里重采样成下一个 2 的幂的正方形（缩小用面积平均，放大用按寻址方式环绕的双线性；超过上限的缩小到上限），进 RGBA8 池，只上传第 0 层。RGBA8 池满了就把层数翻倍：GPU 线程建一张更大的图像，用 kernel 逐层逐级拷过去，旧图像进退役列表。压缩池不能被 kernel 写，也没有图像到图像的拷贝，所以一开始就按约 32 MiB 分配层数，满了的新纹理改走解码。移除纹理时它的层回到空闲列表。
- 解码和重采样不在渲染线程上：`TextureLayout` 当场分好池和层，把转换交给后台池，上传要等转换做完，由之后某一帧的 `SceneUpdate` 带走；原样进池的图像当帧就上传。哪一帧带走由 `RendererConfig::textureConversions` 决定：`AsSoonAsReady`（默认）是转换做完之后的第一帧，没有哪一帧等它；`Deterministic` 是带来它的下一帧，转换没做完就等，所以转换快慢不改变纹理在哪一帧出现，测试用这个。转换期间纹理被移除或者又 upsert 了，旧的转换结果就丢掉。debug 构建里一张 3000×2000 的图在渲染线程上要 2.2 s，现在放进去只要 0.03 ms，`AsSoonAsReady` 下每帧最多等 0.01 ms。
- RGBA8 池存纹理自己的编码（sRGB 纹理存 sRGB 值），kernel 采样之后再解码。下采样 kernel 在线性颜色里求 2×2 平均（黑白相间的棋盘格在第 1 层是线性 0.5，而不是 sRGB 平均出来的 0.21），所以 mip 级别是对的；代价是纹素之间的双线性插值发生在 sRGB 值上，高对比边缘的插值偏暗。tc 没有 sRGB 格式，要完全正确只能存 RGBA16F，内存翻倍，现在没有这样做。
- compute 没有导数，`texture()` 只采第 0 层，所以着色时用射线微分算 LOD：像素右边和下边相邻像素中心的相机射线与三角形所在平面相交，求出那里的 UV，与本像素 UV 的距离取较大者就是足迹，LOD = log2(足迹 × 边长)，再用 `textureLod`（`Mip::Linear`）三线性采样。各向同性，斜看时偏模糊。
- 新纹理流入池时，GPU 线程在录制一帧的开头把数据拷进暂存段（`createBuffer` 分配的映射内存，每个在途帧一段，第一次用到时才分配），再在 tc 的传输队列（`device.transfers()`）上 `copyBufferToImage` 写进对应的层和 mip 级别，马上提交。设备有独立的传输队列（`DeviceInfo::separateTransfers`，比如 NVIDIA 的专用传输队列族）时，拷贝和这一帧的 dispatch 同时在设备上进行；没有时拷贝进设备自己的时间线，按调用顺序排序。两条时间线只在 `waitFor` 处排序：拷贝先等上一帧的提交，因为上一帧可能还在采样要被覆盖的层；这一帧之前建了或换了池时，改等到此为止的设备时间线，因为新图像的布局转换录在那上面。这一帧提交之前，设备等拷贝的提交（`device.waitFor(submission)`），再用下采样 kernel 逐级生成 RGBA8 纹理的其余各级（每次 dispatch 一层，因为 llvmpipe 的 `texelFetch` 在一次 dispatch 里层号不同时会读错层）。最后一级写完，纹理才算驻留，材质从下一帧开始引用它，之前只用因子；所以纹理最早在上传它的那一帧的下一帧出现。录制第 k 帧时用的暂存段是第 k - framesInFlight 帧用过的那一段，那一帧已经做完，它的提交又等过它的拷贝，所以主机写暂存段不会和设备读冲突。每帧最多拷 `RendererConfig::textureStreamingBytes`（默认 8 MiB），按 mip 级别排队，一帧放不下的留到下一帧；`copyBufferToImage` 只能整级拷贝，所以比预算还大的单个级别独占一帧，暂存段为它临时变大。
- 材质里的引用由 GPU 线程填：`SceneUpdate` 里的材质带着纹理 ID，`DeviceScene` 按 `TexturePool` 的驻留情况把它们换成引用，驻留情况一变就重写所有材质。替换一个纹理（同一个 ID 再 upsert）时，旧数据马上不再被引用，新数据写完之前只用因子。
- 压缩格式按 `DeviceLimits::compressedArrayFormats` 选，因为要放进数组（Adreno 的 GLES 上 BC4、BC5 只能是 2D）。资产离线烘焙成 KTX2，桌面一份 BC、移动端一份 ASTC 或 ETC2；KTX2 的块顺序就是 `copyBufferToImage` 要求的顺序，可以直接拷贝。lavapipe 没有 ETC2，测试里这条路走解码。经 asset 加载的图片，压缩池放不下的已经在后台池上解成 RGBA8（见 asset 的“压缩格式”）。
- `RenderWorld` 只把图像句柄交给 GPU 线程，自己不留；`TexturePool` 写完最后一级就放掉它，纹理在整个系统里只有设备上这一份（宿主自己留的句柄除外）。设备丢失后，渲染线程在下一帧把放置过的所有纹理记为丢失，`FrameResult::lostTextures` 列出需要重新提交的 `TextureId`，在此之前材质只用因子；那一帧失败的话，由下一个成功的帧报告。用了 scene 时，`World` 留着自己登记的图像句柄，宿主把这个列表交给 `WorldResources::resubmitTextures`，下一个增量重新提交它们（`lrender_view` 就是这样做的）。

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
- 每个内置 kernel 最多用 8 个 storage buffer、4 个存储图像、16 个采样纹理，push constant 不超过 128 字节；8 个 storage buffer 高于规范下限的 4 个，不满足的设备在启动时由 `checkKernel` 报告出来。着色 kernel 正好用满 8 个（实例、顶点、索引、材质、灯光网格、可见实例、像素列表、阴影贴图），所以可见实例列表带上网格的首索引和首顶点，省掉网格表；灯光网格带着灯光表的副本，省掉灯光表；阴影级联的矩阵放在阴影贴图 buffer 的开头，省掉一个参数 buffer；模型的像素列表每项是（像素、三角形），着色 kernel 因此不必再读逐像素的三角形 buffer，列表的表头（每个模型的长度和起点）放在同一个 buffer 的开头。透明混合 kernel 的第 8 个是片元链表，片元和它们的表头放在同一个 buffer 里。TAA 的运动 pass 要三个矩阵，超过 128 字节，所以放进一个小 buffer，由 pass 每帧上传。
- 每帧变化的小块数据由 pass 写进 upload buffer 的主机端副本，再 `uploadBuffer` 一段：帧图 buffer 按在途帧各取一份，上传在调用时拷走，所以不会和在途帧冲突。阴影级联的矩阵在只在设备上的阴影贴图 buffer 的开头，所以先上传进一个 upload buffer，再 `copyBuffer` 过去。
- HDR 中间目标用 `RGBA16F`，它在 GLES 3.1 和 Vulkan 上都是强制支持的存储格式，也能线性过滤。

#### 回读

窗口目标的 Readback 路径和离屏目标都不调用 `downloadImage`，因为它总会等待。最后一个 pass 把打包的 RGBA8 直接写进 `ViewContext::output`：渲染线程给每个视图声明的一个 `createReadbackBuffer`，也就是 `createBuffer` 分配的映射内存。它和别的帧图资源一样从资源池里按描述取，在途的每一帧各持有一份，帧收尾之后才回到池里，所以池里自然有 `framesInFlight` 份，不需要再按偏移分段。这个回读由 Renderer 自己登记，不归某个 feature：`whenComplete` 完成之后回到 GPU 线程，窗口目标把这段内存直接交给 Presenter，离屏目标拷进 `ViewResult::image`。

#### 计时

- `FrameResult::wallTime` 总是带从 `submit()` 到这一帧收尾的墙钟时间。其中主机上的三段也分开报告：`prepareTime` 是渲染线程应用增量、建视图帧图的时间，`recordTime` 是 GPU 线程从开始录制到提交的时间，`finishTime` 是设备做完之后 GPU 线程收尾（回读、取计时、交给 Presenter）的时间。三段互不重叠，都在 `wallTime` 之内；剩下的是排队和设备执行。`FrameResult::sceneMemory` 是帧收尾时场景占的设备内存：几何、簇、网格实例材质灯光这几张表、软件层次结构，以及纹理池的每一层每一级。
- `RendererConfig::gpuTimers` 打开且 `DeviceInfo::timers` 为真时，`ViewResult::passTimings` 还带每个 pass 的设备耗时。`takeTimings(submission)` 按执行顺序返回这次提交里的每次 dispatch（帧按顺序收尾，失败的帧也取一次再丢掉，所以取到的正好是这一帧的），帧图按同样的顺序记下每次经 `PassContext` 的 dispatch 属于哪个 pass，两者逐个对应。不能按程序名对应，因为同一个 kernel 会出现在多个 pass 里（比如可见性缓冲和阴影的光栅化）；程序名只用来核对：有 pass 绕过 `PassContext` 直接调用设备时，两者对不上，这一帧就不给 pass 耗时。GPU 线程自己的 dispatch 不属于任何 pass：纹理池变大时的拷贝和加速结构在视图之前，到达的纹理的 mip 级别在视图之后，这几个程序在两头按名字跳过。
- 开着计时器时，tc 会在每个计时的 dispatch 前插入完整屏障，dispatch 之间不再重叠，所以 `gpuTimers` 默认关闭，只在开发、剖析和挑选表布局时打开。帧与帧之间的重叠不受影响。
- Adreno 上 GLES 的计时查询几乎总是返回接近 0 的值，又不报告时钟不连续，所以这类设备上只看墙钟时间。llvmpipe 的 OpenGL 和 GLES 也一样。

### present：Presenter 的实现

每种实现是一个单独的 CMake 目标，宿主只链接用得到的。需要特定线程或者可能阻塞的上屏，由 Presenter 自己转交，因为 `present()` 是在 GPU 线程上调用的。

- `VulkanPresenter`（`lr::present_vulkan`，找到 Vulkan 时构建）：自己的 VkInstance、VkDevice 和交换链，一个 Presenter 对应一个窗口目标。窗口的 surface 由宿主建（`lrender_view --present vulkan` 用 `SDL_Vulkan_CreateSurface`）；没有窗口时，用 `VK_EXT_headless_surface`（只有 Mesa 的驱动有，NVIDIA 的驱动问到这种 surface 会崩溃），或者 `offscreen`，拷进自己的三张 BGRA 图像，不经过交换链，供测试和 `lrender_bench` 用。
  - 渲染器的设备和它是同一块 GPU（UUID 相同）、它又能导出 opaque fd 和 sync file 时，走 SharedBuffer：三个槽是可导出的设备内存，用法与 tc 导入时建的 buffer 相同，所以两边的内存类型一致。否则走 Readback：`present()` 把像素拷进主机可见的暂存 buffer。
  - 一个线程专门上屏，每次交换链能接一帧时取交来的最新一帧，更早的直接丢掉，所以渲染器比显示快时只会丢帧，不会等 Presenter。上屏时队列等这一帧的 acquire fence（导入成 semaphore），把 buffer 拷进交换链图像：交换链是 `R8G8B8A8_UNORM` 而且尺寸相同时直接拷，否则先拷进一张 RGBA8 图像再 blit，blit 负责通道顺序和缩放。交换链只用 UNORM 格式，因为像素已经是 sRGB 编码。这次提交另外 signal 一个可导出的 semaphore，导出的 sync file 就是这个槽的 release fence。buffer 在外部队列族和自己的队列族之间转移所有权。
  - 槽按状态轮转：空闲、正在写、等上屏、上屏线程拿着、已上屏。GPU 线程要槽时，先挑 Presenter 已经读完的；SharedBuffer 路径上其次挑拷贝还在跑的，由设备等它的 release fence；最后才挑还没上屏的最旧一帧，那一帧算丢掉。上屏线程拿着的槽不给。
  - `stats()` 给出上屏、走 SharedBuffer、丢掉的帧数，以及从交出到上屏的主机时间；`captureNext()` 把下一帧上屏的图像读回来（测试、截图）；`flush()` 等交出的帧都上屏或丢掉。
  - 测量（RTX 5070 Ti，Sponza 1920×1080，release，与直接丢帧的 Presenter 在同一个进程里交替）：GPU 帧时间 1.218 对 1.115 ms，两者的先后对调之后是 1.204 对 1.095 ms，多出的约 0.1 ms 分散在所有 pass 上，每个不到 3 µs，像是两个设备的上下文轮流占用 GPU 的代价；墙钟多 0.15 到 0.23 ms；从交出到上屏 0.12 到 0.15 ms，全部帧都走 SharedBuffer、没有丢帧。Readback 路径原来每帧收尾要 230 ms 以上，因为 tc 的 `createBuffer` 在开了 resizable BAR 的显卡上把回读 buffer 放在显存里，主机读 8 MB 要 153 ms；tc 加了 `HostAccess::Reads` 之后（`library-requirements.md` 第 5 项），回读 buffer 在主机缓存的内存里，收尾不到 1 ms。
- `SdlPresenter`（Readback，`lr::present_sdl`，只在找到 SDL2 时构建）：`present()` 把像素拷进三缓冲信箱，宿主主线程调用 `blitLatest()` 上屏。SDL 要求在主线程上屏。信箱是三个槽加一个原子交换的“共享槽下标 + 新帧标记”：GPU 线程写自己的槽再换进共享槽，主线程看到新帧标记就换出来，双方都不等对方，主线程总是拿到最新的一帧，中间的帧直接被覆盖。`blitLatest()` 把它传进一张流式纹理（`SDL_PIXELFORMAT_ABGR8888`，就是低字节为红的打包 RGBA8），拉伸到整个输出再 `SDL_RenderPresent`，返回帧号；没有新帧时什么都不画。纹理随 Presenter 销毁，所以最后一个引用要在主线程上、Renderer 之后、`SDL_Renderer` 之前放掉。1080p 的 RGBA8 约 8 MB 一帧，回读本身不到 1 ms，最后一个 pass 经 PCIe 写主机内存多花 0.7 ms 左右；`VulkanPresenter` 的 SharedBuffer 路径两样都省掉。
- `SurfaceControlPresenter`（Android API 29+，SharedBuffer）：引擎直接渲染进一组 RGBA8 的 AHardwareBuffer（通过 `ExternalImageBinding` 写入），这个 Presenter 用 `ASurfaceTransaction_setBuffer` 提交，acquire fence 是 `submitWithFence()` 返回的 `.fence`（`dup` 出来的那一份）。合成器在回调里交还的 release fence，在 GPU 线程下次写这块 buffer 之前交给 `waitFor()`，由设备等待。没有回读，也没有拷贝。Vulkan 第一次以存储图像使用新导入的帧时可能丢弃原有内容，而输出 pass 本来就写满整帧，不受影响。
- `NativeWindowPresenter`（Android API 29 以下，Readback）：在自己的线程上 `ANativeWindow_lock` 并拷贝。
- `ImageSequencePresenter`（Readback，`lr::present_image`）：离线输出。`present()` 拷一份像素，用 `exec::spawn(io, …)` 起一个任务，在 `stackfull::blocking()` 里写 `<directory>/<prefix><帧号>.ppm`，不阻塞 GPU 线程。`flush()` 等到目前交给它的帧都写完（或失败），返回写成和失败的帧数；析构时等还在写的帧，所以宿主要在 Renderer 之后、不在 GPU 线程上放掉最后一个引用。`lrender_cli --sequence <目录>` 用它输出每一帧。

### asset：资产 IO 与解码（可选）

```cpp
struct AssetSource {                    // 同步写法，只在加载任务里调用
    virtual ~AssetSource() = default;
    virtual AssetResult<Bytes> read(AssetPath const &path) = 0;
    virtual std::optional<std::string> localFile(AssetPath const &path) const = 0;   // 热重载监听的本地文件
};
std::unique_ptr<AssetSource> makeFileSource(FileSourceConfig const &config);   // 在 stackfull::blocking() 里整个读
std::unique_ptr<AssetSource> makeMemorySource(MemorySourceConfig config);      // 测试用
std::unique_ptr<AssetSource> makeAndroidAssetSource(AAssetManager *manager);   // 只在 Android 上编译

template <class T>
using AssetResult = tl::expected<T, AssetError>;   // AssetError：kind（NotFound、ReadFailed、Corrupt、Unsupported、Failed）和 message
template <class T>
using LoadSender = lexec::any_sender_of<lexec::set_value_t(AssetResult<T>), lexec::set_stopped_t()>;

struct AssetLibrary {
    virtual ~AssetLibrary() = default;
    virtual LoadSender<MeshHandle> loadMesh(AssetPath const &path) = 0;     // glTF 场景合并成一个网格，没有材质
    virtual LoadSender<ImageHandle> loadImage(AssetPath const &path, ColorEncoding encoding) = 0;   // PNG、JPEG、KTX2
    virtual LoadSender<SceneHandle> loadScene(AssetPath const &path) = 0;   // .gltf、.glb
    virtual void drainChanges(std::vector<AssetChange> &changes) = 0;       // 宿主线程，不等待
    virtual AssetLibraryStats stats() const noexcept = 0;
};
std::unique_ptr<AssetLibrary> makeAssetLibrary(AssetLibraryConfig const &config);
// config：source、runtime、compressedFormats（设备的 compressedArrayFormats）、preferKtx2、watchFiles

SceneDelta sceneDelta(SceneData const &scene, IdAllocator &ids);   // 加入场景的增量，和移除它的 Removals
```

一次加载是 IO 调度器上的一个任务，步骤之间检查停止：

```cpp
AssetResult<ImageHandle> loadImageInTask(LoadContext const &context, AssetPath const &path,
                                         ColorEncoding const encoding, std::vector<AssetPath> &files) {
    auto const file = readImageFile(context, path);   // 阻塞池；可能换成 KTX2 变体
    if (not file) return errorOf(file);
    files.push_back(file->path);
    auto image = onBackgroundPool(context, [&] { return decodeImage(ByteView::of(file->bytes), encoding, context.images); });
    if (not image) return assetError(image.error().kind, file->path + ": " + image.error().message);
    return std::make_shared<ImageData const>(std::move(*image));
}
```

- glTF：任务读 `.gltf` 或 `.glb`，后台池上 cgltf 解析 JSON；任务再按其中的 URI 读外部 buffer 和纹理用到的图片；最后回到后台池装上 buffer（data URI 在这里解码）、`cgltf_validate`，转出网格、材质、图片，展平节点。`loadMesh` 不读图片。
- 去重：在途表按种类和路径（图片再加编码）索引。sender 启动时才加入或发起加载（`let_value(just(), …)`），每个请求方拿到自己的 `Completion`（见 stackfull）。加载结束就从表里拿掉，之后的请求重新读，没有缓存。
- 取消：请求方的停止令牌一触发，它的 `async_get` 立即以 `set_stopped` 完成，加载的等待人数减一。还有人在等，加载照常进行；最后一个也停了，这次加载就被放弃，从表里退出，并请求它自己的 `inplace_stop_source`（经 `write_env` 交给 `exec::spawn`）。任务在下一次读或解码之前以 `stopped_error` 结束，排在后台池里还没开始的解码也不会运行（池在取出工作时检查停止令牌）。之后的同一请求发起新的加载。库析构时停止所有在途加载，还在等的请求方以 `set_stopped` 完成。
- 错误：缺失、损坏、不支持都是 `AssetError` 值，走 `set_value`。任务里的异常（比如内存不足）和任务起不来也变成 `AssetErrorKind::Failed`，所以 `LoadSender` 没有 `set_error`。
- glTF 2.0（cgltf）：每个 primitive 一个 `MeshData`（三角形、带和扇；没有法线时按规范生成平面法线，每个三角形用自己的角）；metallic-roughness 的因子和四张纹理、`normalScale`、alpha 模式和阈值、双面、`KHR_materials_emissive_strength`、`KHR_texture_transform`（材质只有一个 UV 变换，取第一张带变换的纹理的）；节点层级展平成世界空间的实例；`KHR_lights_punctual` 的方向光和点光源（聚光灯当点光源）；透视相机。没有加载或只是近似的内容记在 `SceneData::warnings` 里。材质里的 `TextureId` 是 `SceneData::textures` 的下标，同一张图片既当颜色又当数据时是两张纹理。
- 图片：PNG、JPEG 用 stb_image 解成 RGBA8。编码由用途决定：基础色、自发光是 sRGB，法线、metallic-roughness 是线性，KTX2 格式名里的 SRGB/UNORM 不算数。KTX2 自己解析（头、level index，vkFormat 映射到 RGBA8 或 `tc::CompressedFormat`），只读二维、单层、单面的文件；BasisLZ、Zstandard、ZLIB 超压缩和 Basis Universal 返回 `Unsupported`。
- 压缩格式：KTX2 的块在 render 的 `TextureLayout` 会原样放进压缩池时（格式在 `compressedFormats` 里、正方形、2 的幂、完整 mip 链）原样交出，否则在后台池上用 `tc::decodeCompressed` 解成 RGBA8 的各级，渲染器拿到的总是它能直接用的。`preferKtx2` 打开时，请求 `name.png` 先读 `name.ktx2`：块会被保留就用它，否则用 `name.png`，只有 `name.png` 不存在才解码 KTX2。直接请求 `name.ktx2` 不找别的文件。
- 场景到增量：`sceneDelta` 是纯函数，id 由调用方的 `IdAllocator` 给（`makeSequentialIds` 是最简单的一种）；资产只产出句柄，id 归宿主或 scene。返回的 `removal` 移除这次加入的一切，热重载时放进下一帧增量的 `removed`，新场景用新的 id 一起加入。
- 热重载：`FileWatcher` 是一个 inotify fd（`IN_NONBLOCK | IN_CLOEXEC`），由一个 IO 任务读，事件变成 `FileChange`（Created、Modified、Removed、Overflow）进 `sync::Channel`，`drainChanges` 在宿主线程上 `tryRecv`，队列满了任务就停在发送上，事件留在内核里。它监听目录而不是文件，因为编辑器常常写临时文件再改名过来。`watchFiles` 打开时，加载任务在交出结果之前监听它读过的本地文件所在的目录，并记下文件属于哪些资产；`AssetLibrary::drainChanges` 把 Modified（写完关闭，或改名进来）换成要重新加载的资产，Overflow 报告全部。`FileWatcher` 也能单独用。
- `stats()`：加载次数、加入在途加载的请求数、读次数、后台解码步数、停止的加载数，测试靠它确认“只读一次”“没有解码”。

### scene：世界模型与脚本（可选）

```cpp
struct World {                                     // 只在宿主线程上使用
    virtual EntityId createEntity(Transform const &local, std::optional<EntityId> parent) = 0;
    virtual void destroyEntity(EntityId entity) = 0;                                 // 连同所有后代
    virtual void setParent(EntityId entity, std::optional<EntityId> parent) = 0;     // 保持局部变换
    virtual void setTransform(EntityId entity, Transform const &local) = 0;
    virtual mat4 worldMatrix(EntityId entity) const = 0;
    virtual void setRenderable(EntityId entity, Renderable const &renderable) = 0;   // {MeshId, MaterialId}
    virtual void setLight(EntityId entity, LightDesc const &light) = 0;              // 在实体的空间里
    virtual void setCamera(EntityId entity, CameraLens const &lens) = 0;
    virtual CameraDesc cameraOf(EntityId entity) const = 0;
    virtual void setEnvironment(EnvironmentDesc const &environment) = 0;
    virtual WorldResources &resources() noexcept = 0;                                // 网格、纹理、材质
    virtual RenderWorldDelta extractDelta() = 0;
    // 还有 isAlive、parentOf、removeRenderable、removeLight 等；非虚的 spawnInstance(mesh, transform) 建一个可渲染的根实体
};
std::unique_ptr<World> makeWorld(WorldConfig const &config);   // config：可选的 Runtime、并行传播的门槛
```

- 实体是带代数的 `EntityId { index, generation }`。槽位在实体销毁后复用，代数加一，所以旧 id 一用就抛 `std::invalid_argument`；脚本也可以先用 `isAlive` 检查。
- 组件按槽位以 SoA 存放：局部变换、世界矩阵、父节点、子节点和兄弟链表、过期标记各一个数组；可渲染组件、灯光、相机的镜头也按槽位存放，前两者的数据在按 `InstanceId`、`LightId` 排的稠密表里。每个可渲染的实体是渲染世界里的一个实例。灯光写在实体的空间里，点光源跟着实体移动，方向光跟着实体转动。相机在实体的原点，朝实体的 -z 看，+y 向上；`Rotation::lookingAlong` 给出这样的朝向。
- 层级：改了局部变换或父节点的实体，连同整棵子树标为过期；遇到已经过期的子树就不再往下走，所以过期的实体总是若干整棵子树。`extractDelta()` 从这些子树的根出发按层排好，逐层计算“世界 = 父的世界 × 局部”，同一层之间互不依赖。配了 `Runtime`、这次要算的数目又达到 `WorldConfig::parallelTransforms`（默认 4096）时，256 个以上的层用帧池上的 `bulk` 分发再 `sync_wait`，其余的层在宿主线程上依次算。两条路径调用同一个函数，结果逐位相同，测试用两万个实体的随机层级和两千次随机修改比对。提取之前查询 `worldMatrix`，沿过期的祖先现算，结果也相同。
- 销毁父实体时连同所有后代一起销毁，而不是把孩子挂到别处；它们的实例和灯光在下一个增量里移除。
- 变化跟踪：实例、灯光、网格、纹理、材质各是一张表，记着渲染世界已经有了哪些 id、上次提取以来改了哪些、移除了哪些。世界矩阵按位没变的实例不进增量，设成同样的变换什么也不会发生；两次提取之间建了又删掉的东西不出现在增量里；删掉的 id 在下一次提取之后才再分配，所以同一个增量不会既 upsert 又移除同一个 id。环境只在设置之后的那一个增量里出现。
- 资源：`WorldResources` 给网格、纹理、材质分配 id 并计引用。`add*` 给调用方一个引用，可渲染组件持有它的网格和材质，材质持有它的纹理；同一个网格句柄（纹理是同一张图像加同一个采样器）只有一个 id，所以多个实体共用的网格只提交一次；最后一个引用放掉后，下一个增量移除它。`replace*` 用于热重载，id 不变。`defaultMaterial()` 是 World 自己持有的 `MaterialDesc{}`。
- `World::extractDelta()` 是模拟与渲染之间唯一的交接点。

```cpp
struct ScriptHost {                                    // 只在调用 tick 的宿主线程上使用
    virtual ScriptId start(ScriptTask script) = 0;     // 从下一次 tick 开始运行
    virtual void requestStop(ScriptId script) = 0;
    virtual void tick(SimDuration elapsed) = 0;        // 推进模拟时间，放行到期的等待和该放行的操作，把执行器跑空
    virtual bool isIdle() const noexcept = 0;          // 没有脚本还在运行
    virtual std::vector<ScriptFailure> takeFailures() = 0;   // 以 stopped_error 以外的异常结束的脚本
    virtual void shutdown() = 0;                       // 析构函数也调用它
};
std::unique_ptr<ScriptHost> makeScriptHost(ScriptHostConfig const &config);   // config：放行模式和 k
```

- `ScriptHost` 拥有 `co2::ManualExecutor`、按（模拟时间，等待序号）排序的等待堆，以及在途操作的闸门列表。只有闸门列表加锁，因为 sender 在别的线程上完成。`SimDuration` 是整数纳秒，模拟时间累加没有舍入误差。
- 一次 tick：模拟时间加上 `elapsed`；放行到期的等待（`nextFrame()` 在下一次 tick，`waitFor(d)` 在模拟时间第一次到达的那次 tick）和该放行的操作，按等待的先后排进执行器；把执行器跑空，上次 tick 以来 `start` 的脚本排在最前面，脚本在 tick 里 `start` 的新脚本这次 tick 就开始运行；最后销毁这次结束的脚本的帧。`nextFrame` 和 `waitFor` 返回实际经过的模拟时间。

脚本的生命周期遵守 co2 的销毁契约：挂起在某个操作上的协程不能销毁。

1. `requestStop` 请求脚本的 co2 stop token，也就是 promise 的 `get_stop_token()` 交出的那个。
2. lexec 的 co2 桥把这个 token 放进被等待 sender 的环境里，响应停止的 sender 以 `set_stopped` 完成。完成照样经过闸门。
3. 下一次放行时，闸门后面那一步调度会检查这个 token，所以即使 sender 带回了值，脚本也在宿主线程上收到 `stopped_error`，结束，帧被销毁。等在 `nextFrame` / `waitFor` 上的脚本在下一次 tick 收到它。`stopped_error` 不算失败。还没运行过的脚本被停止时，会运行到第一个等待处再结束。
4. 不响应停止的操作会让脚本停在“停止中”，直到操作完成。实体先被销毁也没关系，脚本之后通过带代数的 `EntityId` 访问时能发现实体已经失效。
5. 关闭时对所有脚本 `requestStop`，然后在模拟时间不走的情况下反复 tick，直到 `isIdle()`；没有可以放行的东西时阻塞，等下一个操作完成，而不是空转。关闭之后 `start` 的脚本不会运行，它的帧随 `ScriptTask` 销毁。析构函数做的就是关闭，所以 `ManualExecutor` 析构时队列一定是空的。

`ScriptHostConfig::release` 有两种取值：

- `AsSoonAsReady`（默认）：已经完成的操作在下一次 tick 放行；同一次 tick 里按等待的先后。
- `Deterministic`：tick t 里等待的操作恰好在 tick t + k 放行（`deterministicDelay`，默认 1），到时还没完成就在那次 tick 里阻塞到它完成，同一次 tick 里按等待的先后。资产在 IO 上快一点还是慢一点，都不会改变脚本的时间线，测试和重放才真正可重复：测试让三个脚本各等六次耗时随机为 0 到 4 ms 的操作，四组随机数得到逐 tick 相同的时间线，正好是 t + k。代价是永远不完成的 sender 会让那次 tick 一直等下去，所以等外部事件的脚本不该用这个模式。

脚本示例。测试按原样运行这一段，`examples/common/src/Scripts.cpp` 里有一份按网格包围盒取景的，`lrender_view --mesh <文件>` 运行它：

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

`lrender_view`（SDL 窗口与输入，加上 scene、asset、Renderer；TAA 默认打开，T 键开关）和 `lrender_cli`（离线，按指定的帧数或采样数输出图片，也是金标准测试的入口；`--taa` 打开 TAA，默认关）。两者默认显示演示场景，它由 `addDemoScene` 建在一个 `World` 里；`lrender_cli` 取这个 World 的第一个增量，`--gltf <文件>` 仍用 `sceneDelta` 直接得到增量。

`lrender_view` 的场景全部在 `World` 里，由 `ScriptHost` 推动，每提交一帧就 tick 一次（`--frames` 时每次 1/60 秒，否则按墙钟时间）：

- 演示场景里，`steerCamera` 每帧让相机沿 `Orbit` 慢慢转，鼠标拖动和滚轮改的也是这个 `Orbit`；`bob` 让发光的球和它上方的点光源一起上下浮动、转动，两者是同一个父实体的孩子。
- `--gltf <文件>` 由 `showGltf` 脚本经 `AssetLibrary` 加载，用 `spawnScene` 把场景放到一个根实体下（每个实例、每盏灯一个子实体；文件里没有灯时加一个太阳），第一次加载时按文件的相机或包围盒设好 `Orbit` 和镜头。之后脚本每帧 `drainChanges`，文件变了就重新加载，销毁旧的根实体、放上新的场景，增量里是旧实例和资源的移除、新的 upsert。
- `--mesh <文件>` 运行上面那个 `orbitCamera` 脚本。
- 设备丢失后，帧报告的 `lostTextures` 交给 `resubmitTextures`。开发构建的 GPU 后端上还会重新加载 kernel。

`lrender_view` 的主循环从不等帧：帧的 sender 用 `lexec::spawn` 放进一个 `counting_scope`，完成时给一个原子计数加一，循环只在未完成的帧少于 `framesInFlight` 时 tick 并提交新帧，每一轮都调 `blitLatest()`。宿主循环也可以写成下面这样，阻塞与否完全由宿主决定：

```cpp
for (;;) {
    if (not pumpSdlEvents(input, orbit)) break;
    world->resources().resubmitTextures(lostTextures());
    scripts->tick(clock.step());
    auto frame = FrameDesc{};
    frame.delta = world->extractDelta();
    frame.views.push_back(mainView(world->cameraOf(camera), window));
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
| 在途操作的闸门 | `ScriptHost` | 任意线程上完成，宿主线程上放行 | 一把锁，只护着这张列表 |
| `RenderWorldDelta` | 按帧移动的值 | 宿主 → 渲染线程 | 不可变 |
| `MeshData` / `ImageData` | `shared_ptr<T const>` | 任意线程只读 | 不可变；热重载产生新对象 |
| `RenderWorld` 镜像、BLAS 缓存、帧图资源池 | render | 渲染线程 | 单一序列 |
| 常驻表、纹理池、临时资源、历史图像 | `FrameQueue` | GPU 线程 | 单一序列；丢掉的进当前帧的退役列表，帧完成之后释放 |
| 回读 buffer、暂存段 | 帧图资源池 | GPU 线程写暂存、设备写回读 | 每个在途帧各持有一份，帧收尾之前不回池 |
| `FrameResult`（统计、回读图像） | 按帧移动的值 | GPU 线程 → 宿主 | 通过 sender 交接 |

## 跨模块约定

- 错误：
  - 可恢复的情况（资产缺失、解码失败）作为值返回（`AssetResult<T>`，`AssetError` 带种类和消息），走 `set_value`；停止的请求走 `set_stopped`。
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
| 光栅化管线 | 超出 compute | 计算光栅化；硬件光栅由 lrender 自己的模块负责，提案见 `hardware-raster.md` |
| 光追管线（raygen、hit 着色器、SBT） | 超出 compute | 路径追踪器在 kernel 里用 ray query（属于 compute），没有它的设备遍历软件 BVH |
| 无异常构建 | 会改变 co2 和 tc 的错误模型 | 引擎要求开启异常 |
| Windows、macOS | 超出 stackfull 声明的目标平台 | 引擎只支持 Linux 与 Android |
| CPU 后端的 kernel 热重载 | CPU 后端运行的是编译好的 C++ | 重新加载只在 GPU 后端提供 |
| 等待 sync file | tc 只提供阻塞的 `wait()` 和非阻塞的 `isSignaled()`，与事件循环集成不是它的事 | gpu 模块在 stackfull 的 Poller 上等待 |

## 放弃的方案

- **全部跑在 stackfull 上（计算、IO、脚本共用 M:N 调度器）。** 它的 lexec 调度器没有对 `bulk` 做并行定制，`bulk` 会在一个任务里串行执行；长计算会占住 worker；宿主无法在帧的固定阶段驱动它，而脚本恰恰需要这一点；每个脚本还要一个独立的栈。
- **全部用 co2。** 无栈协程不能在第三方代码内部挂起，加载器就写不成同步风格。
- **加载器写成 sender 管线（`let_value` 链）。** 读文件、解析、再按引用去读 buffer 和图片、解码，是一个带分支的顺序流程，第三方解码器又都是同步代码。在有栈任务里按同步写法写最直接。
- **在 tc 之外接一套 Vulkan 图形管线。** 那样会失去单一源码、CPU 调试和跨后端等价测试，而这正是这个引擎最有价值的地方。代价是用不上硬件光栅化，性能比传统引擎低，这个项目接受这个代价；RT core 经 kernel 里的 ray query 用上了。
- **每帧提取完整快照，或者渲染线程加锁访问共享场景。** 前者每帧的拷贝量随场景规模增长，后者让模拟和渲染互相等待。改为增量加渲染侧常驻镜像。
- **让宿主线程 `sync_wait` 准备工作来保证串行。** 库会因此阻塞宿主，串行也要靠调用方遵守前置条件。改为渲染线程上的 `run_loop`。
- **一个计算池加任务优先级。** lexec 不提供优先级，自己实现一个带优先级的池又会和 `bulk` 的池定制脱节。两个池最简单。
- **GPU 线程自己等 GPU 完成（`Submission::wait()`，或者轮询 `isComplete()`）。** 前者让 GPU 线程在设备执行期间无法录制下一帧；后者需要定时唤醒，而 `run_loop` 没有定时器。等待交给 IO 调度器，GPU 线程只在完成之后被调度回来收尾。
- **脚本等待的 sender 直接 `continues_on` 到 `ManualExecutor`。** 完成会随时排进执行器，tick 把执行器跑空时，tick 进行中别的线程完成的操作也会在这次 tick 里恢复，一次 tick 做多少事取决于时机；确定性模式还要知道每个操作是在哪次 tick、以什么顺序等待的。所以中间加一道闸门，由 tick 决定放行谁。

## 工程结构

```
lrender/
  CMakeLists.txt  CMakePresets.json
  cmake/LrDependencies.cmake FetchContent，固定提交：lexec、co2、stackfull、compute_shaders，
                             以及 tl-expected、cgltf、stb（提交的源码包加哈希）；doctest 在 tests/ 里拉取；
                             SDL2 用系统里的（find_package(SDL2 CONFIG)），找不到就不构建
                             lr::present_sdl 和 lrender_view
  core/      include/lr/core/                     lr::core     INTERFACE
  runtime/   include/lr/runtime/   src/           lr::runtime  lexec::runtime、stackfull
  gpu/       include/lr/gpu/       src/           lr::gpu      tc、tc_devices、stackfull::exec、stackfull::runtime
  render/    include/lr/render/    src/  kernels/ lr::render   tc_add_kernels(...)；include/lr/render/kernels/ 是用户 kernel 也要用的公开 kernel 头文件
  present/   image_sequence/  sdl/  vulkan/  （android/）
                                                  lr::present_image  lr::present_sdl  lr::present_vulkan  （lr::present_android）
  asset/     include/lr/asset/     src/           lr::asset    lr::core、lr::runtime、tl::expected；私有 cgltf、stb
  scene/     include/lr/scene/     src/           lr::scene    lr::core、lr::runtime、co2；lexec 的 co2 桥随 lexec 而来
  examples/  view/  cli/                          lrender_view  lrender_cli
  probes/    gpu_probe  present_probe  asset_probe  layout_probe  bench（lrender_bench）
  tools/     fetch-scenes.sh  run-benchmarks.sh
  tests/
```

- 命名空间为 `lr::<module>`，每个模块一个 CMake 目标，目标图与模块图一致。
- 作为库交付：要提供 install 规则和 `find_package(lrender)` 用的包配置，每个可选层单独导出为一个组件。现在还没有，因为依赖库还不都有 install 规则，等它们有了再加。
- stackfull 至少要固定到 `acb8861`：在这之前，交给空闲 IO 工作线程的任务偶尔唤不醒它（工作线程在成为 timekeeper 之前被认领，随后睡进 Driver），asset 的加载因此偶尔挂住，见 `library-requirements.md`。
- compute_shaders 至少要固定到 `efc7d78`（`HostAccess::Reads`，回读 buffer 用主机缓存的内存）；更早的几项：只在设备上的 buffer 和 `copyBuffer` 从 `9cec48f` 开始才有，传输队列从 `c3660eb` 开始，ray query 从 `317c74a` 开始；`2f5b9f9` 让同一个 EGL display 上的 OpenGL 设备共用一次初始化，在这之前销毁一个设备（或者找别的设备时跳过一个正在用的 display）会把 display 一起终止，而测试里常常同时开着两个设备；`fb599b6` 让 CPU 后端和 Vulkan 一样把底层引用为 0 的实例当作不参与追踪。
- 平台取各库支持范围的交集：Linux x86_64/aarch64 和 Android arm64。
- 主机构建依赖 libclang-18-dev 和 glslang-tools（tc 转译器）；Android 交叉编译时使用主机构建出的转译器。发布构建加 `-DTC_PROGRAM_RELOAD=OFF`。

## 验证与里程碑

| 模块 | 类型 | 验证方式 |
|---|---|---|
| core、RenderWorld、帧图编译 | 确定 | 单元测试 |
| render kernel | 确定 | CPU 后端的金标准图像，加上跨后端等价测试；子组和 64 位原子的变体与回退变体在同一个设备上互相比对（三角形、深度、可见实例列表、图像逐位相同），路径追踪的 ray query 与软件遍历也在同一个设备上比对；自定义材质模型只改变自己的像素；TAA 与主机上的模拟和解析覆盖率比对，并测拖影、运动矢量和历史的重置；设备图像没有读回接口，纹理池的内容由着色 kernel 采样（自发光纹理贴满一个四边形，像素中心对准纹素中心），再用读 HDR 颜色的 feature 读进 buffer 比对；阴影、透明、环境光的镜面部分、法线贴图与路径追踪器或解析结果比对 |
| Renderer 整体 | 确定 | cpu 后端，以及 llvmpipe、lavapipe 上的 GPU 后端（没有 sync file，走 `Submission` 的等待路径）：连续提交的帧按序完成、累积结果与逐帧提交相同、在途帧不超过上限（用一个数录制而未收尾帧数的 feature）、多视图、窗口目标交给测试用的 Presenter、缩放、设备丢失恢复（用一个录制时抛 `tc::DeviceError` 的 feature）；`ImageSequencePresenter` 写出的文件与离屏目标的图像逐像素相同 |
| scene、ScriptHost | 确定 | 手动 tick 的单元测试：增量里只有该有的条目，层级传播、改父节点、连同后代销毁，旧 id 被拒绝，资源只提交一次、最后一个引用放掉才移除，帧池上的传播与宿主线程上逐位相同；`nextFrame`、`waitFor` 的时机，别的线程上完成的 sender 只在 tick 里、在宿主线程上恢复，停止和失败，析构；`Deterministic` 模式下耗时随机的加载得到逐 tick 相同的时间线；脚本经 `AssetLibrary` 加载生成的 glTF 放进 World，增量交给 Renderer，与解析结果和 CPU 后端比对 |
| asset | 半确定 | 测试自己生成 glTF（内嵌、外部文件、.glb）、PNG、JPEG、KTX2；用一个计数、能扣住读取的测试源测去重、加载重叠和取消；热重载在构建目录下改真实文件，轮询 `drainChanges`；加载的场景经 `sceneDelta` 交给 Renderer，与解析结果和 CPU 后端比对；kernel 重新加载用改过注释的生成目录，前后两帧逐位相同 |
| gpu | 边界 | 单元测试用 eventfd 代替 sync file、用自定义的 `Submission::Completion` 代替还没完成的提交，分别覆盖两条等待路径；`gpu_probe`：选择后端、打印 `checkKernel` 的结果，在真实设备上提交并走 sync file 路径，打开计时器时按提交打印每次 dispatch 的耗时 |
| 表布局 | 边界 | `layout_probe`：在选定的后端和设备上，给剔除和顶点读取的访问模式按各候选布局计时 |
| 性能 | 测量 | `lrender_bench`：release 构建，场景和方法见 `benchmarks.md`；每一项性能工作都拿它和基线比 |
| present | 边界 | `SdlPresenter` 在 SDL 的 offscreen（或 dummy）驱动和软件渲染器上测，读回像素比对；`VulkanPresenter` 在 lavapipe 的 headless surface 和 NVIDIA 的 offscreen 图像上测，两条路径都与同一帧的离屏目标逐像素相同，连续交来的帧要么上屏要么丢掉、最新的一帧上屏，改变尺寸之后重新导入，失败的帧不显示；`present_probe`：每种 Presenter 显示测试图案 |

Mesa 的软件设备没有 sync file，也没有专用的传输队列族。设置 `LRENDER_TEST_DEVICE`（设备名的一部分，比如 `NVIDIA`）后，GPU 后端的测试改在那个设备上跑；Vulkan 上再打开验证层的同步检查（`VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true`），检查两条时间线之间的排序。同步检查不能用 Ubuntu 24.04 自带的 1.3.275 版验证层：它不认 timeline semaphore 上的等待，会把已经排好序的拷贝报成冲突；LunarG SDK 的 1.4.363 没有这个问题，去掉拷贝前的 `waitFor` 时它报出冲突，加上时没有。

1. **M0** 工程骨架：CMake、presets、固定版本的依赖、runtime 模块、CI（debug、clang-debug、asan 三个预设，llvmpipe 与 lavapipe 上跑 GPU 后端，覆盖没有 sync file 的等待路径）。
2. **M1** 渲染核心：`RenderWorld` 增量、帧图、可见性缓冲管线的最小版本（GPU 剔除与间接 dispatch、两遍可见性缓冲、基础 PBR 着色、色调映射），在 cpu 后端上配合 `lrender_cli` 建立金标准测试。
3. **M2** 参考路径追踪，作为画质参照；Vulkan、OpenGL、GLES 与 CPU 的等价测试；打开设备计时器，定下每张表的布局和编码。
4. **M3** 完整的 Renderer 接口：渲染线程、GPU 线程、异步提交与 IO 上的完成等待、回读 buffer、在途帧、多视图、缩放、设备丢失恢复、SDL 窗口目标与图片序列；`lrender_view`。`gpu_probe` 和 `present_probe` 还没有写。
5. **M4** 画质。M4a：设备端纹理池（mip、流式写入、压缩格式与解码回退、设备丢失后列出丢失的纹理）、带纹理的材质（法线贴图、UV 变换、alpha 遮罩）、级联阴影、逐像素链表的顺序无关透明、均匀环境光的镜面部分；路径追踪器同样采样纹理，混合材质按随机不透明处理。M4b：启动时的 kernel 检查与变体选择、一遍写入的可见性缓冲和子组变体、材质分类与间接着色（应用可以加自己的材质模型）、TAA。
6. **M5** asset：流式加载、取消、热重载、KTX2 与按设备选择压缩格式，以及开发期的 kernel 重新加载；`lrender_cli` 和 `lrender_view` 的 `--gltf`。`asset_probe` 还没有写。
7. **M6** scene 与脚本：`World`（SoA 组件、层级与帧池上的并行传播、带代数的实体、变化跟踪、资源的引用计数）、`ScriptTask` 与 `ScriptHost`（闸门、停止与关闭、两种放行模式）；`lrender_view` 改为由 World 和脚本驱动，包括 glTF 的加载和热重载，`--mesh` 运行文档里的 `orbitCamera`。
8. **M7** Android：`SurfaceControlPresenter` 零拷贝上屏、API 29 以下的回读上屏、生命周期。

还没做的：Android（M7）；install 规则和包配置，等依赖库都有 install 规则之后再加；`gpu_probe`、`present_probe`、`asset_probe` 还没有写，`probes/` 里目前只有 `layout_probe` 和基准程序 `lrender_bench`。向 tc 提出的新能力见 `library-requirements.md`：前四项都已在 tc 里实现，lrender 都用上了；第五项（主机读的共享 buffer）还是提案。`VulkanPresenter` 在真实窗口上的上屏（`lrender_view --present vulkan`）在本机的 X11 桌面上跑过一次：debug 构建，RTX 5070 Ti，300 帧 3.6 秒（每秒 83 帧），FIFO 交换链在 60 Hz 下上屏 201 帧，其余的被更新的帧替换，没有报错；自动测试只用 headless surface 和 offscreen 图像，因为本机没有私有的 X 服务器。硬件光栅的提案见 `hardware-raster.md`，实测之后暂缓到 M7。
