# 对依赖库的新需求

前五项是对 compute_shaders 的，最后一节是 stackfull 的一处修正（已经改了）。

lrender 需要 tc 新增五项能力。前三项互相独立；第四项是 `hardware-raster.md` 里硬件光栅方案的前提，现在桌面的零拷贝上屏也用它；第五项是测零拷贝上屏时发现的。每一项写明 lrender 怎么用、为什么在 tc 的职责之内、接口草案（照 tc 现有的写法，命名只是建议）、各后端的语义与回退，以及验收方式。

五项都已在 tc 里实现，接口以 tc 的 README 为准（第 5 项是 `efc7d78`，帧图的回读 buffer 用它，Sponza 1080p 经 Readback 路径的收尾从 232 ms 降到 1 ms 以内）：只在设备上的 buffer、内存堆与资源别名是 `9cec48f`，传输队列是 `c3660eb`，外部 buffer 导入与设备 UUID 是 `93f30dc`，ray query 是 `317c74a`。lrender 都用上了：第 1 项的 ray query（路径追踪器的变体），第 2 项的只在设备上的 buffer、`copyBuffer` 和堆（帧图里生命周期不重叠的资源共用内存），第 3 项的传输队列，第 4 项的外部 buffer（窗口的 SharedBuffer 路径，见 `lr::present_vulkan`）。实现中补了一处：CPU 后端原先不接受底层引用为 0 的实例（Vulkan 规定它不参与追踪），`fb599b6` 改成和 Vulkan 一样；lavapipe 在这种实例上有问题，见 tc 的 README。

文中的设备能力是在本机查到的：RTX 5070 Ti（NVIDIA 595 驱动）和 lavapipe（Mesa 25.2.8）。

| | 能力 | lrender 的用途 | Vulkan | GL / GLES | CPU |
|---|---|---|---|---|---|
| 1 | kernel 里的 ray query 与加速结构 | 路径追踪走硬件遍历 | 有 | 没有 | 软件实现 |
| 2 | 只在设备上的 buffer、内存堆与资源别名 | 帧图的临时资源 | buffer 和图像都别名 | buffer 别名，图像不别名 | 都别名 |
| 3 | 独立的传输队列 | 纹理与几何的流式上传 | 专用的传输队列族 | 退化为主时间线 | `submit()` 时完成 |
| 4 | 导入外部 buffer、设备 UUID | 与硬件光栅模块共享可见性缓冲 | 有 | 桌面 GL 有 | 不适用 |
| 5 | 主机读的共享 buffer | 离屏目标和 Readback 路径的回读 | 主机缓存的内存 | `GL_CLIENT_STORAGE_BIT` | 不变 |

## 1. kernel 里的 ray query 与加速结构

**lrender 怎么用。** 参考路径追踪器现在在 kernel 里遍历自己的两级 BVH（`render/kernels/PathTrace.h`，32 个元素的栈），BVH 在渲染线程上同步构建。有 ray query 的设备上，遍历交给硬件，加速结构也在设备上构建：每个网格建一次 BLAS；TLAS 每帧重建，实例由一个 kernel 从实例表写出，渲染线程不再建 BVH。软件 BVH 留给没有 ray query 的设备（GL、GLES），两条路径按 kernel 变体选择，互为测试参照。

**为什么在 tc 的职责之内。** Ray query（`GL_EXT_ray_query`、`SPV_KHR_ray_query`）是 compute shader 里的一组内建函数加一种描述符，不需要光追管线；加速结构是 kernel 读取的资源，构建是设备时间线上的一条命令，和 `copyBufferToImage` 同类。光追管线（`VK_KHR_ray_tracing_pipeline` 的 raygen、hit 着色器和 SBT）不是 compute，不在这项需求里。

**kernel 侧。** 函数和常量沿用 GLSL 的名字，和 tc 的其他内建函数一样：

```cpp
struct TC_KERNEL Trace {
    static constexpr char fileLocation[] = "trace";
    static constexpr bool usesRayQuery = true;   // 和 usesSubgroups 一样：没声明就用，转译器报错
    uvec3 local_size{64, 1, 1};

    AccelerationStructureBinding<0> scene;       // uniform accelerationStructureEXT
    BufferBinding<vec4, 1> rays;
    BufferBinding<vec4, 2> hits;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        rayQueryEXT query;                       // 不透明类型，只能是局部变量
        rayQueryInitializeEXT(query, scene, gl_RayFlagsNoneEXT, 0xFFu, rays[2u * i].xyz(), 0.0f,
                              rays[2u * i + 1u].xyz(), 1e30f);
        while (rayQueryProceedEXT(query)) {      // 只在不透明标记以外的几何上停下，由 kernel 决定
            if (keepsAlpha(rayQueryGetIntersectionInstanceCustomIndexEXT(query, false),
                           rayQueryGetIntersectionPrimitiveIndexEXT(query, false),
                           rayQueryGetIntersectionBarycentricsEXT(query, false))) {
                rayQueryConfirmIntersectionEXT(query);
            }
        }
        if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
            hits[i] = vec4(rayQueryGetIntersectionBarycentricsEXT(query, true), rayQueryGetIntersectionTEXT(query, true),
                           uintBitsToFloat(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)));
        }
    }
};
```

需要的子集：`rayQueryInitializeEXT`、`rayQueryProceedEXT`、`rayQueryTerminateEXT`、`rayQueryConfirmIntersectionEXT`；候选交点和已提交交点的 `rayQueryGetIntersection` 系列：`Type`、`T`、`InstanceCustomIndex`、`InstanceId`、`PrimitiveIndex`、`GeometryIndex`、`Barycentrics`、`FrontFace`；射线标志 `Opaque`、`NoOpaque`、`TerminateOnFirstHit`、`CullBackFacingTriangles`、`CullFrontFacingTriangles`。不需要 AABB 几何（`rayQueryGenerateIntersectionEXT`）。

**主机侧：**

```cpp
// 位置是 buffer 里按固定步长排列的 xyz（BufferResource<vec4>，或每 3 个 float 一个点），索引是 uint
auto const geometry = tc::TriangleGeometry::of(positions, firstVertex, vertexCount, indices, firstIndex,
                                               triangleCount, /*opaque=*/true);
auto blas = device->createBottomLevel({geometry}, tc::BuildOptions{});   // 在设备时间线上构建
auto tlas = device->createTopLevel(maxInstances, tc::BuildOptions{});
device->buildTopLevel(tlas, instances, count);   // instances 是 BufferResource<tc::TopLevelInstance>，可以由 kernel 写
kernel.scene.attach(tlas);
device->bindAccelerationStructure(kernel.scene);
```

- `TopLevelInstance` 是 64 字节的 std430 结构，与 `VkAccelerationStructureInstanceKHR` 逐字节相同：按行存放的 3×4 变换（`std::array<vec4, 3>`），24 位自定义索引加 8 位掩码，SBT 偏移加标志，以及 `uvec2` 形式的 BLAS 引用（`blas.reference()`）。引用用 `uvec2` 而不是 64 位整数，写实例的 kernel 就不需要 `Int64`。kernel 能写这个结构，TLAS 就能整个在 GPU 上更新。
- 构建与前后的 dispatch 按读写关系排序：它读几何 buffer、写加速结构。scratch 内存由 tc 管理。
- `BuildOptions` 指定偏向遍历快还是构建快、是否允许更新（refit）、是否压缩。lrender 先只用“遍历快”。
- BLAS 和 TLAS 都是值，副本指向同一个，最后一个副本释放、设备也不再使用时才释放。实例 buffer 里的引用 tc 看不到，所以约定：TLAS 还可能被使用期间，它引用的 BLAS 必须活着；lrender 用退役列表保证这一点。
- 能力：`KernelFeature::RayQuery`；`DeviceLimits` 增加每个 kernel 的加速结构绑定数、TLAS 的实例数上限和 BLAS 的三角形数上限。

**后端：**

- Vulkan：`VK_KHR_acceleration_structure`、`VK_KHR_ray_query`，以及它们依赖的扩展（`VK_KHR_deferred_host_operations`；`VK_KHR_buffer_device_address` 和 `VK_KHR_spirv_1_4` 在 Vulkan 1.2 里是核心）。用到 ray query 的 kernel 要编成 SPIR-V 1.4，只有这些 kernel 换目标版本。作为构建输入的 buffer 需要设备地址和构建输入两种用途。本机 RTX 5070 Ti 和 lavapipe 都支持 ray query 和加速结构，CI 上的 lavapipe 就能测。
- GL、GLES：没有对应的扩展。设备不报告 `RayQuery`，`checkKernel` 拒绝这类 kernel，创建加速结构抛 `DeviceError`。
- CPU：构建时在主机上建 BVH；`rayQueryEXT` 对象保存遍历状态，`rayQueryProceedEXT` 每次交出一个非不透明的候选。候选的顺序与 GPU 不同，最近的交点相同（并列的除外）。

**验收：** 同一组射线在 CPU 与 Vulkan（NVIDIA、lavapipe）上的最近交点（实例、三角形）相同，t 和重心坐标在小容差内（例如相对 1e-5）；非不透明几何的确认与拒绝；有交点时 `TerminateOnFirstHit` 一定报告一个；由 kernel 写实例再构建的 TLAS；变换行列式为负（镜像）的实例，正反面判定在 CPU 和 GPU 上一致。

## 2. 只在设备上的 buffer、内存堆与资源别名

**lrender 怎么用。**

- 帧图里主机不读的临时 buffer，现在是 `tc::BufferResource<T>(n)` 加一次 `uploadBuffer`。在 GPU 后端上，主机端那一份只在创建时用过，却和设备端一样大，而且每个在途帧各有一份。主视图的这类 buffer 每像素几十字节：打包的可见性缓冲 8 字节，材质模型的像素列表 8 字节，透明片元链表约 28 字节。只在设备上的 buffer 去掉主机这一份，与 `DeviceImage` 对应。
- 帧图编译时知道每个临时资源第一次和最后一次被哪个 pass 用到。生命周期不重叠的资源放进同一块堆里互相重叠的位置，一帧的临时内存就从所有资源之和降到同时活着的峰值。现在的资源池只在帧与帧之间按描述复用。

**为什么在 tc 的职责之内。** 资源的内存从哪里来、什么时候可以重用，归 tc 管；别名需要的同步只有 tc 的屏障跟踪能插：后一个资源的第一次写要排在前一个资源的最后一次访问之后，Vulkan 上的图像还要换布局。哪个资源放在哪里、偏移多少是策略，归 lrender。

**接口草案：**

```cpp
auto lists = device->createDeviceBuffer<uvec2>(n);                         // 只在设备上，内容未定义
auto heap = device->createHeap(bytes);                                      // 一块设备内存
auto const forLists = device->placementOf<uvec2>(n);                        // {bytes, alignment}
auto const forColour = device->placementOf<gpu::RGBA16F, Dim::D2>(extent, 1);
auto a = device->createDeviceBuffer<uvec2>(n, tc::Placement{heap, 0});
auto b = device->createImage<gpu::RGBA16F, Dim::D2>(extent, 1, tc::Placement{heap, 0});   // 与 a 重叠
kernel.lists.attach(a);                                                     // BufferBinding 接受 DeviceBuffer
```

- `DeviceBuffer<T, D>` 和 `DeviceImage` 一样没有主机副本：不能 `uploadBuffer` / `downloadBuffer`，内容由 kernel 写入或者从别的 buffer 拷贝，初始内容未定义；CPU 后端上它持有唯一的一份数据。从别的 buffer 拷贝需要一个设备上的 `copyBuffer`，第 3 项也用到它。
- 别名：同一个堆里字节区间重叠的资源互为别名。用到其中一个之后，与它重叠的其他资源的内容未定义；一个资源在它自己第一次被写之前，内容也未定义。
- 同步：放置资源的访问按“堆加字节区间”跟踪。`HazardTracker` 已经按键和区间判断冲突，放置资源以堆为键、加上自己的偏移，前一个资源的最后一次访问和后一个资源的第一次写之间就会插入屏障。在 Vulkan 上，区间换了使用者时，图像从 `UNDEFINED` 转到 `GENERAL`。
- 寿命：堆和放置资源都是值。资源是堆的视图；堆和它所有视图的最后一个副本都释放、设备也不再使用时，堆才释放。

**后端：**

- Vulkan：堆是一次 `vkAllocateMemory`，放置资源用 `vkBindBufferMemory` / `vkBindImageMemory` 绑到偏移上。对齐取资源自己的内存需求；buffer 和优化布局的图像相邻时，还要满足 `bufferImageGranularity`。内存类型不兼容的资源放不进同一个堆，`placementOf` 要报告这一点，或者把堆分成只放 buffer 和只放图像两种。
- GL、GLES：buffer 的堆是一个大 GL buffer，放置 buffer 是其中一段（`glBindBufferRange`，偏移满足 `GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT`），buffer 照样别名。图像没有可移植的放置方式，各自分配，不别名；不别名也满足“内容未定义”，语义不变，`DeviceInfo` 报告图像不别名。lrender 的临时资源大多是 buffer，所以 GL 上也能省下大部分。
- CPU：堆是一块主机内存，buffer 和图像都是它的视图，别名是真实的。区间换了使用者时，CPU 后端把它填成固定的垃圾值，“读到别名旧内容”的 bug 在 CPU 上必现。

**验收：** 两个 kernel 先后写互为别名的 A 和 B，B 只读到自己写的值；A 的最后一次读在 B 的第一次写之前完成（让 A 的读足够长，再检查 A 读到的结果）；Vulkan 上 buffer 与图像互为别名；GL 上两个放置 buffer 确实共用字节；CPU 后端填入的垃圾值；`DeviceBuffer` 能绑定、能 `copyBuffer`，对它上传或下载时抛 `DeviceError`。

## 3. 独立的传输队列

**lrender 怎么用。** 纹理的流式写入（每帧至多 `textureStreamingBytes`，默认 8 MiB 的 `copyBufferToImage`）和常驻表的上传现在都录在帧的命令流里，和这一帧的 dispatch 在同一个队列上排队，大纹理进来的那一帧会变长。本机 RTX 5070 Ti 有 4 个只做传输的队列族。拷贝放到那里，由拷贝引擎和 compute 同时执行，主时间线只在用到新数据之前在设备上等一次。

**为什么在 tc 的职责之内。** 用设备的哪个队列、队列之间怎么同步，是 tc 后端的事；lrender 碰不到队列。

**接口草案：**

```cpp
tc::TransferQueue &transfers = device->transfers();
transfers.waitFor(lastUse);                          // 设备上等主时间线的一次提交，比如要覆盖的层已经没有人采样
transfers.copyBufferToImage(staging, offset, pool, level, layer);
transfers.copyBuffer(staging, stagingOffset, table.words(), wordOffset, bytes);
tc::Submission const copied = transfers.submit();    // 不等待
device->waitFor(copied);                             // 之后排队的主时间线工作在设备上等这次拷贝
```

- 传输时间线上只有拷贝：buffer 到 buffer、buffer 到图像，以及部分 `uploadBuffer`；没有 dispatch。
- 两条时间线之间只在 `waitFor` 处排序。一条时间线写一个资源，另一条读写它之前，要先等写的那次提交；没有这样排序的并发访问是未定义的。所以 `Submission` 要能作为设备上的等待点：Vulkan 上每个队列一个 timeline semaphore，值就是提交序号；本机两个设备都有 `VK_KHR_timeline_semaphore`。
- tc 现在的资源都是 `VK_SHARING_MODE_EXCLUSIVE`，图像都在 `GENERAL` 布局。跨队列族的访问由 tc 处理（用 `CONCURRENT`，或者在 `waitFor` 处自动转移所有权），lrender 不感知。
- `DeviceInfo` 报告传输队列是独立的硬件队列，还是退化成了主队列。
- 可选的第二步：`TransferQueue` 可以在设备线程以外的一个线程上使用。这样纹理的流式上传可以整个放进 lrender 的 IO 任务（读文件、写暂存、提交拷贝），GPU 线程只负责 `waitFor`。代价是 tc 的资源表要能被两个线程访问，GL 后端还要在那个线程上建共享上下文。第一步不需要它。

**后端：**

- Vulkan：有只做传输的队列族就用它，否则用计算队列族的第二个队列，都没有就退化为主队列。lavapipe 只有一个队列族、一个队列，走的是退化路径。只做传输的队列族有 `minImageTransferGranularity` 限制，本机是 1×1×1，整级拷贝总能满足。
- GL、GLES：退化为主时间线，按调用顺序执行，`waitFor` 什么也不用做。
- CPU：拷贝在 `submit()` 时完成。

**验收：** 主时间线在执行一个长 dispatch 时，传输时间线上的拷贝在它结束之前完成（NVIDIA）；两个方向的 `waitFor` 都生效（先采样旧内容再覆盖，采样结果不受覆盖影响）；退化的设备上语义相同；按提交取计时不受影响。

## 4. 导入外部 buffer 与设备 UUID

硬件光栅方案（`hardware-raster.md`）的前提。lrender 的光栅模块在自己的 VkDevice 上分配可以导出的内存，tc 把它导入成 buffer，两边用 sync file 交接（`waitFor` 和 `submitWithFence` 已经有了）。

- 导入：Linux 上是 opaque fd 或 dma-buf，Android 上是 BLOB 格式的 `AHardwareBuffer`，得到一个只在设备上的 buffer（第 2 项的 `DeviceBuffer`），kernel 通过 `BufferBinding` 读写，原子操作照常可用。只要求导入，不要求导出，因为 GL 只能导入。
- `DeviceInfo` 给出 deviceUUID 和 driverUUID（Vulkan 的 `VkPhysicalDeviceIDProperties`，GL 的 `GL_EXT_memory_object`），lrender 用它确认两个设备在同一块 GPU 上。
- Vulkan 上，共享 buffer 在外部队列族与自己的队列族之间转移所有权，和现在的外部帧一样。
- 后端：Vulkan 用 `VK_KHR_external_memory_fd`，本机两个设备都有；桌面 GL 用 `GL_EXT_memory_object_fd`；GLES 取决于扩展；CPU 不适用。
- 为什么在 tc 的职责之内：tc 已经和相机、合成器这类外部使用者共享内存（`ExternalImage`、`SyncFence`），这一项把同样的能力扩展到 buffer 和桌面 Linux。

## 5. 主机读的共享 buffer（已实现，`efc7d78`）

**lrender 怎么用。** 离屏目标和窗口的 Readback 路径，最后一个 pass 都把整帧写进 `createBuffer()` 的内存，帧做完后主机就地读：离屏目标拷进 `ViewResult::image`，Readback 的 Presenter 拷一份再交给自己的线程。1080p 一帧是 8 MB。

**问题。** `createBuffer()` 选内存时只想着主机写、设备读：设备本地的堆足够大（独立显卡开了 resizable BAR，或者统一内存）就用设备本地、主机可见的内存，否则用主机缓存的系统内存。RTX 5070 Ti 开了 resizable BAR，于是回读的 buffer 落在显存里，主机经 PCIe 读它非常慢。本机测的 8 MB（`memcpy` 取三次最快）：

| 内存类型 | 主机读 | 主机写 |
|---|---|---|
| 设备本地、主机可见、一致（现在 `createBuffer()` 选的） | 153 ms | 0.77 ms |
| 系统内存、主机可见、一致、不缓存 | 19.4 ms | 0.21 ms |
| 系统内存、主机可见、一致、缓存 | 0.29 ms | 0.30 ms |

`lrender_bench` 上，Sponza 1080p 经 Readback 路径交给 Vulkan Presenter，每帧的收尾（主要是这次读）要 230 ms 以上，GPU 因此闲下来降到 P8。离屏目标一样。窗口有了 SharedBuffer 路径可以绕开，离屏目标和没有 sync file 的设备绕不开。

**为什么在 tc 的职责之内。** 共享 buffer 用哪种内存是 tc 后端的事，lrender 碰不到内存类型。这一项只是让调用方说出主机怎么用这块内存，不增加新的资源种类。

**接口草案：**

```cpp
enum class HostAccess : std::uint8_t {
    Writes, // 主机写、设备读：上传、暂存（现在的选择）
    Reads,  // 设备写、主机读：回读
};

template <class T, Dim D = Dim::D1>
BufferResource<T, D> createBuffer(typename DimTraits<D>::IndexType extent, HostAccess access = HostAccess::Writes);
```

**后端：**

- Vulkan：`Reads` 选主机可见、一致、缓存的内存，不在设备本地的优先；没有这样的类型时退回现在的选择。缓存而不一致的类型不用，免得 `finish()` 和 `downloadBuffer` 还要 invalidate。
- GL、GLES：`glBufferStorage` 加上 `GL_CLIENT_STORAGE_BIT`，提示驱动把它放在主机内存里；只是提示，语义不变。
- CPU：不变。

**验收：** 各后端上设备写、主机读的结果正确，和 `Writes` 的 buffer 一样；RTX 5070 Ti 上主机读 8 MB 的 `Reads` buffer 不到 1 ms；lrender 的帧图把回读 buffer 改成 `Reads` 之后，Sponza 1080p 的离屏目标和 Readback 路径的收尾时间从 230 ms 降到几毫秒。

## stackfull：交给空闲 IO 工作线程的任务可能唤不醒它（已修正，`acb8861`）

**现象。** lrender 的 asset 测试偶尔挂住：debug 构建里约 60 次一次，ASan 下约 10 次一次，机器忙的时候更多。挂住时所有线程都在睡：主线程在 `sync_wait` 里等一次加载，两个池的线程在条件变量上，IO 工作线程在 `epoll_wait` 里。给每个线程发个信号把它们从系统调用里打断，测试就接着跑完了，所以是丢了一次唤醒。

**原因（`sched/src/Worker.cpp` 和 `SchedulerCore.cpp`，固定在 `67d087c`）。** 空闲的工作线程先 `markIdle`（在 `idleMask` 上置位），再看一次队列，然后 `sleepIdle()`：抢到 timekeeper 的那个睡在 Driver 里（`epoll_wait`，没有定时器时不设超时），其余的睡在自己的 Parker 上。生产者 `notifyIdleWorker()` 清掉一个空闲位，再 `unparkWorker()`：读到 `timekeeper` 就是这个线程时写 Driver 的 eventfd，否则只给它的 Parker 一个令牌。如果生产者清位、读 `timekeeper` 都发生在这个线程看完队列之后、抢到 timekeeper 之前，它读到的 timekeeper 不是这个线程，只给了 Parker 令牌；这个线程随后成为 timekeeper，在 `epoll_wait` 里一直睡下去，令牌没人看。

**修正。** 在 `sleepIdle()` 里，成为 timekeeper 之后、进 Driver 之前，再读一次自己的空闲位（和生产者一样用 `idleMask` 上的 seq_cst RMW）：已经被清掉，说明有生产者认领了自己，这次就不阻塞（超时取 0）。生产者是“清位，再读 timekeeper”，这里是“写 timekeeper，再读空闲位”，都是 seq_cst，所以至少有一方看到对方：要么生产者看到 timekeeper 写 eventfd，要么这里看到位被清掉。多出来的 Parker 令牌照旧只让下一次 `park()` 早返回一次。

```cpp
    gathering = true;
    // A producer that claimed us (cleared our idle bit) before we became the timekeeper
    // woke our Parker, not the Driver: do not block in the Driver then.
    if ((core.idleMask.fetch_or(0, std::memory_order_seq_cst) & (std::uint64_t{1} << index)) == 0) {
        timeout = std::chrono::nanoseconds{0};
    }
    if (core.driver != nullptr) {
        core.driver->wait(timeout);
```

**验证。** 一个只有一个 IO 工作线程的 runtime，从另一个线程连续交 2 万次任务、每次等它完成，进度停住 10 秒就算失败：没改的 stackfull 5 次里失败 2 次（第 1193 和 1569 次交接没有唤醒它）；改了之后 20 次都没有失败，lrender 的 asset 测试并行跑 400 次也没有再挂住。这个测试应当放进 stackfull 自己的测试里（`stackfull::runtime` 带 Driver、一个工作线程），lrender 的 `RuntimeTest` 也加一份，等 lrender 固定到修正后的提交再加，免得现在的 CI 时红时绿。

**实现。** stackfull 的 `acb8861` 按上面的写法修正，`IoTest.TasksSpawnedFromOutsideWakeTheWorkerSleepingInTheDriver`（一个带 Driver 的工作线程，从外面连续派生 2 万个任务）在修正前每次都在几百次之内挂住，修正后 epoll 和 poll 两种 Poller 上都通过，stackfull 的 308 个测试在 debug、asan、tsan 下都通过。lrender 固定到这个提交，`RuntimeTest` 加了同样的测试；asset 测试连跑 100 次不再挂住。

**为什么在 stackfull 的职责之内。** 调度器的唤醒协议是 stackfull 的；lrender 只是通过 `lr::runtime` 用它，绕不开。
