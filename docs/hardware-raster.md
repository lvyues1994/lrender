# 硬件光栅（提案）

现在可见性缓冲和阴影贴图完全由计算光栅化生成。这份提案让有图形管线的设备把大三角形交给硬件光栅化，其余部分不变。还没有实现，定下来之后并入 `architecture.md`。

## 不放弃什么

`architecture.md` 的“放弃的方案”否掉了“在 tc 之外接一套 Vulkan 图形管线”，理由是会失去单一源码、CPU 调试和跨后端等价测试。这个方案把硬件光栅当作计算光栅化的加速器，三样都保留：

- 剔除、裁剪、分流、着色仍然是 tc kernel，新写的只有两段很短的 GLSL（顶点和片元）。
- CPU 后端和不满足条件的设备走纯计算路径，和现在一样。
- 硬件光栅模式与纯计算模式的可见性缓冲直接比对，见“验证”。

## 归属

光栅化管线超出 compute，不向 tc 要。硬件光栅由 lrender 的一个可选模块负责，做法和 Presenter 一样：render 定义接口 `HardwareRasterizer`，`lr::raster_vulkan` 用 Vulkan 实现它，宿主链接这个目标并通过 `RendererConfig` 注入，render 本身不依赖 Vulkan。着色器在构建时用 glslangValidator 编成 SPIR-V，tc 的构建本来就依赖它。

光栅模块在 GPU 线程上使用，有自己的 VkDevice，和 tc 的设备通过共享内存与 sync file 交接，与 tc 和相机、合成器交接的方式相同；tc 需要提供的只有 `library-requirements.md` 的第 4 项，tc 已经实现（`93f30dc`）。不和 tc 共用一个 VkDevice，是因为那需要 tc 公开后端句柄、命令缓冲和布局约定，并让外部命令进入它的屏障跟踪，侵入性太大。代价是每帧多一次设备之间的交接，开销要实测。

## 硬件只负责覆盖

1. **分流。** 光栅化 kernel（`LR_TRIANGLE_RASTERIZER`）做完近平面裁剪后，有每个三角形的屏幕坐标。包围盒大于阈值、材质不是 `Mask` 的三角形，kernel 不再逐像素扫描，而是把它追加进大三角形列表：每项是三个角的屏幕坐标和深度（`toScreen` 的结果）加上可见三角形号，用 `atomicAdd` 分配位置，同时累加 `DrawIndirectCommand` 的顶点数。列表满了，多出来的三角形照旧在 kernel 里扫描。阈值先取 32×32 像素，实测再定。
2. **硬件 pass。** 一次 `vkCmdDrawIndirect`，没有顶点输入，也没有附件。顶点着色器按 `gl_VertexIndex` 从列表里取角，输出它在 NDC 里的位置，z 固定为 0.5、w 为 1，三角形号作为 flat 变量传下去。深度不交给硬件，所以 z 不会被裁剪。片元着色器读出这个三角形的三个角，在像素中心用 kernel 里同样的公式算边函数和深度，对同一个打包的可见性缓冲做 `atomicMin(depth << 32 | 三角形)`。
3. **为什么结果不变。** `atomicMin` 与执行顺序无关。只要每个三角形覆盖的像素和算出的深度都与 kernel 一样，无论哪些三角形由硬件画，可见性缓冲都和纯计算的结果相同。
   - 覆盖：设备有保守光栅化（`VK_EXT_conservative_rasterization`）时，硬件给出的像素包含 kernel 会覆盖的所有像素，片元着色器再用 kernel 的覆盖判断丢掉多余的，包括“边上的像素中心两侧都算覆盖”这条规则，覆盖与 kernel 完全一致。本机 RTX 5070 Ti 有这个扩展，过估约 0.002 像素。没有保守光栅化时（lavapipe 就没有），用硬件自己的覆盖：本机两个设备的子像素精度都是 8 位，共享边上的采样点只归一个三角形，所以只在像素中心离边不到约 1/256 像素的地方与 kernel 不同。
   - 深度：片元着色器用 `precise` 禁止乘加融合，但 tc 生成的 GLSL 没有这个限定，驱动可能把 kernel 里的乘加融合成 FMA，深度的最低位就可能不同。能否做到逐位相同要实测，做不到时测试按容差比较。
4. **留在 kernel 里的：**
   - 小三角形，计算光栅化比硬件快的正是这部分。
   - `Mask` 材质，因为要采样纹理池，而纹理池在 tc 的设备上。
   - 透明片元链表。
   - 没有 64 位原子的设备：两遍的做法要在两个设备之间来回交接四次，不做。
5. **阴影。** 级联阴影贴图同样分流，硬件 pass 对阴影 buffer 做 32 位 `atomicMin`，和主视图的大三角形在同一次提交里画。

## 一帧

```
tc 第一段   剔除、压缩、主视图与各级联的光栅化 kernel（只画小三角形，写大三角形列表）   submitWithFence → F1
光栅设备    等 F1，画主视图和各级联的大三角形                                           提交，导出 F2
tc 第二段   waitFor(F2)，拆分可见性缓冲、材质分类、着色、透明、TAA、色调映射、回读       submit
```

- 帧图多一种硬件光栅 pass。它读写的资源（大三角形列表、绘制参数、打包的可见性缓冲、阴影 buffer）是共享 buffer：光栅模块分配并导出，tc 导入，资源池按“是否共享”分开复用。编译时把硬件 pass 尽量往后放，合并成一次光栅提交，所以一帧只多一次来回。
- 录制都在 GPU 线程上：遇到硬件 pass 时，tc 用 `submitWithFence()` 结束第一段；光栅模块等这个 fence、录制并提交，交回自己的 fence；tc 对它 `waitFor` 之后录第二段。主机不等设备。
- 一帧因此有两次 tc 提交。`FrameQueue` 等第二次，它在设备上已经等过光栅提交；两次提交的计时都能从第二次的 `takeTimings` 取到，光栅 pass 用光栅设备自己的时间戳计时。
- 两边都要能导出和导入 sync file。本机 NVIDIA 可以；lavapipe 没有 `VK_KHR_external_semaphore_fd`，只能在主机上等，GPU 线程会阻塞，所以只在测试里（`Required`）允许，`Auto` 在这种设备上不启用。
- 设备丢失：任何一个设备丢失，两个都重建，共享 buffer 重新分配并导入，其余照现有的恢复流程。

## 启用条件与配置

- `RendererConfig::hardwareRaster`（`VariantUse`：`Auto`、`Off`、`Required`），注入的 `HardwareRasterizer`，以及分流阈值 `hardwareRasterMinPixels`。
- 启用条件：注入了实现；tc 设备能导入外部 buffer，并且有 64 位原子；光栅设备与 tc 设备的 deviceUUID 相同，有 `fragmentStoresAndAtomics` 和 `shaderBufferInt64Atomics`；两边都有 sync file（`Required` 时可以没有）。
- `RendererCapabilities` 报告是否启用了硬件光栅，以及覆盖是否与 kernel 一致（也就是有没有保守光栅化）。

## 验证

- lavapipe（CI）：硬件光栅模式与纯计算模式比对可见性缓冲，三角形号不同的像素只能出现在三角形的边上，并且不超过一个比例上限；深度在容差之内；最终图像比对；大三角形列表溢出时结果不变；阴影。
- 本机 NVIDIA：开保守光栅化，与纯计算模式逐项比对；按 pass 计时，场景至少包括铺满屏幕的地面、近景的大三角形、密集的小三角形和阴影。
- CPU 后端和现有的金标准测试不受影响。

## 移动端（以后）

在片元着色器里做原子操作，会绕开 TBDR 的 tile 内存和隐藏面消除。移动端更合适的是传统做法：不透明、非 `Mask` 的三角形全部交给硬件，用深度附件和 R32UI 的三角形号附件做深度测试，画完把两个附件拷进共享 buffer（`VisibilityOutputs` 的格式）。这时深度来自硬件插值，和计算路径的差别比桌面模式大。等做 M7（Android）时在 Adreno 830 上实测再定。

## 与计算光栅化分块的关系

有硬件光栅的设备上，一个线程扫描整个大三角形的情况不再出现；但 GL、GLES、CPU 后端和不满足条件的设备仍然全靠 kernel，所以给光栅化 kernel 做分块（把大三角形按屏幕块分给多个工作组）仍然要做。两者共用同一个分流判断。
