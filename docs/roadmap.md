# 接下来的工作

从现在的 lrender（`a00f614`）走向高性能游戏渲染引擎要做的事，按阶段排好。每一阶段的工作项大体按优先级排列；同一阶段里没有依赖关系的可以并行。“现状与证据”指出代码里的位置或已有的测量，“目标与验收”是做完的判据，通用的验证要求（三套预设、跨后端比对、验证层、性能测量的方法）见 `next-phase-prompt.md`，这里不重复。

规模是一个 agent 的大致工作量：S 不到一天，M 两到四天，L 一到两周，XL 两周以上。“库边界”一栏说明要不要依赖库新增能力：“无”表示 lrender 自己能做；需要的能力按 `library-requirements.md` 的写法先提出，经同意再动依赖库。

## 第 0 阶段：基准与测量

后面每一项的收益都要拿这里的结果来证明，所以最先做。已经完成：场景、方法和基线见 `benchmarks.md`。B1 用 Khronos 的 Sponza 和 ABeautifulGame 代替了 Intel Sponza 和 Bistro，原因在那里说明。测基线时还发现渲染线程每帧都把世界里的所有 id 复制一遍（100 万个实例时每帧 33 ms），已经修正。

| 编号 | 工作项 | 现状与证据 | 目标与验收 | 涉及模块 | 依赖 | 库边界 | 规模 |
|---|---|---|---|---|---|---|---|
| B1 | 基准场景 | 现有的测量都是小场景（几百个实例、几千个三角形，`VisibilityPipelineTest` 里的地面加 400 个小球）。没有能代表游戏的场景 | 接入 Intel Sponza、Amazon Bistro（外景和内景）两个 glTF 场景，不放进仓库，由脚本下载到构建目录并校验哈希；再加一个合成的规模场景：10 万到 100 万个实例、几百盏点光源、可调的三角形密度 | asset、examples | 无 | 无 | M |
| B2 | 基准程序 | 只有 `layout_probe` 和 `lrender_cli --timers on`；GPU 在测试循环里常停在 P5、P8，同一个 pass 的时间能差到六倍（`hardware-raster.md` 的实测） | `probes/bench`（名字在设计里定）：release 构建，先持续渲染到 GPU 频率稳定再计时，A/B 两种配置在同一个进程里隔帧交替；输出 JSON：每个 pass 的设备时间、GPU 帧时间、CPU 帧时间（渲染线程准备、GPU 线程录制）、`graphBytes`、几何和纹理显存。同一配置连跑三次，主要指标的差别在 3% 以内 | probes、render | B1 | 无 | M |
| B3 | CPU 侧计时 | `FrameResult::wallTime` 只有提交到完成的总时间，看不出渲染线程和 GPU 线程各花了多少 | `FrameResult` 增加渲染线程准备和 GPU 线程录制的时间；基准程序报告它们 | render | 无 | 无 | S |
| B4 | 基线报告 | 无 | 在 RTX 5070 Ti 和 lavapipe 上，用 B2 测 B1 的场景，写进 `docs/benchmarks.md`，作为后面所有工作项的对照 | docs | B1–B3 | 无 | S |

## 第 1 阶段：大场景下的性能结构

决定大场景能不能跑起来。P1 已经完成（`lr::present_vulkan`，测量见 `benchmarks.md`），只差在真实窗口上看一次；P2 已经完成，100 万个实例时压缩从 15.1 ms 降到 0.084 ms；P3 已经完成，簇是网格三角形按索引顺序的连续区间，簇表和场景显存由 `FrameResult::sceneMemory` 报告；P4 的簇级视锥和背面剔除已经完成（逐像素与不剔除时相同），两遍 HiZ 遮挡剔除在做；测 P1 时发现离屏目标和 Readback 路径的回读在 NVIDIA 上要两百多毫秒，向 tc 提了第 5 项。

| 编号 | 工作项 | 现状与证据 | 目标与验收 | 涉及模块 | 依赖 | 库边界 | 规模 |
|---|---|---|---|---|---|---|---|
| P1 | 桌面零拷贝上屏 | 桌面只有 Readback 路径：`SdlPresenter` 每帧把整张图读回主机再传进 SDL 纹理（1080p 约 8 MB）。`architecture.md` 的“目前只有 Readback 路径”一段已经为 SharedBuffer 路径留了位置 | 新模块（如 `lr::present_vulkan`）用自己的 VkDevice 建交换链，分配可导出的内存；tc 用 `importBuffer` 导入，最后一个 pass 直接写进去；帧末用 sync file 交接。`Presenter` 加上 SharedBuffer 路径，现有 Presenter 不改。验收：1080p 不再回读；与 Readback 路径的图像逐像素相同；在途帧之间不互相等待；B2 报告的帧时间和延迟 | present、render、gpu | B2 | tc 已有 `importBuffer`、GPU UUID、sync file。交换链和窗口表面在 present 模块里，不向 tc 要 | L |
| P2 | 并行压缩可见实例 | `CompactVisible` 只用一个工作组，按 256 个一批串行扫完所有实例（`Features.cpp` 里的 `dispatch(kernel, {1,1,1})`）。基线：100 万个实例时每帧四次压缩共 15.1 ms，是 GPU 帧的 47% | 多工作组的前缀和（先分块求和，再扫块和，再写出；不用依赖前向进度保证的 decoupled look-back，因为不是每个设备都保证）。可见实例仍按实例顺序编号，因为“同一深度取编号最小的三角形”依赖这个顺序，所以可见性缓冲逐位不变。验收：现有的子组和共享内存两种变体都换掉，比对测试照旧通过；100 万个实例时压缩的时间 | render（Visibility.h、Features.cpp） | B2 | 无 | M |
| P3 | 簇（meshlet）数据 | 网格只有顶点和索引（`core::MeshData`），剔除和光栅化都以实例为单位 | 加载或 upsert 时把网格切成簇（每簇不超过 64 个顶点、124 个三角形），算出包围盒和法线锥；`RenderWorld` 和 `DeviceScene` 增加簇表，几何回收（P6）要一并考虑。验收：切簇在帧池上按网格并行，结果确定；簇表的显存在 B2 里报告 | core、render、asset | 无 | 无 | M |
| P4 | 簇剔除与两遍 HiZ 遮挡剔除 | 只有实例级的视锥剔除（`CullInstances`），没有遮挡剔除；一个大网格只要露出一角，所有三角形都进光栅化。基线：Sponza 的三级阴影占 GPU 帧的 57%（lavapipe 上 81%），因为每一级都光栅化整个场景；规模场景 100 万个实例时可见性光栅化 16.3 ms，可见三角形 7366 万个，大部分被挡住 | 实例剔除之后再按簇做视锥、背面锥和 HiZ 剔除。两遍：先画上一帧可见的簇，建 HiZ，再测试其余的簇并画通过的。光栅化 kernel 改为读簇列表；阴影级联同样按簇剔除，阴影不做遮挡剔除。验收：与不剔除时的可见性缓冲逐位相同（遮挡剔除只能去掉确实看不见的簇）；Bistro 上可见三角形数和光栅化时间的变化 | render | P2、P3 | 无 | L |
| P5 | 分簇光照 | 着色时每个像素循环所有灯（`SurfaceShading.h` 第 265 行附近的 `for (uint i = 0u; i < lightCount.value; ++i)`），代价是像素数乘以灯数 | 一个 compute pass 把灯分进视锥体素（froxel）的列表，着色只看本格子的灯；没有 `range` 的点光源和方向光留在全局列表。路径追踪器不变。验收：与逐灯循环的图像相同（有限范围的灯在范围外贡献为零，所以应当逐位相同，做不到时说明原因并给出容差）；灯数从 1 到 1000 时着色时间的变化 | render | B2 | 无 | M |
| P6 | 几何内存回收 | 顶点表和索引表只往后追加，被替换或删除的网格不回收（`RenderWorld.h` 的注释明说了）；软件 BVH 的节点同样只增不减 | 顶点、索引（以及 P3 的簇、软件 BVH）用子分配器管理，删除的网格归还空闲区间，必要时整理；`AccelerationScene` 和设备丢失后的全量更新跟着改。验收：反复加载和卸载网格 1000 次，表的容量稳定；现有测试全部通过 | render | P3 | 无 | M |
| P7 | 场景规模压力测试 | 没有超过几百个实例的测试。基线：GPU 线程每帧录制 0.7 到 1.1 ms，几乎与场景大小无关（lavapipe 上 0.3 ms） | 用 B1 的规模场景把实例数推到 100 万、灯推到 1000，找出 CPU 和 GPU 两侧的下一个瓶颈（实例表上传、帧图每帧重建、TLAS 构建等），按测量结果补工作项 | 全部 | P2、P4、P5 | 无 | M |

## 第 2 阶段：现代游戏的画面

| 编号 | 工作项 | 现状与证据 | 目标与验收 | 涉及模块 | 依赖 | 库边界 | 规模 |
|---|---|---|---|---|---|---|---|
| F1 | 基于图像的环境光（IBL） | `core::EnvironmentDesc` 只有一个均匀的 `radiance`；镜面部分用环境 BRDF 表积分均匀天空 | 支持 HDR 环境图（等距柱状投影，KTX2 或 `.hdr`）：compute 预滤波出镜面的 mip 链，漫反射用二阶球谐；均匀环境作为特例保留。路径追踪器按亮度重要性采样环境图。验收：白炉测试照旧通过；可见性缓冲管线和路径追踪器在环境图下的比对 | core、asset、render | 无 | 无 | L |
| F2 | 环境光遮蔽 | 没有 | GTAO（半分辨率加时域累积）；路径追踪器作为参照比对 | render | B2 | 无 | M |
| F3 | 泛光、自动曝光、颜色分级 | 后处理只有 TAA 和色调映射 | 泛光（逐级降采样再升采样）、直方图自动曝光（随时间适应）、3D LUT 颜色分级、可选锐化 | render | 无 | 无 | M |
| F4 | 聚光灯和更多阴影 | 灯只有方向光和点光源；只有第一盏投影的方向光有阴影（`RenderWorld::shadowLight`） | 加聚光灯；点光源和聚光灯的阴影放进一张阴影图集，按屏幕重要性分配分辨率；静态物体的阴影缓存。验收：与路径追踪器的阴影比对 | core、render | P5 | 无 | L |
| F5 | 骨骼动画与形变目标 | `MeshData` 只有位置、法线、UV，没有骨骼权重、形变目标、切线、顶点色；asset 不读 glTF 的 skin 和 animation | asset 读 skin、animation、形变目标；scene 有动画的求值（可以由脚本驱动）；render 加一个 compute 蒙皮 pass，蒙皮后的位置供光栅化、阴影和光追使用，变形网格输出自己的运动矢量；ray query 路径对蒙皮网格的 BLAS 做更新（refit）。验收：蒙皮结果与主机上的参照计算比对；TAA 下运动的角色没有拖影 | core、asset、scene、render | P3（簇的包围盒要随形变更新） | tc 已有 BLAS 构建；refit 若 tc 没有，先提出 | XL |
| F6 | 材质扩展和各向异性采样 | 纹理 LOD 是各向同性的，斜看时发糊（`architecture.md`）；只有 metallic-roughness | 多次采样的各向异性过滤；glTF 的 clearcoat、transmission、sheen、emissive_strength 等扩展作为材质模型 | render、asset | 无 | 无 | L |
| F7 | 透明的容量 | 片元链表平均每像素两个片元的容量，满了就丢（`kTransparentFragmentsPerPixel`） | 容量按上一帧的用量调整，溢出时退到加权混合的 OIT；报告丢掉的片元数 | render | 无 | 无 | M |
| F8 | 其他游戏模块 | 没有 | 粒子、贴花、天空和大气散射、体积雾、调试用的 UI 和文字叠加；按需要排期，每个先写设计 | render | — | 无 | 各 M–L |

## 第 3 阶段：进一步的性能和光追

| 编号 | 工作项 | 现状与证据 | 目标与验收 | 涉及模块 | 依赖 | 库边界 | 规模 |
|---|---|---|---|---|---|---|---|
| R1 | 时域超分 | 只有同分辨率的 TAA | 视图按缩放比例渲染（50% 到 100%），由时域超分放大到目标分辨率（FSR2 一类的做法，全部是 compute），TAA 的运动矢量和历史复用。验收：B2 报告 67% 和 50% 下的帧时间；静止和运动画面与全分辨率的差别 | render | B2 | 无 | L |
| R2 | 光追反射和阴影 | ray query 只用在参考路径追踪器上 | 可见性缓冲管线里，Vulkan 上用 ray query 追踪反射和阴影（与光栅混合），GL、GLES 退到屏幕空间反射和阴影图 | render | F4 | 无 | L |
| R3 | 全局光照 | 没有 | 有 ray query 的设备上做 DDGI 一类的探针 GI；其他设备用烘焙的探针。路径追踪器作为参照 | render | R2 | 无 | XL |
| R4 | LOD | 没有 | 加载时用简化算法生成离散 LOD（或者簇的层次），按屏幕误差选择 | asset、render | P3、P4 | 无 | L |
| R5 | 纹理流式的优先级 | 新纹理按到达顺序上传，每帧有字节预算（`TexturePool`）；没有按距离或屏幕占比排序，也没有显存预算 | 按屏幕占比和 mip 需求排序，显存超出预算时降低远处纹理的 mip | render | B2 | 无 | L |
| R6 | 异步 compute 队列 | tc 只有一条 compute 队列和一条传输队列 | 只有当 B2 显示 GPU 有明显空闲时才做：按 `library-requirements.md` 的写法向 tc 提出第二条 compute 队列（和传输队列同一套 `waitFor` 语义），再让帧图把互不依赖的 pass 分到两条队列 | render、tc | B2、P7 | 需要 tc 新增，属于 compute 的职责 | L |

## 第 4 阶段：平台与工程化

| 编号 | 工作项 | 现状与证据 | 目标与验收 | 涉及模块 | 依赖 | 库边界 | 规模 |
|---|---|---|---|---|---|---|---|
| A1 | Android（M7） | 没有做（`architecture.md` 的里程碑） | `SurfaceControlPresenter` 零拷贝上屏、API 29 以下的回读上屏、生命周期；针对 tile 架构的 GPU 调优；在 Adreno 上重新评估 `hardware-raster.md` 的移动端做法 | present、render | P1 | 视测量而定 | XL |
| A2 | 打包 | install 规则和 `find_package(lrender)` 的包配置还没有，因为 lexec、stackfull、compute_shaders 没有 install 规则（只有 co2 有） | 依赖库补上 install 规则之后，每个可选层导出为一个组件，用一个外部的示例工程验证 | cmake | 依赖库 | 依赖库各自补 install 规则 | M |
| A3 | 探针和调试工具 | `probes/` 里只有 `layout_probe`；已有每个 pass 的计时和帧图导出 | `gpu_probe`、`present_probe`、`asset_probe`；常驻的统计叠加层（帧时间、pass 时间、显存）；确认 RenderDoc 能抓到 Vulkan 帧 | probes、render | B2 | 无 | M |
| A4 | 帧节奏 | 没有 vsync、延迟和帧节奏的控制 | 零拷贝上屏之后，加呈现模式（FIFO、Mailbox）、最大在途帧、延迟的测量 | present、render | P1 | 无 | M |

## 暂缓和不做的

- 桌面的硬件光栅：两个设备之间每帧多一次来回（实测 0.92 ms），比它能替换的块阶段（满频率时约 0.2 到 0.3 ms）还贵，见 `hardware-raster.md`。到 A1 时在移动端重新评估。
- mesh shader、可变速率着色：属于图形管线，超出 tc 只做 compute 的职责。
