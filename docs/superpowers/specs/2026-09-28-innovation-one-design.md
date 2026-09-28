# 创新点一：内容与运行状态感知的耗时预测及 EFT 调度设计

## 目标

实现文档中“创新点一”的可复现实验闭环：从不可变请求轨迹和单请求性能画像中提取外部可观测特征，训练或在线更新每 GPU 渲染耗时预测器，用预测的 P95 服务时间维护队列剩余工作量，并比较 Round Robin、Least Queue、Static Weighted、EWMA-EFT、Feature-EFT 和 Oracle-EFT 对 P50/P95/P99、SLO、吞吐量及负载均衡的影响。

当前开发机没有 NVIDIA GPU、VTK、NVML 或 Linux EGL。真实渲染及设备遥测通过窄接口隔离；接口在缺少硬件后端时必须抛出可诊断异常，不能生成伪装成实测值的数据。算法、回放、指标和 CLI 使用确定性模拟后端完整运行。

## 范围

### 包含

- 五类特征：模型、视角、渲染参数、设备状态和在线历史。
- 五类预测方法：按模型/GPU 的历史均值、按模型/GPU 的 EWMA、岭回归、均值梯度提升、P50/P95 分位数梯度提升。
- 模型、视角、设备和在线历史四组特征的自动消融训练与独立测试集评价。
- 六类调度策略：Round Robin、Least Queue、Static Weighted、EWMA-EFT、Feature-EFT 和仅供离线回放的 Oracle-EFT。
- 每 GPU 运行中请求与排队请求的显式状态，以及基于 P95 的剩余工作量估计。
- 按轨迹/会话分组的数据切分，防止连续帧泄漏。
- 固定种子的开放式泊松轨迹生成、离散事件回放和相同轨迹的策略对比。
- 预测及调度评价指标、CSV/JSON 输入输出、模型持久化和命令行入口。
- 可替换的渲染器与设备状态提供者协议，以及明确的无硬件后端。

### 不包含

- 创新点二的模型加载、显存缓存、淘汰、热点复制和执行前重分派。
- 真实 VTK/EGL 双 GPU 绑定实现；其构造入口和数据契约保留，待 Linux 双 RTX 5090 机器补齐。
- GemPy 模型生成、HTTP 服务、Parquet 日志和论文作图。这些不是验证创新点一算法闭环的必要依赖。
- 人为生成或填充任何 RTX 5090 性能结论。

## 架构与模块

项目使用 `src` 布局，核心模块保持硬件无关：

```text
src/geo_render/
├── common/       # 稳定数据对象、异常和序列化
├── rendering/    # Renderer / DeviceStateProvider 协议及无硬件实现
├── prediction/   # 特征、在线统计、预测器、训练和评价
├── scheduling/   # worker 状态、统一策略接口及六种策略
├── workload/     # 轨迹生成、分组切分和回放
├── analysis/     # 预测、延迟、公平性和队列指标
└── cli.py        # profile/train/replay/compare 命令
```

`RenderRequest`、`ModelManifest`、`DeviceState`、`WorkerSnapshot`、`CostEstimate`、`RenderResult` 和 `TraceRecord` 是跨模块稳定接口。时间以毫秒表示，运行时事件用相对单调时间；持久化数据同时允许保存 ISO 8601 墙上时间作为元数据。

## 数据契约

### 请求与模型

`RenderRequest` 保存请求、用户、会话、轨迹、模型、相机、裁剪比例、输出尺寸、采样步长、阴影、传递函数、到达时间和可选 SLO。相机包含位置、焦点、向上方向、投影类型和视野角。

`ModelManifest` 保存体素维度、有效体素比例、字节数、地层类别数、世界坐标包围盒和校验值。所有比例字段限制在 `[0, 1]`，尺寸、字节数和采样步长必须为正数。

### 设备与结果

`DeviceState` 保存逻辑 GPU ID、型号、显存、利用率、显存利用率、温度、功耗和相对历史速度。未知的硬件遥测用 `None`，不能用零伪装。

`RenderResult` 保存 GPU、开始/结束时间，以及 render/readback/encode 三阶段耗时。创新点一默认模型已驻留，所以不包含加载和淘汰决策。

### 调度代价

每个候选 GPU 生成一个 `CostEstimate`：

```text
ECT = running_remaining_ms
    + sum(queued_predicted_p95_ms)
    + candidate_predicted_p95_ms
    + predicted_readback_ms
    + predicted_encode_ms
```

策略返回所选 GPU、所有候选代价和机器可读原因。相同代价按 GPU ID 稳定打破平局，以保证重复实验一致。

## 特征工程与预测

特征提取是纯函数。视角特征包括相机到包围盒中心距离、单位视线向量、归一化投影面积近似、裁剪体积比例和投影类型；模型和渲染参数直接来自请求与 manifest；设备特征来自候选 GPU 的最新状态；历史特征来自仅含已完成请求的在线统计表。

分类字段采用显式字典和 one-hot 编码。训练器把完整轨迹/会话作为不可分组，先按组划分训练、验证和测试，再拟合预处理器和模型。测试组在拟合期间不可见。

统一 `DurationPredictor` 接口输出 `DurationPrediction(p50_ms, p95_ms, model_version)`。预测必须为有限正数并满足 `p95 >= p50`：

- GlobalMean：按 GPU 和模型回退；层级为 `(gpu, model)`、`gpu`、全局。
- EWMA：只由已完成结果更新，同样使用分层回退，并维护近期样本的经验 P95。
- Ridge：统一模型显式包含 GPU 特征，输出 P50；验证集残差分位数用于构造 P95。
- Mean GBDT：以平方误差训练均值模型，并用验证集上界残差构造 P95。
- Quantile GBDT：分别以 0.50 和 0.95 分位数损失训练两个模型。

模型工件保存训练配置、特征模式版本、依赖版本、训练数据 SHA-256 和预测器本体。加载时校验模式版本。

## 调度策略

所有策略实现同一个 `SchedulerPolicy.choose(request, context)` 接口：

- Round Robin：按稳定 GPU 顺序循环。
- Least Queue：选择运行中与排队请求总数最少的 GPU，并按 GPU ID 稳定打破平局。
- Static Weighted：使用离线给定的相对服务率，通过最小化 `(running + queued + 1) / rate` 分派。
- EWMA-EFT：使用 EWMA 的 P95 估计计算 ECT。
- Feature-EFT：使用内容、设备与历史特征预测器的 P95 计算 ECT；这是创新点一的主策略。
- Oracle-EFT：只在离线回放中读取轨迹内真实 render、readback 和 encode 时间，不允许实时入口构造。

运行中剩余时间以 `max(0, predicted_finish - now)` 计算。排队请求保存其在目标 GPU 上决策时的 P95 预测值；请求开始或完成时状态机校验请求 ID 和 GPU 一致，非法转换立即报错。

## 回放与实验公平性

确定性离散事件模拟器按 `(事件时间, 事件类型优先级, 请求 ID)` 排序。到达事件调用策略；每张 GPU 同一时刻最多运行一个请求；完成事件更新在线统计并启动队首请求。实际服务时间由轨迹记录提供，或由带固定随机种子的 `SyntheticRenderer` 根据内容特征和 GPU 速度生成。

所有策略必须读取同一份冻结轨迹。轨迹生成器支持泊松到达、多个用户/会话、模型混合、分辨率和采样步长组合；输出包含真实服务时间，供模拟执行和 Oracle 使用。模拟值必须在元数据中标记 `source=synthetic`，与将来的硬件实测数据分开。

## 指标与输出

预测评价输出 MAE、MAPE、P95 绝对误差、低估比例、P95 覆盖率和单次预测耗时。调度评价输出完成数、吞吐量、P50/P95/P99 端到端时间、P50/P95/P99 queue time、SLO 违约率、每用户 slowdown、Jain 公平指数、两 GPU busy ratio 和平均/最大预计工作量差。

每次运行创建自包含目录，至少保存 `config.json`、`trace.csv`、`requests.csv`、`decisions.jsonl`、`summary.json` 和 `metadata.json`。元数据记录随机种子、数据哈希、策略和模型版本；若目录不属于 Git 仓库，代码提交号显式为 `null`。

## 错误处理

- 数据验证错误包含字段路径和非法值。
- 未拟合预测器拒绝预测；未知类别走固定的 unknown 编码，不重新拟合。
- 预测值非有限、非正或 P95 小于 P50 时立即拒绝调度。
- 空 worker 集合、重复请求 ID、时间倒退和状态机不一致立即失败。
- 真实硬件后端在当前实现中抛出 `HardwareBackendUnavailable`，消息列出所需平台、依赖和待实现适配器；CLI 返回非零退出码。
- 单个实验配置失败时写入错误元数据，策略批量比较继续运行其余配置并在最终摘要列出失败项。

## 测试与验收

单元测试覆盖所有数据约束、几何特征、EWMA 更新与回退、分组切分无泄漏、预测器输出约束、模型保存/加载、队列状态机、六种策略和平局规则。集成测试使用一份小型冻结轨迹，验证：

1. 相同种子生成完全相同的轨迹；
2. 所有在线策略无法读取未来实际耗时；
3. Oracle 只能由离线回放创建；
4. Feature-EFT 在构造的异质请求序列中选择最小 ECT；
5. 每种策略完成相同请求集合且输出所需日志字段；
6. 当前无 GPU 环境下算法闭环成功、真实硬件入口明确失败；
7. CLI 可生成轨迹、训练模型、比较策略并产生自包含结果目录。

验收不以模拟数据证明论文结论。模拟闭环只证明实现的确定性、接口和算法逻辑；论文数值必须替换为双 RTX 5090 上按 E0/E1 采集的真实画像与轨迹结果。
