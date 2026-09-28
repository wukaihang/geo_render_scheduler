# 创新点一实施计划

> **供智能体执行：** 使用 `subagent-driven-development`（推荐）或 `executing-plans` 按任务实施。复选框用于记录完成状态。

**目标：** 构建可复现、与硬件解耦的内容与运行状态感知单 GPU 耗时预测，以及基于预计完成时间（EFT）的多 GPU 调度实现。

**架构：** 使用稳定的数据类和协议隔离请求、设备、结果与硬件契约。预测和调度模块保持纯净且可测试；确定性离散事件回放提供完整实验闭环，但模拟值不得宣称为 RTX 5090 实测结果。

**技术栈：** Python 3.9+、NumPy、scikit-learn、joblib、pytest，以及标准库 argparse、CSV 和 JSON。

## 全局约束

- 创新点一假定每个模型已驻留在每张 GPU 上，不包含加载、淘汰、复制或缓存感知评分。
- EFT 和排队工作量使用 P95 渲染耗时预测；P50 仅用于预测评价。
- 只有已完成请求可以更新在线历史。
- 训练集、验证集和测试集按完整会话或轨迹分组切分。
- 模拟结果必须标记 `source=synthetic`；真实硬件后端在 Linux EGL/NVML 适配器完成前必须明确失败。
- 所有策略使用相同的不可变轨迹和稳定 GPU 顺序。
- 未知遥测使用 `None`，不得伪造为零。

---

### 任务 1：包结构、领域类型与硬件边界

**文件：**
- 创建：`pyproject.toml`、`README.md`
- 创建：`src/geo_render/common/*.py`
- 创建：`src/geo_render/rendering/*.py`
- 测试：`tests/unit/test_types.py`、`tests/unit/test_hardware_boundary.py`

**接口：**
- 产出不可变数据类 `Camera`、`RenderRequest`、`ModelManifest`、`DeviceState`、`DurationPrediction`、`QueuedRequest`、`WorkerSnapshot`、`CostEstimate`、`ScheduleDecision`、`RenderResult` 和 `TraceRecord`。
- 产出 `Renderer.render(request, gpu_id) -> RenderResult` 与 `DeviceStateProvider.snapshot() -> tuple[DeviceState, ...]` 协议。
- 产出 `HardwareBackendUnavailable` 及始终给出可操作平台要求的无硬件适配器。

- [x] 编写数据校验和硬件边界失败测试。
- [x] 运行测试，确认模块缺失时测试为红。
- [x] 添加打包配置、受校验的数据类、协议和无硬件适配器。
- [x] 运行任务测试，确认全部通过。
- [x] 提交 `feat: add innovation one domain contracts`。

### 任务 2：特征提取、在线历史与防泄漏切分

**文件：**
- 创建：`src/geo_render/prediction/features.py`
- 创建：`src/geo_render/prediction/online_stats.py`
- 创建：`src/geo_render/prediction/dataset.py`
- 测试：`tests/unit/test_features.py`、`tests/unit/test_online_stats.py`、`tests/unit/test_dataset.py`

**接口：**
- 产出 `extract_features(request, manifest, device, history) -> dict[str, float | str]`。
- 产出带 `observe(...)` 与分层 `estimate(...)` 的 `OnlineDurationStats`。
- 产出按完整分组返回三个互斥数据分区的 `group_split(...)`。

- [x] 编写几何特征、更新顺序和数据泄漏测试。
- [x] 实现纯函数特征、仅完成请求更新的在线统计及分组切分。
- [x] 历史回退顺序固定为 `(gpu, model) -> gpu -> global -> configured default`。
- [x] 运行任务测试并提交 `feat: extract render features and online history`。

### 任务 3：耗时预测器、评价与工件持久化

**文件：**
- 创建：`src/geo_render/prediction/interface.py`、`baselines.py`、`models.py`、`artifacts.py`
- 创建：`src/geo_render/analysis/prediction_metrics.py`
- 测试：`tests/unit/test_predictors.py`、`tests/unit/test_prediction_metrics.py`

**接口：**
- 产出统一的 `DurationPredictor.predict(features) -> DurationPrediction`。
- 产出 `GlobalMeanPredictor`、`EWMAPredictor`、`RidgeDurationPredictor`、`MeanGBDTPredictor` 和 `QuantileGBDTPredictor`。
- 产出版本化的 `save_artifact(...)`、`load_artifact(...)` 与 `prediction_metrics(...)`。

- [x] 编写预测约束、持久化往返和指标测试。
- [x] 使用 `DictVectorizer`、Ridge 和梯度提升实现预测器。
- [x] 将预测值限制为有限正数，并强制 `p95 >= p50`。
- [x] 工件保存特征模式版本、训练数据哈希、依赖版本和模型。
- [x] 运行任务测试并提交 `feat: add duration prediction models`。

### 任务 4：Worker 状态机与六种调度策略

**文件：**
- 创建：`src/geo_render/scheduling/interface.py`、`state.py`、`policies.py`
- 测试：`tests/unit/test_scheduler_state.py`、`tests/unit/test_policies.py`

**接口：**
- 产出 `SchedulerContext`、`SchedulerPolicy`、`WorkerState` 和 `ClusterState`。
- 产出 Round Robin、Least Queue、Static Weighted、EWMA-EFT、Feature-EFT 和仅离线可用的 Oracle-EFT。

- [x] 编写状态转换、最低 ECT 决策和 Oracle 隔离测试。
- [x] 排队时保存 P95，运行中剩余时间由预测完成时刻与 `now_ms` 计算。
- [x] 相同代价按 GPU ID 稳定打破平局。
- [x] 在线请求不得暴露 `actual_render_ms_by_gpu`。
- [x] 运行任务测试并提交 `feat: implement EFT and baseline schedulers`。

### 任务 5：确定性轨迹生成与离散事件回放

**文件：**
- 创建：`src/geo_render/workload/trace.py`、`synthetic.py`、`replay.py`
- 测试：`tests/unit/test_trace.py`、`tests/integration/test_replay.py`

**接口：**
- 产出 `generate_synthetic_trace(config)`、`read_trace_csv`、`write_trace_csv` 与 `ReplayEngine.run(trace, policy)`。
- 产出不可变的已完成请求记录和决策事件记录。

- [x] 编写固定种子确定性、策略公平性和未来信息隔离测试。
- [x] 泊松到达使用 `random.Random(seed).expovariate(arrival_rate_per_ms)`。
- [x] 事件按 `(time_ms, completion_before_arrival, sequence)` 排序。
- [x] 完成事件只更新所选在线预测器或历史观察器，再启动该 GPU 队首请求。
- [x] 生成轨迹和回放元数据强制使用 `source="synthetic"`。
- [x] 运行任务测试并提交 `feat: add deterministic workload replay`。

### 任务 6：调度指标与自包含实验输出

**文件：**
- 创建：`src/geo_render/analysis/scheduling_metrics.py`
- 创建：`src/geo_render/experiments/io.py`、`compare.py`
- 测试：`tests/unit/test_scheduling_metrics.py`、`tests/integration/test_experiment_outputs.py`

**接口：**
- 产出 `scheduling_metrics(result) -> dict[str, object]`。
- 产出 `write_run_directory(...)` 与 `compare_policies(...)`。

- [x] 编写指标及实验目录完整性测试。
- [x] 实现延迟、SLO、利用率、公平性与工作量均衡指标。
- [x] 百分位数使用 NumPy 线性方法；缺少 isolated P50 时 slowdown 指标为 `null`。
- [x] 通过同级临时目录和最终重命名实现原子写入，拒绝覆盖已有输出目录。
- [x] `metadata.json` 记录来源、种子、轨迹 SHA-256、依赖版本和可用的 Git 提交号。
- [x] 运行任务测试并提交 `feat: report innovation one experiment metrics`。

### 任务 7：CLI、示例配置、文档与端到端验证

**文件：**
- 创建：`src/geo_render/cli.py`、`src/geo_render/__main__.py`
- 创建：`configs/synthetic_example.json`
- 修改：`README.md`
- 测试：`tests/integration/test_cli.py`

**接口：**
- 产出 `geo-render generate-trace`、`geo-render train`、`geo-render compare` 与 `geo-render check-hardware`。

- [x] 编写 CLI 冒烟测试。
- [x] `generate-trace` 生成并哈希冻结 CSV。
- [x] `train` 按组切分画像数据，保存所有预测器工件及指标。
- [x] `compare` 在相同轨迹上运行六种策略并生成顶层汇总。
- [x] `check-hardware` 在真实适配器缺失时以状态码 2 退出。
- [x] README 明确区分模拟验证和论文证据，并说明硬件扩展入口。
- [x] 运行 CLI 测试、完整测试、策略比较和硬件失败检查。
- [x] 审计未实现标记并提交 `feat: complete innovation one experiment workflow`。

### 任务 8：完工审计与缺口闭合

**文件：**
- 修改：预测、特征、训练、调度、数据类型与模拟轨迹模块。
- 修改：`README.md` 与创新点一设计文档。
- 测试：预测器、特征、策略、类型与轨迹单元测试。

**接口：**
- 补充 `MeanGBDTPredictor` 和四种命名的 Quantile GBDT 特征消融工件。
- 强化 Oracle 对 render/readback/encode 全阶段真实耗时的使用，以及 Least Queue 对运行中任务的计数。

- [x] 添加遗漏预测器、消融、队列和 Oracle 要求的失败测试。
- [x] 实现 Mean GBDT、特征组消融及工件参数元数据。
- [x] 修正 Least Queue 与全阶段离线 Oracle 语义。
- [x] 重新运行 CLI 工作流、静态检查、编译检查和完整测试集。
- [x] 提交完整实现。
