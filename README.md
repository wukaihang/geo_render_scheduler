# Geo Render Scheduler

面向单机多 GPU 三维地质体绘制的内容与运行状态感知耗时预测、预计完成时间（EFT）调度实验代码。当前版本实现论文“创新点一”，并把真实 VTK/EGL/NVML 依赖隔离在硬件适配器边界之外。

## 当前实现

- 模型、视角、渲染参数、设备状态和在线历史五组外部特征；
- Global Mean、EWMA、Ridge、Mean GBDT 和 P50/P95 Quantile GBDT 五类预测器；
- 模型、视角、设备与在线历史特征组的自动化消融模型及测试集误差；
- Round Robin、Least Queue、Static Weighted、EWMA-EFT、Feature-EFT、Oracle-EFT 六种策略；
- 按完整轨迹分组的训练/验证/测试切分；
- 固定种子开放式泊松请求流和确定性离散事件回放；
- 预测误差、P50/P95/P99、SLO、slowdown、Jain 指数、GPU busy ratio 与剩余工作量差；
- 自包含实验目录和可追溯的模型工件。

创新点一假定模型已经驻留在两张 GPU 上，不实现显存加载、缓存、淘汰或热点复制；这些属于创新点二。

## 安装

推荐 Python 3.10 或 3.11；代码同时兼容当前验证环境的 Python 3.9。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

## 可运行的无硬件闭环

生成不可变轨迹：

```bash
.venv/bin/python -m geo_render generate-trace \
  --config configs/synthetic_example.json \
  --output build/profile.csv \
  --requests 120 \
  --seed 20260928
```

按轨迹分组训练并评价五类预测器及四组特征消融：

```bash
.venv/bin/python -m geo_render train \
  --config configs/synthetic_example.json \
  --trace build/profile.csv \
  --output build/models
```

在完全相同的请求轨迹上比较六种调度策略：

```bash
.venv/bin/python -m geo_render compare \
  --config configs/synthetic_example.json \
  --output build/example-results
```

每个策略目录包含 `config.json`、`trace.csv`、`requests.csv`、`decisions.jsonl`、`summary.json` 和 `metadata.json`。顶层 `comparison.json` 汇总所有策略。

## 数据边界与真实性

示例配置生成的数据始终标记为 `source=synthetic`。它只用于验证实现、确定性和实验管线，不能作为 RTX 5090 性能证据，也不能写入论文结果。

真实画像 CSV 使用同一 `trace.csv` 模式：每行包含请求/相机/渲染参数、每张 GPU 的实际驻留命中 `render_ms` 与 `readback_ms`、CPU `encode_ms`、来源和 isolated P50。连续相机帧必须共享 `trajectory_id`，训练切分按整个轨迹完成。

未知遥测在输入对象中使用 `null`；特征工程会同时生成缺失指示器，不会把未知值解释成真实的零利用率。

## 硬件接入占位边界

```bash
.venv/bin/python -m geo_render check-hardware
```

在当前实现中该命令以状态码 2 明确失败，并列出 Linux、NVIDIA、VTK EGL、NVML 与 GPU 绑定要求。真实双 RTX 5090 机器需要实现：

- `src/geo_render/rendering/interface.py` 中的 `Renderer`；
- `src/geo_render/rendering/interface.py` 中的 `DeviceStateProvider`。

适配器必须在创建第一个 VTK RenderWindow 前固定 EGL 设备，输出 PID、目标设备、OpenGL renderer 和 GPU UUID，并用 `nvidia-smi`/NVML 验证两个常驻进程确实落在不同 GPU。禁止通过修改模拟配置伪造实测结果。
