# 人类可读内容全面中文化实施计划

> **供智能体执行：** 必须使用 `executing-plans` 或 `subagent-driven-development` 按任务执行。步骤使用复选框（`- [ ]`）跟踪。

**目标：** 将仓库内面向使用者和开发者的说明统一为中文，同时保持代码 API、命令、配置和实验数据格式兼容。

**架构：** 本次只修改展示层文本，不改变算法和数据流。Markdown 按原结构逐段翻译；Python 中仅翻译注释、文档字符串、CLI 帮助、异常和诊断文本，并同步调整依赖消息匹配的测试。

**技术栈：** Markdown、Python 3.9、argparse、pytest、Ruff、Git。

## 全局约束

- 保留 Python 标识符、模块路径、命令、参数、文件名和目录名。
- 保留 JSON/CSV 字段、配置键、环境变量和机器可读枚举值。
- 保留通用技术名称、正式策略名和代码块中的可执行内容。
- 不做语言转换以外的重构，不改变算法行为与实验数据格式。

---

### 任务 1：翻译 Markdown 文档

**文件：**
- 修改：`README.md`
- 修改：`docs/superpowers/specs/2026-09-28-innovation-one-design.md`
- 修改：`docs/superpowers/plans/2026-09-28-innovation-one.md`
- 检查：`docs/superpowers/specs/2026-09-28-chinese-human-facing-content-design.md`
- 修改：`docs/superpowers/plans/2026-09-28-chinese-human-facing-content.md`

**接口：**
- 输入：现有 Markdown 结构、命令和代码示例。
- 输出：正文、标题、提示语和步骤说明均为中文的等价文档。

- [x] **步骤 1：逐段翻译英文标题、正文、列表和执行说明**
- [x] **步骤 2：保留代码块、路径、标识符和正式技术名称**
- [x] **步骤 3：扫描 Markdown 中的英文句子并逐项复核**
- [x] **步骤 4：运行 `git diff --check`，预期无空白错误**

### 任务 2：翻译 Python 人类可读文本

**文件：**
- 修改：`src/geo_render/**/*.py`
- 修改：`tests/**/*.py`

**接口：**
- 输入：现有注释、文档字符串、`argparse` 帮助文本、异常消息和诊断输出。
- 输出：中文人类可读文本；函数签名、数据字段和机器值保持不变。

- [x] **步骤 1：翻译所有模块、类、函数和协议文档字符串及注释**
- [x] **步骤 2：翻译 CLI 描述、子命令帮助、参数帮助和错误前缀**
- [x] **步骤 3：翻译异常与校验消息，字段名保持原样**
- [x] **步骤 4：更新测试中依赖英文正则匹配的断言**
- [x] **步骤 5：扫描 Python 中剩余的人类可读英文并逐项复核**

### 任务 3：验证、提交与同步远端

**文件：**
- 验证：`README.md`、`docs/**/*.md`、`src/**/*.py`、`tests/**/*.py`

**接口：**
- 输入：完成中文化的仓库。
- 输出：静态检查、测试、CLI 帮助和 Git 状态证据。

- [x] **步骤 1：运行 `.venv/bin/ruff check src tests`，预期显示 `All checks passed!`**
- [x] **步骤 2：运行 `.venv/bin/python -m pytest -q -W error`，预期 68 项测试全部通过**
- [x] **步骤 3：运行总帮助及四个子命令的 `--help`，确认中文可见且退出码为 0**
- [x] **步骤 4：运行英文残留扫描，确认剩余内容均属于全局约束中的保留项**
- [x] **步骤 5：提交中文化修改并推送 `main`，核对本地和远端提交 SHA 一致**
