# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Agent skills

### Issue tracker

Issues 使用仓库内的 Local Markdown 工作流，存放在本项目根目录下的 `.scratch/<feature-slug>/`；知识库、设计与实施票据分别放在 `knowledge/`、`design/`、`tickets/`，执行入口为各任务的 `tickets/overview.md`。路径均相对于本项目根目录。

### Triage labels

Triage 使用默认五项标签：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`，记录在对应本地任务文档中。

### Domain docs

Domain docs 的入口为根目录 `CONTEXT.md`；已确认决策保存在对应任务的 `.scratch/<feature-slug>/design/`，V3实验记录位于 `UniFP_v3/docs/experiments/`，Go2-W实验记录自2026-11-08起位于 `UniFP_go2w/docs/experiments/`。`CONTEXT.md` 中按日期追加的内容应区分历史状态与最新结论。本仓库目前没有 `docs/agents/` 或 `docs/adr/`，不得将其作为现有文档入口。

---

## 个性化设置

以下为项目/个人偏好,优先级高于上述通用准则中的冲突项。

### 语言与输出

**所有自然语言输出必须使用简体中文。** 包括但不限于:
- 对话回复和解释说明
- 写入磁盘的文档、报告、分析文件
- 代码审查意见和反馈
- 计划、设计文档、规格说明
- 任何包含自然语言段落的生成文件

以下场景可使用英文:
- 代码本身 (变量名、函数名、注释可按团队惯例)
- Git commit message
- 无自然中文对应词的术语
- 配置文件与机器可读格式

适用范围: 主对话、子 agent、skill 输出均须遵守。特别是生成的文档文件必须以中文输出。

**Why:** 主要开发者以中文为母语,英文输出严重降低阅读效率

### 代码风格

- 所有新增或修改的函数（包括私有函数）必须具有完整类型标注和 Google 风格 docstring
- docstring 必须使用英文, 并说明函数职责、主要处理流程、每个参数的业务含义及必要的单位/shape/默认行为、返回值和可能抛出的异常；有副作用时补充 `Note:`。
- 复杂函数必须按实际执行阶段添加代码逻辑注释，说明每一步为什么存在、输入如何转换、状态如何变化；不要只用注释复述显而易见的赋值或控制流。
- 每个局部变量必须通过清晰命名或紧邻注释表达含义；坐标系、单位、shape、索引含义、缓存状态和中间推导值等无法从名称直接判断的信息必须写变量注释。
- 使用 `snake_case` 命名函数和变量，使用 `PascalCase` 命名类，使用 `UPPER_SNAKE_CASE` 命名常量；避免无语义缩写、魔法数和过深嵌套。
- 保持函数职责单一并优先直白实现；不要为一次性逻辑增加额外抽象、兼容层，或增加调用方未使用的参数、字段和配置。

### 任务开发流程

每个开发任务在修改业务代码前，必须按以下顺序完成知识整理、设计拉齐和实施拆分：

1. **建立任务知识库**: 在 `.scratch/<feature-slug>/knowledge/` 下创建 `PRD.md` 和 `research.md`。
- `PRD.md` 是稳定的任务开发简介，结合当前代码说明开发目的、范围、现有调用链、关键约束、注意事项和开发完成的基本含义；不得混入调研过程、未确认的设计方案或实施计划。
- `research.md` 是当前源码依据汇编。根据任务需求阅读当前工作树中的实际代码、测试、日志和本地依赖中的一手参考实现，按代码区域或调用链整理可追溯的代码事实、实际行为及直接接口对应关系，并标注具体文件路径和行号；不得写入待设计问题、方案建议、实验计划、执行顺序或验收标准。
2. **逐项拉齐设计**: 首先基于知识库确认本次设计需要讨论的全部议题，并在 `.scratch/<feature-slug>/design/` 下为每个议题创建按讨论顺序编号的独立 Markdown 文档。除测试外，议题应按照开发内容从顶层运行逻辑、模块与数据架构到具体组件实现逐层向下组织；最后一个议题必须是测试，并明确分为单元测试和整体测试两部分。随后按顺序对每个议题使用 `/grill-with-docs`：沿设计依赖一次只讨论一个决策，为每个问题给出推荐答案，并在对应文档中使用 Q&A 形式逐步记录 grilling 问题和用户确认后的结论；当前议题未形成共同理解前不得进入下一议题，在全部设计确认前不得修改业务代码。每个议题完成 grilling 后，在对应文档开头补充一段覆盖该文档全部结论的总结。上述设计文档整体必须明确当前与目标调用链、模块边界、所有新增或修改接口、数据 shape/单位/坐标系、状态变化、失败语义、测试 seam 和验收方式，并用接近真实调用关系的伪代码给出接口完整但不包含业务实现的最小代码框架。
3. **拆分实施计划**: 将已确认的设计拆成 `.scratch/<feature-slug>/tickets/<NN>-<slug>.md`。每个子任务必须写明具体实现范围、涉及的代码位置与接口、测试和验收标准，以及 `Blocked by` 依赖；任务应是可独立验证的纵向切片，并能在单个新上下文中完成。在 `.scratch/<feature-slug>/tickets/overview.md` 汇总所有子任务、依赖图、执行顺序和当前可执行任务，优先执行所有前置任务已完成的子任务。要求：在实现时，删除无意义的防御性编程和过度的封装，保持代码逻辑的简介和直
4. **按计划实施**: 仅在知识库、设计文档和实施计划完成并经用户确认后开始编码；每个子任务使用独立上下文，按测试驱动实现、验证和代码审查，完成后再推进依赖它的任务。实现中发现新的事实、约束或设计冲突时，先回写对应文档并重新确认，不得静默偏离已确认设计。

---

## 项目相关

### 本地默认路径

本项目的唯一开发根目录为 `/home/sorry/python_ws/interactive/unifp_reproduction`。即使客户端初始工作目录是 `/home/sorry/python_ws/interactive/research`，执行本项目命令时也必须显式指定本项目工作目录；不得把 `research`、`gaze_vr_isaaclab` 或上级目录当成本项目根目录。

| 用途 | 默认路径（相对于本项目根目录） |
|---|---|
| 原始官方代码副本，保留作为对照 | `UniFP/` |
| V3 迁移开发代码及其命令工作目录 | `UniFP_v3/` |
| Go2-W＋Piper独立迁移目录（当前为讨论准备，尚无训练入口） | `UniFP_go2w/` |
| Go2-W迁移知识库、设计与执行入口 | `.scratch/go2w-unifp-migration-v1/tickets/overview.md` |
| Go2-W迁移可行性与方案 | `UniFP_go2w/docs/experiments/迁移可行性与讨论方案.md` |
| 官方流程讲解与参考资料 | `introduction/` |
| 项目术语、阶段边界与按日期追加的状态 | `CONTEXT.md` |
| V3统一进度、修复结论、复跑命令与下一步计划 | `docs/progress/2026-11-06_v3_status.md` |
| A／B 复现知识库、分模块设计与实施计划 | `.scratch/unifp-v3-stage1-reproduction/` |
| 已结束的模型创建、控制修复与A训练记录 | `.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/README.md` |
| 保留的r3／V8原始碰撞诊断证据 | `.scratch/unifp-v3-stage1-reproduction/knowledge/model_history/unifp-v3-collision-proxy-repair/` |
| 当前B设计与实施准备 | `.scratch/unifp-v3-stage1-reproduction/design/part-b/`、`.scratch/unifp-v3-stage1-reproduction/tickets/part-b/overview.md` |
| 实验记录 | `UniFP_v3/docs/experiments/` |
| 项目进度与版本归档说明 | `docs/progress/` |
| 机器人资产 | `UniFP_v3/resources/robots/` |

2026-11-08用户要求开始Go2-W迁移讨论并建立独立目录，原 `UniFP_v3/docs/experiments/go2w/` 已整体迁至 `UniFP_go2w/docs/experiments/go2w/`，20个文件搬迁前后SHA256一致。资产和业务源码本轮未迁移；当前唯一Go2-W组合模型仍为 `UniFP_v3/resources/robots/go2w_piper_unifp_v3/`。新任务记录 `.scratch/go2w-unifp-migration-v1/` 用于当前迁移设计，不是此前已删除的旧模型创建任务。方案尚待讨论，不把旧候选的运动结果视为最终41-shape资产训练准入。

当前 V3 A 配置引用的活动资产为 `UniFP_v3/resources/robots/v3_a1_desktop_train_v6_coplanar_r2/`，具体路径以 `UniFP_v3/legged_gym/envs/v3/v3_pos_force_config.py` 为准。`v3_a1_desktop/` 是冻结桌面基准；其他候选资产、失败实验与旧训练记录须按各自说明阅读，不得仅因不再活动就认定为无用文件。

2026-11-07用户已明确选择V6→B，授权删除旧中间资产和已结束创建任务目录，保留r3／V8诊断证据。V1—V5、失败V7候选及旧v3_a1已删除，来源记录在主线`knowledge/model_history/asset_sources/`；五个旧独立任务的设计／票据合并为历史创建记录，原目录不再作为活动入口。诊断可视化工具仍位于`.scratch/unifp-v3-failure-visualization/`。这些删除是本轮明确授权的处置，不授权删除Go2W/Piper内容。

### 环境与远程运行路径

- 正式训练只在远程服务器执行；本机可做代码开发、单元测试和已授权的 CPU／GPU 仿真短测。
- `tavis` 是已有的通用环境，不是为本项目创建的 Isaac Gym 环境，不能作为本项目默认训练环境；不得为解决 Gym 导入问题擅自修改其依赖。
- 本机已有 Isaac Gym Preview4 短测缓存位于 `/home/sorry/.cache/unifp/isaacgym_preview4/`。使用前核对其中的解释器、SDK 与依赖，不因目录存在就假定任意 Python 环境可导入 Gym。
- 已记录的远程服务器为 `gpuf3`，校园网 SSH 入口为 `ssh myl@10.106.11.248`。每次操作须确认实际远程身份和目标路径，历史连接记录不代表实时可达性。

| 远程用途 | 路径 |
|---|---|
| 项目运行根目录 | `/mnt/data_8THDD/myl_unifp_v3_stage1/` |
| 项目 Python 解释器 | `/mnt/data_8THDD/conda_envs/unifp_v3_stage1/bin/python` |
| Isaac Gym Python SDK | `/mnt/data_8THDD/myl_unifp/isaacgym/python` |
| Torch 扩展缓存 | `/mnt/data_8THDD/myl_unifp_v3_stage1/cache/torch_extensions` |
| 独立源码与验证快照 | `/mnt/data_8THDD/myl_unifp_v3_stage1/validation/` |
| 训练输出 | `/mnt/data_8THDD/myl_unifp_v3_stage1/runs/` |
| 正式评估输出 | `/mnt/data_8THDD/myl_unifp_v3_stage1/evaluations/` |

具体训练使用的冻结源码、run 目录、启动参数与观察安排，以 `docs/progress/2026-11-06_v3_status.md` 和对应运行身份记录为准；不得覆盖正在使用的远程源码快照、资产、检查点或输出目录。

### 工作区维护

清理仅针对已确认可再生成的缓存、一次性临时文件及经 SHA256 核验的重复副本；清理重复证据时记录保留位置。保留官方副本、有效资产、知识库、已确认设计、tickets、训练数据及必要失败证据。Go2W／Piper 等其他任务的未提交内容不得随 V3 清理而删除、回滚或批量提交。Git 中存在真实开发改动不等于工作区存在垃圾文件，不用 `git clean` 或 `git reset --hard` 制造表面干净。
