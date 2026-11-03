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

Issues 使用仓库内的 Local Markdown 工作流，存放在 `.scratch/<feature-slug>/`。见 `docs/agents/issue-tracker.md`。

### Triage labels

Triage 使用默认五项标签：`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`。见 `docs/agents/triage-labels.md`。

### Domain docs

Domain docs 使用 single-context 布局：根目录 `CONTEXT.md` 与 `docs/adr/`。见 `docs/agents/domain.md`。

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

目前在名为tavis的conda虚拟环境进行开发
