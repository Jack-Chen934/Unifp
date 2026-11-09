# UniFP V3 开发目录

基于官方提交`68847a070f88d731058c3d8476929bc3b205f5bd`开发。原官方快照位于`../UniFP/`，本目录随根项目仓库统一版本管理。

截至2026-11-09，V6 r2的seed1 A8000已经完成，S0／S1／S2冻结评估通过；B受力复现已实施，首版冻结为`B1`分支并继续原参数训练。B1绑定17800轮完整checkpoint，受力／接触效果尚未通过验收。见[B1冻结说明](../docs/releases/B1/README.md)、[统一报告](../docs/progress/2026-11-06_v3_status.md)与[阶段入口](../.scratch/unifp-v3-stage1-reproduction/README.md)。

## 环境与入口

训练只在远程进行，环境为Python3.8／PyTorch2.3.1+cu121／Isaac Gym Preview4；启动需`PYTHONUTF8=1`。本机`tavis`是通用开发环境，不是本项目仿真训练环境。本机另有Preview4短测环境缓存。Isaac Gym SDK和权重不随Git仓库分发。

- 正式训练：`legged_gym/scripts/train_v3.py`。
- 启动门控：`legged_gym/scripts/preflight_v3.py`。
- 冻结评估：`legged_gym/scripts/evaluate_v3.py`。
- 参数解析：`legged_gym/scripts/v3_args.py`。
- A阶段冻结manifest：`configs/v3_a_initial_v1.json`。
- B受力／接触manifest：`configs/v3_b_initial_v1.json`、`configs/v3_b_contact_v1.json`；正式观察工具`tools/train_v3_b_observed.py`。
- 已完成训练的精确环境、身份和命令：[统一报告](../docs/progress/2026-11-06_v3_status.md)。不要依据历史启动记录重复启动训练。

当前配置资产为`resources/robots/v3_a1_desktop_train_v6_coplanar_r2/`，从冻结桌面基准独立派生，保留外观、28 links和14 DOF，已完成轮区、共面与低层控制修复。`v3_a1_desktop/`保留桌面基准；`v3_a1/`存在已记录的历史问题。r3／V8仅为独立碰撞候选，正式配置未切换；V9路线暂停。内部候选manifest保持验证时字节，最终验证状态记录在资产外部文档。

## 验证和实验记录

V6最终修复身份通过恢复、容量及控制门控，8000轮三场景TCP L2均值为2.06／2.09／2.27cm，共288个episode零物理失败；多训练seed及B尚未完成。4096规模V3接触对容量为`2**25`。旧r02的80-shape性能结果不能套用到当前V6身份。

- [V3实验汇总：资产、环境、PPO、恢复与正式结果](docs/experiments/v3/实验汇总.md)
- [A阶段票据总览](../.scratch/unifp-v3-stage1-reproduction/tickets/part-a/overview.md)

早期927-shape旧训练在1423轮停止，与已完成的正式V6 A8000不同；完整数据保留远程。早期物理接触分类采用V6第1400轮探索权重，不能视为8000轮成熟策略或当前B性能。B正式训练已启动且继续观察，普通物理失败局部reset；第一阶段全部验收前不开展轮式创新。
