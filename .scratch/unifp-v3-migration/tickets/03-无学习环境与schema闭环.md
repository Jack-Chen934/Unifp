# T03 无学习环境与schema闭环

状态：实施草案，待相关设计确认；本轮未执行业务代码修改。

## Blocked by

T01、T02。此外，涉及模块设计需先确认。

## 具体范围

实现一个明确正式后端的V3观测、命令/目标、历史、动作adapter、reset/终止与metrics接口。先固定/脚本动作端到端运行，不接PPO；区分相对底盘基线与世界固定任务。

## 涉及位置与接口

`unifp_observations.py`、`RangerPrivilegedInputs`、`unifp_task_signals.py`、环境step/reset及正式action adapter；旧桥只作对照。

Actor[N,H,D]、critic[N,Hv,Dv]、label[N,L]与valid；raw/applied action分别存；reset前final value observation与terminated/truncated分离。

## 测试

字段shape/单位/frame、90°yaw/四元数转换、每env历史reset、标签时刻、动作映射/饱和、目标采样可达性；随机有界脚本动作烟雾。

## 验收与产物

无特权泄漏/历史串回合/重复入队；脚本动作与反馈、力作用点和原/补偿目标可追踪；不存在正常step直接写root运动。

交付最小可独立复查的代码差异、配置/schema版本、测试/评估结果及原始证据路径；不覆盖旧实验。知识库发现新事实时先更新对应源码审计，再调整设计。对应设计模块：00、02、06、10。

