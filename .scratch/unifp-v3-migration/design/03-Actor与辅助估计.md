# 03 Actor、历史编码与辅助估计器

状态：待确认设计草案。推荐结论：保留官方共享latent结构，重建V3的观测/动作/监督契约；先证明确定性动作和非零力估计有效，再探索显式物理状态反馈或新编码器。

## 当前与目标网络

官方真实结构是`历史2336→512→256→128→64 latent`，动作头`[最新73,latent64]→512→256→128→17`；decoder为`64→128→64→12`。当前V3换成1856历史、最新58、输出14。**decoder的12维估计没有显式拼回动作头。** [源码与梯度依据](../knowledge/official_source_audit.md#3-actor历史编码与辅助状态估计)

建议首版：

```text
history[N,H,D] → encoder(512,256,128) → latent[N,64]
[latest_frame[N,D],latent] → actor(512,256,128) → mean[N,A]
learned_valid_std[A] → joint diagonal Normal(mean,std)
latent → decoder(128,64) → estimates[N,9]
```

`H=32`、`A=9`或14由动作边界确定，D由schema确定。历史仍对应约0.64s名义采样窗口；最旧至最新时间差为31dt。保留MLP/latent维度是为了控制变量，不是已证明V3最优。

## 分模块修改清单

| 子模块 | 保留 | 修改/核查 |
|---|---|---|
| 历史缓存 | per-env FIFO、reset清零 | 有效历史mask、时延对齐、部署时间戳；不跨回合 |
| encoder | 共享64 latent、PPO与辅助监督均回传 | 输入schema与物理尺度，记录两支梯度/参数更新量 |
| actor head | 最新本体帧＋latent、联合动作分布 | 输出维度、动作分组尺度/限速；初版不增加多个独立决策器 |
| std | 可学习探索 | 按底盘/轮/转向/臂的物理1σ分别初始化，正数性、上下界/记录可审计；不可统一照抄1.0 |
| decoder | MLP辅助物理监督 | 首版建议9维：base velocity3、EE position3、EE external force3；基座外力未激励不保留成功假象 |
| 估计目标 | 仅训练时真值监督 | EE位置建议统一局部笛卡尔m，力在yaw或任务frame的N；与原球坐标标签差异单独版本化 |
| loss | 辅助监督独立于PPO，encoder共享 | 单位缩放、valid mask、非零力分层、零力回归；报告原单位MAE/RMSE |
| 推理 | 仅Actor与encoder；decoder按需估计 | deterministic mean为部署主指标，sample另作诊断；禁止在线使用Critic真值 |

若要最严格对照官方表示，可先以12维旧标签/球坐标保持一个诊断baseline，再在正式9维任务使用清楚定义的标签。二者不能共用同一个“官方完全不变”标记。

## Loss与梯度接口

原实现将每组prediction和target都乘权重后做MSE，`.2`实际贡献`.04`倍平方误差；力另有`.01`单位缩放。新配置应显式区分`physical_scale`与`loss_coefficient`，不要把这个行为悄悄改成另一公式。[原实现](../knowledge/official_source_audit.md#53-辅助损失及实际更新范围)

```text
latent = encoder(history)
mean = actor(concat(latest_frame,latent))   # 不detach latent
prediction = decoder(latent)
auxiliary_loss = sum(alpha_k * masked_mean(square((prediction_k-truth_k)/scale_k)))
```

norm/scale只用训练数据或事先冻结物理尺度，评估不更新统计。保持一套PPO Adam和一套辅助Adam交替更新encoder，并保存两者状态。actor-head/std不与辅助预测损失相连；decoder不与PPO动作计算相连。显式拼接`F_hat`或detach latent均属于后续结构消融，不是首版默认。

## 可观测性与失败语义

大质量底盘、摩擦/滑移/内环、负载和电流饱和可能造成相似状态历史；不能把有decoder就等同于可准确辨识力。使用多个方向/幅值/姿态/接触开关，比较恒零、传统观测器、无历史、无辅助估计等基线。真值标签只表示实际建模/施加的力；注入力、接触反力及合外力须有各自mask。

历史不足、传感器失效或物理非有限不产生“可信力为0”的输出语义；部署降级由监督器处理，具体阈值在接口测量后定。力预测不确定性网络仅在有校准与决策收益证据后添加。

## 测试与Q&A

测试seam：shape、缓存时间、标签t/t+1对齐、梯度归属、actor无特权路径、std正数性、同一输入CPU/GPU/导出一致性。mean/sample在同一验证种子和初始设置下对照，不能用sample曲线替代mean部署能力。

| 问题 | 推荐答案 | 用户确认 |
|---|---|---|
| 首版换Transformer/GRU吗？ | 先保留官方MLP；历史模型升级需在稳定基线上做等预算消融 | 待确认 |
| 力估计是否直接回灌Actor？ | 首版保留共享latent；显式力回灌作为单独、可比较的实验 | 待确认 |
| 首版是否保留base外力头？ | 无独立激励/真值与任务时移除；后续开启需整体扩展命令、标签与测试 | 待确认 |
