# 动作学习方案与运行时能力清单

状态：当前动作学习与挂载的权威记录

审计日期：2026-09-05

适用环境：WSL2 Ubuntu 22.04、Booster T1、RCSSServerMJ、`my3d-rl`

训练产物目录：`/home/win98/rl_runs`

## 1. 结论

动作学习采用“分层决策 + 可替换低层技能”，不训练一个端到端七人球队模型：

1. 比赛规则、球队策略、职责分配和动作选择保持确定性；
2. 学习模型只实现有明确输入、输出和物理包线的运动技能；
3. ONNX 是目标主执行器，程序化轨迹和现有稳定动作是回退；
4. 训练资产只有经过 C++ runner 和服务器链路后才算“已挂载”；
5. 当前最高价值训练目标是可部署的目标条件短传，其次是接球、射门/解围和
   高速多方向行走；
6. 动作训练与 `docs/team-excellence-roadmap.md` 的 D1/D2 球队决策开发并行，
   不阻塞能利用现有动作完成的策略代码。

“队伍决策优先”只决定主开发队列，不代表冻结或删除训练。当前采用双轨：决策轨
优先完成共享比赛状态、职责和动作生命周期；动作轨持续完成教师搜索、数据生成、
监督初始化、ONNX runner 和服务器域适配。两轨通过能力注册表与物理结果反馈汇合。

当前不应继续直接延长 K2-B 的 PPO。它在固定 2 m smoke checkpoint 上虽然有
正确脚触球，但 `target_success=0.0`、评估跌倒比例为 `0.375`，说明缺少可学习的
目标动作先验。正确顺序是先产生稳定、覆盖目标空间的教师轨迹，再做监督初始化，
最后使用受约束强化学习微调。

## 2. 状态定义

本文严格区分以下状态：

| 状态 | 含义 |
| --- | --- |
| 默认挂载 | 标准启动无需额外开关，决策能够到达且运动层真实执行 |
| 可选挂载 | C++ 调用链已接通，但必须显式配置，不能视为正式默认能力 |
| 工作树挂载 | 当前未提交工作树可运行，尚未进入远端接受基线 |
| 训练候选 | 有 checkpoint/ONNX/轨迹和评估工具，但 C++ 正式链路没有使用 |
| 教师/证据 | 可指导训练或证明物理动作存在，不是可部署策略 |
| 仅有接口 | 枚举、命令或能力类型存在，但没有可执行资产或选择路径 |

“有 ONNX 文件”“训练 reward 上升”“精确物理某个固定窗口成功”都不自动等于
已挂载能力。

## 3. 当前实际挂载进度

### 3.1 默认挂载

| 能力 | 实现与资产 | 决策/运动链路 | 当前事实 |
| --- | --- | --- | --- |
| 稳定行走/转向/停止 | `runtime/apollo/assets/networks/walk/policy.onnx`，`[1,78] -> [1,23]` | 所有 `WalkCommand` 经 `WalkRunner` 执行 | 正式默认步态；仍是比赛运动基线 |
| 四向起身 | `runtime/apollo/assets/networks/getup/policy.onnx` | 跌倒检测产生 `GetUpCommand`，`MotionManager` 返回完成或超时 | 正式默认恢复能力 |
| 原地保持 | `keyframes/neutral.yaml` | `NeutralCommand -> NeutralRunner` | 用于安全保持和不支持动作拒绝 |
| 接近/绕障/触球前对准 | 确定性步行规划、球后槽位和稳定驻留 | AP、定位球和门将路径共用 | 已使用，但最终精度受步态摆动和观测遮挡影响 |
| 固定前向触球 | 无独立踢球模型；使用稳定步态前推、稳定和恢复宏 | `ForwardContact -> MotionManager -> WalkRunner` | 可触球、可完成定位球回退，但距离弱，不能称为短传/射门 |
| 起身和动作反馈 | `Running/Completed/Rejected/TimedOut` | 下一决策周期回传 | 失败的匹配传球会取消并重规划 |

两个上游 ONNX 的 SHA-256 分别为：

- walk：`6df65fa7d36fd4989fcb022e385de797d51f35c8375532841034716e4bc0d850`；
- get-up：`ae6ade761e50fccb432e118cebd456d9e96e87de0fa3a3adc2d5f92ef496a83d`。

### 3.2 可选或有界实验挂载

| 能力 | 启用方式 | 证据 | 限制与结论 |
| --- | --- | --- | --- |
| `FastWalkV2` 高速行走 | 显式设置 `APOLLO_ENABLE_FAST_WALK=1` 后传入 `--enable-fast-walk --fast-walk-model <path>` | 指定候选 CPU 32/32 直立、中位速度 1.499 m/s；7v7 调用链可运行 | 10 秒横向漂移中位 5.452 m，服务器跌倒率过高；已从默认比赛配置撤下，仅保留更严格前向域和稳定步态回退，不能称为跑步 |
| 残差表目标传球 | 源码树 WSL 启动脚本默认传入 `--enable-parameterized-kick` | 152/153 条件表的精确评估三种子约 94.7%–96.3% | 服务器保存结果的实际出球仍弱；作为 learned kick 不匹配/失败时的实验回退 |
| 程序化 `DribbleTouch` | 同一参数化踢球开关；AP 无传球提交时可发出短触 | 精确 MuJoCo 球位扰动 20/20；真实 7v7 为 14/14、63 个执行样本、1 次接触，执行者无起身；该事件约前进 0.831 m、横向 0.016 m、方向误差约 1.08° | 资产、能力包线、决策和 C++ runner 均已挂载；仍只有右脚、近零角度和一个 0.55 m 锚点，不等同于连续学习带球 |
| 程序化 `Shot` | 同一参数化踢球开关；AP 距球门 3.5–4.5 m 且进入释放槽位后选择 | 4 m 教师在预声明球位区间独立留出 100/100；真实 1200-cycle 7v7 有 12 个射门决策样本和 1 次物理接触 | 右脚、近零角度、2.50 m/s 的窄包线；已作为有界实验动作挂载和 ONNX 训练教师，不外推成通用射门 |
| 程序化 `Clear` | 同一参数化踢球开关；球在自家防守三区时由 AP 优先选择 | 独立 6 m 教师按安全解围语义留出 100/100；真实 1200-cycle 7v7 有 13 个解围样本和 1 次物理接触 | 右脚、近零角度、3.50 m/s；保证至少 4.5 m 前进和 1.5 m 半通道，不声称精确 6 m 落点 |
| 学习目标传球 runner | 源码树 WSL 启动脚本默认 active；可用 `APOLLO_LEARNED_KICK_MODE=shadow` 改为只推理 | `kick_policy_v3` 的 98→23 观测、推理、残差解码、限位和同周期回退已接通；外部 r2 ONNX 加载/推理通过 | r2 冻结评估仅 27/92 且 1 次跌倒；主动执行被限制在实际固定 2 m 训练切片，其他请求不覆盖残差/程序化回退 |
| 目标动作安全拒绝 | 直接运行二进制/保守配置时参数化动作关闭，或请求超出共享包线 | 运动层返回 `RejectedTargetedKickHold` | 已实现且正确；不会把目标传球静默变成固定前踢 |

本阶段交付验证：

- Apollo C++ 构建成功，16/16 CTest 通过；训练目录 233/233 Python 测试通过，
  包含本轮新增课程、纯转向相位和着地脚滑移覆盖；
- `KickTeacherEvaluator` 已支持显式鲁棒球位范围和逐代进度回调，新增回归测试通过；
- 4 m 程序化射门的冻结教师 manifest SHA-256 为
  `2b6b40b78c3acb4b62c87b2a1145ea6ade0169043d1245707f5611aca063c978`；
- 独立 seed `20261054` 在球局部 x=`0.312..0.340 m`、
  y=`0.028..0.052 m` 内为 100/100，报告 SHA-256 为
  `5cc692c8677a26f64e43a21f88cb0ccab46f744c33d06d7c6cc9dcda77fd183c`；
- 7v7 服务器证据在
  `/home/win98/rl_runs/player-action-foundation/server-shot-s20261053-v1`：
  14/14 客户端干净退出、12 个 `Shot` 样本、1 次物理接触、430 个
  FastWalkV2 样本和 75 个起身样本；
- 6 m 解围教师 manifest SHA-256 为
  `8c40e2bb86c00dbbd733b41d933a955f408ff3c3bad6ce61fdc6a8d52c054c51`，
  seed `20261055` 的预声明释放槽位评测 100/100；真实服务器证据在
  `/home/win98/rl_runs/player-action-foundation/server-clear-s20261059-v1`，
  13 个 `Clear` 样本、1 次物理接触、14/14 客户端干净退出；
- 当前最佳 transition r2 ONNX 已通过真实 C++ `LearnedKickRunner` 外部模型加载和
  一次完整 98→23 推理；
- 7v7 shadow 启动样本 14/14 客户端完成连接、入场和退出，但旧定点场景未产生
  `KickCommand`，因此只能证明进程级加载，不能写成比赛内 shadow 已触发；
- 历史全能力默认启动的 900-cycle 7v7 为 14/14 干净退出、212 个
  FastWalkV2 采样和 97 个起身采样；该跌倒证据已触发默认配置回退；
- 本轮稳定默认、转向—前进组合的 900-cycle 7v7 为 14/14 干净退出、0 个
  FastWalkV2 样本、0 个起身样本、157 个传球规划、13 个 Ready 和 1 次真实触球；
- 程序化短触资产仍是 `server_status: contact_observed`，不是正式晋级状态。

### 3.3 只有接口、没有实际动作

| 能力 | 已有表面 | 实际缺口 |
| --- | --- | --- |
| 学习版 `Dribble` | 动作类型和程序化短触存在 | 没有连续闭环学习策略；当前只是“接近—短触—重获球”的确定性循环基础 |
| `Receive/FirstTouch` | 接球者能走到目标并面球 | 没有接触模型、缓冲/停球策略和学习 runner |
| 学习版门将扑挡 | 门将有门线交点与步行拦截 | 没有侧扑、前扑或封堵模型 |
| 通用学习踢球能力 | `LearnedKickRunner` 已支持 v3 传球 actor，程序化 Shot/Clear/Dribble 已有独立模式、教师和窄能力包线 | 尚无覆盖多距离、多方向、左右脚并达到比赛质量的统一 ONNX |

## 4. 已有资产的挂载状态与复用边界

### 4.1 可以继续利用的高价值资产

| 资产/能力 | 位置与证据 | 当前状态 | 正确复用方式 |
| --- | --- | --- | --- |
| PAiD K1-D 动作跟踪 actor | `/home/win98/rl_runs/paid-k1/k1d-crossfit-bc-s20260982-v2/checkpoints/000000001000` | 依赖外部有限动作参考；只通过动作跟踪选择/确认，未通过 Apollo 服务器动作链 | 作为教师、初始化和轨迹结构参考，不直接替换运行时 |
| K2-A 固定强触球窗口 | motion 12、帧 113–118；精确 MuJoCo 360/360 正确脚触球、稳定、零跌倒，中位前进约 4.1 m | 只证明固定窗口；未证明目标距离/方向、接近入口、移动球和服务器执行；参考资产受本地非再分发约束 | 用于验证接触/恢复设计、教师搜索和训练诊断，不冒充参数化射门 |
| K2-B 球/目标条件 checkpoint | `/home/win98/rl_runs/paid-k2/k2b-fixed2m-smoke-s20260990-v1/checkpoints/000000001024` | `target_success=0.0`，跌倒与恢复均不合格；无 ONNX runtime | 保留为失败基线和环境回归，不从该 checkpoint 继续长训 |
| 高速行走候选 | `/home/win98/rl_runs/run-phase-v2-formal-s71-20260831-01/policy-best.onnx` | 仅在显式开关下有界挂载；横向漂移和服务器跌倒/起身问题已使其退出默认步态 | 作为离线对照和训练初始化；稳定比赛默认继续使用 78→23 walk |
| 程序化短触轨迹 | `/home/win98/rl_runs/procedural-kick/` 与仓库 YAML | 已挂载为窄 `DribbleTouch` fallback；只有一个右脚锚点和一次服务器接触 | 保留为 fallback，并生成监督学习教师数据 |
| 残差表目标传球 | `kick_residual_table.yaml` | 已挂载为目标传球实验回退；精确评估与真实服务器出球仍有差距，表还缺一个条件 | 用作球位敏感性数据、teacher/ablation 和短期实验回退 |

### 4.2 已挂载的 ONNX 与仍未晋级的候选

当前将 r2 transition 候选挂到通用 runner：

- `/home/win98/rl_runs/kick-transition-dagger-r2-bc-s10002/policy.onnx`；
- SHA-256：`b89b67ad78766615cebdb3e340ebf40305fbf01b5ffa6cf927a8737b18d4aea1`；
- 冻结 exact CPU：27/92（29.35%）、92 次触球、1 次跌倒；
- runner 只接受 1.90–2.10 m、相对方向 ±12° 和 transition corpus 实际球槽位，
  不能把固定 2 m 候选外推到整个残差表包线；
- 源码树 WSL 启动器按用户要求默认启用，但只有上述窄包线才取得关节控制权；
  `APOLLO_LEARNED_KICK_MODE=shadow` 可切为仅推理，任何不匹配、非有限输出或
  运行失败均同周期回退到残差表/程序化动作；该默认启用不代表模型已晋级；
- 锁定记录：`training/locks/learned_kick_runtime_candidates_2026_09_04.yaml`。

其余 `/home/win98/rl_runs` 导出家族：

- `kick-bc/*.onnx`：v2 行为克隆、角度网格和 DAgger 候选；
- `kick-transition-*/policy.onnx`：接近到触球的切换候选；
- `kick-physical-residual-*/correction-*.onnx`：残差 PPO 候选；
- `kick-switch-selector-*/selector.onnx`：动作/时机选择器；
- `striker-action-bank-*/selector*.onnx`、`outcome-selector*.onnx`：前锋动作库
  选择器；
- 多个 `run-*/*.onnx`：不同参考、相位和 curriculum 的高速运动候选。

不授予关节控制权不是因为文件格式不可用，而是因为结果不足：

- 五动作前锋库的 oracle 为 928/1023（90.71%），说明动作库中通常存在可行动作；
- 当前状态选择器在冻结验证上为 153/205（74.63%）；
- privileged/history 选择器约 71%–72%；
- 连续 outcome selector 为 128/205（62.44%）；
- 这证明主要瓶颈是稳定动作表示、状态可辨识性和切换，而不是缺少一个更大
  MLP；
- v3/残差 PPO 多轮精确评估没有稳定超过冻结先验；
- 除 `FastWalkV2` 外，多数 run ONNX 已因转向模式、漂移、飞行相位或完成率
  被明确拒绝。

这些 ONNX 的最佳用途是离线对照、hard-negative 数据和消融实验。正式运行时不
应扫描目录自动选“最好看的模型”。

### 4.3 不是能力的资产

以下内容有价值，但不能列入机器人现有动作：

- PAiD/GMR/Holosoma 动作片段和重定向 corpus；
- PPO smoke checkpoints；
- 只有 checkpoint 迁移奇偶性的 bootstrap；
- action-bank oracle；
- TensorBoard 曲线；
- 尚未通过 held-out 或服务器检查的 teacher manifest。

## 5. 统一动作学习架构

### 5.1 运行时接口

每个学习动作必须服从同一契约：

```text
ActionRequest
  mode, target point/range, requested launch/arrival speed,
  action id, participant, deadline
        |
        v
ActionCapabilityRegistry
  state + measured envelope + model/asset revision
        |
        v
Learned runner (ONNX, preferred)
        |
        +-- invalid/unavailable --> procedural or retained stable fallback
        |
        +-- no executable fallback --> Neutral + Rejected
        v
Motion feedback + observed ball/body outcome
```

当前代码已新增通用 `LearnedKickRunner`，后续还要补齐：

- 加载模型和 manifest，而不是只接收一个裸 ONNX 路径；
- 校验输入/输出 shape、观察顺序、单位、归一化、关节顺序、PD gain 和 SHA-256；
- 已支持 shadow inference、显式 active、球/姿态/有限值检查、关节限位以及
  同周期残差/程序化回退；
- 待增加 companion manifest 的自动散列校验，而不是只由比赛脚本锁定 SHA；
- 待输出动作 ID、阶段、模型散列、fallback 原因和观测到的物理结果；
- 待把 shadow 输出与执行回退的实际球结果写成可训练的 paired trace。

### 5.2 学习边界

短期不学习球队战术。模型只处理低层难以手工稳定实现的部分：

- 全身接触轨迹与平衡；
- 球位/目标变化下的关节残差；
- 接近、制动、触球和恢复之间的短时切换；
- 移动球 first touch；
- 方向相关高速步态和门将快速封堵。

规则合法性、传球对象、射门时机、职责归属、定位球双触规则和不支持动作拒绝
继续由 C++ 决定。

## 6. 目标条件球动作训练方案

### B0：冻结证据与基线

在新训练前固定以下不可变对照：

- 默认 walk/get-up/fixed-contact；
- 程序化 0.55 m 右脚短触；
- 实验 residual table；
- K2-A 固定窗口（教师证据，不是运行时）；
- K2-B 1024-step 失败 checkpoint；
- 五动作 action-bank 和所有被冻结的选择器结果。

每次新候选使用同一批未参与训练的球位、方向、摩擦和入口状态对比。

### B1：程序化教师覆盖

先扩展确定性 CEM/轨迹搜索，不直接启动长 PPO：

1. 保持当前 0.55 m 短触锚点；
2. 改进轨迹表示后重新搜索 1.5–2.5 m 短传；
3. 当前两次 2 m 搜索一个触球后跌倒且仅约 1.15 m，另一个完全未触球，不能
   直接发布；
4. 增加支撑腿屈伸、分阶段时长、足端高度/摆幅和恢复姿态等自由度；
5. 先做右脚 0°，再开 ±10°/±20°，最后做左脚镜像和独立确认；
6. 分别搜索短触、传球、射门和解围，不用同一轨迹改名；
7. 保存完整 joint/ball/body/contact 轨迹，作为监督数据和 deterministic fallback。

2026-09-05 已完成一个独立 `Shot` 教师闭环：冻结 4 m 轨迹、预声明释放槽位
100/100 留出成功，并在真实 7v7 服务器观察到球从约 0.32 m 局部槽位以约
3 m/s 离开。另一次 30 代、256 population、17 个鲁棒样本的搜索得到更小横向
误差候选，但仍标记 `promotable=false`，仅保留为训练候选，不替换已验证资产。
`Clear` 使用独立目标和验收定义完成搜索：名义前进约 5.18 m、峰值约
3.48 m/s、无跌倒；按“至少前进 4.5 m、1.5 m 半通道、保持可控姿态”的
安全解围语义独立留出 100/100，并通过真实服务器接触。它不是把射门轨迹改名，
也不宣称精确 6 m 落点。

工具基线：

- `training/tools/optimize_kick_teacher.py`；
- `training/tools/evaluate_kick_teacher.py`；
- `training/tools/generate_kick_teacher_dataset.py`；
- `training/tools/export_kick_residual_table.py`。

### B2：监督初始化

从成功教师轨迹建立目标条件数据集：

- 输入只使用可部署信号：关节位置/速度、身体角速度、重力投影、上一动作、球在
  torso-yaw 坐标系的位置/速度、目标方向/距离、请求初速/到达速度、动作模式、
  球观测 age/mask、动作进度和支撑脚提示；
- critic 可使用精确接触、身体速度和无噪球状态，actor 不可使用；
- 输出为 23 维有界关节残差；
- 初始候选复用 `kick_policy_v3` 的 `[1,98] -> [1,23]` 工具链；需要把
  `DribbleTouch` 纳入统一模型时再版本化为四模式契约，不静默改变 v3；
- 首先行为克隆教师，再用 DAgger 收集模型偏离后的状态；
- action-bank 失败状态作为 hard negative，训练是否执行、选脚和 fallback，
  不直接训练其已失败的选择器标签。

已有 `train_kick_bc.py` 和 ONNX 导出器可复用。K2 的 126 维外部参考 actor 保留为
teacher/ablation；正式目标应是自包含或只依赖可随运行时合法分发资产的 ONNX。

### B3：受约束 PPO 微调

只有监督模型在精确 CPU 上已经能稳定触球、保持直立并回到可控姿态后才开始：

1. 固定 2 m、0°、静止球；
2. 打开 1.5–2.5 m 距离；
3. 打开方向和球位偏差；
4. 打开到达速度与 pass/shot/clear 模式；
5. 打开摩擦、质量、PD、延迟、观测噪声和轻推；
6. 最后加入移动球与慢速接近入口。

采用 progressive unfreeze：先冻结已学好的动作主干，只训练新增球/目标输入行和
小残差头；确认目标信号能改变物理结果后再逐步解冻。每轮同时记录：

- 正确脚/错误脚/无触球；
- 方向、距离和到达速度误差分布；
- 摔倒、最低躯干高度、支撑脚滑移和恢复完成；
- 动作变化、限位/力矩成本和非有限输出；
- 对基线新增成功、基线独有成功和灾难性回归。

不再用单一总 reward 或在线小批次成功率决定继续训练。

### B4：接近与切换

静态触球模型稳定后，先保持确定性 FSM：

```text
Walk/fast-walk -> precision approach -> neutral settle
-> learned/procedural contact -> recovery -> walk/get-up
```

然后才训练 `striker_policy_v1` 类的短时切换残差。动作开始前仍由 C++ 检查
球槽位、朝向、平面速度、比赛状态和 cooldown。旧 action-bank 已证明开环窗口
选择难以从当前状态可靠辨识，因此不能再以“训练一个 selector”替代闭环切换。

### B5：ONNX 与服务器挂载

1. 导出 ONNX、manifest、SHA、训练 revision、seed 和数据散列；
2. C++/Python 固定 observation corpus 做数值一致性；
3. 使用现有 `LearnedKickRunner` 先 shadow inference；
4. 单人服务器按球位/方向/入口速度重复执行；
5. 发布最小可靠子包线到 `ActionCapabilityRegistry`；
6. 选择顺序为 learned ONNX、程序化/残差回退、固定动作或安全拒绝；
7. 再让 AP、传球、射门、解围和门将分配使用该子包线。

## 7. 其他动作学习路线

### L：高速、多方向行走

不重新追求更高直线标称速度。按“快速转身、稳定前进、横向/全向、比赛域适配”
分阶段训练，优先解决：

1. 服务器横向漂移；
2. 横移、倒退、转向、制动和 command switch；
3. 从高速到触球槽位的稳定过渡；
4. 推扰、轻碰撞、观测噪声和不同角色速度限制；
5. 方向相关到达时间与跌倒概率标定。

训练数据应加入真实服务器中触发 fallback/get-up 的入口状态。默认 walk 继续负责
所有正式比赛移动；大偏角目标先转身再前进，直到新模型逐方向通过验证。

2026-09-05 已开始 L1 正式批次：新增 `soccer_omni` 课程，保持现有
`run_policy_v2` 的 80→23 ONNX/运行时边界，从已挂载 phase-v2 checkpoint 继续
训练。课程覆盖 `vx=-0.25..1.65 m/s`、`vy=±0.45 m/s`、`yaw=±0.75 rad/s`、
1.5 秒指令重采样、25% 停止命令、推扰和一周期动作延迟。

冻结的旧候选八命令 CPU 基线位于
`/home/win98/rl_runs/run-soccer-omni-baseline-s20260941-v1`：5/8 命令通过；
高速前行偏航、倒退和左转失败，其中倒退 8/8 均未完成。后续模型必须在同一命令
集上报告最差值，不能再由单一 1.5 m/s 直线结果选出。评测入口为
`training/tools/evaluate_run_command_suite.py`。

首个 500 万步 broad curriculum 位于
`/home/win98/rl_runs/run-soccer-omni-s20260951-v1`。MJWarp 随机命令评估的跌倒率
由 28.1% 降到 12.5%，但两个候选的精确 CPU 命令集仍是相同的 5/8，故明确拒绝
挂载。根因是连续三轴均匀采样几乎不产生纯直行、纯横移或纯转向命令，同时旧
`legacy_phase_warmstart_v2` 的 adaptive-KL 下限高于名义学习率，更新实际放大。
下一批使用 `soccer_omni_axis`（50% 轴对齐命令）和
`legacy_phase_soccer_v3`（`5e-7..5e-6` 学习率、较小 KL）从冻结运行时 checkpoint
重新开始，而不是在拒绝模型上继续。

2026-09-05 的复盘进一步表明仍不应直接打开宽三轴课程。训练环境已加入真正的
纯 yaw 步态相位、着地脚滑移代价，以及 `rapid_turn -> stable_forward ->
soccer_lateral` 三阶段课程；冻结命令集也新增纯横移和纯原地转身。第一次快速转身
探索的左转已明显改善，但右转 16 次仅 11 次直立，故拒绝挂载。完整证据、服务器
对比和强触球路线见 `docs/stable-motion-strong-kick-development.md`。

### R：接球与 first touch

先完成 D2 的接球意图和来球轨迹预测，再建立确定性接球姿态。只有失败主要来自
接触与平衡而不是通信/站位时才训练模型：

- 输入为来球相对位置/速度、期望停球或下一动作方向、身体状态和观测 age；
- 动作为短时全身残差；
- 结果按控球、反弹距离、下一动作可执行时间、直立和边界风险评估；
- 训练来球由已经标定的传球分布产生，不能使用理想直线球替代。

### G：门将封堵

步行门线拦截、前出和站位保持确定性。只有不可步行覆盖的射门样本足够后，训练
左/右侧扑或前扑封堵；恢复继续复用 get-up。模型输出不得决定是否离开禁区、是否
接球或传给谁。

### 不单独重训 get-up

现有四向起身已挂载。除非出现明确机器人姿态缺口或需要消除上游 GPL 资产依赖，
不把计算资源投入一个已可用的动作。

## 8. 证据与发布原则

这里的检查用于说明能力边界，不作为阻塞其他策略开发的总门槛。

每个动作至少回答：

- 哪些目标、球位、入口速度、左右脚和比赛状态可执行；
- 物理结果的中位数、分位数和失败类型；
- 与当前 fallback 相比新增了什么，破坏了什么；
- ONNX 与训练 checkpoint 是否一致；
- RCSSServerMJ 是否观察到相同动作类别；
- 请求超出包线或推理失败时会执行什么；
- 决策层是否真正能选择它。

固定 90% 不适合所有动作：短传、短触和起身需要高可靠性，远射可接受较低命中
但必须保持直立和方向安全，解围重视离开危险区而非精确落点。阈值按动作用途和
基线风险设定，并同时报告原始计数与置信区间。

## 9. 产物和磁盘规则

- 所有 checkpoint、ONNX 候选、NPZ、TensorBoard、服务器日志和大报告写入
  `/home/win98/rl_runs`；
- C 盘仓库只保存代码、小型契约、通过审核的运行时资产和摘要；
- 每个 run 必须有 `run-manifest.json`、git revision、环境/数据/源 checkpoint
  散列、seed、状态和明确的 `runtime_promotion`；
- `failed`、`teacher_only`、`training_candidate`、`shadow`、`experimental`、
  `stable` 不得混用；
- 不删除失败模型，但从活动清单移入实验日志，避免下轮再次误用；
- 受许可限制的参考动作不得提交或作为运行时隐式依赖。

## 10. 完整并行实施计划

### 主线 S：球队策略与动作利用

1. 完成共享 possession owner、全队计划 revision 和唯一职责分配；
2. 完成传球 `Committed -> Commanded -> 物理结果 -> Cancelled/Timeout` 生命周期；
3. 让 AP 在能力包线内选择 `DribbleTouch`、TargetedPass 或 ForwardContact，超出
   包线时显式重规划；
4. 接入进攻宽度/深度、二过一支持、丢球反抢、退防、门将出球和定位球执行者；
5. 将动作完成、拒绝、跌倒、球是否进入接收区反馈给下一周期决策。

### 并行线 M：动作训练与部署

1. 冻结已挂载动作和 ONNX 候选清单，保留所有失败基线；
2. 扩展程序化教师：右脚 0.55 m 短触之后优先完成稳定 2 m 传球，再扩方向与左脚；
3. 从成功轨迹做 BC/DAgger；现有 29.35% r2 只作基线，不再直接长训；
4. 收集 `LearnedKickRunner` shadow 的服务器 observation/action/fallback/outcome
   paired trace，缩小 exact CPU 与服务器差异；
5. 监督模型稳定后再做受约束 PPO，并分别训练 pass、shot、clear；
6. 用实际传球分布训练 receive/first-touch；
7. 同步进行 FastWalkV2 的漂移、制动、转向和进入精确接近的域适配；
8. 最后扩展门将非步行可覆盖区域的扑挡模型。

### 依赖关系和算力顺序

```text
程序化成功教师 -> BC/DAgger -> shadow trace -> 受约束 PPO -> 最小可靠包线
        |                                                |
        +-------------- 策略能力注册表 <----------------+

真实传球包线 -> 接球来球分布 -> first-touch 模型
FastWalk 域适配 -> 更准 ReachTime -> 球队职责/传球时机修正
```

- 日常 CPU 优先用于 C++/服务器回归、教师优化和数据检查；
- GPU/JAX 长训只在教师覆盖、数据散列和冻结评估集已准备后启动；
- 每个长任务必须能从 `/home/win98/rl_runs` checkpoint 恢复，日志和模型不写 C 盘；
- 决策开发不等待长训；训练完成后通过能力包线替换执行器，而不是重写高层策略。

### 交付批次

| 批次 | 必须交付 | 训练结果进入比赛的方式 |
| --- | --- | --- |
| M0 已完成 | 稳定 walk/get-up、FastWalk 可选、残差传球、程序化短触、4 m 程序化射门、LearnedKickRunner | 能力注册表 + 显式开关 + fallback |
| M1 | 2 m 教师覆盖、数据集、三种子 BC/DAgger、shadow paired trace | 先发布一个窄 TargetedPass 子包线 |
| M2 | 距离/角度/左右脚课程、有限 PPO、服务器重复结果 | 扩展传球包线并校准 ReachTime/接球目标 |
| M3 已完成 | shot/clear 均有独立教师、独立留出和服务器物理接触 | Shot/Clear 均成为有界实验能力，超出包线仍 fail closed |
| M4 | receive/first-touch + FastWalk 域适配 | 提升完整传接闭环和攻防转换速度 |
| M5 | 门将封堵与高级移动球动作 | 只补步行/现有动作无法覆盖的低层缺口 |

## 11. 当前紧接的实施顺序

1. 提交程序化短触、4 m 射门、6 m 解围、`LearnedKickRunner`、共享契约和证据锁；
2. 扩展程序化教师表示并重新搜索稳定 2 m 短传，生成目标条件教师数据；
3. 行为克隆一个自包含短传 actor，使用已实现 runner 收集 shadow paired trace，
   并补 companion manifest 自动校验；
4. 仅对监督模型做有限、逐阶段 PPO 微调，在服务器发布最小可靠传球子包线；
5. 用真实传球分布开发接球/first-touch；
6. 并行进行 `FastWalkV2` 的漂移、制动和方向域适配；
7. 最后才考虑门将扑挡和更高层学习策略。

详细程序化轨迹设计见 `docs/model-free-parameterized-kick-plan.md`；历史实验与拒绝
原因见 `docs/rl-experiment-log.md` 和 `docs/kick-transition-development.md`。
