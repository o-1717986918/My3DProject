# 球员动作训练主线：实战状态、技能与过渡

日期：2026-09-20。执行环境：WSL Ubuntu 22.04、`my3d-rl`、RCSSServerMJ Booster T1；大体积日志和 checkpoint 只放 `/home/win98/rl_runs`。

## 目标与边界

目标是让球队更快到球、更常在运动中完成有效触球，并逐步获得远距离地面传球、强射门、挑射和可靠的动作间切换。门将短距出击、封堵、解围与恢复是并行专项。**比赛能力以端到端结果定义**：到位时间、控球/出球结果、跌倒后的总耗时及比赛中的触发频率；既不因一次跌倒否定快动作，也不因 ONNX 可加载就宣称技能可用。

当前重建版二进制默认保留 Apollo Walk/GetUp，动态直传 selector 和门将拦截默认开启。FastWalk、RapidTurn、参数化踢球和 learned kick 有接口及候选模型，但在 `runtime/apollo_rebuild/src/app/runtime_config.h` 中默认关闭。现有模型继续作为训练初始化和可比较候选，不删除。

已有服务器结果比 CPU 教师更严格：先前 113 次覆盖十个 prototype 的 RCSS 动作结果中，只有 7 次满足窄域 2 m 直传标签（见 `docs/rl-experiment-log.md` 的 2026-09-13 节）。因此当前 CPU 回放只能定位训练缺口，不能用它的成功率推断 RCSS 传球成功率。

## 检索形成的技术选择

- [T1 striker 官方代码](https://github.com/Daffan/humanoid-soccer)：追球教师、定向踢、DAgger 学生、噪声观测下的适应分阶段进行。应借鉴运动中触球及教师—学生的训练顺序，不直接搬其 Isaac Gym 模型。
- [RoboNaldo 论文](https://arxiv.org/abs/2606.11092)与[公开代码](https://github.com/OpenDriveLab/RoboNaldo)：动作跟踪先验 → 小范围静止球 → 扩大位置 → 动态来球/触发。已有 2/3.5/5 m 踢球档位可充当本项目的初始先验；G1 权重和球物理不能直接使用。
- [PAiD 官方代码](https://github.com/TeleHuman/HumanoidSoccer)现已公开训练任务、动作数据、checkpoint 和 MuJoCo sim2sim 入口。其“动作跟踪 → 球位/滚动球课程 → sim2sim”的结构可用于设计本项目课程；机器人是 Unitree G1，且许可证为 CC BY-NC 4.0，故不复制权重或源码进 T1 比赛运行时，只借鉴训练分期并以本项目物理复现。
- [BeyondMimic 开源跟踪框架](https://github.com/HybridRobotics/whole_body_tracking)：参考动作先要满足动力学和控制约束，失败阶段自适应取样。我们此前原始“跑步”参考在精确物理中不可持续，不应重复把奖励权重加大当作解决办法。
- [SkillX 预印本](https://arxiv.org/abs/2609.06718)：单策略组合多种足球技能是远期方向。目前其证据不能代替本项目的可用过渡数据，先做好分项技能及共享状态合同。
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)：继续用现有 MJX/Warp 向量环境训练；TensorBoard 看曲线，单轨迹 MuJoCo Viewer 看动作，RCSSServerMJ 验证实际比赛。可选 rscope，不把实时渲染绑到所有训练环境。

## 已实施：比赛指令回放入口（2026-09-20）

新工具 `training/tools/collect_match_kick_handoff.py` 从重建版球员的连续 `PlayOn/Walk` 状态日志取**新鲜且有效的近球观测**，把记录的身体局部走路指令及首帧局部球位放入精确 CPU RCSS T1 场景，从已有 Apollo Walk handoff 状态起始，按 50 Hz 回放并采集完整 `qpos/qvel`、上一动作、指令、球相对位置、步态相位、支撑提示及来源 ID。输出有 SHA-256、参数、来源日志与按连续 trace 分组的划分；`release_like_geometry` 只是数据量诊断，**不是新的比赛释放门槛**。

来源界限必须保持清楚：指令和首帧球位来自服务器日志，之后的关节与球轨迹来自单机器人 CPU 物理重建，**不是服务器直接记录的真实关节状态**。因此该语料适合入口分布探索与训练初态扩充，仍需独立服务器检验。

`training/tools/evaluate_match_kick_handoff.py` 选每条回放中最接近 2 m 教师锚点的一帧，运行已冻结的教师动作，避免把相邻帧当成独立成功次数。状态读取工具同时修复了 0.20 s 日志间隔的浮点误切段，并禁止无效或过期球位进入近球样本。

| 来源 | 回放状态 | 类触球几何帧 | 独立回放 | 冻结 2 m 教师 |
| --- | ---: | ---: | ---: | --- |
| 正常 7v7，45 s 高频日志 | 156 | 9 | 仅 1 条真正类触球回放 | 1/1 触球，0/1 达目标，0 跌倒 |
| 受控近球场景，25 s 高频日志 | 395 | 69 | 5 条类触球回放 | 5/5 触球，1/5 达目标，0 跌倒 |

自然比赛来自 `/home/win98/rl_runs/training-transition/rebuild-match-s20260920-v1`；受控场景来自 `/home/win98/rl_runs/training-transition/near-ball-fixture-s20260922-v1`。对应语料与教师探针分别保存在同级 `match-kick-handoff-s20260920-v4` 和 `match-kick-handoff-fixture-s20260922-v2`。受控数据中有约 0.52–1.04 m/s 的类触球入口速度，说明它确实覆盖“运动中释放”的问题，但 5 条回放和 1 条自然比赛回放远不足以评价训练后的泛化。教师结果尤其不是比赛射门/传球成功率。

## 接下来的最短可行训练闭环

### 服务器真实关节遥测（同日增量）

重建版新增默认关闭的 `--training-telemetry-interval N`。在 `PlayOn` 每 N 个周期以固定 23 关节顺序记录**服务器实际观测**的关节角/角速度、躯干位置/朝向/机体速度、陀螺仪、球估计，以及本周期真正发送的电机目标。遥测 schema 2 补齐目标速度、PD 增益与附加力矩；旧 schema 1 缺这些控制量，不能用于精确单步对照。与上面的单机器人回放不同，这些观测来自比赛现场；但它们仍**不是完整的** MuJoCo `qpos/qvel`：脚接触、其他球员和部分球速不可观测。首关节在运行时叫 `Head_yaw`，训练合同叫 `AAHead_yaw`，顺序一致但名称需显式映射。

启动脚本可用 `APOLLO_REBUILD_TRAINING_TELEMETRY_INTERVAL=1` 开启；`training/tools/collect_server_motion_telemetry.py` 把一个或多个比赛日志转换为带来源哈希的 NPZ。只在**不同比赛**之间划分训练/验证；同场的不同球员、相邻时间帧不是独立样本。两场 18 秒 wall-time 的 7v7 烟测均有 14/14 客户端存活：正常比赛得到 7,133 帧、76 帧新鲜近球状态，日志 6.6 MB；受控近球比赛得到 9,415 帧、115 帧新鲜近球状态。合并归档位于 `/home/win98/rl_runs/training-transition/server-telemetry-corpus-s20260920-v2`，受控赛为训练、正常赛为留出，合计 16,548 帧、191 帧新鲜近球状态。两场仍不足以报泛化成功率，也不能把旧动作发送的目标关节角当作优秀动作标签。

### 实测状态校准与现有模型复测（同日第二阶段）

已核对 RCSSServerMJ 安装源码：位置/朝向感知器绑定在 T1 的 `torso` site，该 site 与 free-joint root 同位；陀螺仪为躯干局部角速度。`training/my3d_rl/server_motion_state.py` 将这几项与服务器关节传感器、球估计投到精确单 T1 场景；只有连续稳定 Walk 的上一周期电机目标可反算 23 维上一动作，头两维因头部跟踪器覆写而置零。不把此反算用于其他运动模式。

`training/tools/evaluate_server_motion_parity.py` 用**当前帧发送的完整 PD 控制**预测下一帧。在 2 场分组、424 个诊断帧中，留出正常赛近球 40 帧的中位误差：关节角 0.45°（“完全不动”基线 0.97°）、躯干位置 0.15 cm、朝向 0.56°；关节角速度仍差约 20°/s。陀螺仪的 body-frame 映射误差约 `1e-8 rad/s`，world-frame 假设约 `0.05 rad/s`。这些是相邻帧、不是 424 次独立试验；单步一致性足以允许**探索**实测初态，但速度/接触误差不允许声称完整仿真一致。

又跑了两个未用过的左右侧/横偏受控比赛，共 5 场、29,022 帧、630 帧新鲜近球；真正满足面向前方、近触球几何的只有 29 帧，合并为 10 次接近，**全部来自 4 场受控赛**，自然赛为 0 次。`training/tools/evaluate_server_kick_handoff.py` 从每次接近只取一帧，使用重建的上一 Walk 动作比较：冻结程序教师 8/10 触球、1/10 达 2 m 标签；旧 r2 ONNX 7/10、0/10；r3 ONNX 9/10、1/10；三者均无跌倒。r3 的触球覆盖值得作为训练先验保留，距离、方向及不同初态的出球差异仍大，**没有独立自然赛释放样本，不能由此升级默认球队**。报告位于 `/home/win98/rl_runs/training-transition/server-telemetry-v2-five-matches-s20260920/teacher-vs-onnx-probe-action-reconstructed.json`。

用两场不同侧别的动态入口做 8×4 次低维 CEM 教师搜索，只在一条训练入口得到达目标结果，全部 10 次接近仍是 1/10；见同目录 `server-teacher-cem-previous-action-v2.json`。这说明再对少量固定初态强行调教师会过拟合。现已可用 `training/tools/render_soccer_motion_replay.py` 将任意保存的 50 Hz 踢球轨迹渲染为 MuJoCo 接触图；同目录有 `teacher-contact-sheet-previous-action.png` 和 `r3-model-contact-sheet.png`，后者为 r3 一次窄域成功的可视化，不代表统计结论。

下一步优先增加**独立且多样的真实释放入口**，特别是正常比赛中能够到达触球几何的进入过程；随后以 r3 作为接触先验，针对方向/距离/触球后恢复训练修正或学生，而不是继续调一个对 10 状态都不稳的固定程序轨迹。继续并行推进快速移动参考与过渡训练；最终依据球权和净到位/出球收益比较，不把单一窄 2 m 标签当作全部比赛价值。

1. **补数据而非先加网络层数。** 用左右镜像、不同接近距离/横偏角、走路指令变化和多场正常比赛扩展入口。受控场景可用于训练，自然比赛按整场或连续 trace 留出。先确保训练与留出两侧都有多条独立类触球回放；现有自然场景虽有 78/78 帧划分，真正类触球样本只来自一条回放，不能据此报泛化率。
2. **获得可学的动作标签。** 对近球入口先复用固定 2 m 教师和已存档的距离动作档位；按实际触球、落点、方向、速度、恢复与跌倒分别记录。为成功轨迹做阶段条件 BC，再对学生偏离状态做 DAgger，最后用受约束 RL 微调。静态程序动作和学习过渡动作要有独立入口合同；不能先用静态腿速条件拦掉学习策略的动态输入。
3. **快速移动与过渡并行。** 从现有 Apollo Walk/回放状态和物理上可行的参考起步，训练连续加速、刹停、转身、Walk↔FastWalk/Run↔Kick↔Walk。比较净到位时间、侧漂、跌倒/起身耗时；真奔跑的腾空相位是专项，不替代比赛速度指标。先在精确 CPU 检查动作参考，再在 MJX/Warp 扩规模。
4. **扩展球路而不混淆任务。** 地面短传 → 中长距离传球与强射门，分别测落点及到达速度；挑射另测腾空高度、越过障碍后的落点和落地速度。门将手抛球若列入“抛射”，需单独动作与规则验证。不同球路可以最终共享条件策略，但第一版不假设一个固定踢球轨迹可连续覆盖全部距离。
5. **三层验证。** MJX/Warp 训练曲线与可视化回放 → 独立精确 CPU 初态/物理评价 → RCSS 自然比赛 A/B。记录成功动作的使用率和比赛收益，不以单场比分或任意高准确率阈值替代因果比较。

这一阶段交付的是**可复现的真实指令分布回放、直接服务器关节遥测、单步物理校准、已有 ONNX 的实测入口复测和可视化、教师小搜索的负结果**。后续快走真实入口续训见下文；默认比赛动作尚未更改。踢球下一轮仍需补独立入口和高质量触球序列；没有多样的成功衔接标签时，不继续做弱监督的长 PPO。

## 快走真实入口配对诊断（2026-09-21）

新增 `training/tools/evaluate_onnx_run.py --server-corpus`：从上述 schema-2
遥测中抽取直立、正在 Walk、身体平面速度至少 0.2 m/s 且前一帧连续的真实入口；
同一球员同一场每 2 秒最多取一帧，并保留近球和远球的分组。初态用已经做过
单步校准的映射器投到精确 CPU T1；Apollo Walk 反算上一动作，FastWalkV2 按
实际运行时从零上一动作、零步态相位接管。保持原观测球位，再做移走球与步态相位
消融。报告包含逐入口存活、速度、偏航、横漂、球—机器人接触及来源行号；这是
**单机器人接管诊断，不是 RCSS 比赛成绩**。

从五场归档抽取 40 个起点（近球 7、远球 33，最后一场 7 个），同一批起点各
模拟 3 秒。原版 Apollo 用其运行时前向命令 `1.0 m/s`，FastWalkV2 用其运行时
固定前向命令 `1.5 m/s`；指令不同，因此速度只能解释为**两个部署路径的输出**，
不是同命令模型优劣。原版 40/40 直立、前速中位 0.868 m/s、绝对横漂中位
0.076 m；`fast_walk_transition_recovery` 37/40 直立、前速中位 1.366 m/s、
绝对横漂中位 0.416 m、偏航跟踪 RMSE 中位 0.195 rad/s（原版 0.071）。
两次跌倒来自近球受控场景，一次来自远球入口；移走球后仍有两次跌倒，说明
接管后的持续步态/横向偏差本身就有缺口，另一次可能与球接触有关。把初始步态相位
改为 0.25/0.5/0.75 的假设性消融分别为 39/40、38/40、38/40 直立，横漂
中位仍为 0.392/0.369/0.462 m；不能仅调整相位常数挂载。

**跌倒不是一票否决。** 以跌倒时立即停止计算、不给快走补偿起身后的任何位移，
快走在 36/40 个配对起点的 3 秒前向位移仍更大，均值 3.576 m 对原版
2.515 m；三次跌倒起点则是明显的局部损失。下一轮优化的是“前进/到球收益减去
横漂、失衡与起身损失”，而不是把跌倒率压到零作为唯一目标。本诊断的旧静态
`candidate_gate`/十命令 `soccer_command_gate` 对实测入口不适用，报告置为 `null`；
仍保留逐入口原始数据供比较。

最终配对报告为 `/home/win98/rl_runs/training-transition/server-run-{apollo,fastwalk}-final-40x3-s20260921.json`；
球和步态相位消融位于同目录的 `server-run-*-40x3-s20260921.json`。
例如复测当前候选：

```bash
PYTHONPATH=training /home/win98/miniconda3/envs/my3d-rl/bin/python \
  training/tools/evaluate_onnx_run.py \
  --server-corpus /home/win98/rl_runs/training-transition/server-telemetry-v2-five-matches-s20260920/server-motion-telemetry.npz \
  --episodes 40 --duration-s 3 --warmup-s 0.4 --seed 20260921 \
  --vx 1.5 --model /home/win98/rl_runs/stable-motion/fast-walk-transition-recovery-s20261160-v1/policy.onnx \
  --contract training/contracts/run_policy_v2.yaml \
  --output /home/win98/rl_runs/training-transition/<new-report>.json
```

这 40 条虽按时间稀疏，仍来自仅五场、且多条来自同一球员；近球初态全是受控场景，
仅七条属于最后一场留出。单机器人精确 CPU 缺少对手和队友碰撞，球位是服务器估计；
不能把 37/40 当真实比赛跌倒率，更不能据此自动启用 FastWalk。下一轮以现有
checkpoint 为初始化，不重头训练：先加入**真实 Walk 姿态和球位的 reset/replay**，
课程混合纯前进、接球前刹停、球旁绕行与恢复，优化前速同时约束横漂、偏航和倒地
总耗时；同一批入口先跑本工具配对，随后跑原有十命令面与真实 7v7。只保留确实
改善净到位/控球收益的候选。这个选择也与 [ICRA 2026 T1 striker 官方训练结构](https://github.com/Daffan/humanoid-soccer)
中的追球步态 → 定向踢 → 感知学生分阶段路线一致；训练数据与物理仍使用本项目
RCSS T1，不复制其 Isaac Gym 权重。

同日两场 25 秒 wall-time 的真实 7v7 开/关快走烟测均保持 14/14 客户端、
周期状态中均无 GetUp。开启版仅记录 29 个 FastWalkV2 状态样本（总 2,535），
而两场实际模拟进度不同；它证明接口可运行，却不足以比较球队净收益。
日志分别位于 `/home/win98/rl_runs/training-transition/rebuild-fastwalk-ab-{off,on}-s20260921`。
`scripts/run_apollo_rebuild_match.sh` 新增仅本次运行生效的
`APOLLO_REBUILD_ENABLE_FAST_WALK=1` 与可选 `APOLLO_REBUILD_FAST_WALK_MODEL`；
不改变重建版默认设置。
下一轮真实比赛需延长或设计更多确会进入高速前向职责的场景，再看到球、球权和
恢复成本，不能用这两场无跌倒或比分作晋级依据。

## 真实比赛状态快走续训（2026-09-21）

`training/tools/build_server_run_reset_corpus.py` 从上述五场 schema-2 服务器遥测中，
每名球员每 0.2 秒最多保留一个动态、直立且速度至少 0.2 m/s 的 Walk 状态。
它复用已校准的单 T1 状态投影，保留原有**按比赛划分**的训练/验证标签和来源行号；
共 2,514 个初态，训练 2,213、验证 301，其中近球分别只有 63 和 4 个。
这是量化观测的单机物理重建，不是完整 RCSS 状态；脚接触、其他球员仍有缺失。
语料及哈希清单位于
`/home/win98/rl_runs/training-transition/server-run-reset-corpus-s20260921/`。

`ServerFastWalkRun` 在这些初态上按运行时合同重置 FastWalkV2：固定前向指令
1.5 m/s、步态相位零、上一动作零；训练时以 20% 概率取近球初态，验证时按原始
比例取样。`training/tools/train_server_fastwalk.py` 从已有
`fast-walk-transition-recovery` checkpoint 继续 196,608 步的 MJX/Warp PPO，
损失同时关注前进、偏航和路径横漂，跌倒有有限代价而非绝对禁止。
产物在 `/home/win98/rl_runs/training-transition/server-fastwalk-continuation-s20260921-v1/`：
训练完成，80→23 ONNX 导出数值一致性通过（最大绝对误差 `1.05e-5`）。
8 个验证 rollout 最终均到 150 步上限，但验证近球仅 4 个初态，不能据此称比赛稳定。

同一 40 个服务器入口、各 3 秒的精确 CPU 配对结果如下。候选只作为**前进专项**
评估，不能把静态十命令测试或单机模拟结果当作比赛净收益。

| 模型 | 直立结束 | 平均前向位移 | 绝对横漂中位 / P90 | 偏航 RMSE 中位 / P90 |
| --- | ---: | ---: | ---: | ---: |
| 续训前 FastWalkV2 | 37/40 | 3.576 m | 0.416 / 0.970 m | 0.195 / 0.353 rad/s |
| 实测入口续训候选 | 38/40 | 3.693 m | 0.429 / 1.031 m | 0.193 / 0.289 rad/s |

候选在 29/40 个起点增加前向位移、少了一个跌倒终局，但横漂总体略差。配对
十命令套件从 8/10 项通过变为 7/10：前向两者均 8/8，候选纯右横移
7/8（原模型 8/8），倒退两者均 0/8。它说明当前训练目标过窄，不能把该 ONNX
替换成所有方向的默认动作。报告分别为同产物目录下的 `server-40x3.json`、
`parent-command-suite/summary.json` 与 `candidate-command-suite/summary.json`；
旧模型对照是 `server-run-fastwalk-final-40x3-s20260921.json`。

**决策原则修订：偶发跌倒可接受，但必须计算净效益。** 比较相同任务的实际到球
时间、路径偏移导致的丢球、倒地到重新参与比赛的耗时和动作触发率；不设零跌倒
硬门槛，也不单凭更快的直线前进晋级。下一轮优先改善横漂与转向、刹停，并用
足够长且角色/侧别配对的 RCSS 比赛检验球权收益；当前默认 FastWalk 仍关闭，
该候选保存在外置训练目录，可通过比赛脚本的模型路径变量单次试用。

同脚本各 50 秒墙钟的两次探索性真实 7v7 均为 14/14 客户端存活：续训候选有
68 次 FastWalk 状态采样、1 次 GetUp 状态采样；旧模型分别为 67 和 26 次。
状态采样**不是独立的跌倒次数**，候选那次 GetUp 发生在快走最后一次触发约
1.9 秒之后、当时执行的是 Walk；旧模型的 GetUp 也可能混有后续转向和接触。
两场球路、角色轨迹不相同，且 FastWalk 使用占比很低，不能从 1 对 26 推断
跌倒率改善或比赛胜率改善。完整日志保存在候选目录的
`live-rebuild-vs-base-50s/` 和 `live-parent-vs-base-50s/`；下一轮需要同侧别、
多场景配对及到球/出球净收益分析。

## 快走入口分布与航向偏置消融（2026-09-21）

上一轮 2,514 个真实 Walk 重置状态中，只有 1,312 个满足**观测运动学近似**：
前向速度至少 0.5 m/s、侧速绝对值不超过 0.25 m/s、陀螺仪不超过 55°/s、
躯干倾斜不超过 6°。遥测没有高层 WalkCommand，此标签**不是**真正运行时
FastWalk 门控。`build_server_run_reset_corpus.py` 现在保留此标签，语料 schema 2
位于 `/home/win98/rl_runs/training-transition/server-run-reset-corpus-forward-proxy-s20260921-v2/`；
按整场划分仍为训练 2,213、验证 301，其中近似正向分别 1,140 和 172。
训练环境可将正向近似作为主要初态，仍抽取其他侧移/转向状态用于恢复。

从原 FastWalk checkpoint **独立**续训的 v2 使用非近球样本中 80% 正向近似、
10% 近球样本，路径横漂与朝向惩罚分别从 12/8 调到 24/12，完成 196,608 步。
ONNX 导出校验通过，最大误差 `8.58e-6`；产物及配置清单在
`/home/win98/rl_runs/training-transition/server-fastwalk-continuation-s20260921-v2-forwardmix/`。
续训命令的关键参数如下；输出目录必须是新的 WSL 路径：

```bash
PYTHONPATH=training XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/win98/miniconda3/envs/my3d-rl/bin/python training/tools/train_server_fastwalk.py \
  --reset-corpus /home/win98/rl_runs/training-transition/server-run-reset-corpus-forward-proxy-s20260921-v2/server-run-resets.npz \
  --restore-checkpoint /home/win98/rl_runs/stable-motion/fast-walk-transition-recovery-s20261160-v1/checkpoints/000001179648 \
  --run-dir /home/win98/rl_runs/training-transition/server-fastwalk-continuation-s20260921-v2-forwardmix \
  --num-envs 64 --num-timesteps 196608 --num-evals 2 --num-eval-envs 16 \
  --near-ball-probability 0.10 --forward-entry-probability 0.80 \
  --path-lateral-penalty 24 --heading-drift-penalty 12 --seed 20260922
```

`evaluate_onnx_run.py` 新增只抽**留出比赛**与正向近似的选项，32 个按球员/时间
稀疏的相同初态各跑 3 秒：

| 模型 | 直立结束 | 平均前向位移 | 绝对横漂中位 / P90 |
| --- | ---: | ---: | ---: |
| 原 FastWalk | 32/32 | 3.873 m | 0.302 / 0.712 m |
| v1 实测入口续训 | 32/32 | 3.875 m | 0.315 / 0.653 m |
| v2 正向混合续训 | 32/32 | 3.871 m | 0.346 / 0.647 m |

v2 未改善留出集前进或典型横漂，不能晋级。这里模型偏转呈正向偏置：原模型
偏航速率中位 `+0.097 rad/s`，横漂也同向。对原模型做**推理时**镜像双路平均，
留出正向入口横漂中位降至 0.230 m，却多 1 次倒地，宽域 40 入口则从
37/40 降到 35/40 直立、十命令套件从 8/10 降到 6/10；不采用为通用方案。

更轻的 `−0.05 rad/s` FastWalk 策略输入补偿，在相同的留出正向入口得到
32/32 直立、平均前进 3.888 m、横漂中位/P90 为 0.256/0.578 m；宽域
40 入口为 38/40 直立、平均前进 3.622 m、横漂中位 0.290 m，但横漂
P90 从 0.970 升到 1.007 m。`−0.10` 和 `−0.15` 在部分入口退步，故只保留
小幅值作为**试验选项**。`scripts/run_apollo_rebuild_match.sh` 可用
`APOLLO_REBUILD_ENABLE_FAST_WALK=1 APOLLO_REBUILD_FAST_WALK_YAW_BIAS=-0.05`
单次启用；二进制默认偏置仍为零、FastWalk 仍默认关闭。此补偿属于部署控制
消融，不是新训练出更强全向模型，也不能替代真实转身和横向移动训练。

同侧 50 秒墙钟的真实 7v7 有/无补偿试跑均为 14/14 客户端存活，但 FastWalk
状态采样分别只有 36 与 90、GetUp 状态采样分别为 31 与 30；轨迹分叉明显，
不能从这些数字归因比赛收益或跌倒率。CPU 同初态结果足以支持保留试验入口，
不足以默认启用。下一步要么在可重复的定点跑向球场景中增加有效快走暴露，
要么用服务器状态重新训练能直接接管转向、刹停的策略，而非继续调整静态惩罚。
现有定点球场景也试跑了各 20 秒：7 号队员有/无补偿分别仅 16/15 个
FastWalk 采样帧，不能用于净到位比较；需要重新设计更长的可重复前向职责。

## 移动中触球入口扩充与动作库诊断（2026-09-21）

`scripts/collect_rebuild_kick_approaches.sh` 在现有重建版对原版 Apollo 的 7v7
比赛内做重复 reset，保留真实服务器 Walk 状态、感知和电机目标。`train`、
`holdout` 使用两组固定且不同的机器人姿态/滚动球速度；`random` 用显式 seed
生成新的比赛级分布。它们都是**受控比赛入口**，不是自然比赛中自主形成的射门。
脚本不改变正式比赛启动器，且大文件只写 `/home/win98/rl_runs`。

首两场共 41,083 帧、1,269 帧新鲜近球观测；按连续接近去重后为训练 8、留出
7 次释放入口。冻结程序教师与 r3 学习策略均为 12/15 触球、0 跌倒；狭义 2 m
目标分别 0/15 与 3/15。随后增加两个新 seed 的随机场景，四场合计 86,198 帧、
2,395 帧新鲜近球观测及 30 次独立接近；固定教师为 25/30 触球、1/30 狭义
成功，r3 为 27/30、3/30，仍均为 0 跌倒。语料和报告分别在：

- `/home/win98/rl_runs/training-transition/kick-approach-corpus-s20260921-v2-multimatch/`
- `/home/win98/rl_runs/training-transition/kick-approach-corpus-s20260924-v3-four-matches/`

新增诊断可按入口的局部球位，从已有静态 2 m 教师网格选最近 8 个动作。直接取
最近动作，首两场有 3/15 狭义成功；若事后查看 8 个候选的最好结果，10/15 至少
存在一个成功动作。扩展到四场后，事后上限为 17/30。这个 oracle **不能部署**，
但说明动作库仍有可复用信号：当前缺口同时包括移动状态下的动作选择和动作衔接，
不能只靠进一步放宽触发条件。8 候选中偶有跌倒；遵照比赛目标，跌倒记入失去球权、
起身和重新参与比赛的总成本，不作一票否决。

下一步用多场入口训练一个只读取释放时已有观测的动作选择/修正学生，按整场划分；
先比较触球、前向进展、横向偏差和恢复成本，再扩展 3.5/5 m 地面球、强射与挑射。
只有精确 CPU 留出和 RCSS A/B 均显示净比赛收益，才默认挂载；零跌倒或单一 2 m
命中率都不是独立晋级门槛。

### 首轮服务器入口 DAgger

`training/tools/collect_server_kick_dagger.py` 只从前三场训练分区选择在精确物理中
确实成功且未跌倒的动作库轨迹，用旧 r3 学生自己的闭环访问状态生成教师动作标签；
13 个入口增加 1,963 帧，原 185,365 帧数据仍保留。第一候选将新入口权重设为
12，训练产物在 `/home/win98/rl_runs/training-transition/kick-server-dagger-s20260925-v1/`。

| 评价集 | r3 触球 / 2 m 窄域 / 跌倒 | 新候选触球 / 2 m 窄域 / 跌倒 |
| --- | ---: | ---: |
| 四场 30 次接近（第 4 场整场留出） | 27 / 3 / 0 | 29 / 4 / 0 |
| 第 4 场留出 7 次 | 7 / 0 / 0 | 7 / 1 / 0 |
| 训练完成后新采 seed 的 6 次 | 3 / 0 / 0 | 5 / 0 / 0 |

新 seed 上最大前向进展中位数由 0.287 m 增至 1.245 m、到 2 m 的距离误差中位数
由 1.713 m 降至 0.755 m，但横向误差中位数由 0.154 m 增至 0.623 m。因此这是
“更常有效触球、方向尚不稳定”的训练候选，不是可默认启用的精确传球。保持旧 r3
为回退，不接入正式比赛。

为排除单一超参数偶然性，又以和旧 r3 相同的 seed、batch size、学习率训练了
权重 6 的 v2；四场窄域结果退到 1/30，新 seed 虽也是 5/6 触球但 0/6 命中。
该负结果保留在 `kick-server-dagger-s20260925-v2-weight6/`，不晋级。下一轮扩大
独立 seed/侧别并训练带显式方向、距离和触球后恢复目标的修正策略；不再只调 BC
样本权重。

### 真实 Walk 上一动作与方向修正试验

复查训练环境发现：服务器姿态虽已进入 `DirectionalKick`，Apollo Walk 的上一
动作仍被重置为零；而精确 CPU 评估会保留该 23 维状态。现在 transition corpus、
训练环境和 CPU/Warp parity 工具统一传递 `walk_previous_action`，旧语料没有该字段
时才显式回退为零。`build_server_kick_transition_corpus.py` 从上述四场生成 30 个
入口（训练 23、整场留出 7），30/30 都成功反算上一动作；产物位于
`/home/win98/rl_runs/training-transition/kick-server-transition-corpus-s20260925-v1/`。
同一入口 60 步的 CPU/Warp 对照通过，最大关节角误差约 `1.68e-5 rad`、球位误差
约 `2.27e-5 m`。

在首轮 DAgger 候选上训练 0.1 幅度 PPO 修正，横向成本提高到 2、跌倒代价降为
有限的 5；262,144 请求步实际产生最终 294,912 步 checkpoint，全程留出评估无
跌倒。加速后端窄域事件却由零修正的 0.25 降到最终 0.125，累计横向误差基本
未改善。精确 CPU 的已知四场从 4/30 降到 2/30；训练后新 seed 虽从 0/6 变为
1/6，但横向误差中位仍由 0.623 m 增到 0.892 m，不能晋级。缩放消融中 0.025
在已知四场为 5/30、但新 seed 仍 0/6 且触球从 5/6 降到 4/6，也没有稳定收益。
完整负结果在 `kick-server-correction-s20260926-v1/`。

这次失败说明当前 30 个入口不足以用稀疏 PPO 事件学好方向，并非跌倒惩罚不够。
下一轮先增加左右侧、不同球速的独立入口和可用动作标签，再做显式落点/横向监督
或更密集的方向课程；保持 DAgger v1 为训练候选、旧 r3 为正式回退。

### 左右侧扩展与第二轮 DAgger

采集脚本现支持 `KICK_APPROACH_SIDE=left|right`：右侧的 raw monitor 坐标、朝向和
球速做 180° 旋转，遥测仍由运行时输出统一球队坐标。两场右侧及一场后采左侧均
保持 14/14 客户端；第一场右侧新 seed 上，r3 为 5/5 触球、0/5 窄域、1 次跌倒，
DAgger v1 为 5/5、1/5、0 跌倒。第二场右侧被完整保留为下一轮验证。

七场合并语料有 151,928 帧、4,182 帧新鲜近球观测、50 次独立接近；最后一场
右侧包含 9 次留出接近。总体 r3 为 42/50 触球、4/50 窄域、1 跌倒，v1 为
48/50、5/50、0 跌倒；但右侧留出分别是 1/9 和 0/9，证明 v1 的总体改善没有
稳定迁移到新右侧分布。语料位于
`/home/win98/rl_runs/training-transition/kick-approach-corpus-s20260928-v4-seven-matches/`。

第二轮以 v1 学生访问状态、21 个成功动作库教师入口重新生成 3,171 帧 DAgger
数据，训练得到 `kick-server-dagger-s20260930-v3-sevenmatch`。总体提升到 47/50
触球、9/50 窄域、0 跌倒；右侧留出仍为 8/9 触球、0/9 窄域。其留出前向进展
中位数约 3.45 m，但横向末端误差中位约 0.61 m，不能当作精确 2 m 传球。
训练后新采左侧 seed 上 v3 为 7/7 触球、0/7 窄域，并没有形成可复现的长传走廊。
因此 v3 也只保留为接触/强出球研究候选，不接入正式比赛。

当前证据进一步收敛了问题：动作衔接学习可稳定提高触球覆盖，但距离和方向仍耦合；
下一轮需要把“触球”“落点/走廊”“恢复”分阶段或多头优化，并增加不同目标距离的
真实标签，而不是继续用同一个 2 m BC 数据集加权。

### 3.5/5 m 动态进入状态教师

静态 3.5 m 与 5 m 程序教师在标准初态分别达到 3.552 m / 0.024 m 横偏和
4.970 m / 0.042 m 横偏，均未跌倒；它们可作为腿部发力先验，但不是动态长传
能力。直接从 50 个真实服务器接近状态释放时，两者均为 43/50 触球、1/50
狭义命中。此前工具还会丢掉 Apollo Walk 的上一动作，使优化和回放与线上入口
不一致；现在逐状态优化、教师回放、ONNX 留出评价和动作原型评价均显式继承这
23 维状态。静态距离教师的单文件 manifest 也统一进入同一套可复现工具链。

在七场语料的 41 个训练入口上，以静态动作作初值、两轮有限预算 CEM 生成标签：

| 目标 | 静态初值命中 | 第一轮 | 失败入口修复后 | 触球 / 跌倒 |
| --- | ---: | ---: | ---: | ---: |
| 3.5 m 地面长传 | 1/41 | 18/41 | 29/41 | 41/41 / 0 |
| 5 m 强出球 | 1/41 | 15/41 | 26/41 | 40/41 / 0 |

这证明现有机器人和动作参数空间能从比赛步态直接产生长球，不必先回到静止姿态。
但 29 个 3.5 m 动作原型在未参与优化的右侧整场入口上，全部原型的事后上限仅
2/9，按训练集选出的 6 个原型仅 1/9。由 29 条成功轨迹生成的 4,379 帧 BC
学生，虽然验证 MSE 为 0.00216，闭环留出仍为 0/9 命中、5/9 触球、0 跌倒；
低监督损失不能替代闭环结果。

随后又用新 seed 采集一场右侧随机接近，14/14 客户端正常，增加 8 个训练入口，
同时保持原 9 个右侧入口完全留出。八场语料共 58 个独立接近（49 训练、9 留出），
3.5 m 逐状态标签修复到 31/49，其中新增右侧为 5/8。动作库留出事后上限提高到
3/9，训练选择的 8 个原型为 2/9；加入右侧标签的 BC 学生触球提高到 7/9，
但仍为 0/9 命中。其留出中位最大前向进展仅 0.274 m、中位距离误差 3.226 m、
中位横偏 0.088 m，说明单头回归把多套相位相关的强动作平均成了弱动作。

因此本阶段不挂载这些长传 ONNX。下一步优先增加独立右侧、不同进入速度和支撑脚
数据，并用可部署的离散原型门控/混合专家保持踢球力度；门控必须在未见比赛上优于
固定动作后再进 RCSS A/B。跌倒继续按有限比赛成本处理：允许偶发跌倒，但统计
倒地到恢复、丢球和重新参与比赛的总损失，不设置零跌倒硬门槛。本轮恰好没有跌倒，
不把它解释为能力已安全晋级。

主要产物：

- `/home/win98/rl_runs/training-transition/kick-server-teacher-labels-3p5m-s20260932-v2-repair/`
- `/home/win98/rl_runs/training-transition/kick-server-teacher-labels-5m-s20260938-v2-repair/`
- `/home/win98/rl_runs/training-transition/kick-approach-corpus-s20260939-v5-eight-matches/`
- `/home/win98/rl_runs/training-transition/kick-server-teacher-labels-3p5m-s20260941-v4-repair/`
- `/home/win98/rl_runs/training-transition/kick-server-3p5m-prototype-bank-s20260941-v2-righttrain/`
- `/home/win98/rl_runs/training-transition/kick-server-bc-3p5m-s20260943-v2-righttrain/`

### 16 场动态长传扩充与连续效用动作库

后续又加入左右侧各三场独立随机接近，并把服务器遥测收集器改为可按**多个完整
比赛**留出，而不是只能留出最后一场。最终语料有 16 场、119 个去重交接状态：
13 场 99 条用于训练，最后 3 场 20 条完全留出；119/119 都成功反算 Apollo Walk
上一动作。留出状态的球局部前向距离为 `0.560..0.647 m`（中位 `0.608 m`），
20/20 都不在现有 DynamicPass `0.25..0.50 m` 释放窗内。把场景停留从 2.2 秒
延长到 3.2 秒仍得到同样距离，说明这是真实接近器的动作交接位置，不是 reset
过早造成的假象。训练新过渡动作必须覆盖该入口；是否扩大比赛运行时释放窗则仍需
动作本身通过盲测，不能先放宽一个弱 selector。

动作库评价同时保留严格 3.5 m 成功标签和连续物理得分。得分包含推进、距离、
横偏、球速、触球以及有限的跌倒代价；跌倒不再一票否决，但其起身和丢球损失仍
被强惩罚。初始 49 个成功原型在三场留出上的事后上限为 12/20；按训练集连续效用
选出的 15 个动作也保留 12/20，优于硬成功覆盖选出的 10/20。增量标签工具现在
能按 rollout ID、`qpos/qvel` 和 Walk 上一动作逐项校验后复用旧标签，只优化新增
状态和旧失败状态。49 条成功旧标签加 50 次增量搜索后得到 72/99 成功教师、
99/99 触球、0 跌倒；扩展 72 动作库把留出事后上限提高到 14/20。15 动作压缩
只能保留 11/20；30 个效用动作才能保留 14/20，说明过早压缩动作库会丢失真实
相位覆盖。

两类可部署 selector 均未接近该上限。98 维硬标签 MLP 在留出集释放 8 次、成功
2 次；连续效用软标签版本释放 13 次、成功 1 次，均为 0 跌倒。由 72 条成功教师
生成 10,872 帧完整轨迹后，单 actor BC 为 0/20 成功、16/20 触球。修复 DAgger
对单教师 manifest 的兼容，并把真实 Walk 上一动作传入 learner rollout 后，第一轮
提高到 3/20 成功、20/20 触球、0 跌倒；第二轮仍为 3/20，停止继续堆同一训练。

这批结果不挂载比赛。它证明 3.5 m 动态长传的物理覆盖已到 70% 的盲测 oracle，
主要缺口是多模态动作选择和连续闭环，而非零跌倒。下一模型改为“释放时冻结
latent/专家，执行期连续反馈”的混合专家：先用完整 30 动作效用集蒸馏专家簇，
再训练低维门控并让整段动作保持同一 latent；评价仍用这三场 20 条冻结留出，
随后另采新比赛做最终确认。只有可执行策略而非事后 oracle 明显超过固定动作
`164464` 的 6/20，才进入 RCSS shadow/active A/B。

本轮主要产物：

- `/home/win98/rl_runs/training-transition/kick-server-transition-corpus-s20260959-v7-three-match-validation/`
- `/home/win98/rl_runs/training-transition/kick-server-teacher-labels-3p5m-s20260962-v7-expanded-repair/`
- `/home/win98/rl_runs/training-transition/kick-server-3p5m-prototype-bank-s20260962-v7-expanded/`
- `/home/win98/rl_runs/training-transition/kick-server-bc-3p5m-s20260964-v4-expanded/`
- `/home/win98/rl_runs/training-transition/kick-server-dagger-3p5m-s20260966-v1/`
- `/home/win98/rl_runs/training-transition/kick-server-dagger-3p5m-s20260968-v2/`

### 冻结专家门控、第二批盲测与服务器影子结果

在上述 16 场语料上实现了释放时只选择一次、整段保持同一动作的低秩专家门控。
门控只读释放时已有的 98 维观测，不在执行中事后换专家；数据按
`source_match_id` 整场切分。此前部分 selector 实验把每一帧唯一的
`rollout_id` 当成分组键，实质仍是帧级切分，该结果不再作为泛化证据。

第一版物理分数门控在旧 20 条开发集上为 8/20 严格命中，高于固定动作
`164464` 的 6/20；冻结后新采 3 场 23 条盲测却只有 2/23，对照为 1/23，配对
单侧符号检验 `p=0.5`。加入这 3 场后形成 19 场、142 条入口：16 场 119 条训练，
3 场 23 条开发。严格边界 margin、低秩维数和 ridge 消融都只有 2/23，说明不是
继续调一个线性门控超参数就能解决。

将用途改成“向前强力出球/解围”后，成功契约为触球、最大前进不少于 2.5 m、
绝对横偏不超过 1.0 m、方向速度不少于 1.5 m/s；是否倒地另列，不把一次倒地
从球结果中抹掉。冻结 margin 门控在开发集为 10/23，最佳静态原型 `60467` 为
9/23。第二批全新 3 场 22 条盲测中，门控为 10/22，`60467` 为 11/22；门控没有
超过更简单的固定原型。它虽以 10/22 显著超过旧固定 `164464` 的 3/22
（gate-only 7、baseline-only 0，`p=0.0078125`），仍不能据此上线一个比当前
最佳静态动作更复杂的模型。两批均为 0 跌倒，这里拒绝原因是收益不足而非追求
零跌倒。

`60467` 使用与运行时 `DynamicPassRunner` 相同的 14 参数解码，被加入源码中的
**仅强制影子**原型；旧十输出 selector 永远不会选到它，默认球队行为没有改变。
强制时单独开放真实 Walk 交接所需的 `0.50..0.68 m` 前向、`±0.25 m` 横向窗口。
RCSSServerMJ 在左侧正中、左侧 0.10 m 偏置和镜像右侧三种场景中记录到 4 次完整
执行，4/4 直立，但强力出球为 0/4。只有一次结束时球观测仍新鲜可完整计量：
前进 1.639 m、横移 0.445 m、峰值球速 2.205 m/s；其余执行没有可靠触球证据。
因此 `60467` 也不进入默认策略，影子日志位于：

- `/home/win98/rl_runs/rcss-shadow-60467-left-y0-s20260921-v2/`
- `/home/win98/rl_runs/rcss-shadow-60467-left-ym010-s20260921-v1/`
- `/home/win98/rl_runs/rcss-shadow-60467-right-y0-s20260921-v1/`

评估器现在同时报告 `measurement_valid`、窄 2 m、向前强力出球和 `upright`；球在
动作结束时已陈旧的轨迹不会再伪装成物理成功。下一轮不继续拟合同一个门控，而是
采集运行时逐周期的 Walk 基底目标、接触时刻及服务器真值球轨迹，针对
`0.50..0.68 m` 真实交接窗训练“对齐—发力—恢复”动作；先把固定动作在真实服务器
做成可靠基线，再恢复混合专家。主要冻结产物为：

- `/home/win98/rl_runs/training-transition/kick-server-expert-gate-s20260969-v1/`
- `/home/win98/rl_runs/training-transition/kick-server-expert-gate-s20260970-v7-forward-drive-frozen/`
- `/home/win98/rl_runs/training-transition/kick-server-transition-corpus-s20260970-v10-forward-blind/`
