# nero_dh116 测试脚本技术文档

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 文档名称 | nero_dh116 测试脚本技术文档 |
| 文档版本 | v1.0 |
| 对应入口 | `tests/dh116/test_nero_dh116_lift.py` |
| 适用模型 | Nero 7 自由度机械臂 + DH116 灵巧手 |
| 运行环境 | robosuite Lift + MuJoCo + EGL |
| 控制频率 | 20 Hz |

## 2. 文档目的

本文件说明 `test_nero_dh116_lift.py` 的模块结构、类与函数职责、主循环设计、接触判定、稳定抓取、视频录制和实验日志机制。它既用于复现实验，也用于后续修改轨迹时判断改动是否影响抓取逻辑。

脚本是一个确定性的回归测试和演示程序，不是通用的运动规划器。方块位置、关键关节姿态和控制阶段均被显式固定，以便在模型接入阶段快速定位碰撞、控制器和渲染问题。

## 3. 总体执行流程

脚本启动后按以下顺序执行：

```text
解析命令行参数
    ↓
创建 ExperimentLogger，立即写入 run_started
    ↓
创建 Lift 环境、注册 Nero 和 DH116
    ↓
固定方块到掌心参考位置并 reset
    ↓
把机械臂放到 Nero XML home 姿态，关闭接近阶段碰撞
    ↓
按 home → pregrasp → approach → close → lift 执行主循环
    ↓
接触确认后启动 ContactGraspStabilizer
    ↓
同步记录阶段、状态、视频帧和结束结果
    ↓
写入 run_finished，关闭 FFmpeg 和 MuJoCo 环境
```

主循环每次迭代代表一个控制周期。20 Hz 下，每一步约 0.05 秒；动作向量由 Nero 的 7 个关节位置目标和 DH116 的 6 个手指控制量组成，总维度为 13。

## 4. 常量与轨迹参数

脚本将关键姿态和阶段边界集中放在文件顶部：

| 名称 | 作用 |
|---|---|
| `HOME_QPOS` | Nero 默认零关节姿态，视频起点 |
| `PREGRASP_QPOS` | 预抓取姿态 |
| `GRASP_QPOS` | 掌心对准方块的目标姿态 |
| `PALM_CUBE_XY` | 方块在桌面上的固定平面位置 |
| `HOME_END` | home 保持结束步，当前为 100 |
| `PREGRASP_END` | home 到预抓取插值结束步，当前为 280 |
| `APPROACH_END` | 预抓取到掌心接近姿态插值结束步，当前为 500 |
| `CLOSE_END` | 闭合结束步，当前为 580 |
| `LIFT_MAX_DELTA` | 抬升阶段第 2 关节的最大变化量，当前为 0.30 rad |
| `DEFAULT_STEPS` | 默认总步数，当前为 900，即 45 秒视频 |

机械臂不再在阶段边界直接跳到完整关键帧。`interpolate_qpos()` 对相邻关键帧进行分段线性插值：

```text
home：       0    ───── 100
pregrasp：   100  ───── 280    HOME_QPOS → PREGRASP_QPOS
approach：   280  ───── 500    PREGRASP_QPOS → GRASP_QPOS
close：      500  ───── 580    保持 GRASP_QPOS，夹爪闭合
lift：       580  ───── 900    第 2 关节逐步减少 0.30 rad
```

这样做有两个目的：第一，视频可以清晰表现从初始姿态到目标的连续运动；第二，动作目标的变化率受轨迹时间控制，不完全依赖控制器的饱和和动力学滞后。相比旧版 560 步、0.15 rad 抬升，当前演示的末端移动更慢，方块抬升高度也更明显。

## 5. 类的职责

### 5.1 `ContactGraspStabilizer`

该类负责“接触确认后的演示稳定”，不是接触前的运动控制器。

初始化时，它记录方块自由关节在 MuJoCo `qpos` 和 `qvel` 中的地址，并把状态设为未激活。

`activate_if_grasped(observation, step)` 的职责：

1. 调用 `env._check_grasp(gripper, env.cube)`；
2. 只有左右接触组都满足 robosuite 的抓取条件时才激活；
3. 记录接触步数、方块位置锚点和末端位置锚点；
4. 返回是否在当前步首次激活。

`apply(observation)` 的职责：

1. 未激活时不修改仿真状态；
2. 激活后计算当前末端相对初始末端的平移量；
3. 将方块位置设置为“初始方块锚点 + 末端平移量”；
4. 清零方块自由关节速度并调用 `sim.forward()`；
5. 刷新观测后返回。

因此，稳定器不会让机械臂跳过接近、闭合和真实接触判定。它只在接触已经确认后，抑制当前简化碰撞几何造成的方块滑脱，保证视频能够稳定展示抬升结果。

## 6. 函数职责

| 函数 | 作用 |
|---|---|
| `fixed_palm_placement()` | 创建确定性的 `UniformRandomSampler`，把方块放到掌心参考位置 |
| `set_nero_contact_mode(env, enabled)` | 成组开关掌心、支撑面和基座接触代理的 `contype/conaffinity` |
| `interpolate_qpos(...)` | 在两个 7 维关节姿态之间生成当前步的线性插值目标 |
| `start_video_writer(...)` | 启动 FFmpeg，通过 stdin 接收连续 RGB 帧并编码为 H.264 |
| `run_lift_task(...)` | 创建环境、执行主循环、写日志、写视频并返回结果字典；`stabilize=False` 时进入纯物理诊断 |
| `test_nero_dh116_initializes()` | 验证 Nero、DH116、动作维度和初始观测 |
| `test_nero_dh116_lift_succeeds()` | 验证接触、稳定抓取、Lift 成功和抬升高度 |
| `parse_args()` | 解析步数、随机种子、视频、分辨率和帧率参数 |
| `main()` | 校验参数、调用 rollout、打印摘要和视频路径 |

## 7. 主循环设计

主循环每步执行以下操作：

1. 根据步数选择阶段和机械臂关节参考值。
2. 在 `CLOSE_START`（当前第 500 步）开启 Nero 掌心与桌面接触代理；接近阶段保持关闭，避免桌面反力干扰轨迹。
3. 通过 `robot.create_action_vector()` 拼接 7 维机械臂动作和 6 维夹爪动作。
4. 调用 `env.step(action)` 推进一步 MuJoCo 仿真。
5. 调用 `activate_if_grasped()` 检查是否首次形成左右掌心接触。
6. 调用 `stabilizer.apply()`，必要时维持方块和末端的相对平移。
7. 更新最高方块高度、Lift 成功标志和最大动作幅度。
8. 每 20 步写一条状态日志；阶段切换、接触确认和碰撞开关变化立即写日志。
9. 如果开启视频，把 `sideview_image` 连续写入 FFmpeg stdin。

当使用 `--physical-only` 时，脚本不调用 `ContactGraspStabilizer.apply()`，并额外启用六个 Nero 指尖末端碰撞代理。此模式保留真实 MuJoCo 接触、摩擦和重力结果；只有方块在抬升阶段连续满足成功条件至少 `PHYSICAL_HOLD_STEPS=20` 步，结果才记为成功。这样可以区分“瞬时碰撞弹起”和“持续抓住”。

伪代码如下：

```python
for step in range(steps):
    phase, arm_target, gripper_target = trajectory(step)
    if step == CLOSE_START:
        set_nero_contact_mode(env, True)

    action = make_13d_action(arm_target, gripper_target)
    observation, reward, _, _ = env.step(action)

    stabilizer.activate_if_grasped(observation, step)
    observation = stabilizer.apply(observation)
    success |= env._check_success()
    logger_periodic_state(...)
    write_video_frame_if_needed(observation)
```

## 8. 日志、异常和资源释放

`ExperimentLogger` 在 `run_lift_task()` 进入环境创建前就写入 `run_started`，因此环境初始化失败也有记录。正常完成写入 `completed`；异常写入 `failed`；收到 SIGINT/SIGTERM 时写入 `interrupted`。如果进程被强制杀死到无法执行清理，最后一个 JSONL 文件仍会保留未完成状态，不能被当作成功结果。

日志事件至少包括：配置、项目文件 SHA-256、reset、阶段切换、周期状态、接触确认、结束结果和结束时 SHA-256。脚本会先等待 FFmpeg 正常结束，再把运行标记为 `completed`；如果视频编码失败，会走 `failed` 分支，避免把半成品视频记录成成功结果。异常退出时，资源仍在 `finally` 中关闭或等待。

## 9. 设计边界

1. 方块位置是固定的，当前目标是可重复的模型接入回归，不是随机抓取性能评估。
2. 掌心接触使用简化代理；DH116 原始薄 STL 网格不直接承担 Nero 组合下的主要动态碰撞。
3. `ContactGraspStabilizer` 是接触后的演示稳定机制，不能等价为完整摩擦动力学或真实机器人力控。
4. 轨迹是固定关节空间序列，未实现目标检测、在线 IK 或反馈式末端轨迹规划。
5. 当前 `--physical-only` 基线已经能检测掌心和指尖接触，但固定实验落点下方块最大抬升约 0.0083 m，尚未形成稳定的纯物理抓取。

## 10. 推荐验证命令

```bash
cd /root/gpufree-data/robosuite
export MUJOCO_GL=egl

# 运行 45 秒无头演示并写 JSONL 日志
/root/gpufree-data/conda_envs/robosuite_test/bin/python \
  tests/dh116/test_nero_dh116_lift.py --steps 900

# 录制完整 home-to-lift 视频
/root/gpufree-data/conda_envs/robosuite_test/bin/python \
  tests/dh116/test_nero_dh116_lift.py --steps 900 --video

# 评估不使用稳定器的真实接触结果
/root/gpufree-data/conda_envs/robosuite_test/bin/python \
  tests/dh116/test_nero_dh116_lift.py --steps 900 --physical-only

# 完整回归
/root/gpufree-data/conda_envs/robosuite_test/bin/pytest -q \
  tests/dh116/test_dh116_standalone.py \
  tests/dh116/test_dh116_integration.py \
  tests/dh116/test_dh116_lift_task.py \
  tests/dh116/test_nero_dh116_lift.py
```
