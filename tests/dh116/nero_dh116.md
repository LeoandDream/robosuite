# nero_dh116 实验报告

## 文档信息

| 项目 | 内容 |
|---|---|
| 实验名称 | nero_dh116 |
| 报告版本 | v1.0 |
| 执行修订 | v1.3：纯物理诊断与指尖接触代理 |
| 实验类型 | MuJoCo / robosuite 机器人模型接入与抓取抬升验证 |
| 实验日期 | 2026-09-19 |
| 实验状态 | 已完成，可复现 |
| 目标机器人 | AgileX Nero，7 自由度 |
| 目标末端 | DH116 灵巧手 |
| 任务环境 | robosuite Lift |

## 摘要

本实验将 AgileX Nero 机械臂模型接入本地 robosuite，并完成 Nero 与 DH116 灵巧手的组合、控制器配置、Lift 抓取任务和侧视视频录制。

实验结果表明：Nero 模型能够被 robosuite 正确注册和实例化，Nero+DH116 的动作空间为 13 维，稳定演示流程可以检测到掌心两侧接触并完成方块抬升。最终回归测试共 7 项，全部通过；录制得到 640×480、20 FPS、45 秒的 H.264 视频。新增的 `--physical-only` 诊断显示，当前真实接触基线尚不能持续抓住方块。

需要说明的是，DH116 原始 STL 碰撞面较薄，在 Nero 组合姿态下容易产生 MuJoCo 网格穿透和数值冲量。因此 v1.3 延续掌心两侧低体积接触代理和掌心支撑面，并新增仅供 Nero 物理诊断使用的透明指尖代理：接近阶段暂时关闭接触几何，闭合阶段再启用 DH116 与方块、桌面的接触；确认两侧接触后，稳定演示使用相对位姿约束，纯物理诊断则完全不改写方块位姿。轨迹仍采用关键帧分段线性插值，抬升关节变化量为 0.30 rad。

## 1. 实验目的

1. 将 AgileX Nero MuJoCo 模型接入 robosuite 机器人注册体系。
2. 将 DH116 灵巧手安装到 Nero 末端，并保持 Panda+DH116 的原有兼容性。
3. 配置 Nero 的关节位置控制器和动作接口。
4. 在 Lift 环境中完成方块接触、闭合和抬升流程。
5. 生成可复现的侧视 MP4，并建立自动化回归测试。

## 2. 实验环境

### 2.1 软件环境

| 组件 | 版本或配置 |
|---|---|
| Python | 3.10.21 |
| robosuite | 1.5.2，本地源码路径 `/root/gpufree-data/robosuite` |
| MuJoCo | 3.9.0 |
| FFmpeg | 4.4.2 |
| OpenGL 后端 | EGL，无头渲染 |
| 控制频率 | 20 Hz |
| 随机种子 | 3 |

### 2.2 模型来源

Nero 的 MuJoCo 模型和网格资源来自 AgileX Robotics 的公开仓库：

[agx_arm_sim/mujoco/agilex_arm/agilex_nero](https://github.com/agilexrobotics/agx_arm_sim/tree/master/mujoco/agilex_arm/agilex_nero)

导入后的主要资源位于：

- `robosuite/models/assets/robots/nero/robot.xml`
- `robosuite/models/assets/robots/nero/*.obj`
- `robosuite/models/robots/manipulators/nero_robot.py`
- `robosuite/controllers/config/robots/default_nero.json`

## 3. 模型与软件实现

### 3.1 Nero 机器人注册

Nero 作为单臂、7 自由度固定基座机器人接入 robosuite：

| 配置项 | 实现结果 |
|---|---|
| 机器人名称 | `Nero` |
| 机械臂数量 | 单臂，`right` |
| 自由度 | 7 |
| 默认基座 | `RethinkMount` |
| 默认夹爪 | `DH116` |
| 默认控制器 | `default_nero` |
| 机器人类型 | `FixedBaseRobot` |
| 末端名称 | `right_hand` |

### 3.2 控制器配置

Nero 使用绝对关节位置控制器 `JOINT_POSITION`。控制器配置文件为：

`robosuite/controllers/config/robots/default_nero.json`

动作向量由两部分构成：

| 部分 | 维度 | 说明 |
|---|---:|---|
| Nero 机械臂 | 7 | 绝对关节位置目标 |
| DH116 灵巧手 | 6 | 归一化手指控制量 |
| 总动作维度 | 13 | `env.action_dim == 13` |

### 3.3 DH116 碰撞适配

为兼顾 Panda 和 Nero 两种组合方式，DH116 采用按机器人模型适配的碰撞策略：

- Panda 使用原 DH116 末端实体网格参与接触。
- Nero 组合时关闭 DH116 薄 STL 指部网格的碰撞属性，保留视觉网格。
- Nero 使用 `palm_left`、`palm_right` 和 `palm_support` 掌心碰撞代理。
- `palm_support` 在闭合阶段允许与桌面接触；接近阶段由实验脚本临时关闭，避免桌面反作用力改变接近轨迹。
- `finger13_pad`、`finger14_pad`、`finger23_pad`、`finger33_pad`、`finger43_pad`、`finger53_pad` 是 Nero 专用透明指尖接触代理，默认关闭，仅在 `--physical-only` 模式启用。
- `Lift._check_grasp` 仍通过左右两侧接触组判断接触事件。

这样可以避免 DH116 原始网格在 Nero 安装姿态下与机械臂、桌面产生过深穿透，同时不影响视觉模型显示。

## 4. 实验任务与控制流程

实验使用 robosuite `Lift` 环境，随机种子设为 3，最大控制步数为 900。为保证掌心位置和视频过程可重复，v1.2 将方块固定放置在 DH116 掌心参考位置附近，平面坐标为 `(-0.069, 0.039)`。

| 阶段 | 控制步 | 夹爪指令 | 机械臂行为 |
|---|---:|---:|---|
| Home | 0–99 | 打开，`-1` | 保持 Nero XML home 位姿，作为视频起点 |
| 预抓取 | 100–279 | 打开，`-1` | 从 home 平滑移动至安全预抓取位姿 |
| 接近 | 280–499 | 打开，`-1` | 平滑接近掌心目标；掌心/桌面碰撞暂时关闭 |
| 闭合 | 500–579 | 闭合，`+1` | 启用掌心接触和桌面接触，确认两侧接触 |
| 抬升 | 580–899 | 闭合，`+1` | 第 2 关节逐步减少最多 0.30 rad，抬高末端 |

使用的关键关节目标为：

```text
PREGRASP_QPOS = [-2.497, 1.913, 1.594, -1.010, -0.032, 0.353, -1.144]
GRASP_QPOS    = [-2.479, 2.060, 1.746, -1.010, -0.358, 0.522, -1.145]
```

关键帧之间通过 `interpolate_qpos()` 进行分段线性插值，避免直接跳变。对应的类、函数和主循环说明见：[nero_dh116_technical.md](./nero_dh116_technical.md)。

### 4.1 接触稳定策略

接近阶段先通过正常 robosuite 动作接口推进仿真。进入闭合阶段后启用 DH116 掌心接触集合，再调用 `Lift._check_grasp` 检查 `palm_left` 和 `palm_right` 是否都与方块接触。只有在接触检测成功后，才记录方块与末端的相对位置，并在抬升阶段维持平移相对位姿。

该过程封装在：

`tests/dh116/test_nero_dh116_lift.py::ContactGraspStabilizer`

每次运行同步写入 `tests/dh116/experiment_logs/`。JSONL 日志包含启动快照、阶段切换、周期状态、结束状态和项目文件 SHA-256 前后快照；中断运行会保留 `running` 或 `interrupted` 状态，不能被误判为成功。

因此，稳定约束不会在接触前生效，也不会替代初始接近和夹爪闭合过程。

`--physical-only` 模式不调用稳定器，只保留 MuJoCo 的碰撞、摩擦和重力；结果需要在抬升阶段连续成功至少 20 步才计为纯物理成功。

## 5. 评价指标

本实验使用以下指标：

1. **模型初始化**：Nero 和 DH116 可以正常实例化。
2. **动作空间**：动作维度为 13。
3. **数值稳定性**：关键观测和仿真状态保持有限值。
4. **接触确认**：`Lift._check_grasp` 检测到左右两侧接触。
5. **任务成功**：`Lift._check_success()` 返回真，即方块高度超过桌面高度 `0.04 m`。
6. **视频完整性**：MP4 能被 FFmpeg 正常封装并被 `ffprobe` 读取。

## 6. 实验结果

### 6.1 Nero+DH116 任务结果

| 指标 | 结果 |
|---|---:|
| 初始化测试 | 通过 |
| 接触检测 | 通过 |
| 接触步数 | 500 |
| 稳定抓取状态 | `True` |
| Lift 成功状态 | `True` |
| 方块最大抬升量 | 约 0.1276 m |
| 最终奖励 | 1.0000 |

### 6.2 回归测试结果

执行以下四个测试文件后，共 7 项测试全部通过：

```bash
pytest -q \
  tests/dh116/test_dh116_standalone.py \
  tests/dh116/test_dh116_integration.py \
  tests/dh116/test_dh116_lift_task.py \
  tests/dh116/test_nero_dh116_lift.py
```

结果：

```text
7 passed in 20.68s
```

这同时验证了：

- DH116 独立模型结构未被破坏。
- Panda+DH116 原有集成测试通过。
- Panda+DH116 原有 Lift 任务通过。
- Nero+DH116 初始化和 Lift 任务通过。

### 6.3 视频结果

视频文件：

[nero_dh116_lift.mp4](./vedio/nero_dh116_lift.mp4)

视频参数：

| 项目 | 结果 |
|---|---:|
| 分辨率 | 640×480 |
| 帧率 | 20 FPS |
| 总帧数 | 900 |
| 时长 | 45.000 秒 |
| 编码 | H.264 / yuv420p |
| 文件大小 | 1082421 bytes（约 1.03 MiB） |

### 6.4 纯物理诊断结果

执行：

```bash
/root/gpufree-data/conda_envs/robosuite_test/bin/python \
  tests/dh116/test_nero_dh116_lift.py --steps 900 --physical-only
```

| 指标 | 结果 |
|---|---:|
| 接触步数 | 500 |
| 稳定器 | 未启用 |
| 最大抬升量 | 约 0.0083 m |
| 抬升阶段连续成功步数 | 0 |
| 纯物理成功 | `False` |

该结果说明当前方块仍主要受到掌心/桌面碰撞冲量，指尖代理尚未形成足够的包络和摩擦保持；因此 45 秒视频仍标记为“接触后稳定演示”，不宣称为纯物理抓取。

## 7. 复现实验

进入 robosuite 本地源码目录后执行：

```bash
cd /root/gpufree-data/robosuite
export MUJOCO_GL=egl

# 无视频，快速验证任务结果
/root/gpufree-data/conda_envs/robosuite_test/bin/python \
  tests/dh116/test_nero_dh116_lift.py --steps 900

# 录制侧视视频
/root/gpufree-data/conda_envs/robosuite_test/bin/python \
  tests/dh116/test_nero_dh116_lift.py --steps 900 --video
```

脚本结构说明：[nero_dh116_technical.md](./nero_dh116_technical.md)

默认视频输出路径：

```text
tests/dh116/vedio/nero_dh116_lift.mp4
```

实验日志和变更记录：

- `tests/dh116/experiment_logs/latest.json`
- `tests/dh116/experiment_logs/*.jsonl`
- `tests/dh116/experiment_logs/project_changes.md`

完整回归测试命令：

```bash
/root/gpufree-data/conda_envs/robosuite_test/bin/pytest -q \
  tests/dh116/test_dh116_standalone.py \
  tests/dh116/test_dh116_integration.py \
  tests/dh116/test_dh116_lift_task.py \
  tests/dh116/test_nero_dh116_lift.py
```

## 8. 已知限制与后续工作

### 8.1 当前限制

1. 当前结果以随机种子 3 和固定掌心放置点为确定性回归样例，尚未覆盖多种方块位置和姿态。
2. Nero 的 DH116 抓取视频使用接触后的稳定约束，尚未完成完全依靠 STL 网格、摩擦和阻尼参数的纯物理稳定抓取。
3. 当前控制策略是固定关节目标序列，尚未加入通用的目标检测、逆运动学或轨迹规划器。
4. 夹爪的六维动作目前作为统一手指控制量发送，尚未针对每根手指实现独立的力控或位置闭环。

### 8.2 建议后续工作

1. 根据 DH116 实际指尖尺寸重新生成简化碰撞网格。
2. 标定手指与方块、桌面的摩擦参数和接触刚度。
3. 使用 MuJoCo equality weld 或经过标定的接触动力学替代当前演示稳定器。
4. 加入多随机种子、多方块位置和不同方块尺寸的批量测试。
5. 将固定关节位姿策略替换为基于末端位姿的 IK 或 OSC 控制策略。

## 9. 结论

版本 v1.0 报告经 v1.3 执行修订后，已完成 Nero 模型在 robosuite 中的基础接入、DH116 掌心接触适配、平滑 home-to-lift 视频录制、实验日志和纯物理诊断接口。当前版本可用于模型联调、接口验证和抓取流程演示；纯物理抓取仍需要继续标定指尖碰撞几何、摩擦参数和接近姿态。
