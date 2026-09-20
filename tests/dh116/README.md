# DH116 测试说明

本目录只保留独立模型、robosuite 集成、真实任务、渲染诊断和视频录制五类入口。所有 Python
文件都有模块说明，测试文件导入时不会启动 Viewer 或进入长循环。

## 文件

| 文件 | 用途 |
|---|---|
| `test_dh116_standalone.py` | 直接加载 DH116 MJCF，检查关节、执行器、耦合约束和数值稳定性 |
| `test_dh116_integration.py` | 检查 Panda + DH116 的注册、模型拼接、动作维度和控制稳定性 |
| `test_dh116_lift_task.py` | 在真实 Lift 环境执行带随机小增量和三层限幅的抓取抬升 rollout |
| `test_nero_dh116_lift.py` | Nero + DH116 的接触确认、抬升回归测试和侧视 MP4 录制 |
| `nero_dh116.md` | Nero + DH116 v1.0 标准实验报告 |
| `nero_dh116_technical.md` | 测试脚本的类、函数、主循环和轨迹设计技术文档 |
| `nero_dh116_vla.py` | 无额外训练依赖的小型视觉语言动作数据闭环 |
| `VLA_DATA_LOOP.md` | VLA 阶段目标、反作弊边界、数据格式和完成条件 |
| `VLA_EXPERIMENT_REPORT.md` | 小型 VLA 闭环的结果、反事实对照和限制说明 |
| `experiment_logging.py` | 运行日志、项目文件校验和中断状态记录 |
| `experiment_logs/` | 每次运行的 JSONL 日志与项目变更记录 |
| `debug_rendering.py` | 检查图形环境、GLFW、robosuite Viewer 调用入口和 GUI 刷新 |
| `view_dh116_panda_lift.py` | 使用 EGL 无头渲染并输出 H.264 视频 |

## 常用命令

```bash
# 自动测试（不打开窗口）
pytest -q tests/dh116/test_dh116_standalone.py tests/dh116/test_dh116_integration.py

# 查看独立模型的关节和执行器
python tests/dh116/test_dh116_standalone.py --details

# 查看集成后的 DH116 关节、执行器和 geom
python tests/dh116/test_dh116_integration.py --details

# 执行真实 Lift rollout；增加 --video 可录制完整侧视视频
python tests/dh116/test_dh116_lift_task.py --steps 400 --video

# 打开 10 秒 GUI；需要有效的 DISPLAY / Wayland 环境
python tests/dh116/test_dh116_integration.py --viewer --seconds 10

# 输出渲染环境，按需增加 --glfw、--source 或 --viewer --dh116
python tests/dh116/debug_rendering.py

# 无头录制，默认写入 vedio/dh116_panda_lift.mp4
python tests/dh116/view_dh116_panda_lift.py

# Nero + DH116 无头录制，默认 900 步、45 秒，写入 vedio/nero_dh116_lift.mp4
python tests/dh116/test_nero_dh116_lift.py --steps 900 --video

# 纯物理诊断：关闭接触后稳定器，结果只认定抬升阶段持续成功
python tests/dh116/test_nero_dh116_lift.py --steps 900 --physical-only

# 查看最近一次运行状态和项目文件校验快照
cat tests/dh116/experiment_logs/latest.json
```

## 整合记录

- 原模型、关节、geom 和可视化检查已合并到 `test_dh116_integration.py`。
- 原 MuJoCo、GLFW、`MUJOCO_GL` 和 robosuite Viewer 排障脚本已合并到
  `debug_rendering.py`。
- 历史排错脚本中的固定绝对路径、导入即执行、固定 20–60 秒循环和重复输出均已移除。
