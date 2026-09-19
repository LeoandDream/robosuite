# nero_dh116 项目变更记录

| 日期 | 版本/批次 | 文件或范围 | 修改内容 | 原因与验证 |
|---|---|---|---|---|
| 2026-09-18 | v1.0 | `robosuite/models/assets/robots/nero/` | 导入 AgileX Nero 的 MJCF 和 OBJ 网格 | `Nero` 可在 Lift 中实例化 |
| 2026-09-18 | v1.0 | `nero_robot.py`、机器人注册、`default_nero.json` | 注册 Nero、配置 7 自由度关节位置控制 | 动作维度为 13，初始化测试通过 |
| 2026-09-18 | v1.0 | `dh116.xml`、`dh116_gripper.py` | 接入 DH116 站点、执行器和接触组 | DH116 独立测试及 Panda 集成测试通过 |
| 2026-09-18 | v1.0 | `test_nero_dh116_lift.py` | 增加 Nero+DH116 Lift rollout 和 MP4 录制 | 任务成功，视频可由 `ffprobe` 读取 |
| 2026-09-19 | v1.1 修订 | `dh116.xml`、`nero_robot.py` | 增加掌心两侧接触点和掌心支撑面；保留 DH116 桌面接触能力 | 先关闭接触几何完成接近，闭合阶段再启用并记录接触 |
| 2026-09-19 | v1.1 修订 | `test_nero_dh116_lift.py` | 视频从 home 位姿开始，补齐预抓取、接近、闭合和抬升阶段；固定方块到掌心实验位 | 解决视频缺少接近过程和擦边抓取问题 |
| 2026-09-19 | v1.1 修订 | `experiment_logging.py`、`experiment_logs/` | 增加 JSONL 运行日志、中断状态和项目文件 SHA-256 快照 | 每次运行可追溯，异常退出仍保留运行状态 |
| 2026-09-19 | v1.1 文档同步 | `nero_dh116.md`、`README.md`、本变更记录 | 同步 home-to-lift 流程、掌心接触结果、560 步视频参数和复现实验命令 | 文档结果与最新视频、回归测试和日志保持一致 |
| 2026-09-19 | v1.2 轨迹修订 | `test_nero_dh116_lift.py`、`nero_dh116_technical.md` | 将关键帧跳变改为分段线性插值；默认 900 步；抬升关节变化量从 0.15 rad 提高到 0.30 rad；碰撞开关与闭合阶段同步 | 解决初始运动过快和末端抬升不明显；无视频复核成功，接触步数 500 |
| 2026-09-19 | v1.2 文档同步 | `README.md`、`nero_dh116.md`、本变更记录 | 补充类/函数/主循环说明，并同步 900 步、45 秒视频和新结果 | 保持实现、实验报告和复现命令一致 |
| 2026-09-19 | v1.2 日志修订 | `test_nero_dh116_lift.py`、`nero_dh116_technical.md` | FFmpeg 关闭并确认成功后才写入 `completed`；编码失败进入 `failed` | 避免半成品视频被误记为成功实验 |
| 2026-09-19 | v1.3 物理诊断 | `dh116.xml`、`dh116_gripper.py`、`nero_robot.py`、`test_nero_dh116_lift.py` | 增加 Nero 专用透明指尖接触代理、`--physical-only` 模式和持续成功判定 | 分离接触后稳定演示与真实 MuJoCo 抓取；当前基线接触后仍会滑落 |

## 环境污染控制

实验脚本只允许将以下运行产物写入项目目录：

- `tests/dh116/vedio/*.mp4`：视频结果；
- `tests/dh116/experiment_logs/*.jsonl`：运行日志；
- `tests/dh116/experiment_logs/latest.json`：最近运行索引。

项目源码、MJCF、控制器和报告的修改必须先更新本表，再执行回归测试。日志中的 before/after manifest 用于检查运行期间是否发生了未记录的源码变化。
