# nero_dh116 实验日志

本目录保存 `nero_dh116` 的运行日志和项目变更记录。

## 运行日志

每次执行 `test_nero_dh116_lift.py` 都会先创建一个 JSONL 文件，并立即写入：

- `run_started`：运行时间、进程号、实验配置和项目文件 SHA-256 快照；
- `phase_changed`：home、pregrasp、approach、close、lift 阶段切换；
- `step`：周期性的末端位置、方块位置、接触状态和成功状态；
- `run_finished`：完成、失败或中断状态，以及结束时的项目文件 SHA-256 快照。

`latest.json` 始终指向最近一次运行。若程序被强制终止，日志可能保留
`status=running`；这表示该运行未正常写入结束事件，不能视为成功。

## 项目变更

`project_changes.md` 记录实验相关文件的修改原因、验证方式和影响范围。新增模型、控制器、碰撞代理、测试脚本或报告时，应同步更新该文件。

