# Nero + DH116 小型 VLA 闭环实验报告

## 结论

阶段任务已经在“视觉到达 + 语言回家”的范围内跑通。正常 RGB 在 6 个未见随机种子上完成 6/6 到达和 6/6 回家；将 RGB 全部置零后，到达变为 0/6。结果说明策略不是只靠场景中心、时间步或本体状态猜目标。

这不是抓取成功报告。现有 DH116 纯物理 Lift 基线仍然失败：检测到接触后方块抬升高度为 0。为了不把接触稳定器制造的视觉效果当作真实抓取，本阶段只声明完成可物理验证的“移动到方块上方”和“返回零位”。

## 实验设置

- 机器人：Nero + DH116，DH116 保持张开；
- 指令：`move above the cube`、`return home`；
- 方块范围：桌面 XY 各 ±8 cm；
- 相机：128×128 `agentview`，数据保存为 32×32 RGB；
- 采集：16 个 episode、2240 个样本，种子 1000–1015；
- 训练/验证：前 12 个 episode 训练，后 4 个 episode 验证；
- 闭环评测：6 个未见种子 9000–9005；
- 到达标准：末端位于方块初始位置上方 12 cm，三维误差小于 2 cm；
- 回家标准：七轴相对零位的联合误差小于 0.12 rad。

## 模型与控制

模型先从首帧普通 RGB 中计算红色响应质心，再用训练集拟合的二次标定预测方块 XY。预测值在 episode 内锁存，不会通过仿真真值持续纠偏。语言指令选择动作技能：到达指令使用视觉目标和 Nero 自身 Jacobian，回家指令使用已知零位目标。所有动作均通过 `env.step()` 执行。

方块 XY 真值只用于两件事：训练视觉定位器的监督标签，以及 episode 结束后的外部评分。环境设置 `use_object_obs=False`，策略输入只有 RGB、关节位置和末端位置。

## 结果

| 项目 | 结果 |
|---|---:|
| 留出集视觉定位平均误差 | 2.06 mm |
| 留出集视觉定位最大误差 | 3.00 mm |
| 正常 RGB 到达成功率 | 6/6（100%） |
| 正常 RGB 回家成功率 | 6/6（100%） |
| 全黑 RGB 到达成功率 | 0/6（0%） |
| 正常 RGB 最小/最大到达误差 | 4.23 / 18.74 mm |
| 正常 RGB 最大在线定位误差 | 3.56 mm |
| 回家最大联合误差 | 0.00183 rad |

正常 RGB 与全黑 RGB 使用完全相同的种子和初始方块位置。全黑组最终到达误差为 11.87–17.91 cm，明显超过 2 cm 阈值。

## 反作弊检查

- observation 中不存在 `cube_pos` 或其他物体真值；
- 训练集与验证集按 episode 分开；
- 采集种子与闭环评测种子互斥；
- 评测不调用专家动作，不在失败时切换专家；
- 不使用 segmentation、geom id、接触稳定器或焊接约束；
- 不修改方块 qpos/qvel；
- 评分目标固定为 reset 后的初始方块位置；
- 全黑反事实对照确认到达依赖 RGB；
- `latest_report.json` 中的 `completion_audit` 对以上关键门禁给出机器可读结果。

## 产物

- `nero_dh116_reach_dataset.npz`：图像、本体状态、语言编号、动作和监督标签；
- `nero_dh116_reach_locator.npz`：视觉定位标定参数；
- `latest_report.json`：逐 episode 采集、训练、闭环和消融结果；
- `nero_dh116_vla.py`：可从头复现实验的单文件入口。
- `vedio/vla/vla_rgb_seed9000_ep001_001.mp4` … `vla_rgb_seed9005_ep006_001.mp4`：6 个正常 RGB 完整闭环视频，每个 181 帧、20 fps；
- `vedio/vla/vla_video_manifest.json`：视频编号、种子、帧数和逐指令结果；脚本会寻找下一个空编号并禁止覆盖旧文件；
- `VLA_TECHNICAL_GUIDE.md`：系统设计、数据字段、代码职责、流程图和复现实验说明。

## 视频记录核验

本次使用命令：

```bash
python tests/dh116/nero_dh116_vla.py evaluate \
  --eval-episodes 6 --eval-seed-start 9000 --record-video
```

视频目录是 `tests/dh116/vedio/vla/`。文件名格式为
`vla_rgb_seed<seed>_ep<episode>_<run_number>.mp4`；再次运行同一条件时编号递增，旧的
`tests/dh116/vedio/nero_dh116_lift.mp4` 不会被写入。`ffprobe` 核验首个文件为 128×128、20 fps、
181 帧、9.05 秒，6 个文件路径均唯一。

## 下一阶段

下一步若继续做抓取，必须先让 `--physical-only` 在多个位置真实抬升成功，再采集抓取数据。不能把 `ContactGraspStabilizer` 生成的轨迹标记为物理抓取成功。
