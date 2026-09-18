"""Panda + DH116 的真实 Lift 任务 rollout。

策略依次执行转腕、接近、闭合和抬升，只通过 robosuite 动作接口控制仿真。
平移命令叠加可复现的小步随机游走，并对随机状态、平移命令和完整动作做三层限幅。
任务是否成功完全采用 Lift 环境自身的成功条件，不直接修改方块状态。
"""

import argparse
import os
import subprocess
from pathlib import Path

# 视频模式需要在导入 robosuite 前选择无头 OpenGL 后端。
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import robosuite as suite
import robosuite.macros as macros


macros.IMAGE_CONVENTION = "opencv"
DEFAULT_VIDEO = Path(__file__).resolve().parent / "vedio" / "dh116_panda_lift_task.mp4"


class BoundedRandomIncrement:
    """Generate smooth zero-mean noise without allowing an unbounded random walk."""

    def __init__(self, rng, dimensions=3, step_limit=0.004, state_limit=0.02):
        self.rng = rng
        self.step_limit = float(step_limit)
        self.state_limit = float(state_limit)
        self.state = np.zeros(dimensions, dtype=np.float64)

    def sample(self):
        """Add one small random increment, then clamp the accumulated noise."""
        increment = self.rng.uniform(-self.step_limit, self.step_limit, self.state.shape)
        self.state = np.clip(self.state + increment, -self.state_limit, self.state_limit)
        return self.state.copy()


class LiftPolicy:
    """Small-delta Cartesian policy for a four-phase DH116 lift attempt."""

    def __init__(self, robot, arm, gripper, cube_position, total_steps, seed):
        self.robot = robot
        self.arm = arm
        self.gripper = gripper
        self.gripper_name = robot.get_gripper_name(arm)
        self.cube_position = np.asarray(cube_position, dtype=np.float64).copy()
        self.total_steps = total_steps
        self.noise = BoundedRandomIncrement(np.random.default_rng(seed))

        # OSC_POSE receives normalized increments. A limit of 0.30 corresponds
        # to at most 1.5 cm because Panda's default controller output_max is 5 cm.
        self.position_gain = 5.0
        self.translation_limit = 0.30

    def phase(self, step):
        """Return phase target, finger command, and Y-axis rotation increment."""
        xy_offset = np.array([-0.052, 0.0, 0.0])

        # Rotating first aligns the thumb and opposing fingers at a similar
        # height. Without this phase the lower fingers touch the table before
        # the thumb reaches the cube, so the hand cannot establish a grasp.
        if step < 50:
            target = self.cube_position + xy_offset + [0.0, 0.0, 0.12]
            return "rotate", target, -1.0, 0.20
        if step < 150:
            target = self.cube_position + xy_offset + [0.0, 0.0, 0.03]
            return "approach", target, -1.0, 0.0
        if step < 230:
            target = self.cube_position + xy_offset + [0.0, 0.0, 0.03]
            return "close", target, 1.0, 0.0
        target = self.cube_position + xy_offset + [0.0, 0.0, 0.23]
        return "lift", target, 1.0, 0.0

    def action(self, observation, step):
        """Build one bounded action from position feedback and smooth noise."""
        phase, target, gripper_value, rotation_y = self.phase(step)
        position_error = target - observation["robot0_eef_pos"]

        # First limit the random walk, then limit the combined translation.
        translation = self.position_gain * position_error + self.noise.sample()
        translation = np.clip(translation, -self.translation_limit, self.translation_limit)
        arm_action = np.concatenate([translation, [0.0, rotation_y, 0.0]])

        action = self.robot.create_action_vector(
            {
                self.arm: arm_action,
                self.gripper_name: np.full(self.gripper.dof, gripper_value),
            }
        )
        # Final safety limit protects every controller dimension.
        action = np.clip(action, -1.0, 1.0)
        return action, phase


def start_video_writer(path, width, height, fps):
    """Start FFmpeg and accept contiguous RGB frames through stdin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def run_lift_task(steps=400, seed=3, video_path=None, width=640, height=480, fps=20):
    """Execute one real Lift rollout and return its measured task statistics."""
    record_video = video_path is not None
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        gripper_types="DH116",
        has_renderer=False,
        has_offscreen_renderer=record_video,
        use_camera_obs=record_video,
        use_object_obs=True,
        camera_names="sideview",
        camera_widths=width,
        camera_heights=height,
        reward_shaping=True,
        control_freq=fps,
        horizon=steps,
        ignore_done=True,
        seed=seed,
    )

    writer = None
    return_code = 0
    try:
        observation = env.reset()
        robot = env.robots[0]
        arm = robot.arms[0]
        gripper = robot.gripper[arm]
        initial_cube_height = float(observation["cube_pos"][2])
        policy = LiftPolicy(robot, arm, gripper, observation["cube_pos"], steps, seed)

        if record_video:
            writer = start_video_writer(Path(video_path), width, height, fps)

        max_cube_height = initial_cube_height
        max_action = 0.0
        max_translation = 0.0
        success = False
        phase = "reset"
        reward = 0.0
        for step in range(steps):
            action, phase = policy.action(observation, step)
            max_action = max(max_action, float(np.max(np.abs(action))))
            arm_start, _ = robot._action_split_indexes[arm]
            max_translation = max(
                max_translation,
                float(np.max(np.abs(action[arm_start : arm_start + 3]))),
            )
            observation, reward, _, _ = env.step(action)
            max_cube_height = max(max_cube_height, float(observation["cube_pos"][2]))
            success = success or bool(env._check_success())

            if writer is not None:
                frame = np.ascontiguousarray(observation["sideview_image"], dtype=np.uint8)
                writer.stdin.write(frame.tobytes())

        return {
            "success": success,
            "final_phase": phase,
            "final_reward": float(reward),
            "height_gain": max_cube_height - initial_cube_height,
            "max_action": max_action,
            "max_translation": max_translation,
        }
    finally:
        if writer is not None:
            writer.stdin.close()
            return_code = writer.wait()
        env.close()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")


def test_bounded_random_increment():
    """Random accumulation must never exceed its configured safety bound."""
    noise = BoundedRandomIncrement(np.random.default_rng(1), step_limit=0.004, state_limit=0.02)
    samples = np.array([noise.sample() for _ in range(1000)])
    assert np.max(np.abs(samples)) <= 0.02
    assert np.max(np.abs(np.diff(samples, axis=0))) <= 0.0041


def test_real_lift_task_succeeds_with_bounded_actions():
    """The deterministic seeded rollout must satisfy Lift's own success rule."""
    result = run_lift_task(steps=400, seed=3)
    assert np.isfinite(result["final_reward"])
    assert result["success"]
    assert result["height_gain"] > 0.04
    assert result["max_action"] <= 1.0
    assert result["max_translation"] <= 0.30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400, help="Total control steps")
    parser.add_argument("--seed", type=int, default=3, help="Environment and noise seed")
    parser.add_argument(
        "--video",
        type=Path,
        nargs="?",
        const=DEFAULT_VIDEO,
        help="Optionally record MP4; omit the value to use the default path",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.fps < 1 or args.width < 1 or args.height < 1:
        raise ValueError("steps, fps, width, and height must be positive")
    if args.video is not None and (args.width % 2 or args.height % 2):
        raise ValueError("video width and height must be even")

    result = run_lift_task(
        steps=args.steps,
        seed=args.seed,
        video_path=args.video,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(
        f"success={result['success']}, height_gain={result['height_gain']:.4f} m, "
        f"reward={result['final_reward']:.4f}, "
        f"max_translation={result['max_translation']:.3f}, max_action={result['max_action']:.3f}"
    )
    if args.video is not None:
        print(f"Video saved to: {args.video.resolve()}")


if __name__ == "__main__":
    main()
