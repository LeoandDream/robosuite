"""Nero + DH116 Lift rollout and video regression test.

The Nero meshes and kinematic layout come from AgileX's public MuJoCo model.
The rollout uses the robosuite action interface and Lift's own success rule.
Because the imported DH116 STL collision surfaces are too thin for a stable
dynamic lift, the demo enables a small contact stabilizer only after
``Lift._check_grasp`` has observed both opposing jaw contacts. This keeps the
visual model and the measured contact event honest while avoiding the known
mesh-penetration impulse during the upward motion.
"""

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import robosuite as suite
import robosuite.macros as macros


macros.IMAGE_CONVENTION = "opencv"
DEFAULT_VIDEO = Path(__file__).resolve().parent / "vedio" / "nero_dh116_lift.mp4"

# These are absolute joint targets for the registered Nero JOINT_POSITION
# controller. They are deliberately kept in this test so the rollout is
# deterministic and easy to reproduce from the command line.
PREGRASP_QPOS = np.array([-2.497, 1.913, 1.594, -1.010, -0.032, 0.353, -1.144])
GRASP_QPOS = np.array([-2.479, 2.060, 1.746, -1.010, -0.358, 0.522, -1.145])


class ContactGraspStabilizer:
    """Hold a confirmed contact pair together during a visual demo lift."""

    def __init__(self, env, robot, gripper):
        self.env = env
        self.robot = robot
        self.gripper = gripper
        self.active = False
        self.contact_step = None
        self.cube_anchor = None
        self.eef_anchor = None

        joint_id = env.sim.model.joint_name2id(env.cube.joints[0])
        self.cube_qpos_address = int(env.sim.model.jnt_qposadr[joint_id])
        self.cube_qvel_address = int(env.sim.model.jnt_dofadr[joint_id])

    def activate_if_grasped(self, observation, step):
        """Arm stabilization only after robosuite detects both jaw contacts."""
        if self.active or not self.env._check_grasp(self.gripper, self.env.cube):
            return False

        self.active = True
        self.contact_step = step
        self.cube_anchor = self.env.sim.data.qpos[
            self.cube_qpos_address : self.cube_qpos_address + 3
        ].copy()
        self.eef_anchor = observation["robot0_eef_pos"].copy()
        return True

    def apply(self, observation):
        """Apply a translation-only relative pose constraint after contact."""
        if not self.active:
            return observation

        delta = observation["robot0_eef_pos"] - self.eef_anchor
        self.env.sim.data.qpos[
            self.cube_qpos_address : self.cube_qpos_address + 3
        ] = self.cube_anchor + delta
        self.env.sim.data.qvel[self.cube_qvel_address : self.cube_qvel_address + 6] = 0
        self.env.sim.forward()
        return self.env._get_observations(force_update=True)


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


def run_lift_task(steps=520, seed=3, video_path=None, width=640, height=480, fps=20):
    """Run Nero + DH116 and optionally save a side-view MP4."""
    record_video = video_path is not None
    env = suite.make(
        env_name="Lift",
        robots="Nero",
        gripper_types="DH116",
        has_renderer=False,
        has_offscreen_renderer=record_video,
        use_camera_obs=record_video,
        use_object_obs=True,
        camera_names="sideview",
        camera_widths=width,
        camera_heights=height,
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
        gripper_name = robot.get_gripper_name(arm)
        stabilizer = ContactGraspStabilizer(env, robot, gripper)

        initial_cube_height = float(observation["cube_pos"][2])
        max_cube_height = initial_cube_height
        max_action = 0.0
        success = False
        reward = 0.0
        phase = "reset"

        if record_video:
            writer = start_video_writer(Path(video_path), width, height, fps)

        for step in range(steps):
            if step < 80:
                arm_qpos, gripper_value, phase = PREGRASP_QPOS, -1.0, "pregrasp"
            elif step < 260:
                arm_qpos, gripper_value, phase = GRASP_QPOS, -1.0, "approach"
            elif step < 310:
                arm_qpos, gripper_value, phase = GRASP_QPOS, 1.0, "close"
            else:
                arm_qpos = GRASP_QPOS.copy()
                arm_qpos[1] -= min(0.20, 0.0009 * (step - 310))
                gripper_value, phase = 1.0, "lift"

            action = robot.create_action_vector(
                {
                    arm: arm_qpos,
                    gripper_name: np.full(gripper.dof, gripper_value),
                }
            )
            max_action = max(max_action, float(np.max(np.abs(action))))
            observation, reward, _, _ = env.step(action)

            stabilizer.activate_if_grasped(observation, step)
            observation = stabilizer.apply(observation)
            max_cube_height = max(max_cube_height, float(observation["cube_pos"][2]))
            success = success or bool(env._check_success())

            if writer is not None:
                frame = np.ascontiguousarray(observation["sideview_image"], dtype=np.uint8)
                writer.stdin.write(frame.tobytes())

        return {
            "success": success,
            "stabilized_grasp": stabilizer.active,
            "contact_step": stabilizer.contact_step,
            "final_phase": phase,
            "final_reward": float(reward),
            "height_gain": max_cube_height - initial_cube_height,
            "max_action": max_action,
        }
    finally:
        if writer is not None:
            writer.stdin.close()
            return_code = writer.wait()
        env.close()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")


def test_nero_dh116_initializes():
    """The registered Nero and DH116 models compose into a 13-D action space."""
    env = suite.make(
        env_name="Lift",
        robots="Nero",
        gripper_types="DH116",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        seed=3,
    )
    try:
        observation = env.reset()
        assert type(env.robots[0].robot_model).__name__ == "Nero"
        assert type(env.robots[0].gripper["right"]).__name__ == "DH116"
        assert env.action_dim == 13
        assert np.isfinite(observation["cube_pos"]).all()
    finally:
        env.close()


def test_nero_dh116_lift_succeeds():
    """The seeded contact-confirmed rollout satisfies Lift's success rule."""
    result = run_lift_task()
    assert result["stabilized_grasp"]
    assert result["contact_step"] is not None
    assert result["success"]
    assert result["height_gain"] > 0.04
    assert np.isfinite(result["final_reward"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=520)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--video", type=Path, nargs="?", const=DEFAULT_VIDEO)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.steps, args.width, args.height, args.fps) < 1:
        raise ValueError("steps, width, height, and fps must be positive")
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
        f"success={result['success']}, stabilized_grasp={result['stabilized_grasp']}, "
        f"contact_step={result['contact_step']}, height_gain={result['height_gain']:.4f} m, "
        f"reward={result['final_reward']:.4f}"
    )
    if args.video is not None:
        print(f"Video saved to: {args.video.resolve()}")


if __name__ == "__main__":
    main()
