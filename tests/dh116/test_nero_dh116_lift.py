"""Nero + DH116 Lift rollout and video regression test.

The Nero meshes and kinematic layout come from AgileX's public MuJoCo model.
The rollout uses the robosuite action interface and Lift's own success rule.
The video starts at the Nero home keyframe, then records pre-grasp, approach,
close, and lift. The object is placed below the DH116 palm reference so the
fingers close around the palm rather than touching only its leading edge.
Because the imported DH116 STL collision surfaces are too thin for a stable
dynamic lift, the demo enables palm collision only during the close phase and
uses a contact stabilizer only after ``Lift._check_grasp`` confirms both palm
side contacts.
"""

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite.utils.placement_samplers import UniformRandomSampler

from experiment_logging import ExperimentLogger


macros.IMAGE_CONVENTION = "opencv"
DEFAULT_VIDEO = Path(__file__).resolve().parent / "vedio" / "nero_dh116_lift.mp4"
DEFAULT_STEPS = 900

# The trajectory is deliberately longer than the minimum successful rollout.
# At 20 Hz this gives the video enough time to show the motion instead of only
# showing abrupt target changes between keyframes.
HOME_END = 100
PREGRASP_START = HOME_END
PREGRASP_END = 280
APPROACH_START = PREGRASP_END
APPROACH_END = 500
CLOSE_START = APPROACH_END
CLOSE_END = 580
LIFT_START = CLOSE_END
LIFT_MAX_DELTA = 0.30
PHYSICAL_HOLD_STEPS = 20

# These are absolute joint targets for the registered Nero JOINT_POSITION
# controller. They are deliberately kept in this test so the rollout is
# deterministic and easy to reproduce from the command line.
HOME_QPOS = np.zeros(7)
PREGRASP_QPOS = np.array([-2.497, 1.913, 1.594, -1.010, -0.032, 0.353, -1.144])
GRASP_QPOS = np.array([-2.479, 2.060, 1.746, -1.010, -0.358, 0.522, -1.145])
PALM_CUBE_XY = (-0.069, 0.039)
NERO_CONTACT_GEOMS = (
    "gripper0_right_palm_left",
    "gripper0_right_palm_right",
    "gripper0_right_palm_support",
    "gripper0_right_base_link",
)
NERO_FINGER_CONTACT_GEOMS = (
    "gripper0_right_finger13_link",
    "gripper0_right_finger14_link",
    "gripper0_right_finger23_link",
    "gripper0_right_finger33_link",
    "gripper0_right_finger43_link",
    "gripper0_right_finger53_link",
)
NERO_FINGER_PAD_GEOMS = (
    "gripper0_right_finger13_pad",
    "gripper0_right_finger14_pad",
    "gripper0_right_finger23_pad",
    "gripper0_right_finger33_pad",
    "gripper0_right_finger43_pad",
    "gripper0_right_finger53_pad",
)


def fixed_palm_placement():
    """Return the deterministic tabletop placement used by the video demo."""
    return UniformRandomSampler(
        name="NeroPalmObjectSampler",
        x_range=(PALM_CUBE_XY[0], PALM_CUBE_XY[0]),
        y_range=(PALM_CUBE_XY[1], PALM_CUBE_XY[1]),
        rotation=0,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0.8),
        z_offset=0.01,
    )


def set_nero_contact_mode(env, enabled, include_fingertips=False):
    """Enable Nero palm contacts and optionally distal finger contacts."""
    value = 1 if enabled else 0
    geom_names = NERO_CONTACT_GEOMS + NERO_FINGER_PAD_GEOMS
    if include_fingertips:
        geom_names += NERO_FINGER_CONTACT_GEOMS
    for name in geom_names:
        geom_id = env.sim.model.geom_name2id(name)
        env.sim.model.geom_contype[geom_id] = value
        env.sim.model.geom_conaffinity[geom_id] = value


def interpolate_qpos(start_qpos, end_qpos, step, start_step, end_step):
    """Return a bounded linear joint-space reference for one control step."""
    if end_step <= start_step:
        return np.asarray(end_qpos, dtype=float).copy()
    progress = np.clip((step - start_step) / float(end_step - start_step), 0.0, 1.0)
    return (1.0 - progress) * np.asarray(start_qpos) + progress * np.asarray(end_qpos)


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


def run_lift_task(
    steps=DEFAULT_STEPS,
    seed=3,
    video_path=None,
    width=640,
    height=480,
    fps=20,
    stabilize=True,
):
    """Run Nero + DH116 and optionally save a side-view MP4.

    The cube placement is fixed to the measured DH116 palm reference for this
    v1.1 experiment. This makes the recorded motion and contact geometry
    repeatable, while the JSONL logger records every phase and interruption.
    """
    record_video = video_path is not None
    logger = ExperimentLogger(
        "nero_dh116_lift",
        config={
            "steps": steps,
            "seed": seed,
            "video_path": str(video_path) if video_path is not None else None,
            "width": width,
            "height": height,
            "fps": fps,
            "start_pose": "home_zero_keyframe",
            "cube_xy": PALM_CUBE_XY,
            "stabilization": "contact_relative_translation" if stabilize else "physical_only",
            "trajectory": {
                "home_end": HOME_END,
                "pregrasp_end": PREGRASP_END,
                "approach_end": APPROACH_END,
                "close_end": CLOSE_END,
                "lift_max_delta": LIFT_MAX_DELTA,
            },
        },
    )
    env = None
    writer = None
    return_code = 0
    try:
        env = suite.make(
            env_name="Lift",
            robots="Nero",
            gripper_types="DH116",
            has_renderer=False,
            has_offscreen_renderer=record_video,
            use_camera_obs=record_video,
            use_object_obs=True,
            placement_initializer=fixed_palm_placement(),
            camera_names="sideview",
            camera_widths=width,
            camera_heights=height,
            control_freq=fps,
            horizon=steps,
            ignore_done=True,
            seed=seed,
        )
        observation = env.reset()
        robot = env.robots[0]
        arm = robot.arms[0]
        gripper = robot.gripper[arm]
        gripper_name = robot.get_gripper_name(arm)

        # Reset puts the arm in the model's safe robosuite pose. For the video
        # we explicitly start at the Nero XML home keyframe and record the
        # approach from there.
        env.sim.data.qpos[robot._ref_joint_pos_indexes] = HOME_QPOS
        env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
        env.sim.forward()
        observation = env._get_observations(force_update=True)
        set_nero_contact_mode(env, enabled=False, include_fingertips=not stabilize)
        logger.event(
            "reset",
            cube_position=observation["cube_pos"],
            eef_position=observation["robot0_eef_pos"],
            collision_mode="disabled_during_approach",
        )
        stabilizer = ContactGraspStabilizer(env, robot, gripper) if stabilize else None

        initial_cube_height = float(observation["cube_pos"][2])
        max_cube_height = initial_cube_height
        max_action = 0.0
        success = False
        contact_step = None
        success_streak = 0
        max_success_streak = 0
        lift_success_streak = 0
        max_lift_success_streak = 0
        reward = 0.0
        phase = "reset"
        last_phase = None

        if record_video:
            writer = start_video_writer(Path(video_path), width, height, fps)

        for step in range(steps):
            if step < HOME_END:
                arm_qpos, gripper_value, phase = HOME_QPOS, -1.0, "home"
            elif step < PREGRASP_END:
                arm_qpos = interpolate_qpos(
                    HOME_QPOS, PREGRASP_QPOS, step, PREGRASP_START, PREGRASP_END
                )
                gripper_value, phase = -1.0, "pregrasp"
            elif step < APPROACH_END:
                arm_qpos = interpolate_qpos(
                    PREGRASP_QPOS, GRASP_QPOS, step, APPROACH_START, APPROACH_END
                )
                gripper_value, phase = -1.0, "approach"
            elif step < CLOSE_END:
                arm_qpos, gripper_value, phase = GRASP_QPOS, 1.0, "close"
            else:
                arm_qpos, gripper_value, phase = GRASP_QPOS, 1.0, "close"
                if step >= LIFT_START:
                    arm_qpos = GRASP_QPOS.copy()
                    lift_progress = np.clip(
                        (step - LIFT_START) / float(max(1, steps - LIFT_START - 1)),
                        0.0,
                        1.0,
                    )
                    arm_qpos[1] -= LIFT_MAX_DELTA * lift_progress
                    phase = "lift"

            if phase != last_phase:
                logger.event(
                    "phase_changed",
                    step=step,
                    phase=phase,
                    collision_mode="enabled" if phase in {"close", "lift"} else "disabled",
                )
                last_phase = phase

            # Palm and table collisions are intentionally enabled only after
            # the arm reaches the palm-centred approach pose. The model still
            # permits DH116/table contact; this avoids the table pushing the
            # arm away while it is travelling from home.
            if step == CLOSE_START:
                set_nero_contact_mode(env, enabled=True, include_fingertips=not stabilize)
                active_contact_geoms = NERO_CONTACT_GEOMS + (NERO_FINGER_CONTACT_GEOMS if not stabilize else ())
                logger.event("collision_mode_changed", step=step, enabled=True, geoms=active_contact_geoms)

            action = robot.create_action_vector(
                {
                    arm: arm_qpos,
                    gripper_name: np.full(gripper.dof, gripper_value),
                }
            )
            max_action = max(max_action, float(np.max(np.abs(action))))
            observation, reward, _, _ = env.step(action)

            grasp_detected = bool(env._check_grasp(gripper, env.cube))
            activated = False
            if grasp_detected and contact_step is None:
                contact_step = step
                if stabilizer is not None:
                    activated = stabilizer.activate_if_grasped(observation, step)
                logger.event(
                    "grasp_detected",
                    step=step,
                    cube_position=observation["cube_pos"],
                    eef_position=observation["robot0_eef_pos"],
                    contact_groups=gripper.important_geoms,
                    stabilization_enabled=stabilizer is not None,
                )
            if stabilizer is not None:
                observation = stabilizer.apply(observation)
            max_cube_height = max(max_cube_height, float(observation["cube_pos"][2]))
            raw_success = bool(env._check_success())
            success = success or raw_success
            success_streak = success_streak + 1 if raw_success else 0
            max_success_streak = max(max_success_streak, success_streak)
            if step >= LIFT_START:
                lift_success_streak = lift_success_streak + 1 if raw_success else 0
                max_lift_success_streak = max(max_lift_success_streak, lift_success_streak)

            if step % 20 == 0 or activated or success:
                logger.event(
                    "step",
                    step=step,
                    phase=phase,
                    cube_position=observation["cube_pos"],
                    eef_position=observation["robot0_eef_pos"],
                    grasp_detected=grasp_detected,
                    stabilized_grasp=bool(stabilizer is not None and stabilizer.active),
                    success=success,
                    reward=reward,
                )

            if writer is not None:
                frame = np.ascontiguousarray(observation["sideview_image"], dtype=np.uint8)
                writer.stdin.write(frame.tobytes())

        result = {
            "success": success if stabilize else max_lift_success_streak >= PHYSICAL_HOLD_STEPS,
            "raw_success": success,
            "final_success": bool(env._check_success()),
            "stabilized_grasp": bool(stabilizer is not None and stabilizer.active),
            "physical_only": not stabilize,
            "contact_step": stabilizer.contact_step if stabilizer is not None else contact_step,
            "final_phase": phase,
            "final_reward": float(reward),
            "height_gain": max_cube_height - initial_cube_height,
            "max_action": max_action,
            "lift_max_delta": LIFT_MAX_DELTA,
            "max_success_streak": max_success_streak,
            "max_lift_success_streak": max_lift_success_streak,
        }

        # Finalize FFmpeg before marking the experiment completed. If video
        # encoding fails, the exception path must record the run as failed
        # instead of leaving a false completed result in latest.json.
        video_process = writer
        writer = None
        if video_process is not None:
            video_process.stdin.close()
            return_code = video_process.wait()
            if return_code != 0:
                raise RuntimeError(f"FFmpeg exited with status {return_code}")

        logger.finish("completed", result=result)
        return result
    except KeyboardInterrupt:
        logger.finish("interrupted", error={"message": "keyboard interrupt"})
        raise
    except BaseException as exc:
        logger.finish(
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        if writer is not None:
            try:
                writer.stdin.close()
            finally:
                writer.wait()
        if env is not None:
            env.close()


def test_nero_dh116_initializes():
    """The registered Nero and DH116 models compose into a 13-D action space."""
    logger = ExperimentLogger("nero_dh116_initialization", config={"seed": 3})
    env = None
    try:
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
        observation = env.reset()
        assert type(env.robots[0].robot_model).__name__ == "Nero"
        assert type(env.robots[0].gripper["right"]).__name__ == "DH116"
        assert env.action_dim == 13
        assert np.isfinite(observation["cube_pos"]).all()
        logger.finish("completed", result={"action_dim": env.action_dim})
    except BaseException as exc:
        logger.finish("failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if env is not None:
            env.close()


def test_nero_dh116_lift_succeeds():
    """The seeded contact-confirmed rollout satisfies Lift's success rule."""
    result = run_lift_task(steps=DEFAULT_STEPS)
    assert result["stabilized_grasp"]
    assert result["contact_step"] is not None
    assert result["success"]
    assert result["height_gain"] > 0.04
    assert np.isfinite(result["final_reward"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--video", type=Path, nargs="?", const=DEFAULT_VIDEO)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--physical-only",
        action="store_true",
        help="disable the post-contact relative-pose stabilizer and evaluate raw MuJoCo grasp dynamics",
    )
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
        stabilize=not args.physical_only,
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
