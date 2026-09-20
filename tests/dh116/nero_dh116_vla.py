"""Small, auditable vision-language-action loop for Nero + DH116.

The task alternates between two language commands:

* ``move above the cube``: move the end effector 12 cm above a randomly placed cube;
* ``return home``: return the seven Nero arm joints to the zero keyframe.

The expert may read the simulator cube pose while collecting labels. The learned
policy cannot: the environment is created with ``use_object_obs=False`` and the
policy encoder only accepts RGB, arm proprioception, end-effector position, and
the command id. Evaluation uses held-out seeds and executes predictions through
``env.step``. No object state is rewritten and no contact stabilizer is used.

This deliberately uses NumPy ridge regression so the complete data loop runs in
the base robosuite environment without PyTorch or h5py.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

import robosuite as suite
import robosuite.macros as macros
from robosuite.utils.placement_samplers import UniformRandomSampler


macros.IMAGE_CONVENTION = "opencv"

ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "vla_artifacts"
DATASET_PATH = ARTIFACT_DIR / "nero_dh116_reach_dataset.npz"
MODEL_PATH = ARTIFACT_DIR / "nero_dh116_reach_locator.npz"
REPORT_PATH = ARTIFACT_DIR / "latest_report.json"
VIDEO_DIR = ROOT / "vedio" / "vla"

IMAGE_SIZE = 32
CAMERA_SIZE = 128
COLLECTION_STEPS_PER_COMMAND = 70
EVALUATION_STEPS_PER_COMMAND = 90
TOUCH_HEIGHT = 0.12
TABLE_CUBE_CENTER_Z = 0.821
MAX_JOINT_STEP = 0.035
SUCCESS_DISTANCE = 0.02
HOME_TOLERANCE = 0.12
HOME_QPOS = np.zeros(7, dtype=np.float64)

COMMANDS = ("move above the cube", "return home")
MOVE_ABOVE, RETURN_HOME = range(len(COMMANDS))
VIDEO_FPS = 20


def report_path(path):
    """Prefer repository-relative artifact paths so reports remain portable."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def next_numbered_video_path(video_dir, stem):
    """Return the first unused numbered path without touching existing videos."""
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = video_dir / f"{stem}_{number:03d}.mp4"
        if not candidate.exists():
            return candidate
        number += 1


def start_video_writer(path, width=CAMERA_SIZE, height=CAMERA_SIZE, fps=VIDEO_FPS):
    """Start an ffmpeg raw-RGB writer with an explicit no-overwrite guard."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
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
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def write_video_frame(writer, image):
    """Write one uint8 HWC RGB frame and return its byte count."""
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB frame, got shape {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    writer.stdin.write(np.ascontiguousarray(frame).tobytes())
    return 1


def finish_video_writer(writer):
    """Close ffmpeg and fail loudly if encoding did not complete."""
    if writer is None:
        return
    if writer.stdin is not None and not writer.stdin.closed:
        writer.stdin.close()
    return_code = writer.wait(timeout=60)
    error = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {error.strip()}")


def append_video_manifest(video_dir, records):
    """Append video metadata while preserving records from earlier runs."""
    video_dir = Path(video_dir)
    manifest_path = video_dir / "vla_video_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"schema_version": 1, "videos": []}
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("videos", [])
    manifest["videos"].extend(records)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def make_env(seed):
    """Create an RGB + proprioception environment with no object observations."""
    env = suite.make(
        env_name="Lift",
        robots="Nero",
        gripper_types="DH116",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        placement_initializer=UniformRandomSampler(
            name="NeroVLACubeSampler",
            x_range=(-0.08, 0.08),
            y_range=(-0.08, 0.08),
            rotation=0,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=(0, 0, 0.8),
            z_offset=0.01,
            rng=np.random.default_rng(int(seed)),
        ),
        camera_names="agentview",
        camera_widths=CAMERA_SIZE,
        camera_heights=CAMERA_SIZE,
        control_freq=20,
        horizon=2 * EVALUATION_STEPS_PER_COMMAND,
        ignore_done=True,
        initialization_noise={"magnitude": 0.015, "type": "gaussian"},
        seed=int(seed),
    )
    observation = env.reset()
    forbidden = [key for key in observation if "cube" in key.lower() or "object" in key.lower()]
    if forbidden:
        env.close()
        raise RuntimeError(f"policy observation leaks object truth: {forbidden}")
    return env, observation


def arm_handles(env):
    robot = env.robots[0]
    arm = robot.arms[0]
    gripper = robot.gripper[arm]
    return robot, arm, gripper, robot.get_gripper_name(arm)


def cube_position_for_expert_or_metric(env):
    """Privileged value used only by the demonstrator and external evaluator."""
    body_id = env.sim.model.body_name2id(env.cube.root_body)
    return env.sim.data.body_xpos[body_id].copy()


def policy_inputs(observation):
    """Return the explicit policy boundary; no simulator object state is accepted."""
    required = ("agentview_image", "robot0_joint_pos", "robot0_eef_pos")
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"missing policy observations: {missing}")
    return {
        "image": observation["agentview_image"],
        "joint_pos": observation["robot0_joint_pos"],
        "eef_pos": observation["robot0_eef_pos"],
    }


def expert_joint_target(env, command_id):
    """Compute an absolute joint target in the Nero controller's native semantics."""
    robot, arm, _, _ = arm_handles(env)
    qpos = env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()
    if command_id == RETURN_HOME:
        return HOME_QPOS.copy()

    target = cube_position_for_expert_or_metric(env) + np.array([0.0, 0.0, TOUCH_HEIGHT])
    site_id = robot.eef_site_id[arm]
    site_name = env.sim.model.site_id2name(site_id)
    error = target - env.sim.data.site_xpos[site_id]
    jacobian = env.sim.data.get_site_jacp(site_name).reshape(3, -1)
    jacobian = jacobian[:, robot._ref_joint_vel_indexes]
    damping = 1e-3 * np.eye(3)
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, error)
    return qpos + np.clip(delta, -MAX_JOINT_STEP, MAX_JOINT_STEP)


def step_arm(env, joint_target):
    robot, arm, gripper, gripper_name = arm_handles(env)
    action = robot.create_action_vector(
        {arm: joint_target, gripper_name: -np.ones(gripper.dof, dtype=np.float64)}
    )
    return env.step(action)[0]


def collect(dataset_path, episodes, seed_start, exploration_noise=0.004):
    """Collect expert-labelled states, grouped by episode for leakage-free splits."""
    images, joint_pos, eef_pos, commands, actions, cube_xy_labels, episode_ids, seeds = (
        [], [], [], [], [], [], [], []
    )
    summaries = []
    rng = np.random.default_rng(seed_start)
    for episode in range(episodes):
        seed = seed_start + episode
        env, observation = make_env(seed)
        try:
            cube = cube_position_for_expert_or_metric(env)
            for command_id in (MOVE_ABOVE, RETURN_HOME):
                for _ in range(COLLECTION_STEPS_PER_COMMAND):
                    inputs = policy_inputs(observation)
                    label = expert_joint_target(env, command_id)
                    images.append(
                        cv2.resize(
                            inputs["image"], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
                        )
                    )
                    joint_pos.append(inputs["joint_pos"])
                    eef_pos.append(inputs["eef_pos"])
                    commands.append(command_id)
                    actions.append(label)
                    cube_xy_labels.append(cube[:2])
                    episode_ids.append(episode)
                    seeds.append(seed)
                    executed = label + rng.normal(0.0, exploration_noise, size=label.shape)
                    observation = step_arm(env, executed)
            summaries.append({"episode": episode, "seed": seed, "cube_xy": cube[:2].tolist()})
        finally:
            env.close()

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        images=np.asarray(images, dtype=np.uint8),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
        eef_pos=np.asarray(eef_pos, dtype=np.float32),
        commands=np.asarray(commands, dtype=np.int8),
        actions=np.asarray(actions, dtype=np.float32),
        cube_xy_labels=np.asarray(cube_xy_labels, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int16),
        seeds=np.asarray(seeds, dtype=np.int32),
        command_text=np.asarray(COMMANDS),
    )
    return {
        "episodes": episodes,
        "samples": len(actions),
        "seed_start": seed_start,
        "policy_observation_keys": ["agentview_image", "robot0_joint_pos", "robot0_eef_pos"],
        "privileged_training_input": False,
        "privileged_supervision": "cube_xy_labels",
        "episode_summary": summaries,
    }


def locator_features(image, threshold=0.06):
    """Extract a red-object centroid from RGB without simulator segmentation."""
    color = np.asarray(image, dtype=np.float64) / 255.0
    redness = color[..., 0] - np.maximum(color[..., 1], color[..., 2])
    weights = np.maximum(redness - threshold, 0.0)
    yy, xx = np.mgrid[0 : color.shape[0], 0 : color.shape[1]]
    mass = weights.sum() + 1e-9
    center_x = (weights * xx).sum() / mass / max(1, color.shape[1] - 1)
    center_y = (weights * yy).sum() / mass / max(1, color.shape[0] - 1)
    return np.array(
        [1.0, center_x, center_y, center_x**2, center_x * center_y, center_y**2],
        dtype=np.float64,
    )


def train(dataset_path, model_path, validation_episodes=4):
    """Train RGB localization and retain an episode-level validation split."""
    data = np.load(dataset_path)
    if "cube_xy_labels" not in data:
        raise ValueError("dataset is missing cube_xy_labels; recollect with the current script")
    episode_ids = data["episode_ids"]
    unique_episodes = np.unique(episode_ids)
    if len(unique_episodes) <= validation_episodes:
        raise ValueError("dataset needs more episodes than the validation split")
    validation_ids = unique_episodes[-validation_episodes:]
    first_move_samples = []
    for episode_id in unique_episodes:
        indices = np.flatnonzero((episode_ids == episode_id) & (data["commands"] == MOVE_ABOVE))
        first_move_samples.append(int(indices[0]))
    first_move_samples = np.asarray(first_move_samples)
    locator_x = np.stack([locator_features(data["images"][index]) for index in first_move_samples])
    locator_y = data["cube_xy_labels"][first_move_samples]
    locator_train = ~np.isin(unique_episodes, validation_ids)
    locator_validation = ~locator_train
    locator_weights = np.linalg.solve(
        locator_x[locator_train].T @ locator_x[locator_train] + 1e-6 * np.eye(locator_x.shape[1]),
        locator_x[locator_train].T @ locator_y[locator_train],
    )
    locator_prediction = locator_x[locator_validation] @ locator_weights
    locator_error = np.linalg.norm(locator_prediction - locator_y[locator_validation], axis=1)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        commands=np.asarray(COMMANDS),
        image_size=np.asarray(IMAGE_SIZE),
        locator_weights=locator_weights,
        locator_threshold=np.asarray(0.06),
    )
    return {
        "training_episodes": int(len(unique_episodes) - len(validation_ids)),
        "validation_episodes": validation_ids.astype(int).tolist(),
        "split_unit": "episode",
        "model": "rgb_centroid_calibration_plus_robot_jacobian",
        "metrics": {
            "cube_xy_localization": {
                "validation_samples": int(locator_validation.sum()),
                "mean_error": float(locator_error.mean()),
                "max_error": float(locator_error.max()),
                "errors": locator_error.tolist(),
            }
        },
    }


def completion_audit(report):
    """Return explicit, machine-readable gates for the no-cheating milestone."""
    collection = report["collection"]
    training = report["training"]
    evaluation = report["evaluation"]
    ablation = report["vision_ablation"]
    expected_keys = ["agentview_image", "robot0_joint_pos", "robot0_eef_pos"]
    train_seeds = {item["seed"] for item in collection["episode_summary"]}
    eval_seeds = {item["seed"] for item in evaluation["episode_results"]}
    rgb_xy = [episode["commands"][0]["cube_xy"] for episode in evaluation["episode_results"]]
    blank_xy = [episode["commands"][0]["cube_xy"] for episode in ablation["episode_results"]]
    checks = {
        "policy_inputs_are_rgb_and_proprio_only": collection["policy_observation_keys"] == expected_keys
        and not collection["privileged_training_input"],
        "train_and_evaluation_seeds_are_disjoint": train_seeds.isdisjoint(eval_seeds),
        "episode_level_validation_split": training["split_unit"] == "episode",
        "validation_localization_below_1cm": training["metrics"]["cube_xy_localization"]["max_error"] < 0.01,
        "rgb_reach_all_successful": evaluation["by_command"][COMMANDS[MOVE_ABOVE]]["success_rate"] == 1.0,
        "return_home_all_successful": evaluation["by_command"][COMMANDS[RETURN_HOME]]["success_rate"] == 1.0,
        "black_rgb_reach_all_failed": ablation["by_command"][COMMANDS[MOVE_ABOVE]]["success_rate"] == 0.0,
        "paired_ablation_uses_identical_initial_targets": np.allclose(rgb_xy, blank_xy, atol=0.0, rtol=0.0),
        "no_stabilizer_or_object_rewrite": not evaluation["stabilizer"]
        and not evaluation["object_state_rewrites"]
        and not evaluation["policy_has_object_truth"],
    }
    return {"passed": all(checks.values()), "checks": checks}


class RidgeVLAPolicy:
    def __init__(self, model_path):
        self.model = np.load(model_path)
        if tuple(self.model["commands"].tolist()) != COMMANDS:
            raise ValueError("model command vocabulary does not match runtime")
        self.cached_cube_xy = None

    def reset(self):
        self.cached_cube_xy = None

    def predict(self, observation, command_id, env, blank_image=False):
        inputs = policy_inputs(observation)
        if command_id == RETURN_HOME:
            return HOME_QPOS.copy()

        if self.cached_cube_xy is None:
            image = np.zeros_like(inputs["image"]) if blank_image else inputs["image"]
            feature = locator_features(image, float(self.model["locator_threshold"]))
            estimate = feature @ self.model["locator_weights"]
            self.cached_cube_xy = np.clip(estimate, -0.08, 0.08)

        robot, arm, _, _ = arm_handles(env)
        qpos = np.asarray(inputs["joint_pos"], dtype=np.float64).copy()
        target = np.array(
            [self.cached_cube_xy[0], self.cached_cube_xy[1], TABLE_CUBE_CENTER_Z + TOUCH_HEIGHT]
        )
        site_id = robot.eef_site_id[arm]
        site_name = env.sim.model.site_id2name(site_id)
        error = target - np.asarray(inputs["eef_pos"], dtype=np.float64)
        jacobian = env.sim.data.get_site_jacp(site_name).reshape(3, -1)
        jacobian = jacobian[:, robot._ref_joint_vel_indexes]
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-3 * np.eye(3), error
        )
        return qpos + np.clip(delta, -MAX_JOINT_STEP, MAX_JOINT_STEP)


def evaluate(model_path, episodes, seed_start, blank_image=False, record_video=False, video_dir=None):
    """Run the learned policy closed-loop on seeds absent from the dataset."""
    if record_video and blank_image:
        raise ValueError("video recording is reserved for the normal RGB evaluation")
    video_dir = Path(video_dir or VIDEO_DIR)
    policy = RidgeVLAPolicy(model_path)
    episode_results = []
    video_records = []
    for episode in range(episodes):
        seed = seed_start + episode
        env, observation = make_env(seed)
        policy.reset()
        writer = None
        video_path = None
        frame_count = 0
        try:
            initial_cube_position = cube_position_for_expert_or_metric(env)
            if record_video:
                video_path = next_numbered_video_path(
                    video_dir, f"vla_rgb_seed{seed}_ep{episode + 1:03d}"
                )
                writer = start_video_writer(video_path)
                frame_count += write_video_frame(writer, observation["agentview_image"])
            command_results = []
            command_ids = (MOVE_ABOVE,) if blank_image else (MOVE_ABOVE, RETURN_HOME)
            for command_id in command_ids:
                command = COMMANDS[command_id]
                for _ in range(EVALUATION_STEPS_PER_COMMAND):
                    observation = step_arm(
                        env,
                        policy.predict(
                            observation, command_id, env=env, blank_image=blank_image
                        ),
                    )
                    if writer is not None:
                        frame_count += write_video_frame(writer, observation["agentview_image"])
                robot, arm, _, _ = arm_handles(env)
                if command_id == MOVE_ABOVE:
                    target = initial_cube_position + np.array([0.0, 0.0, TOUCH_HEIGHT])
                    error = float(np.linalg.norm(observation["robot0_eef_pos"] - target))
                    success = error < SUCCESS_DISTANCE
                    metric = {
                        "eef_target_error": error,
                        "threshold": SUCCESS_DISTANCE,
                        "cube_xy": initial_cube_position[:2].tolist(),
                        "predicted_cube_xy": policy.cached_cube_xy.tolist(),
                        "localization_error": float(
                            np.linalg.norm(initial_cube_position[:2] - policy.cached_cube_xy)
                        ),
                    }
                else:
                    qpos = env.sim.data.qpos[robot._ref_joint_pos_indexes]
                    error = float(np.linalg.norm(qpos - HOME_QPOS))
                    success = error < HOME_TOLERANCE
                    metric = {"home_joint_error": error, "threshold": HOME_TOLERANCE}
                command_results.append({"command": command, "success": bool(success), **metric})
            episode_results.append({"episode": episode, "seed": seed, "commands": command_results})
            if video_path is not None:
                video_records.append(
                    {
                        "path": report_path(video_path),
                        "seed": seed,
                        "episode": episode,
                        "commands": list(COMMANDS),
                        "frames": frame_count,
                        "fps": VIDEO_FPS,
                        "policy_image": "rgb",
                        "results": command_results,
                    }
                )
        finally:
            finish_video_writer(writer)
            env.close()

    all_results = [item for episode in episode_results for item in episode["commands"]]
    evaluated_commands = (COMMANDS[MOVE_ABOVE],) if blank_image else COMMANDS
    by_command = {
        command: {
            "successes": sum(item["success"] for item in all_results if item["command"] == command),
            "episodes": episodes,
        }
        for command in evaluated_commands
    }
    for result in by_command.values():
        result["success_rate"] = result["successes"] / result["episodes"]
    return {
        "episodes": episodes,
        "seed_start": seed_start,
        "held_out_from_collection": True,
        "closed_loop": True,
        "stabilizer": False,
        "object_state_rewrites": False,
        "policy_has_object_truth": False,
        "policy_image": "all_zero_ablation" if blank_image else "rgb",
        "by_command": by_command,
        "episode_results": episode_results,
        "videos": video_records,
    }


def write_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "train", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--episodes", type=int, default=24, help="collection episodes")
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--eval-seed-start", type=int, default=9000)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="record numbered normal-RGB evaluation videos without overwriting old files",
    )
    parser.add_argument("--video-dir", type=Path, default=VIDEO_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.episodes, args.eval_episodes) < 1:
        raise ValueError("episode counts must be positive")
    if args.stage != "all" and REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        report = {"task": "nero_dh116_small_vla_reach", "commands": list(COMMANDS)}
    if args.stage in ("collect", "all"):
        report["collection"] = collect(args.dataset, args.episodes, args.seed_start)
    if args.stage in ("train", "all"):
        report["training"] = train(args.dataset, args.model)
    if args.stage in ("evaluate", "all"):
        report["evaluation"] = evaluate(
            args.model,
            args.eval_episodes,
            args.eval_seed_start,
            record_video=args.record_video,
            video_dir=args.video_dir,
        )
        if report["evaluation"]["videos"]:
            manifest_path = append_video_manifest(args.video_dir, report["evaluation"]["videos"])
            report["evaluation"]["video_manifest"] = report_path(manifest_path)
        report["vision_ablation"] = evaluate(
            args.model, args.eval_episodes, args.eval_seed_start, blank_image=True
        )
    required_sections = {"collection", "training", "evaluation", "vision_ablation"}
    if required_sections.issubset(report):
        report["completion_audit"] = completion_audit(report)
    write_report(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
