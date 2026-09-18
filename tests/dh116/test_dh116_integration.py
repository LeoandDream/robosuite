"""Panda + DH116 集成测试。

验证 DH116 能被 robosuite 注册、拼接和控制；直接运行时可打印模型明细，
也可通过 ``--viewer`` 启动一个短时 GUI 检查。默认测试不创建渲染上下文。
"""

import argparse
import time

import numpy as np

import robosuite as suite


EXPECTED_JOINT_SUFFIXES = (
    "finger11",
    "finger12",
    "finger13",
    "finger21",
    "finger22",
    "finger31",
    "finger32",
    "finger41",
    "finger42",
    "finger51",
    "finger52",
)


def make_environment(has_renderer=False):
    """Create the common Lift environment used by tests and diagnostics."""
    return suite.make(
        env_name="Lift",
        robots="Panda",
        gripper_types="DH116",
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=1000,
        ignore_done=True,
    )


def gripper_parts(env):
    """Return the active robot, arm name, and DH116 model."""
    robot = env.robots[0]
    arm = robot.arms[0]
    return robot, arm, robot.gripper[arm]


def make_gripper_action(robot, arm, gripper, value=0.5):
    """Build an action without depending on gripper dimensions being last."""
    return robot.create_action_vector(
        {robot.get_gripper_name(arm): np.full(gripper.dof, value)}
    )


def validate_integration(env, steps=20):
    """Assert model structure and numerical stability for several control steps."""
    env.reset()
    robot, arm, gripper = gripper_parts(env)

    assert type(gripper).__name__ == "DH116"
    assert gripper.dof == 6
    assert len(gripper.actuators) == 6
    assert len(gripper.joints) == len(EXPECTED_JOINT_SUFFIXES)
    assert gripper.important_geoms["left_fingerpad"]
    assert gripper.important_geoms["right_fingerpad"]
    for suffix in EXPECTED_JOINT_SUFFIXES:
        assert any(name.endswith(suffix) for name in gripper.joints), suffix

    action = make_gripper_action(robot, arm, gripper)
    assert action.shape == (env.action_dim,)

    for _ in range(steps):
        env.step(action)

    assert np.isfinite(env.sim.data.qpos).all()
    assert np.isfinite(env.sim.data.qvel).all()
    return robot, arm, gripper


def print_model_summary(env, details=False):
    """Print a compact summary, optionally including every DH116 model element."""
    model = env.sim.model
    _, _, gripper = gripper_parts(env)
    print(
        "Model: "
        f"bodies={model.nbody}, joints={model.njnt}, geoms={model.ngeom}, "
        f"actuators={model.nu}, action_dim={env.action_dim}"
    )
    print(f"DH116: joints={len(gripper.joints)}, actuators={len(gripper.actuators)}, dof={gripper.dof}")

    if not details:
        return

    gripper_body_ids = {model.body_name2id(name) for name in gripper.bodies}

    print("\n[Joints]")
    for name in gripper.joints:
        joint_id = model.joint_name2id(name)
        print(
            f"{joint_id:3d} {name:40s} "
            f"range={model.jnt_range[joint_id]} axis={model.jnt_axis[joint_id]}"
        )

    print("\n[Actuators]")
    for name in gripper.actuators:
        actuator_id = model.actuator_name2id(name)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        print(
            f"{actuator_id:3d} {name:40s} "
            f"joint={model.joint(joint_id).name} ctrl={model.actuator_ctrlrange[actuator_id]}"
        )

    print("\n[Geoms]")
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id in gripper_body_ids:
            print(
                f"{geom_id:3d} {str(model.geom(geom_id).name):40s} "
                f"body={model.body(body_id).name} group={model.geom_group[geom_id]}"
            )


def test_dh116_integrates_with_panda():
    """Pytest entry point for the complete non-rendering integration check."""
    env = make_environment()
    try:
        validate_integration(env)
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="Control steps to validate")
    parser.add_argument("--details", action="store_true", help="Print DH116 joints, actuators, and geoms")
    parser.add_argument("--viewer", action="store_true", help="Open the GUI after validation")
    parser.add_argument("--seconds", type=float, default=10.0, help="GUI duration")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1 or args.seconds <= 0:
        raise ValueError("steps and seconds must be positive")

    env = make_environment(has_renderer=args.viewer)
    try:
        robot, arm, gripper = validate_integration(env, args.steps)
        print_model_summary(env, args.details)
        print("Integration check passed.")

        if args.viewer:
            env.viewer.update()
            deadline = time.monotonic() + args.seconds
            action = make_gripper_action(robot, arm, gripper)
            while time.monotonic() < deadline:
                env.step(action)
                env.viewer.update()
                time.sleep(0.01)
    finally:
        env.close()


if __name__ == "__main__":
    main()
