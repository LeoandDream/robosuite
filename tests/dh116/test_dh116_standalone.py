"""DH116 独立 MJCF 测试。

不经过 robosuite，直接验证 XML 能被 MuJoCo 加载，关节和执行器集合完整，
并在若干仿真步内保持数值稳定。既可由 pytest 收集，也可直接运行查看明细。
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np


XML_PATH = Path(__file__).resolve().parents[2] / "robosuite/models/assets/grippers/dh116.xml"
EXPECTED_JOINTS = {
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
}
EXPECTED_ACTUATORS = {
    "act_finger11",
    "act_finger12",
    "act_finger21",
    "act_finger31",
    "act_finger41",
    "act_finger51",
}


def load_model():
    """Load a fresh standalone DH116 model and data pair."""
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    return model, mujoco.MjData(model)


def model_names(model, kind, count):
    """Collect named MuJoCo elements of one kind."""
    getter = getattr(model, kind)
    return {getter(index).name for index in range(count)}


def validate_structure(model):
    """Check the exact controllable joint and actuator sets."""
    assert model_names(model, "joint", model.njnt) == EXPECTED_JOINTS
    assert model_names(model, "actuator", model.nu) == EXPECTED_ACTUATORS
    assert model.neq == 5

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        assert 0 <= joint_id < model.njnt


def simulate(model, data, steps):
    """Advance the model and fail immediately on NaN or infinity."""
    for step in range(steps):
        mujoco.mj_step(model, data)
        if not (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        ):
            raise AssertionError(f"DH116 became unstable at simulation step {step}")


def print_model_details(model):
    """Print compact joint-to-actuator diagnostics."""
    print(f"XML: {XML_PATH}")
    print(f"nq={model.nq}, nv={model.nv}, joints={model.njnt}, actuators={model.nu}, equalities={model.neq}")
    print("\n[Joints]")
    for joint_id in range(model.njnt):
        print(
            f"{joint_id:2d} {model.joint(joint_id).name:12s} "
            f"qpos={model.jnt_qposadr[joint_id]:2d} dof={model.jnt_dofadr[joint_id]:2d} "
            f"range={model.jnt_range[joint_id]}"
        )
    print("\n[Actuators]")
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        print(
            f"{actuator_id:2d} {model.actuator(actuator_id).name:14s} "
            f"-> {model.joint(joint_id).name}"
        )


def test_dh116_xml_structure():
    """Pytest entry point for standalone model structure."""
    model, _ = load_model()
    validate_structure(model)


def test_dh116_stability():
    """Pytest entry point for passive numerical stability."""
    model, data = load_model()
    simulate(model, data, steps=1000)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--details", action="store_true", help="Print joints and actuators")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    model, data = load_model()
    validate_structure(model)
    if args.details:
        print_model_details(model)
    simulate(model, data, args.steps)
    print(f"Standalone check passed for {args.steps} simulation steps.")


if __name__ == "__main__":
    main()
