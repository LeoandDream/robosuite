"""AgileX Nero 7-DoF manipulator model."""

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Nero(ManipulatorModel):
    """AgileX Nero arm with a robosuite-compatible DH116 end-effector mount."""

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/nero/robot.xml"), idn=idn)

        # Use moderate physical damping for robosuite's torque-space controller;
        # the source position actuators already supplied additional damping.
        self.set_joint_attribute(
            attrib="damping",
            values=np.array((5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0)),
        )

    def configure_gripper(self, gripper):
        """Use compact collision jaws for DH116 on the Nero mount.

        The source DH116 meshes are retained as the physical contact model for
        other robots. On Nero they intersect the arm/table during the imported
        hand's neutral pose, so only the two low-volume jaw proxies participate
        in collision while all visual meshes remain unchanged.
        """
        if type(gripper).__name__ != "DH116":
            return
        for geom in gripper.worldbody.iter("geom"):
            name = geom.get("name", "")
            if name.endswith(("jaw_left", "jaw_right")):
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
            elif geom.get("type") == "mesh":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "DH116"}

    @property
    def default_controller_config(self):
        return {"right": "default_nero"}

    @property
    def init_qpos(self):
        # Safe pre-grasp pose with the DH116 palm above the Lift workspace.
        # The all-zero source pose drives joints 2 and 4 into their lower limits
        # before the hand can descend to the table.
        return np.array([-2.497, 1.913, 1.594, -1.010, -0.032, 0.353, -1.144])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.6

    @property
    def _eef_name(self):
        return {"right": "right_hand"}

    @property
    def arm_type(self):
        return "single"
