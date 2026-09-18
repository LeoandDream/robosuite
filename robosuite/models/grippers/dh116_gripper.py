import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class DH116(GripperModel):

    def __init__(self, idn=0):
        super().__init__(
            fname=xml_path_completion("grippers/dh116.xml"),
            idn=idn,
        )

    def format_action(self, action):
        """Pass six normalized finger commands to robosuite's gripper controller.

        The controller performs the ``[-1, 1]`` to actuator-range scaling, so
        converting to ``[0, 1]`` here would incorrectly scale the command twice.
        """
        return np.clip(action, -1.0, 1.0)

    @property
    def _important_sites(self):
        return {
            "grip_site": "grip_site",
            "grip_cylinder": "grip_cylinder",
        }

    @property
    def _important_geoms(self):
        """Define opposing contact groups used by Lift's grasp detector."""
        return {
            # The original DH116 distal meshes remain the physical contact
            # model for existing Panda rollouts. Nero disables those meshes at
            # model composition time and uses the compact jaw proxies below.
            "left_finger": [
                "jaw_left",
                "finger11_link",
                "finger12_link",
                "finger13_link",
                "finger14_link",
            ],
            "right_finger": [
                "jaw_right",
                "finger21_link",
                "finger22_link",
                "finger23_link",
                "finger31_link",
                "finger32_link",
                "finger33_link",
                "finger41_link",
                "finger42_link",
                "finger43_link",
                "finger51_link",
                "finger52_link",
                "finger53_link",
            ],
            "left_fingerpad": [
                "jaw_left",
                "finger11_link",
                "finger12_link",
                "finger13_link",
                "finger14_link",
            ],
            "right_fingerpad": [
                "jaw_right",
                "finger21_link",
                "finger22_link",
                "finger23_link",
                "finger31_link",
                "finger32_link",
                "finger33_link",
                "finger41_link",
                "finger42_link",
                "finger43_link",
                "finger51_link",
                "finger52_link",
                "finger53_link",
            ],
        }

    @property
    def init_qpos(self):
        return np.zeros(len(self.joints))
