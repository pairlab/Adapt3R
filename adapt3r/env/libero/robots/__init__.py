from .mounted_ur5e import MountedUR5e
from .on_the_ground_ur5e import OnTheGroundUR5e
from .mounted_sawyer import MountedSawyer
from .on_the_ground_sawyer import OnTheGroundSawyer
from .mounted_kinova3 import MountedKinova3
from .on_the_ground_kinova3 import OnTheGroundKinova3
from .mounted_iiwa import MountedIIWA
from .on_the_ground_iiwa import OnTheGroundIIWA

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedUR5e": SingleArm,
        "OnTheGroundUR5e": SingleArm,
        "MountedSawyer": SingleArm,
        "OnTheGroundSawyer": SingleArm,
        "MountedSawyer": SingleArm,
        "OnTheGroundSawyer": SingleArm,
        "MountedKinova3": SingleArm,
        "OnTheGroundKinova3": SingleArm,
        "MountedIIWA": SingleArm,
        "OnTheGroundIIWA": SingleArm,
    }
)
