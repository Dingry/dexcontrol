# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script for basic admittance control without system identification.

This script shows a simple admittance control that does not require dedicated system
identification. For optimal performance, it is recommended to:
1. Perform proper system identification
2. Run this code on a Jetson or PC connected to the robot via ethernet, as admittance
   control performance is sensitive to network latency

Tuning guide:
    - If the arm is **shaky/oscillating**, increase ``damping_scale`` (e.g. 2.0–5.0)
      or decrease ``kd_gain``. Also try raising ``filter_alpha`` towards 1.0 for
      heavier smoothing.
    - If the arm feels **sluggish**, decrease ``damping_scale`` or increase ``kd_gain``.
      Lower ``filter_alpha`` towards 0.0 for less smoothing.
    - ``kd_gain`` scales both stiffness (K) and damping (D) together.
    - ``damping_scale`` scales **only** the damping (D) on top of ``kd_gain``, so it
      directly controls the K/D ratio and system bandwidth.
"""

from typing import Literal

import numpy as np
import tyro
from dexcomm import RateLimiter
from dexmotion.ik import LocalPinkIKSolver
from dexmotion.motion_manager import MotionManager
from loguru import logger
from pytransform3d import rotations
from pytransform3d import transformations as pt

from dexcontrol.core.arm import Arm
from dexcontrol.robot import Robot


class WrenchLowPassFilter:
    """Exponential moving average (EMA) low-pass filter for wrench signals.

    Smooths noisy force/torque sensor readings to prevent high-frequency
    oscillations in admittance control.  The filter equation is::

        y[k] = alpha * y[k-1] + (1 - alpha) * x[k]

    where ``alpha`` ∈ [0, 1) controls the amount of smoothing
    (higher = smoother but more lag).
    """

    def __init__(self, alpha: float = 0.5, dim: int = 6) -> None:
        """Initialize the low-pass filter.

        Args:
            alpha: Smoothing factor in [0, 1). 0 = no filtering,
                0.9 = very heavy filtering. Good starting range: 0.3–0.7.
            dim: Dimension of the wrench vector (default 6).
        """
        assert 0.0 <= alpha < 1.0, "alpha must be in [0, 1)"
        self._alpha = alpha
        self._prev: np.ndarray | None = None
        self._dim = dim

    def filter(self, wrench: np.ndarray) -> np.ndarray:
        """Apply one step of the EMA filter.

        Args:
            wrench: Raw wrench reading of shape ``(dim,)``.

        Returns:
            Filtered wrench of shape ``(dim,)``.
        """
        if self._prev is None:
            self._prev = wrench.copy()
            return wrench.copy()
        self._prev = self._alpha * self._prev + (1.0 - self._alpha) * wrench
        return self._prev.copy()


class AdmittanceController:
    """Admittance controller for robot end-effector force control.

    This controller implements a simple admittance control scheme that allows
    the robot to respond to external forces while optionally maintaining
    a desired pose.
    """

    def __init__(
        self,
        kd_gain: float = 1.0,
        damping_scale: float = 3.0,
        dt: float = 0.01,
        zero_force_control: bool = False,
    ) -> None:
        """Initialize the admittance controller.

        Args:
            kd_gain: Gain multiplier for stiffness and damping matrices.
            damping_scale: Extra multiplier applied **only** to the damping
                matrix D.  Increasing this lowers the K/D ratio (= lower
                bandwidth) and suppresses oscillations.  Values in 1.0–5.0
                are typical; higher = smoother but slower response.
            dt: Control loop time step in seconds.
            zero_force_control: If True, reduces stiffness for zero-force mode.
        """
        self.is_zero_force_control = zero_force_control
        self.K = np.diag([500.0, 500.0, 500.0, 20.0, 20.0, 10.0]) * kd_gain
        # Angular damping raised from 0.05 → 0.5 to reduce rotational shaking,
        # then further scaled by damping_scale for easy tuning.
        self.D = (
            np.diag([5.0, 5.0, 5.0, 0.5, 0.5, 0.5]) * kd_gain * damping_scale
        )
        self.dt = dt
        if zero_force_control:
            self.K = self.K / 5

    def get_admittance_pose(
        self,
        pose_cur: np.ndarray,
        wrench_ext: np.ndarray,
        pose_des: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the admittance-controlled pose.

        Args:
            pose_cur: Current end-effector pose as 4x4 transformation matrix.
            wrench_ext: External wrench as [fx, fy, fz, tx, ty, tz].
            pose_des: Desired pose as 4x4 transformation matrix. If None,
                     only external forces are considered.

        Returns:
            Corrected pose as 4x4 transformation matrix.
        """
        if pose_des is None:
            virtual_wrench = np.zeros(6)
        else:
            virtual_wrench = self.K @ self._compute_pose_dist(pose_des, pose_cur)

        virtual_wrench += wrench_ext
        correction_motion = np.linalg.inv(self.D) @ virtual_wrench * self.dt
        corrected_pose = self._apply_correction(pose_cur, correction_motion)
        return corrected_pose

    def _compute_pose_dist(
        self, pose_des: np.ndarray, pose_cur: np.ndarray
    ) -> np.ndarray:
        """Compute pose distance in SE(3) space.

        Args:
            pose_des: Desired pose as 4x4 transformation matrix.
            pose_cur: Current pose as 4x4 transformation matrix.

        Returns:
            Pose distance as [dx, dy, dz, rx, ry, rz].
        """
        pos_diff = pose_cur[:3, :3].T @ (pose_des[:3, 3] - pose_cur[:3, 3])
        rot_diff = rotations.compact_axis_angle_from_matrix(
            pose_cur[:3, :3].T @ pose_des[:3, :3]
        )
        return np.concatenate([pos_diff, rot_diff])

    def _apply_correction(
        self, pose_cur: np.ndarray, correction_motion: np.ndarray
    ) -> np.ndarray:
        """Apply motion correction to current pose.

        Args:
            pose_cur: Current pose as 4x4 transformation matrix.
            correction_motion: Correction motion as [dx, dy, dz, rx, ry, rz].

        Returns:
            Corrected pose as 4x4 transformation matrix.
        """
        mat_correction = rotations.matrix_from_compact_axis_angle(correction_motion[3:])
        pose_correction = pt.transform_from(mat_correction, correction_motion[:3])
        return pose_cur @ pose_correction


def preprocess_wrench(
    wrench: np.ndarray,
    init_wrench: np.ndarray,
    arm: Literal["left", "right"],
    force_threshold: float = 10.0,
    torque_threshold: float = 0.8,
) -> np.ndarray:
    """Preprocess raw wrench readings by removing initial offset and thresholding.

    Args:
        wrench: Raw wrench reading as [fx, fy, fz, tx, ty, tz].
        init_wrench: Initial wrench reading to subtract as offset.
        arm: Arm side ("left" or "right").
        force_threshold: Threshold for clipping forces (N).
        torque_threshold: Threshold for clipping torques (Nm).

    Returns:
        Processed wrench as [fx, fy, fz, tx, ty, tz].
    """
    # Remove initial offset
    wrench = wrench - init_wrench

    # Reorder and threshold forces
    force_reorder = [-1, 0, 2] if arm == "left" else [1, 0, 2]
    force = np.array([np.sign(i) * wrench[abs(i)] for i in force_reorder])
    force_margin = np.clip(force, -force_threshold, force_threshold)
    force = force - force_margin

    # Reorder and threshold torques
    torque_reorder = [-4, 3, 5] if arm == "left" else [4, -3, 5]
    torque = np.array([np.sign(i) * wrench[abs(i)] for i in torque_reorder])
    torque_margin = np.clip(torque, -torque_threshold, torque_threshold)
    torque = torque - torque_margin

    return np.concatenate([force, torque])


def _initialize_robot_and_motion_manager(
    bot: Robot,
) -> tuple[MotionManager, LocalPinkIKSolver]:
    """Initialize motion manager and IK solver.

    Args:
        bot: Robot instance.

    Returns:
        Tuple of (motion_manager, ik_solver).

    Raises:
        ValueError: If initialization fails.
    """
    if bot.has_component("torso"):
        qpos_dict = bot.get_joint_pos_dict(
            component=["head", "left_arm", "right_arm", "torso"]
        )
    else:
        qpos_dict = bot.get_joint_pos_dict(component=["head", "left_arm", "right_arm"])

    motion_manager = MotionManager(
        init_visualizer=False,
        initial_joint_configuration_dict=qpos_dict,
    )

    if motion_manager.pin_robot is None:
        raise ValueError("Motion Manager is not initialized")

    ik_solver = motion_manager.local_ik_solver
    if not isinstance(ik_solver, LocalPinkIKSolver):
        raise ValueError("Local IK solver is not initialized")

    return motion_manager, ik_solver


def _get_initial_ee_poses(motion_manager: MotionManager) -> dict[str, np.ndarray]:
    """Get initial end-effector poses.

    Args:
        motion_manager: Initialized motion manager.

    Returns:
        Dictionary mapping arm names to initial poses.
    """
    fk_result = motion_manager.fk(
        frame_names=motion_manager.target_frames,
        qpos=motion_manager.get_joint_pos(),
    )

    return {
        "left": fk_result["L_ee"].np,  # type: ignore
        "right": fk_result["R_ee"].np,  # type: ignore
    }


def _get_initial_wrench_readings(bot: Robot) -> dict[str, np.ndarray]:
    """Get initial wrench sensor readings for offset compensation.

    Args:
        bot: Robot instance.

    Returns:
        Dictionary mapping arm names to initial wrench readings.
    """
    if bot.left_arm.wrench_sensor is None or bot.right_arm.wrench_sensor is None:
        raise ValueError("Wrench sensors are required but not available")

    return {
        "left": bot.left_arm.wrench_sensor.get_wrench_state(),
        "right": bot.right_arm.wrench_sensor.get_wrench_state(),
    }


def _update_ee_poses_with_admittance(
    ee_pose: dict[str, np.ndarray],
    wrench_states: dict[str, dict],
    init_wrench: dict[str, np.ndarray],
    admittance_controller: AdmittanceController,
    init_ee_pose: dict[str, np.ndarray],
    zero_force: bool,
    wrench_filters: dict[str, WrenchLowPassFilter] | None = None,
) -> None:
    """Update end-effector poses using admittance control.

    Args:
        ee_pose: Dictionary of current end-effector poses (modified in-place).
        wrench_states: Dictionary of wrench sensor states.
        init_wrench: Dictionary of initial wrench readings.
        admittance_controller: Admittance controller instance.
        init_ee_pose: Dictionary of initial end-effector poses.
        zero_force: Whether to use zero-force mode.
        wrench_filters: Optional per-arm low-pass filters for wrench smoothing.
    """
    for arm in ("left", "right"):
        wrench = preprocess_wrench(wrench_states[arm]["wrench"], init_wrench[arm], arm)
        if wrench_filters is not None:
            wrench = wrench_filters[arm].filter(wrench)
        new_pose = admittance_controller.get_admittance_pose(
            pose_cur=ee_pose[arm],
            wrench_ext=wrench,
            pose_des=None if zero_force else init_ee_pose[arm],
        )
        ee_pose[arm] = new_pose


def _send_joint_commands(
    target_qpos_dict: dict[str, float],
    wrench_states: dict[str, dict],
    arms: dict[str, Arm],
    need_button: bool,
) -> None:
    """Send joint position commands to robot arms.

    Args:
        target_qpos_dict: Dictionary of target joint positions.
        wrench_states: Dictionary of wrench sensor states.
        arms: Dictionary of arm objects.
        need_button: Whether button press is required for activation.
    """
    for arm in ("left", "right"):
        if wrench_states[arm]["blue_button"] or not need_button:
            prefix = "L" if arm == "left" else "R"
            target_qpos = [
                v for k, v in target_qpos_dict.items() if f"{prefix}_arm" in k
            ]
            arms[arm].set_joint_pos(target_qpos)


def main(
    zero_force: bool = True,
    kd_gain: float = 1.0,
    damping_scale: float = 3.0,
    filter_alpha: float = 0.5,
    need_button: bool = False,
) -> None:
    """Main function for admittance control demo.

    This function demonstrates admittance control with two modes:
    1. Zero force control (default): The robot arm moves only in response to external
       forces detected by the force-torque sensor at the end effector, without trying
       to reach any target pose.
    2. Initial pose admittance control: The robot arm tries to go back to initial pose
       while also responding to external forces.

    Args:
        zero_force: If True, enables zero force control mode.
            If False, maintain initial pose while responding to external forces.
        kd_gain: Gain for the admittance controller. The larger the gain, the stiffer
            the robot arm will move under external force.
        damping_scale: Extra multiplier applied only to the damping matrix.
            Increase (e.g. 3.0–5.0) to suppress oscillations / shakiness.
            Decrease (e.g. 1.0) for faster but less stable response.
        filter_alpha: EMA smoothing factor for wrench readings in [0, 1).
            0.0 = no filtering, 0.9 = very heavy smoothing.
            Recommended range: 0.3–0.7. Increase if shaky.
        need_button: If True, the robot arm will move only when the blue button is
            pressed. If False, the robot arm will move continuously.
    """
    if kd_gain < 0.3:
        logger.warning("kd_gain is too small. Setting to 0.3.")
        kd_gain = 0.3

    # Initialize robot and components
    bot = Robot()
    arms = {"left": bot.left_arm, "right": bot.right_arm}
    arm_ee_name = {"left": "L_ee", "right": "R_ee"}

    # Validate wrench sensors
    if bot.left_arm.wrench_sensor is None or bot.right_arm.wrench_sensor is None:
        raise ValueError(
            "Admittance control requires both left and right wrench "
            "sensors to be present."
        )

    # Initialize motion manager and IK solver
    motion_manager, ik_solver = _initialize_robot_and_motion_manager(bot)

    # Get initial poses and wrench readings
    init_ee_pose = _get_initial_ee_poses(motion_manager)
    init_wrench = _get_initial_wrench_readings(bot)

    # Initialize admittance controller
    admittance_controller = AdmittanceController(
        zero_force_control=zero_force,
        kd_gain=kd_gain,
        damping_scale=damping_scale,
        dt=ik_solver.dt,
    )

    # Initialize wrench low-pass filters (one per arm)
    wrench_filters: dict[str, WrenchLowPassFilter] = {
        "left": WrenchLowPassFilter(alpha=filter_alpha),
        "right": WrenchLowPassFilter(alpha=filter_alpha),
    }

    # Setup rate limiter
    ik_hz = 1 / ik_solver.dt
    rate_limiter = RateLimiter(ik_hz)

    logger.info("Admittance control started.")
    pin_robot = motion_manager.pin_robot
    assert pin_robot is not None

    try:
        while True:
            # Get current wrench states
            wrench_states = {
                "left": bot.left_arm.wrench_sensor.get_state(),
                "right": bot.right_arm.wrench_sensor.get_state(),
            }

            # Check activation condition
            activated = any(
                wrench_states[arm]["blue_button"] for arm in ["left", "right"]
            )
            activated = True if not need_button else activated

            if activated:
                # Update motion manager with current joint positions
                motion_manager.set_joint_pos(
                    bot.get_joint_pos_dict(component=["left_arm", "right_arm"])
                )

                # Compute current end-effector poses
                ee_pose_result = motion_manager.fk(
                    frame_names=motion_manager.target_frames,
                    qpos=motion_manager.get_joint_pos(),
                )
                ee_pose: dict[str, np.ndarray] = {
                    arm: ee_pose_result[arm_ee_name[arm]].np  # type: ignore
                    for arm in ("left", "right")
                }

                # Update poses with admittance control
                _update_ee_poses_with_admittance(
                    ee_pose,
                    wrench_states,
                    init_wrench,
                    admittance_controller,
                    init_ee_pose,
                    zero_force,
                    wrench_filters,
                )

                # Solve inverse kinematics
                target_qpos_dict = ik_solver.solve_ik(
                    target_pose_dict={
                        "L_ee": ee_pose["left"],
                        "R_ee": ee_pose["right"],
                    },
                )[0]

                # Send joint commands
                _send_joint_commands(target_qpos_dict, wrench_states, arms, need_button)

            rate_limiter.sleep()

    except KeyboardInterrupt:
        logger.info("Admittance control stopped by user.")
    finally:
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
