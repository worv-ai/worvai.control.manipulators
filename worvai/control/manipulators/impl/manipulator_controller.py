from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import carb
import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.robot_motion.motion_generation as mg

from .keyboard_driver import ManipulatorKeyBindings, ManipulatorKeyboardDriver

# Collision detection defaults
_EFFORT_SPIKE_THRESHOLD = 50.0
_VELOCITY_STALL_THRESHOLD = 0.005
_STALL_FRAMES_BEFORE_BLOCKED = 10
_SAFETY_CHECK_INTERVAL = 3
_REACH_SAFETY_FACTOR = 0.95


@dataclass(frozen=True)
class RobotProfile:
    """
    Describes the kinematic structure of a specific manipulator robot.

    All joint indices are zero-based DOF indices as reported by
    ``SingleArticulation.dof_names``.
    """

    # Identity — passed to ``load_supported_motion_policy_config``
    robot_name: str = "Franka"
    policy_name: str = "RMPflow"

    # Arm geometry
    arm_dof_count: int = 7
    max_reach_m: float = 0.855

    # Wrist override joints — indices and limits (radians)
    wrist_joint_indices: Tuple[int, int] = (5, 6)
    wrist_joint_lower: Tuple[float, float] = (-0.0175, -2.8973)
    wrist_joint_upper: Tuple[float, float] = (3.7525, 2.8973)

    # Gripper finger joint names and positions (metres)
    finger_joint_names: Tuple[str, ...] = ("panda_finger_joint1", "panda_finger_joint2")
    finger_open_position: float = 0.04
    finger_closed_position: float = 0.0


# Convenience presets
FRANKA_PROFILE = RobotProfile()


@dataclass
class ManipulatorControlConfig:
    """
    Tuneable runtime parameters for the manipulator keyboard controller.

    All speeds are per-physics-step deltas.
    """

    ee_step_size_m: float = 0.0005
    wrist_joint_speed_rad: float = 0.01
    gripper_speed: float = 0.1
    physics_dt: float = 1.0 / 60.0
    effort_spike_threshold: float = _EFFORT_SPIKE_THRESHOLD
    velocity_stall_threshold: float = _VELOCITY_STALL_THRESHOLD
    stall_frames_before_blocked: int = _STALL_FRAMES_BEFORE_BLOCKED


# Backwards-compatible alias
FrankaControlConfig = ManipulatorControlConfig


class ManipulatorKeyboardController:
    """
    Keyboard-driven manipulator controller combining RMPflow end-effector tracking,
    direct wrist joint overrides, and parallel gripper toggle,
    with collision-aware safety monitoring.

    The robot-specific geometry (DOF count, joint limits, finger names, reach) is
    defined by a ``RobotProfile``. The default profile targets the Franka Panda.

    Controls:
        W/A/S/D     -- move end-effector target in XY plane
        Q/E         -- move end-effector up/down (Z axis)
        Arrow keys  -- alternative forward/back/left/right
        Z/X         -- rotate wrist joint (profile.wrist_joint_indices[1])
        C/V         -- twist forearm joint (profile.wrist_joint_indices[0])
        K           -- toggle gripper open/close

    The controller must be stepped explicitly each physics frame via ``step()``.
    """

    def __init__(
        self,
        robot_prim_path: str,
        config: Optional[ManipulatorControlConfig] = None,
        profile: Optional[RobotProfile] = None,
        key_bindings: Optional[ManipulatorKeyBindings] = None,
    ) -> None:
        self._robot_prim_path = robot_prim_path
        self._config = config or ManipulatorControlConfig()
        self._profile = profile or FRANKA_PROFILE

        self._robot: Optional[SingleArticulation] = None
        self._controller: Optional[mg.MotionPolicyController] = None
        self._rmpflow: Optional[mg.lula.motion_policies.RmpFlow] = None

        self._keyboard_driver = ManipulatorKeyboardDriver(key_bindings)

        # Derived from profile — pre-allocated at init for zero per-frame alloc
        self._arm_indices = np.arange(self._profile.arm_dof_count, dtype=np.int32)
        self._wrist_indices = np.array(self._profile.wrist_joint_indices, dtype=np.int32)
        self._max_clamped_reach = self._profile.max_reach_m * _REACH_SAFETY_FACTOR

        # Motion state
        self._ee_target = np.zeros(3, dtype=np.float64)
        self._safe_ee_target = np.zeros(3, dtype=np.float64)
        self._robot_base_position = np.zeros(3, dtype=np.float64)
        self._desired_wrist = np.zeros(2, dtype=np.float64)  # [joint_a, joint_b]
        self._safe_wrist = np.zeros(2, dtype=np.float64)
        self._gripper_open: bool = True
        self._gripper_at_target: bool = True
        self._finger_joint_indices: Optional[np.ndarray] = None

        # Pre-allocated action buffers
        self._wrist_positions_buf = np.zeros(2, dtype=np.float64)
        self._gripper_positions_buf = np.zeros(2, dtype=np.float64)

        # Collision / safety state
        self._is_blocked: bool = False
        self._stall_frame_count: int = 0
        self._is_commanding_motion: bool = False
        self._max_measured_effort: float = 0.0
        self._frame_counter: int = 0

        self._initialized = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def robot(self) -> Optional[SingleArticulation]:
        return self._robot

    @property
    def ee_target(self) -> np.ndarray:
        """Current EE target. Returns read-only view -- do not modify."""
        self._ee_target.flags.writeable = False
        return self._ee_target

    @property
    def gripper_open(self) -> bool:
        return self._gripper_open

    @property
    def config(self) -> ManipulatorControlConfig:
        return self._config

    @property
    def profile(self) -> RobotProfile:
        return self._profile

    @property
    def keyboard_driver(self) -> ManipulatorKeyboardDriver:
        return self._keyboard_driver

    @property
    def is_blocked(self) -> bool:
        return self._is_blocked

    @property
    def max_measured_effort(self) -> float:
        return self._max_measured_effort

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Initialize the robot articulation, RMPflow policy, and keyboard driver.

        Returns:
            True if initialization succeeded.
        """
        try:
            self._robot = SingleArticulation(self._robot_prim_path)
            self._robot.initialize()
        except Exception as exc:
            carb.log_warn(f"[ManipulatorKeyboardController] Robot init deferred: {exc}")
            return False

        if not self._init_rmpflow():
            return False

        self._init_robot_state()
        self._resolve_finger_joint_indices()
        self._keyboard_driver.connect()
        self._initialized = True
        carb.log_info(
            f"[ManipulatorKeyboardController] Initialized '{self._profile.robot_name}' "
            f"at {self._robot_prim_path}. WASD/QE/Arrows=move, ZX=wrist, CV=forearm, K=gripper"
        )
        return True

    def step(self) -> None:
        """
        Execute one control step. Call from a physics callback each simulation frame.

        Order of operations:
            1. Collision safety check (using previous frame sensor data)
            2. If blocked, skip all new commands this frame
            3. Snapshot safe state
            4. Apply new EE target, RMPflow, wrist overrides, gripper
        """
        if not self._initialized or self._robot is None:
            return
        if not self._robot.handles_initialized:
            self._robot.initialize()
            return

        self._frame_counter += 1
        self._is_commanding_motion = self._any_motion_key_pressed()

        # Collision safety check FIRST — uses previous frame's sensor data
        if self._is_commanding_motion and self._frame_counter % _SAFETY_CHECK_INTERVAL == 0:
            if self._check_collision_safety():
                self._revert_to_safe_state()
                return  # Skip all new commands while blocked
        elif not self._is_commanding_motion:
            self._stall_frame_count = 0
            if self._is_blocked:
                self._is_blocked = False

        # Snapshot safe state before applying new commands
        self._safe_ee_target[:] = self._ee_target
        self._safe_wrist[:] = self._desired_wrist

        # Apply new commands
        self._ee_target.flags.writeable = True
        self._update_ee_target_from_keyboard()
        clamped_target = self._clamp_target_to_reach(self._ee_target)
        self._apply_rmpflow(clamped_target)
        self._apply_wrist_overrides()
        self._apply_gripper()

    def reset(self) -> None:
        """
        Re-read the robot base pose and reset the EE target to current end-effector position.
        """
        if self._robot is None or self._controller is None:
            return
        self._ee_target.flags.writeable = True
        self._init_robot_state()
        if self._rmpflow is not None:
            self._rmpflow.reset()
        self._gripper_open = True
        self._gripper_at_target = True
        self._is_blocked = False
        self._stall_frame_count = 0
        self._max_measured_effort = 0.0
        self._frame_counter = 0

    def shutdown(self) -> None:
        """
        Disconnect keyboard and release references.
        """
        self._keyboard_driver.disconnect()
        self._initialized = False
        self._robot = None
        self._controller = None
        self._rmpflow = None
        self._finger_joint_indices = None
        self._is_blocked = False
        self._stall_frame_count = 0

    # ------------------------------------------------------------------
    # Collision safety
    # ------------------------------------------------------------------

    def _check_collision_safety(self) -> bool:
        """
        Check joint efforts and velocities to detect if the arm is blocked.
        Only called when motion is being commanded (gated by caller).

        Returns:
            True if the current command should be reverted.
        """
        should_revert = False

        try:
            measured_efforts = self._robot.get_measured_joint_efforts(joint_indices=self._arm_indices)
            self._max_measured_effort = float(np.max(np.abs(measured_efforts)))
        except Exception as exc:
            carb.log_warn(f"[ManipulatorKeyboardController] Effort read failed: {exc}. Halting as precaution.")
            self._max_measured_effort = 0.0
            self._is_blocked = True
            return True  # Fail-safe: treat unknown sensor state as unsafe

        # Effort spike -- the arm is fighting a collision or singularity
        if self._max_measured_effort > self._config.effort_spike_threshold:
            if not self._is_blocked:
                carb.log_warn(
                    f"[ManipulatorKeyboardController] Effort spike detected "
                    f"({self._max_measured_effort:.1f} N*m > {self._config.effort_spike_threshold} N*m). "
                    "Reverting to safe position."
                )
            self._is_blocked = True
            should_revert = True

        # Stall detection -- joints are stationary despite active commands
        try:
            joint_velocities = self._robot.get_joint_velocities(joint_indices=self._arm_indices)
            max_velocity = float(np.max(np.abs(joint_velocities)))
        except Exception as exc:
            carb.log_warn(f"[ManipulatorKeyboardController] Velocity read failed: {exc}. Assuming stall.")
            max_velocity = 0.0  # Fail-safe: treat unknown as stalled

        if max_velocity < self._config.velocity_stall_threshold:
            self._stall_frame_count += 1
        else:
            self._stall_frame_count = 0

        if self._stall_frame_count >= self._config.stall_frames_before_blocked:
            if not self._is_blocked:
                carb.log_warn(
                    "[ManipulatorKeyboardController] Motion stall detected -- arm joints "
                    "are stationary despite active commands. Reverting target."
                )
            self._is_blocked = True
            should_revert = True

        return should_revert

    def _revert_to_safe_state(self) -> None:
        self._ee_target[:] = self._safe_ee_target
        self._desired_wrist[:] = self._safe_wrist

    def _any_motion_key_pressed(self) -> bool:
        driver = self._keyboard_driver
        for action in (
            "move_forward", "move_backward", "move_left", "move_right",
            "move_up", "move_down",
            "move_forward_alt", "move_backward_alt", "move_left_alt", "move_right_alt",
            "wrist_rotate_positive", "wrist_rotate_negative",
            "forearm_twist_positive", "forearm_twist_negative",
        ):
            if driver.is_pressed(action):
                return True
        return False

    # ------------------------------------------------------------------
    # RMPflow & robot init
    # ------------------------------------------------------------------

    def _init_rmpflow(self) -> bool:
        p = self._profile
        try:
            rmp_config = mg.interface_config_loader.load_supported_motion_policy_config(
                p.robot_name, p.policy_name
            )
            self._rmpflow = mg.lula.motion_policies.RmpFlow(**rmp_config)
            art_policy = mg.ArticulationMotionPolicy(
                self._robot, self._rmpflow, self._config.physics_dt
            )
            self._controller = mg.MotionPolicyController(
                name=f"{p.robot_name}_{p.policy_name}", articulation_motion_policy=art_policy
            )
            return True
        except Exception as exc:
            carb.log_error(f"[ManipulatorKeyboardController] RMPflow config failed: {exc}")
            return False

    def _init_robot_state(self) -> None:
        p = self._profile
        base_pos, base_rot = self._robot.get_world_pose()
        self._robot_base_position[:] = base_pos  # in-place to preserve array identity

        self._controller.get_motion_policy().set_robot_base_pose(base_pos, base_rot)

        # Seed EE target from current end-effector position via forward kinematics
        # get_end_effector_pose returns world-frame position (relative to USD stage origin)
        joint_positions = self._robot.get_joint_positions()
        active_joints = joint_positions[:p.arm_dof_count]
        ee_pos, _ = self._rmpflow.get_end_effector_pose(active_joints)
        self._ee_target[:] = ee_pos  # in-place to preserve array identity

        wi = p.wrist_joint_indices
        self._desired_wrist[0] = float(joint_positions[wi[0]])
        self._desired_wrist[1] = float(joint_positions[wi[1]])

        self._safe_ee_target[:] = self._ee_target
        self._safe_wrist[:] = self._desired_wrist

    def _resolve_finger_joint_indices(self) -> None:
        p = self._profile
        dof_names = self._robot.dof_names
        indices = []
        for finger_name in p.finger_joint_names:
            for idx, name in enumerate(dof_names):
                if name == finger_name:
                    indices.append(idx)
                    break
        if len(indices) == len(p.finger_joint_names):
            self._finger_joint_indices = np.array(indices, dtype=np.int32)
        else:
            carb.log_warn(
                f"[ManipulatorKeyboardController] Could not resolve finger joints {p.finger_joint_names} "
                f"in DOF names {dof_names}. Gripper control disabled."
            )
            self._finger_joint_indices = None

    # ------------------------------------------------------------------
    # Per-step control
    # ------------------------------------------------------------------

    def _update_ee_target_from_keyboard(self) -> None:
        if not self._is_commanding_motion:
            return
        driver = self._keyboard_driver
        step = self._config.ee_step_size_m

        dx = 0.0
        dy = 0.0
        dz = 0.0
        if driver.is_pressed("move_forward") or driver.is_pressed("move_forward_alt"):
            dx += step
        if driver.is_pressed("move_backward") or driver.is_pressed("move_backward_alt"):
            dx -= step
        if driver.is_pressed("move_left") or driver.is_pressed("move_left_alt"):
            dy += step
        if driver.is_pressed("move_right") or driver.is_pressed("move_right_alt"):
            dy -= step
        if driver.is_pressed("move_up"):
            dz += step
        if driver.is_pressed("move_down"):
            dz -= step

        self._ee_target[0] += dx
        self._ee_target[1] += dy
        self._ee_target[2] += dz

    def _clamp_target_to_reach(self, target: np.ndarray) -> np.ndarray:
        offset = target - self._robot_base_position
        distance = np.linalg.norm(offset)
        if distance < 1e-9:
            return target  # Target at robot base — no direction to clamp along
        if distance > self._max_clamped_reach:
            return self._robot_base_position + (offset / distance) * self._max_clamped_reach
        return target

    def _apply_rmpflow(self, target_position: np.ndarray) -> None:
        try:
            action = self._controller.forward(target_position, None)
            self._robot.apply_action(action)
        except Exception as exc:
            carb.log_error(f"[ManipulatorKeyboardController] RMPflow action failed: {exc}")

    def _apply_wrist_overrides(self) -> None:
        p = self._profile
        driver = self._keyboard_driver
        speed = self._config.wrist_joint_speed_rad

        # wrist_joint_indices[1] — wrist rotation (Z/X keys)
        j1_delta = 0.0
        if driver.is_pressed("wrist_rotate_positive"):
            j1_delta += speed
        if driver.is_pressed("wrist_rotate_negative"):
            j1_delta -= speed

        # wrist_joint_indices[0] — forearm twist (C/V keys)
        j0_delta = 0.0
        if driver.is_pressed("forearm_twist_positive"):
            j0_delta += speed
        if driver.is_pressed("forearm_twist_negative"):
            j0_delta -= speed

        if j0_delta == 0.0 and j1_delta == 0.0:
            return

        self._desired_wrist[0] = float(np.clip(
            self._desired_wrist[0] + j0_delta, p.wrist_joint_lower[0], p.wrist_joint_upper[0]
        ))
        self._desired_wrist[1] = float(np.clip(
            self._desired_wrist[1] + j1_delta, p.wrist_joint_lower[1], p.wrist_joint_upper[1]
        ))

        self._wrist_positions_buf[0] = self._desired_wrist[0]
        self._wrist_positions_buf[1] = self._desired_wrist[1]
        wrist_action = ArticulationAction(
            joint_positions=self._wrist_positions_buf,
            joint_indices=self._wrist_indices,
        )
        self._robot.apply_action(wrist_action)

    def _apply_gripper(self) -> None:
        if self._finger_joint_indices is None:
            return

        if self._keyboard_driver.consume_gripper_toggle():
            self._gripper_open = not self._gripper_open
            self._gripper_at_target = False

        if self._gripper_at_target:
            return

        p = self._profile
        target_pos = p.finger_open_position if self._gripper_open else p.finger_closed_position
        current_positions = self._robot.get_joint_positions(joint_indices=self._finger_joint_indices)
        current_finger = float(current_positions[0])

        diff = target_pos - current_finger
        if abs(diff) < 1e-4:
            self._gripper_at_target = True
            return

        step = np.sign(diff) * min(abs(diff), self._config.gripper_speed * self._config.physics_dt)
        new_pos = current_finger + step

        self._gripper_positions_buf[:] = new_pos
        gripper_action = ArticulationAction(
            joint_positions=self._gripper_positions_buf,
            joint_indices=self._finger_joint_indices,
        )
        self._robot.apply_action(gripper_action)


# Backwards-compatible alias
FrankaKeyboardController = ManipulatorKeyboardController
