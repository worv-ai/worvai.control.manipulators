from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import carb
import carb.input
import omni.appwindow


@dataclass(frozen=True)
class ManipulatorKeyBindings:
    """
    Key binding configuration for manipulator keyboard control.

    Each field maps a logical action to a ``carb.input.KeyboardInput`` constant.
    """

    # End-effector translation (WASD + EQ for vertical)
    move_forward: int = carb.input.KeyboardInput.W
    move_backward: int = carb.input.KeyboardInput.S
    move_left: int = carb.input.KeyboardInput.A
    move_right: int = carb.input.KeyboardInput.D
    move_up: int = carb.input.KeyboardInput.E
    move_down: int = carb.input.KeyboardInput.Q

    # Arrow key aliases for EE translation
    move_forward_alt: int = carb.input.KeyboardInput.UP
    move_backward_alt: int = carb.input.KeyboardInput.DOWN
    move_left_alt: int = carb.input.KeyboardInput.LEFT
    move_right_alt: int = carb.input.KeyboardInput.RIGHT

    # Wrist joint overrides (ZXCV)
    wrist_rotate_positive: int = carb.input.KeyboardInput.Z
    wrist_rotate_negative: int = carb.input.KeyboardInput.X
    forearm_twist_positive: int = carb.input.KeyboardInput.C
    forearm_twist_negative: int = carb.input.KeyboardInput.V

    # Gripper toggle
    gripper_toggle: int = carb.input.KeyboardInput.K

    # Reset EE target to initial position
    reset_target: int = carb.input.KeyboardInput.R


@dataclass
class _KeyState:
    """Mutable press state for a single key."""

    key: int
    pressed: bool = False


class ManipulatorKeyboardDriver:
    """
    Low-level keyboard driver that tracks press/release state for manipulator control keys.

    Subscribes to the application keyboard and exposes per-key boolean state.
    Designed to be polled each physics step rather than driving actions from callbacks,
    keeping input handling decoupled from control logic.
    """

    def __init__(self, bindings: Optional[ManipulatorKeyBindings] = None) -> None:
        self._bindings = bindings or ManipulatorKeyBindings()
        self._appwindow: Optional[omni.appwindow.IAppWindow] = None
        self._input: Optional[carb.input.IInput] = None
        self._keyboard = None
        self._event_handle = None

        # Build lookup: carb key constant -> _KeyState
        self._key_states: Dict[int, _KeyState] = {}
        self._action_map: Dict[str, _KeyState] = {}
        for action_name in (
            "move_forward", "move_backward", "move_left", "move_right",
            "move_up", "move_down",
            "move_forward_alt", "move_backward_alt", "move_left_alt", "move_right_alt",
            "wrist_rotate_positive", "wrist_rotate_negative",
            "forearm_twist_positive", "forearm_twist_negative",
            "gripper_toggle", "reset_target",
        ):
            key_constant = getattr(self._bindings, action_name)
            state = _KeyState(key=key_constant)
            self._key_states[key_constant] = state
            self._action_map[action_name] = state

        self._gripper_toggled_this_press = False

    @property
    def bindings(self) -> ManipulatorKeyBindings:
        return self._bindings

    @property
    def is_connected(self) -> bool:
        return self._event_handle is not None

    def connect(self) -> bool:
        """
        Subscribe to keyboard events. Returns True on success.
        """
        if self._event_handle is not None:
            return True
        try:
            self._appwindow = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._keyboard = self._appwindow.get_keyboard()
            self._event_handle = self._input.subscribe_to_keyboard_events(
                self._keyboard, self._on_keyboard_event
            )
            return True
        except Exception as exc:
            carb.log_error(f"[ManipulatorKeyboardDriver] Failed to connect: {exc}")
            self._event_handle = None
            return False

    def disconnect(self) -> None:
        """
        Unsubscribe from keyboard events and reset all key states.
        """
        if self._input and self._keyboard and self._event_handle:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._event_handle)
        self._event_handle = None
        for state in self._key_states.values():
            state.pressed = False
        self._gripper_toggled_this_press = False

    def is_pressed(self, action_name: str) -> bool:
        """
        Check whether the key bound to *action_name* is currently held.

        Args:
            action_name: One of the field names from ``ManipulatorKeyBindings``.

        Returns:
            True if the key is pressed, False otherwise or if the action is unknown.
        """
        state = self._action_map.get(action_name)
        return state.pressed if state else False

    def consume_gripper_toggle(self) -> bool:
        """
        Return True exactly once per physical key-press of the gripper toggle key.

        This prevents repeated toggling while the key is held down.
        The flag resets on key release.
        """
        if self._gripper_toggled_this_press:
            return False
        if self.is_pressed("gripper_toggle"):
            self._gripper_toggled_this_press = True
            return True
        return False

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        key_input = event.input
        state = self._key_states.get(key_input)
        if state is None:
            return False  # Not our key — allow propagation to other handlers

        if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
            state.pressed = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            state.pressed = False
            # Reset single-shot gripper flag on release
            if key_input == self._bindings.gripper_toggle:
                self._gripper_toggled_this_press = False

        return True
