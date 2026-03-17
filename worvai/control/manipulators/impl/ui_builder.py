from __future__ import annotations

from typing import Optional

import carb
import omni.kit.app
import omni.timeline
import omni.ui as ui
from isaacsim.core.utils.stage import add_reference_to_stage

from .manipulator_controller import ManipulatorControlConfig, ManipulatorKeyboardController
from .target_visualizer import TargetVisualizer

_DEFAULT_ROBOT_USD_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
)
_DEFAULT_ROBOT_PRIM_PATH = "/World/Franka"
_LABEL_WIDTH = 140
_UI_UPDATE_INTERVAL = 10  # update status labels every N frames


class ManipulatorUIBuilder:
    """
    Builds the Manipulator Control panel UI and manages the lifecycle of
    ``ManipulatorKeyboardController`` and ``TargetVisualizer``.

    Uses a single update-stream subscription for both physics stepping and UI updates
    to minimise per-frame overhead.
    """

    def __init__(self) -> None:
        self._controller: Optional[ManipulatorKeyboardController] = None
        self._visualizer: Optional[TargetVisualizer] = None
        self._update_sub = None
        self._timeline_sub = None
        self._frame_counter: int = 0

        # UI models
        self._usd_url_model = ui.SimpleStringModel(_DEFAULT_ROBOT_USD_URL)
        self._prim_path_model = ui.SimpleStringModel(_DEFAULT_ROBOT_PRIM_PATH)
        self._ee_speed_model = ui.SimpleFloatModel(0.0005)
        self._wrist_speed_model = ui.SimpleFloatModel(0.01)
        self._gripper_speed_model = ui.SimpleFloatModel(0.1)
        self._effort_threshold_model = ui.SimpleFloatModel(50.0)
        self._show_target_model = ui.SimpleBoolModel(True)

        # Status labels
        self._status_label: Optional[ui.Label] = None
        self._ee_pos_label: Optional[ui.Label] = None
        self._gripper_label: Optional[ui.Label] = None
        self._effort_label: Optional[ui.Label] = None
        self._blocked_label: Optional[ui.Label] = None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def build_ui(self) -> None:
        with ui.VStack(spacing=8):
            ui.Spacer(height=4)
            self._build_section("Robot Setup", self._build_robot_setup_frame)
            self._build_section("Control Parameters", self._build_params_frame)
            self._build_section("Visualisation", self._build_vis_frame)
            self._build_section("Status", self._build_status_frame)
            self._build_section("Key Bindings", self._build_keybindings_frame)
            ui.Spacer(height=4)

    def _build_robot_setup_frame(self) -> None:
        with ui.VStack(spacing=4):
            with ui.HStack(height=22):
                ui.Label("Robot USD URL", width=_LABEL_WIDTH)
                ui.StringField(model=self._usd_url_model)
            with ui.HStack(height=22):
                ui.Label("Robot Prim Path", width=_LABEL_WIDTH)
                ui.StringField(model=self._prim_path_model)
            with ui.HStack(spacing=6, height=28):
                ui.Button("Load Robot", clicked_fn=self._on_load_robot, height=26)
                ui.Button("Start Control", clicked_fn=self._on_start_control, height=26)
                ui.Button("Stop Control", clicked_fn=self._on_stop_control, height=26)

    def _build_params_frame(self) -> None:
        with ui.VStack(spacing=4):
            self._float_row("EE Step Size (m)", self._ee_speed_model, 0.0001, 0.01, self._on_param_changed)
            self._float_row("Wrist Speed (rad)", self._wrist_speed_model, 0.001, 0.1, self._on_param_changed)
            self._float_row("Gripper Speed", self._gripper_speed_model, 0.01, 1.0, self._on_param_changed)
            self._float_row("Effort Threshold", self._effort_threshold_model, 10.0, 200.0, self._on_param_changed)

    def _build_vis_frame(self) -> None:
        with ui.HStack(height=22):
            ui.Label("Show EE Target Marker", width=_LABEL_WIDTH)
            ui.CheckBox(model=self._show_target_model, width=20)
            self._show_target_model.add_value_changed_fn(self._on_show_target_changed)

    def _build_status_frame(self) -> None:
        with ui.VStack(spacing=3):
            with ui.HStack(height=18):
                ui.Label("Status:", width=_LABEL_WIDTH)
                self._status_label = ui.Label("Idle", style={"color": 0xFFAAAAAA})
            with ui.HStack(height=18):
                ui.Label("EE Target:", width=_LABEL_WIDTH)
                self._ee_pos_label = ui.Label("---")
            with ui.HStack(height=18):
                ui.Label("Gripper:", width=_LABEL_WIDTH)
                self._gripper_label = ui.Label("---")
            with ui.HStack(height=18):
                ui.Label("Peak Effort:", width=_LABEL_WIDTH)
                self._effort_label = ui.Label("---")
            with ui.HStack(height=18):
                ui.Label("Collision:", width=_LABEL_WIDTH)
                self._blocked_label = ui.Label("---")

    def _build_keybindings_frame(self) -> None:
        bindings = [
            ("W / S", "Move EE forward / backward"),
            ("A / D", "Move EE left / right"),
            ("Q / E", "Move EE up / down"),
            ("Arrows", "Alt forward / back / left / right"),
            ("Z / X", "Rotate wrist (joint 7)"),
            ("C / V", "Twist forearm (joint 6)"),
            ("K", "Toggle gripper open / close"),
        ]
        with ui.VStack(spacing=2):
            for key, desc in bindings:
                with ui.HStack(height=16):
                    ui.Label(key, width=70, style={"color": 0xFFCCCC00})
                    ui.Label(desc)

    @staticmethod
    def _build_section(title: str, build_fn) -> None:
        with ui.CollapsableFrame(title=title, collapsed=False):
            with ui.VStack(spacing=4):
                ui.Spacer(height=2)
                build_fn()
                ui.Spacer(height=2)

    @staticmethod
    def _float_row(label: str, model: ui.SimpleFloatModel, min_val: float, max_val: float, changed_fn) -> None:
        with ui.HStack(height=22):
            ui.Label(label, width=_LABEL_WIDTH)
            ui.FloatDrag(model=model, min=min_val, max=max_val, step=(max_val - min_val) / 100.0)
            model.add_value_changed_fn(changed_fn)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_load_robot(self) -> None:
        prim_path = self._prim_path_model.get_value_as_string().strip()
        if not prim_path:
            prim_path = _DEFAULT_ROBOT_PRIM_PATH
            self._prim_path_model.set_value(prim_path)
        usd_url = self._usd_url_model.get_value_as_string().strip()
        if not usd_url:
            usd_url = _DEFAULT_ROBOT_USD_URL
            self._usd_url_model.set_value(usd_url)
        try:
            add_reference_to_stage(usd_path=usd_url, prim_path=prim_path)
            self._set_status("Robot loaded", ok=True)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", ok=False)
            carb.log_error(f"[ManipulatorUI] Failed to load robot: {exc}")

    def _on_start_control(self) -> None:
        if self._controller is not None and self._controller.is_initialized:
            self._set_status("Already running", ok=True)
            return

        prim_path = self._prim_path_model.get_value_as_string().strip()
        if not prim_path:
            self._set_status("Set prim path first", ok=False)
            return

        config = ManipulatorControlConfig(
            ee_step_size_m=self._ee_speed_model.get_value_as_float(),
            wrist_joint_speed_rad=self._wrist_speed_model.get_value_as_float(),
            gripper_speed=self._gripper_speed_model.get_value_as_float(),
            effort_spike_threshold=self._effort_threshold_model.get_value_as_float(),
        )
        self._controller = ManipulatorKeyboardController(robot_prim_path=prim_path, config=config)

        timeline = omni.timeline.get_timeline_interface()
        self._timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )

        if timeline.is_playing():
            self._try_initialize()
        else:
            self._set_status("Press Play to activate", ok=True)

    def _on_stop_control(self) -> None:
        self.stop_control()
        self._set_status("Stopped", ok=True)

    def _on_param_changed(self, model) -> None:
        if self._controller is None:
            return
        cfg = self._controller.config
        cfg.ee_step_size_m = self._ee_speed_model.get_value_as_float()
        cfg.wrist_joint_speed_rad = self._wrist_speed_model.get_value_as_float()
        cfg.gripper_speed = self._gripper_speed_model.get_value_as_float()
        cfg.effort_spike_threshold = self._effort_threshold_model.get_value_as_float()

    def _on_show_target_changed(self, model) -> None:
        show = model.get_value_as_bool()
        if self._visualizer is not None and self._visualizer.is_created:
            self._visualizer.set_visible(show)
        elif show and self._controller is not None and self._controller.is_initialized:
            self._ensure_visualizer()

    # ------------------------------------------------------------------
    # Timeline integration
    # ------------------------------------------------------------------

    def _on_timeline_event(self, event) -> None:
        if event.type == int(omni.timeline.TimelineEventType.PLAY):
            self._try_initialize()
        elif event.type == int(omni.timeline.TimelineEventType.STOP):
            self._detach_update()
            if self._controller is not None:
                self._controller.reset()
            if self._visualizer is not None:
                self._visualizer.destroy()
            self._set_status("Stopped (press Play)", ok=True)

    def _try_initialize(self) -> None:
        if self._controller is None:
            return
        if not self._controller.is_initialized:
            if not self._controller.initialize():
                self._set_status("Init failed -- check console", ok=False)
                return
        else:
            self._controller.reset()

        self._ensure_visualizer()
        self._attach_update()
        self._set_status("Running", ok=True)

    # ------------------------------------------------------------------
    # Single unified update loop (physics + UI in one subscription)
    # ------------------------------------------------------------------

    def _attach_update(self) -> None:
        if self._update_sub is not None:
            return
        self._frame_counter = 0
        try:
            app = omni.kit.app.get_app()
            self._update_sub = app.get_update_event_stream().create_subscription_to_pop(
                self._on_update
            )
        except Exception as exc:
            carb.log_error(f"[ManipulatorUI] Failed to subscribe update: {exc}")

    def _detach_update(self) -> None:
        self._update_sub = None

    def _on_update(self, event) -> None:
        if self._controller is None or not self._controller.is_initialized:
            return

        # Physics step
        self._controller.step()

        # Visualizer update (position only -- color change is gated internally)
        show_target = self._show_target_model.get_value_as_bool()
        if show_target and self._visualizer is not None and self._visualizer.is_created:
            self._visualizer.update(self._controller.ee_target, self._controller.is_blocked)

        # Throttled UI label updates
        self._frame_counter += 1
        if self._frame_counter % _UI_UPDATE_INTERVAL == 0:
            self._refresh_status_labels()

    def _refresh_status_labels(self) -> None:
        target = self._controller.ee_target
        if self._ee_pos_label:
            self._ee_pos_label.text = f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
        if self._gripper_label:
            self._gripper_label.text = "Open" if self._controller.gripper_open else "Closed"
        if self._effort_label:
            self._effort_label.text = f"{self._controller.max_measured_effort:.1f} N*m"
        if self._blocked_label:
            if self._controller.is_blocked:
                self._blocked_label.text = "BLOCKED -- reverting target"
                self._blocked_label.style = {"color": 0xFF4444FF}
            else:
                self._blocked_label.text = "Clear"
                self._blocked_label.style = {"color": 0xFF44FF44}

    # ------------------------------------------------------------------
    # Visualizer
    # ------------------------------------------------------------------

    def _ensure_visualizer(self) -> None:
        if not self._show_target_model.get_value_as_bool():
            return
        if self._visualizer is None:
            prim_path = self._prim_path_model.get_value_as_string().strip()
            marker_path = prim_path.rsplit("/", 1)[0] + "/_ee_target_marker"
            self._visualizer = TargetVisualizer(prim_path=marker_path)
        self._visualizer.create()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop_control(self) -> None:
        self._detach_update()
        if self._controller is not None:
            self._controller.shutdown()
            self._controller = None
        if self._visualizer is not None:
            self._visualizer.destroy()
            self._visualizer = None
        self._timeline_sub = None

    def shutdown(self) -> None:
        self.stop_control()

    def _set_status(self, text: str, ok: bool = True) -> None:
        if self._status_label:
            self._status_label.text = text
            self._status_label.style = {"color": 0xFF44FF44 if ok else 0xFF4444FF}
