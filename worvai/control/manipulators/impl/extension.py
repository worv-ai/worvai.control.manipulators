from __future__ import annotations

from typing import Optional

import carb
import omni.ext
import omni.ui as ui
import omni.usd
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from .ui_builder import ManipulatorUIBuilder

EXTENSION_TITLE = "Manipulator Control"


class ManipulatorControlExtension(omni.ext.IExt):
    """
    Extension for keyboard-driven Franka manipulator control with a dockable GUI panel.

    Accessible via Robot > Manipulator Control in the menu bar.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ext_id: Optional[str] = None
        self._usd_context = None
        self._window: Optional[ui.Window] = None
        self._ui_builder = ManipulatorUIBuilder()
        self._menu_items = []
        self._stage_event_sub = None

    def on_startup(self, ext_id: str) -> None:
        self._ext_id = ext_id
        carb.log_info(f"[{EXTENSION_TITLE}] Extension loaded ({ext_id}).")

        self._usd_context = omni.usd.get_context()
        self._window = ui.Window(
            EXTENSION_TITLE, width=380, height=520, visible=False,
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        self._window.set_visibility_changed_fn(self._on_window_visibility)

        self._menu_items = [
            MenuItemDescription(name=EXTENSION_TITLE, onclick_fn=self._toggle_window)
        ]
        add_menu_items(self._menu_items, "Robot")

    def on_shutdown(self) -> None:
        carb.log_info(f"[{EXTENSION_TITLE}] Shutting down.")
        remove_menu_items(self._menu_items, "Robot")
        self._menu_items.clear()
        self._stage_event_sub = None
        self._ui_builder.shutdown()

        if self._window:
            self._window.destroy()
            self._window = None

    def _toggle_window(self) -> None:
        if self._window:
            self._window.visible = not self._window.visible

    def _on_window_visibility(self, visible: bool) -> None:
        if not self._window:
            return
        if visible:
            events = self._usd_context.get_stage_event_stream()
            self._stage_event_sub = events.create_subscription_to_pop(self._on_stage_event)
            self._build_ui()
        else:
            self._stage_event_sub = None
            self._ui_builder.stop_control()

    def _build_ui(self) -> None:
        if not self._window:
            return
        with self._window.frame:
            with ui.VStack(spacing=5, height=ui.Fraction(1)):
                with ui.ScrollingFrame(height=ui.Fraction(1)):
                    with ui.VStack(spacing=6, height=0):
                        self._ui_builder.build_ui()

    def _on_stage_event(self, event) -> None:
        closed = int(omni.usd.StageEventType.CLOSED)
        opened = int(omni.usd.StageEventType.OPENED)
        if event.type in (closed, opened):
            self._ui_builder.shutdown()
