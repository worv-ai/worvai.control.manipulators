from __future__ import annotations

from typing import Optional

import carb
import numpy as np
from isaacsim.core.utils.prims import is_prim_path_valid
from pxr import Gf, UsdGeom

_DEFAULT_RADIUS = 0.015
_DEFAULT_COLOR = Gf.Vec3f(0.2, 0.8, 0.2)   # green
_BLOCKED_COLOR = Gf.Vec3f(0.9, 0.15, 0.15)  # red


class TargetVisualizer:
    """
    Optional visual marker showing the current end-effector target position.

    Creates a small ``UsdGeom.Sphere`` prim. Purely visual -- no collider or rigid body.
    Colour switches to red when the controller reports a blocked state.

    Performance: position is updated via a cached xform op every frame. Colour is only
    written when the blocked state changes, avoiding expensive USD attribute writes.
    """

    def __init__(
        self,
        prim_path: str = "/World/_ee_target_marker",
        radius: float = _DEFAULT_RADIUS,
    ) -> None:
        self._prim_path = prim_path
        self._radius = radius
        self._sphere_prim = None
        self._translate_op = None
        self._color_attr = None
        self._last_blocked: Optional[bool] = None
        self._created = False

    @property
    def is_created(self) -> bool:
        return self._created

    @property
    def prim_path(self) -> str:
        return self._prim_path

    def create(self) -> bool:
        """
        Create the sphere prim on the current stage. Idempotent.

        Returns:
            True if the marker prim exists.
        """
        if self._created and is_prim_path_valid(self._prim_path):
            return True

        try:
            from isaacsim.core.utils.stage import get_current_stage
            stage = get_current_stage()
            if stage is None:
                return False

            if is_prim_path_valid(self._prim_path):
                self._sphere_prim = stage.GetPrimAtPath(self._prim_path)
            else:
                sphere = UsdGeom.Sphere.Define(stage, self._prim_path)
                sphere.GetRadiusAttr().Set(self._radius)
                self._sphere_prim = sphere.GetPrim()

                gprim = UsdGeom.Gprim(self._sphere_prim)
                gprim.GetDisplayColorAttr().Set([_DEFAULT_COLOR])
                gprim.GetDisplayOpacityAttr().Set([0.6])

            xformable = UsdGeom.Xformable(self._sphere_prim)
            xformable.ClearXformOpOrder()
            self._translate_op = xformable.AddTranslateOp()

            # Cache the color attribute for fast writes
            self._color_attr = UsdGeom.Gprim(self._sphere_prim).GetDisplayColorAttr()
            self._last_blocked = None

            self._created = True
            return True

        except Exception as exc:
            carb.log_warn(f"[TargetVisualizer] Failed to create marker: {exc}")
            return False

    def update(self, position: np.ndarray, is_blocked: bool = False) -> None:
        """
        Move the marker to *position*. Only updates colour when blocked state changes.
        """
        if not self._created or self._translate_op is None:
            return

        # Position update -- fast, single xform op write
        self._translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

        # Colour update -- only on state change
        if is_blocked != self._last_blocked:
            self._last_blocked = is_blocked
            color = _BLOCKED_COLOR if is_blocked else _DEFAULT_COLOR
            self._color_attr.Set([color])

    def set_visible(self, visible: bool) -> None:
        if not self._created or self._sphere_prim is None:
            return
        imageable = UsdGeom.Imageable(self._sphere_prim)
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def destroy(self) -> None:
        if not self._created:
            return
        try:
            if is_prim_path_valid(self._prim_path):
                from isaacsim.core.utils.stage import get_current_stage
                stage = get_current_stage()
                if stage is not None:
                    stage.RemovePrim(self._prim_path)
        except Exception as exc:
            carb.log_warn(f"[TargetVisualizer] Failed to remove marker: {exc}")
        self._sphere_prim = None
        self._translate_op = None
        self._color_attr = None
        self._last_blocked = None
        self._created = False
