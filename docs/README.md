# Manipulator Control

Core library for keyboard-driven manipulator control in Isaac Sim using RMPflow.

## Features

- **FrankaKeyboardController**: Combines RMPflow end-effector tracking, direct wrist joint overrides, and parallel gripper toggle
- **Collision-aware safety**: Monitors joint efforts and velocities to detect and revert blocked states
- **ManipulatorKeyboardDriver**: Reusable keyboard input layer with configurable key bindings
- **TargetVisualizer**: Optional visual marker showing the current EE target position

## Key Bindings (default)

| Key | Action |
|-----|--------|
| W/S | Move EE forward/backward |
| A/D | Move EE left/right |
| Z/X | Rotate wrist joint 7 |
| C/V | Twist forearm joint 6 |
| K   | Toggle gripper open/close |

## Usage

This extension is a library — it provides no GUI on its own. Use `worvai.gui.manipulators` for the control panel, or import directly in scripts:

```python
from worvai.control.manipulators import FrankaKeyboardController, FrankaControlConfig

controller = FrankaKeyboardController("/World/Franka")
controller.initialize()
# Call controller.step() each physics frame
```
