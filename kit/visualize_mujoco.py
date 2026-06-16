#!/usr/bin/env python3
"""Validate and visualize MuJoCo MJCF models."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover - depends on optional runtime package
    raise SystemExit(
        "MuJoCo is required for visualization. Install it with: python -m pip install mujoco"
    ) from exc


class MuJoCoVisualizer:
    """Small helper for loading, validating, and interactively viewing MJCF models."""

    def __init__(self, xml_path: str | Path):
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"MJCF file does not exist: {self.xml_path}")

        try:
            print(f"Loading MJCF model: {self.xml_path}")
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self.data = mujoco.MjData(self.model)
            print("Model loaded successfully")
            print(f"  - generalized coordinates: {self.model.nq}")
            print(f"  - actuators: {self.model.nu}")
            print(f"  - bodies: {self.model.nbody}")
            print(f"  - geoms: {self.model.ngeom}")
        except Exception as exc:  # pragma: no cover - MuJoCo-specific failures
            raise RuntimeError(f"Failed to load MJCF model: {exc}") from exc

    def reset_pose(self) -> None:
        """Reset simulation data to the model defaults."""
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

    def interactive_view(self, demo_mode: bool = False) -> None:
        """Launch the passive MuJoCo viewer."""
        print("\nStarting MuJoCo viewer...")
        print("Controls:")
        print("  - drag mouse: rotate camera")
        print("  - Ctrl + drag mouse: pan camera")
        print("  - mouse wheel: zoom")
        print("  - Space: pause/resume simulation")
        print("  - close window or ESC: exit")
        print()

        self.reset_pose()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            step_count = 0
            while viewer.is_running():
                if demo_mode and self.model.nu > 0:
                    t = step_count * 0.01
                    for actuator_index in range(min(self.model.nu, 10)):
                        amplitude = 0.1 / (actuator_index + 1)
                        frequency = 0.5 + actuator_index * 0.1
                        self.data.ctrl[actuator_index] = amplitude * np.sin(
                            frequency * t
                        )

                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)
                step_count += 1

    def validate_model(self) -> bool:
        """Run basic structural checks against the loaded MuJoCo model."""
        print("\n=== Model validation ===")
        issues: list[str] = []
        warnings: list[str] = []

        for joint_index in range(self.model.njnt):
            if self.model.jnt_limited[joint_index]:
                lower, upper = self.model.jnt_range[joint_index]
                if lower >= upper:
                    issues.append(
                        f"Joint {joint_index} has invalid range: [{lower}, {upper}]"
                    )

        if self.model.ngeom == 0:
            warnings.append("Model has no geoms and may not be visible")
        else:
            print(f"Geoms: {self.model.ngeom}")

        if self.model.nmesh > 0:
            print(f"Meshes: {self.model.nmesh}")
            for mesh_index in range(self.model.nmesh):
                if self.model.mesh_vertnum[mesh_index] == 0:
                    warnings.append(f"Mesh {mesh_index} has no vertices")

        if self.model.nmat > 0:
            print(f"Materials: {self.model.nmat}")
        if self.model.ntex > 0:
            print(f"Textures: {self.model.ntex}")

        if issues:
            print(f"\nFound {len(issues)} issue(s):")
            for issue in issues:
                print(f"  - {issue}")

        if warnings:
            print(f"\nFound {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"  - {warning}")

        if not issues and not warnings:
            print("\nModel validation passed")

        return not issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and visualize a MuJoCo MJCF model."
    )
    parser.add_argument("xml_file", help="Path to the MJCF XML file.")
    parser.add_argument(
        "--demo", action="store_true", help="Drive actuators with simple sine waves."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the model without opening the viewer.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Open the viewer even if validation reports issues.",
    )
    args = parser.parse_args()

    try:
        visualizer = MuJoCoVisualizer(args.xml_file)
        is_valid = visualizer.validate_model()

        if args.validate_only:
            return 0 if is_valid else 1

        if not is_valid and not args.force:
            print(
                "Validation found issues. Re-run with --force to open the viewer anyway."
            )
            return 1

        visualizer.interactive_view(demo_mode=args.demo)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
