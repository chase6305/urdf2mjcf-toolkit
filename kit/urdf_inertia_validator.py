#!/usr/bin/env python3
"""Validate URDF inertial parameters and estimate simple inertia tensors."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class URDFInertiaValidator:
    """Validate masses, centers of mass, joint limits, and inertia matrices in a URDF."""

    def __init__(self, urdf_file: str | Path, verbose: bool = False):
        self.urdf_file = Path(urdf_file)
        self.verbose = verbose
        self.tree = ET.parse(self.urdf_file)
        self.root = self.tree.getroot()
        self.issues: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        print(f"Analyzing URDF file: {self.urdf_file}")
        print("=" * 60)

        links = self.root.findall("link")
        print(f"Found {len(links)} links")
        for link in links:
            self.validate_link(link)

        joints = self.root.findall("joint")
        print(f"Found {len(joints)} joints")
        for joint in joints:
            self.validate_joint(joint)

        self.print_summary()
        return not self.issues

    def validate_link(self, link: ET.Element) -> None:
        """Validate the physical properties of one link."""
        link_name = link.get("name", "unknown")
        if self.verbose:
            print(f"\nChecking link: {link_name}")

        inertial = link.find("inertial")
        if inertial is None:
            self.warnings.append(f"Link '{link_name}' has no inertial element")
            return

        mass_elem = inertial.find("mass")
        if mass_elem is None:
            self.issues.append(f"Link '{link_name}' inertial element has no mass")
            return

        try:
            mass_value = float(mass_elem.get("value", "0"))
        except ValueError:
            self.issues.append(f"Link '{link_name}' has an invalid mass value")
            return

        if mass_value <= 0:
            self.issues.append(
                f"Link '{link_name}' mass must be positive, got {mass_value}"
            )
        elif mass_value < 1e-6:
            self.warnings.append(f"Link '{link_name}' mass is very small: {mass_value}")

        origin = inertial.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            try:
                coords = [float(value) for value in xyz.split()]
                if len(coords) != 3:
                    raise ValueError
                if any(abs(value) > 10 for value in coords):
                    self.warnings.append(
                        f"Link '{link_name}' center of mass looks large: {coords}"
                    )
            except ValueError:
                self.issues.append(
                    f"Link '{link_name}' has invalid inertial origin xyz: {xyz}"
                )

        inertia_elem = inertial.find("inertia")
        if inertia_elem is None:
            self.issues.append(f"Link '{link_name}' has no inertia matrix")
            return

        try:
            inertia = np.array(
                [
                    [
                        float(inertia_elem.get("ixx", "0")),
                        float(inertia_elem.get("ixy", "0")),
                        float(inertia_elem.get("ixz", "0")),
                    ],
                    [
                        float(inertia_elem.get("ixy", "0")),
                        float(inertia_elem.get("iyy", "0")),
                        float(inertia_elem.get("iyz", "0")),
                    ],
                    [
                        float(inertia_elem.get("ixz", "0")),
                        float(inertia_elem.get("iyz", "0")),
                        float(inertia_elem.get("izz", "0")),
                    ],
                ]
            )
        except ValueError as exc:
            self.issues.append(f"Link '{link_name}' has invalid inertia values: {exc}")
            return

        if self.verbose:
            print(f"  mass: {mass_value}")
            print(f"  inertia matrix:\n{inertia}")

        self.check_inertia_properties(link_name, inertia, mass_value)

    def check_inertia_properties(
        self, link_name: str, inertia: np.ndarray, mass: float
    ) -> None:
        """Check mathematical and physical consistency of an inertia matrix."""
        if not np.allclose(inertia, inertia.T, rtol=1e-6, atol=1e-12):
            self.issues.append(f"Link '{link_name}' inertia matrix is not symmetric")

        try:
            eigenvalues = np.linalg.eigvalsh(inertia)
        except np.linalg.LinAlgError:
            self.issues.append(
                f"Link '{link_name}' inertia eigenvalue calculation failed"
            )
            return

        min_eigenvalue = float(np.min(eigenvalues))
        if min_eigenvalue < -1e-6:
            self.issues.append(
                f"Link '{link_name}' inertia matrix is not positive definite; min eigenvalue={min_eigenvalue:.6e}"
            )
        elif min_eigenvalue < 1e-6:
            self.warnings.append(
                f"Link '{link_name}' inertia matrix is near singular; min eigenvalue={min_eigenvalue:.6e}"
            )

        sorted_values = np.sort(eigenvalues)
        if sorted_values[0] + sorted_values[1] < sorted_values[2] - 1e-6:
            self.issues.append(
                f"Link '{link_name}' violates inertia triangle inequality: "
                f"{sorted_values[0]:.6e} + {sorted_values[1]:.6e} < {sorted_values[2]:.6e}"
            )

        if mass > 0 and np.trace(inertia) / mass < 1e-12:
            self.warnings.append(
                f"Link '{link_name}' inertia is very small relative to mass: trace(I)/m={np.trace(inertia)/mass:.6e}"
            )

        diagonal = np.diag(inertia)
        if np.any(diagonal <= 0):
            self.issues.append(
                f"Link '{link_name}' inertia diagonal entries must be positive: {diagonal}"
            )

    def validate_joint(self, joint: ET.Element) -> None:
        """Validate joint limits when present."""
        joint_name = joint.get("name", "unknown")
        joint_type = joint.get("type", "unknown")
        limit = joint.find("limit")
        if limit is None or joint_type not in {"revolute", "prismatic"}:
            return

        try:
            lower = float(limit.get("lower", "0"))
            upper = float(limit.get("upper", "0"))
            effort = float(limit.get("effort", "0"))
            velocity = float(limit.get("velocity", "0"))
        except ValueError:
            self.issues.append(f"Joint '{joint_name}' has invalid limit values")
            return

        if upper < lower:
            self.issues.append(
                f"Joint '{joint_name}' has invalid range: lower={lower}, upper={upper}"
            )
        if effort <= 0:
            self.warnings.append(
                f"Joint '{joint_name}' has non-positive effort limit: {effort}"
            )
        if velocity <= 0:
            self.warnings.append(
                f"Joint '{joint_name}' has non-positive velocity limit: {velocity}"
            )

    def print_summary(self) -> None:
        """Print validation results."""
        print("\n" + "=" * 60)
        print("Validation summary")
        print("=" * 60)

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for index, warning in enumerate(self.warnings, 1):
                print(f"  {index}. {warning}")

        if self.issues:
            print(f"\nErrors ({len(self.issues)}):")
            for index, issue in enumerate(self.issues, 1):
                print(f"  {index}. {issue}")
            print("\nValidation failed. Fix the errors above before simulation.")
        else:
            print("\nBasic validation passed.")
            if self.warnings:
                print("Review warnings before using the model in simulation.")


def estimate_inertia_for_shape(
    shape: str, mass: float, dimensions: list[float]
) -> tuple[float, float, float, float, float, float]:
    """Estimate URDF inertia tensor parameters for a simple primitive shape."""
    if mass <= 0:
        raise ValueError("mass must be positive")

    if shape == "box":
        if len(dimensions) != 3:
            raise ValueError("box dimensions must be: length width height")
        length, width, height = dimensions
        return (
            mass * (width**2 + height**2) / 12,
            0.0,
            0.0,
            mass * (length**2 + height**2) / 12,
            0.0,
            mass * (length**2 + width**2) / 12,
        )

    if shape == "cylinder":
        if len(dimensions) != 2:
            raise ValueError("cylinder dimensions must be: radius height")
        radius, height = dimensions
        ixx = mass * (3 * radius**2 + height**2) / 12
        izz = mass * radius**2 / 2
        return (ixx, 0.0, 0.0, ixx, 0.0, izz)

    if shape == "sphere":
        if len(dimensions) != 1:
            raise ValueError("sphere dimensions must be: radius")
        radius = dimensions[0]
        inertia = 2 * mass * radius**2 / 5
        return (inertia, 0.0, 0.0, inertia, 0.0, inertia)

    raise ValueError(f"unsupported shape: {shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate URDF inertia parameters.")
    parser.add_argument("urdf_file", nargs="?", help="URDF file to validate.")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print detailed link information."
    )
    parser.add_argument(
        "--estimate",
        choices=["box", "cylinder", "sphere"],
        help="Estimate inertia for a primitive shape.",
    )
    parser.add_argument("--mass", type=float, help="Shape mass in kilograms.")
    parser.add_argument(
        "--dimensions", help="Shape dimensions, for example: '0.5 0.3 0.2'."
    )
    args = parser.parse_args()

    if args.estimate:
        if args.mass is None or args.dimensions is None:
            parser.error("--estimate requires --mass and --dimensions")
        dimensions = [float(value) for value in args.dimensions.split()]
        result = estimate_inertia_for_shape(args.estimate, args.mass, dimensions)
        print(
            f'<inertia ixx="{result[0]:.6e}" ixy="{result[1]:.6e}" ixz="{result[2]:.6e}" '
            f'iyy="{result[3]:.6e}" iyz="{result[4]:.6e}" izz="{result[5]:.6e}"/>'
        )
        return 0

    if not args.urdf_file:
        parser.print_help()
        return 1

    urdf_file = Path(args.urdf_file)
    if not urdf_file.exists():
        print(f"Error: file does not exist: {urdf_file}")
        return 1

    validator = URDFInertiaValidator(urdf_file, verbose=args.verbose)
    return 0 if validator.validate_all() else 1


if __name__ == "__main__":
    sys.exit(main())
