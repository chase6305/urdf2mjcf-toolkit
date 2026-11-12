"""
URDF Inertia Calculator Module
Calculate inertia properties in URDF using Trimesh library
"""

import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

from urdf2mjcf.asset_paths import MeshPathResolver
from urdf2mjcf.logging_utils import URDF2MJCFLogger
from urdf2mjcf.physics import (
    numbers,
    origin_transform,
    validate_inertia_tensor,
    validate_inertial,
    vector,
)
from urdf2mjcf.urdf_validation import URDFValidationError, declaration_issues


class URDFInertiaCalculator:
    """URDF Inertia Calculator Class"""

    def __init__(
        self,
        urdf_path: Path,
        geometry_preference: str = "visual",
        density: float | None = None,
        scale: float = 1.0,
        regularize_rel_tol: float = 1e-6,
        regularize_abs_tol: float = 1e-8,
        enforce_min_eig: float | None = None,
        mesh_search_dir: Path | None = None,
    ):
        """
        Initialize calculator

        Args:
            urdf_path: Path to URDF file
            geometry_preference: 'collision' or 'visual' - which geometry to use for inertia calculation
            density: Material density (kg/m³), if None calculate from mass and volume
            scale: Mesh scaling factor
        """
        self.urdf_path = Path(urdf_path)
        self.urdf_dir = (
            Path(mesh_search_dir) if mesh_search_dir else self.urdf_path.parent
        )
        self.geometry_preference = geometry_preference
        self.density = density
        self.scale = scale
        self.last_error: str | None = None
        self.updated_links: list[str] = []
        for name, value in (("scale", scale), ("density", density)):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        self._validate_regularization_parameters(
            regularize_rel_tol, regularize_abs_tol, enforce_min_eig
        )
        # Regularization tolerances for inertia positive-definite enforcement
        self.regularize_rel_tol = regularize_rel_tol
        self.regularize_abs_tol = regularize_abs_tol
        # Optional hard minimum eigenvalue to enforce (overrides computed threshold if set)
        self.enforce_min_eig = enforce_min_eig

        # Validate parameters
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        if geometry_preference not in ["collision", "visual"]:
            raise ValueError(
                f"geometry_preference must be 'collision' or 'visual', not '{geometry_preference}'"
            )

        # Initialize logger
        self.logger = URDF2MJCFLogger.get_logger("InertiaCalculator")
        self.logger.info(f"Initializing URDFInertiaCalculator for {urdf_path}")
        self.logger.info(
            f"Geometry preference: {geometry_preference}, Density: {density}, Scale: {scale}"
        )

    def _mesh_properties(self, mesh_path: Path, scale=1.0) -> dict:
        if not TRIMESH_AVAILABLE:
            raise RuntimeError("Trimesh is required for mesh inertia calculation")
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Mesh contains no triangle geometry: {mesh_path}")
        scaling = np.broadcast_to(np.asarray(scale, dtype=float), (3,))
        if not np.all(np.isfinite(scaling)) or np.any(scaling == 0):
            raise ValueError("Mesh scale must be finite and nonzero")
        mesh.apply_scale(scaling)
        if not mesh.is_volume:
            raise ValueError(
                f"Inertia requires a closed, consistently wound volume: {mesh_path}"
            )
        mesh.density = 1.0
        properties = mesh.mass_properties
        return {
            "volume": float(mesh.volume),
            "center_mass": np.asarray(properties["center_mass"]),
            "inertia": np.asarray(properties["inertia"]),
        }

    def calculate_inertia_from_mesh(
        self,
        mesh_path: Path,
        mass: float = 1.0,
        density: float | None = None,
        scale: float = 1.0,
    ) -> dict[str, Any] | None:
        """Compute solid-mesh inertia at its COM, normalized to the supplied mass.

        A uniform density cancels during mass normalization. Open/zero-volume
        meshes fail explicitly; no placeholder inertia is substituted.
        """
        self.last_error = None
        try:
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError("Mass must be finite and positive")
            if density is not None and (not np.isfinite(density) or density <= 0):
                raise ValueError("Density must be finite and positive")
            properties = self._mesh_properties(mesh_path, scale)
            center = properties["center_mass"]
            if not np.all(np.isfinite(center)):
                raise ValueError("Mesh center of mass must contain finite values")
            tensor = self._mass_scaled_inertia(
                properties["inertia"], mass, properties["volume"]
            )
            validate_inertia_tensor(tensor)
            return {
                "mass": mass,
                "center_mass": center,
                "inertia": tensor,
            }
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.error("Cannot calculate inertia: %s", exc)
            return None

    @staticmethod
    def _mass_scaled_inertia(tensor, mass: float, volume: float) -> np.ndarray:
        """Multiply by mass / volume without forming an overflowing ratio.

        Keep the power of two separate until the final result. This also avoids
        losing a representable tensor when a very small ratio underflows to zero.
        """
        matrix = np.asarray(tensor, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Inertia matrix must contain nine finite values")
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("Geometry volume must be finite and positive")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("Mass must be finite and positive")
        fraction, exponent = np.frexp(matrix)
        mass_fraction, mass_exponent = np.frexp(mass)
        volume_fraction, volume_exponent = np.frexp(volume)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.ldexp(
                fraction * (mass_fraction / volume_fraction),
                exponent + mass_exponent - volume_exponent,
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("Mass-normalized inertia must contain finite values")
        return result

    @staticmethod
    def inertia_matrix_to_urdf_params(inertia_matrix: np.ndarray) -> dict[str, float]:
        """Convert 3x3 inertia matrix to URDF parameters"""
        return {
            "ixx": inertia_matrix[0, 0],
            "iyy": inertia_matrix[1, 1],
            "izz": inertia_matrix[2, 2],
            "ixy": inertia_matrix[0, 1],
            "ixz": inertia_matrix[0, 2],
            "iyz": inertia_matrix[1, 2],
        }

    @staticmethod
    def _validate_regularization_parameters(rel_tol, abs_tol, min_eig) -> None:
        for name, value in (
            ("rel_tol", rel_tol),
            ("abs_tol", abs_tol),
            ("min_eig", min_eig),
        ):
            if name == "min_eig" and value is None:
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @staticmethod
    def regularize_inertia_matrix(
        inertia_matrix: np.ndarray,
        rel_tol: float = 1e-6,
        abs_tol: float = 1e-8,
        min_eig: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Ensure inertia matrix is symmetric positive definite by clamping eigenvalues.

        Args:
            inertia_matrix: 3x3 symmetric inertia matrix
            rel_tol: relative tolerance for minimum eigenvalue as fraction of trace
            abs_tol: absolute minimum eigenvalue

        Returns:
            (regularized_matrix, min_eig_used)
        """
        URDFInertiaCalculator._validate_regularization_parameters(
            rel_tol, abs_tol, min_eig
        )
        matrix = np.asarray(inertia_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("Inertia matrix must contain nine finite values")
        matrix = 0.5 * matrix + 0.5 * matrix.T

        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)

        with np.errstate(over="ignore", invalid="ignore"):
            relative_min = np.sum(np.diag(matrix) * rel_tol)
        if not np.isfinite(relative_min):
            raise ValueError("Regularization threshold is not finite")
        min_allowed = max(abs_tol, relative_min)

        # If caller provided absolute min_eig, use the greater of computed and provided
        if min_eig is not None:
            min_allowed = max(min_allowed, float(min_eig))

        # If enforce_min_eig provided on the instance, use it as a hard minimum
        # Note: we cannot access self here because this is staticmethod; caller may pass desired min_allowed

        # If eigenvalues already OK, return original
        if np.all(eigvals > min_allowed):
            return matrix, min_allowed

        # Clamp eigenvalues
        eigvals_clamped = np.maximum(eigvals, min_allowed)

        # Reconstruct
        with np.errstate(over="ignore", invalid="ignore"):
            I_reg = (eigvecs * eigvals_clamped) @ eigvecs.T
        if not np.all(np.isfinite(I_reg)):
            raise ValueError("Regularized inertia contains non-finite values")

        return I_reg, min_allowed

    @staticmethod
    def is_valid_inertia_matrix(
        ixx: float, iyy: float, izz: float, ixy: float, ixz: float, iyz: float
    ) -> tuple[bool, str]:
        """Check finite positive principal moments and their triangle inequality."""
        matrix = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        try:
            validate_inertia_tensor(matrix)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return False, str(exc)
        return True, "Valid"

    def _geometries(self, link: ET.Element) -> list[ET.Element]:
        preferred = [
            node
            for node in link.findall(self.geometry_preference)
            if node.find("geometry") is not None
        ]
        if preferred:
            return preferred
        fallback = "collision" if self.geometry_preference == "visual" else "visual"
        return [
            node for node in link.findall(fallback) if node.find("geometry") is not None
        ]

    def find_mesh_file_for_link(
        self, link_element
    ) -> tuple[Path | None, np.ndarray | None]:
        """Return the first mesh for legacy callers, rejecting ambiguous paths."""
        resolver = MeshPathResolver(
            self.urdf_dir, self.urdf_dir, prefer_converted=False
        )
        for element in self._geometries(link_element):
            mesh = element.find("geometry/mesh")
            if mesh is not None:
                path = resolver.resolve(mesh.get("filename", mesh.get("file", "")))
                return path, origin_transform(element.find("origin"))[:3, 3]
        return None, None

    def _geometry_properties(
        self,
        element: ET.Element,
        resolver: MeshPathResolver,
        mesh_cache: dict | None = None,
    ) -> dict:
        geometry = element.find("geometry")
        if len(geometry) != 1:
            raise ValueError("Each geometry must contain exactly one shape")
        shape = geometry[0]
        center = np.zeros(3)
        if shape.tag == "mesh":
            path = resolver.resolve(shape.get("filename", shape.get("file", "")))
            scale = vector(shape.get("scale", "1 1 1")) * self.scale
            key = (path, tuple(scale))
            if mesh_cache is None:
                properties = self._mesh_properties(path, scale)
            else:
                if key not in mesh_cache:
                    mesh_cache[key] = self._mesh_properties(path, scale)
                properties = mesh_cache[key]
            volume, center, tensor = (
                properties[key] for key in ("volume", "center_mass", "inertia")
            )
        elif shape.tag == "box":
            dimensions = vector(shape.get("size", "")) * self.scale
            if np.any(dimensions <= 0):
                raise ValueError("Box dimensions must be positive")
            volume = float(np.prod(dimensions))
            tensor = np.diag(volume * (np.sum(dimensions**2) - dimensions**2) / 12)
        elif shape.tag in {"sphere", "cylinder"}:
            radius = vector(shape.get("radius", ""), 1)[0] * self.scale
            if radius <= 0:
                raise ValueError("Radius must be positive")
            if shape.tag == "sphere":
                volume = 4 * np.pi * radius**3 / 3
                tensor = np.eye(3) * (2 * volume * radius**2 / 5)
            else:
                length = vector(shape.get("length", ""), 1)[0] * self.scale
                if length <= 0:
                    raise ValueError("Cylinder length must be positive")
                volume = np.pi * radius**2 * length
                transverse = volume * (3 * radius**2 + length**2) / 12
                tensor = np.diag([transverse, transverse, volume * radius**2 / 2])
        else:
            raise ValueError(f"Unsupported inertia geometry: {shape.tag}")
        transform = origin_transform(element.find("origin"))
        rotation = transform[:3, :3]
        return {
            "volume": volume,
            "center_mass": rotation @ center + transform[:3, 3],
            "inertia": rotation @ tensor @ rotation.T,
        }

    def update_inertia(self, *, only_invalid: bool = False) -> bool:
        """Recompute all available link geometries, preserving mass and inertial axes.

        Components share uniform density, apportioned by volume. Their tensors
        are rotated into the link frame and combined about the compound COM.
        Any failed link aborts the write; links without inertial data or geometry
        are left unchanged. A successful write keeps a backup of the source.
        With only_invalid, valid inertia definitions are left untouched and an
        invalid definition without usable geometry fails instead of being skipped.
        """
        self.last_error = None
        self.updated_links = []
        try:
            tree = ET.parse(self.urdf_path)
            root = tree.getroot()
            if root.tag != "robot":
                raise ValueError("Expected a <robot> root element")
            issues = declaration_issues(root)
            if issues:
                raise URDFValidationError(issues)
            resolver = MeshPathResolver(
                self.urdf_dir, self.urdf_dir, prefer_converted=False
            )
            # Keep only unit-density mass properties, not loaded triangle meshes.
            # Scope the cache to this update so edits are observed on the next run.
            mesh_cache = {}
            updated = []
            for link in root.findall("link"):
                inertial = link.find("inertial")
                geometries = self._geometries(link)
                if inertial is None:
                    continue
                try:
                    if only_invalid:
                        try:
                            validate_inertial(inertial)
                        except (ValueError, np.linalg.LinAlgError):
                            pass
                        else:
                            continue
                        if not geometries:
                            raise ValueError(
                                "Cannot repair invalid inertia without geometry"
                            )
                    elif not geometries:
                        continue
                    mass_element = inertial.find("mass")
                    if mass_element is None:
                        raise ValueError("Missing mass element")
                    mass = vector(mass_element.get("value", ""), 1)[0]
                    if mass <= 0:
                        raise ValueError("Mass must be positive")
                    parts = [
                        self._geometry_properties(element, resolver, mesh_cache)
                        for element in geometries
                    ]
                    volumes = np.array([part["volume"] for part in parts])
                    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
                        raise ValueError("Geometry volumes must be finite and positive")
                    volume = float(np.sum(volumes))
                    if not np.isfinite(volume):
                        raise ValueError("Total geometry volume must be finite")
                    center = sum(
                        (part["volume"] / volume) * part["center_mass"]
                        for part in parts
                    )
                    if not np.all(np.isfinite(center)):
                        raise ValueError("Center of mass must contain finite values")
                    tensor = np.zeros((3, 3))
                    for part in parts:
                        offset = part["center_mass"] - center
                        tensor += part["inertia"] + part["volume"] * (
                            np.dot(offset, offset) * np.eye(3)
                            - np.outer(offset, offset)
                        )
                    tensor = self._mass_scaled_inertia(tensor, mass, volume)
                    origin = inertial.find("origin")
                    rotation = origin_transform(origin)[:3, :3]
                    tensor = rotation.T @ tensor @ rotation
                    valid, message = self.is_valid_inertia_matrix(
                        **self.inertia_matrix_to_urdf_params(tensor)
                    )
                    if not valid:
                        raise ValueError(message)
                    tensor, _ = self.regularize_inertia_matrix(
                        tensor,
                        rel_tol=self.regularize_rel_tol,
                        abs_tol=self.regularize_abs_tol,
                        min_eig=self.enforce_min_eig,
                    )
                    validate_inertia_tensor(tensor)
                    if origin is None:
                        origin = ET.SubElement(inertial, "origin")
                    origin.set("xyz", numbers(center))
                    inertia = inertial.find("inertia")
                    if inertia is None:
                        inertia = ET.SubElement(inertial, "inertia")
                    inertia.attrib.clear()
                    inertia.attrib.update(
                        {
                            key: format(float(value), ".17g")
                            for key, value in self.inertia_matrix_to_urdf_params(
                                tensor
                            ).items()
                        }
                    )
                    updated.append(link.get("name", ""))
                except Exception as exc:
                    raise ValueError(f"Link {link.get('name')!r}: {exc}") from exc
            if not updated:
                self.logger.info("No inertia definitions require recalculation")
                return True
            backup_path = self.urdf_path.with_suffix(".urdf.backup")
            if not backup_path.exists():
                shutil.copy2(self.urdf_path, backup_path)
            with tempfile.NamedTemporaryFile(
                dir=self.urdf_path.parent, suffix=".urdf", delete=False
            ) as pending:
                pending_path = Path(pending.name)
            try:
                tree.write(pending_path, encoding="utf-8", xml_declaration=True)
                pending_path.replace(self.urdf_path)
            finally:
                pending_path.unlink(missing_ok=True)
            self.updated_links = updated
            self.logger.info(
                "Updated inertia for %s links; source masses preserved", len(updated)
            )
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.error("Inertia recalculation failed: %s", exc)
            return False

    def batch_process(self, urdf_list: list) -> dict[str, bool]:
        """Batch process multiple URDF files

        Args:
            urdf_list: List of URDF file paths

        Returns:
            dict: filename -> processing result mapping
        """
        results = {}

        for urdf_path in urdf_list:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing file: {urdf_path}")
            self.logger.info(f"{'='*60}")

            self.urdf_path = Path(urdf_path)
            self.urdf_dir = self.urdf_path.parent

            try:
                success = self.update_inertia()
                results[str(urdf_path)] = success
                if success:
                    self.logger.info(f"Successfully processed: {urdf_path}")
                else:
                    self.logger.error(f"Failed to process: {urdf_path}")
            except Exception as e:
                self.logger.error(f"Processing failed for {urdf_path}: {e}")
                results[str(urdf_path)] = False

        # Summary
        successful = sum(1 for result in results.values() if result)
        total = len(results)
        self.logger.info("\nBatch Processing Summary:")
        self.logger.info(f"  Successful: {successful}/{total}")
        self.logger.info(f"  Failed: {total - successful}/{total}")

        return results


def main() -> int:
    """Run inertia recalculation from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Recalculate URDF inertia values from mesh geometry."
    )
    parser.add_argument("urdf_file", help="URDF file to update in place.")
    parser.add_argument(
        "--only-invalid",
        action="store_true",
        help="Repair only invalid inertia definitions, preserving valid link parameters.",
    )
    parser.add_argument(
        "--geometry",
        choices=["visual", "collision"],
        default="visual",
        help="Geometry source used for inertia calculation. Defaults to visual.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Optional uniform density in kg/m^3; mass normalization cancels it and URDF masses are preserved.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform geometry-size factor; geometry and joint origins are not scaled.",
    )
    parser.add_argument("--regularize-rel-tol", type=float, default=1e-6)
    parser.add_argument("--regularize-abs-tol", type=float, default=1e-8)
    parser.add_argument("--enforce-min-eig", type=float, default=None)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    logger = URDF2MJCFLogger.get_logger("InertiaCalculator")
    if args.verbose:
        logger.setLevel("DEBUG")

    try:
        calculator = URDFInertiaCalculator(
            urdf_path=Path(args.urdf_file),
            geometry_preference=args.geometry,
            density=args.density,
            scale=args.scale,
            regularize_rel_tol=args.regularize_rel_tol,
            regularize_abs_tol=args.regularize_abs_tol,
            enforce_min_eig=args.enforce_min_eig,
        )
        return 0 if calculator.update_inertia(only_invalid=args.only_invalid) else 1
    except Exception as exc:
        logger.error(f"Inertia recalculation failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
