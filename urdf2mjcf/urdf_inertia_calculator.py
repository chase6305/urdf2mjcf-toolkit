"""
URDF Inertia Calculator Module
Calculate inertia properties in URDF using Trimesh library
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from typing import Optional, Dict, Any, Tuple

try:
    import numpy as np
except ImportError:
    print("Error: numpy library not installed")
    print("Install command: pip install numpy")
    exit(1)

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh library not installed, will use existing fix logic")
    print("Install command: pip install trimesh[easy]")

from urdf2mjcf.logging_utils import URDF2MJCFLogger


class URDFInertiaCalculator:
    """URDF Inertia Calculator Class"""

    def __init__(
        self,
        urdf_path: Path,
        geometry_preference: str = "visual",
        density: Optional[float] = None,
        scale: float = 1.0,
        regularize_rel_tol: float = 1e-6,
        regularize_abs_tol: float = 1e-8,
        enforce_min_eig: Optional[float] = None,
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
        self.urdf_dir = self.urdf_path.parent
        self.geometry_preference = geometry_preference
        self.density = density
        self.scale = scale
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

    def calculate_inertia_from_mesh(
        self,
        mesh_path: Path,
        mass: float = 1.0,
        density: Optional[float] = None,
        scale: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate inertia matrix from mesh file while keeping original mass

        Args:
            mesh_path: Path to mesh file
            mass: Target mass (keep unchanged)
            density: Optional density parameter
            scale: Mesh scaling factor

        Returns:
            dict: Contains mass and calculated inertia matrix, or None if failed
        """
        if not TRIMESH_AVAILABLE:
            self.logger.warning(
                "Trimesh not available, cannot calculate inertia from mesh"
            )
            return None

        try:
            # Load mesh
            self.logger.debug(f"Loading mesh file: {mesh_path}")
            mesh = trimesh.load(mesh_path)

            if isinstance(mesh, trimesh.Scene):
                # Merge all geometries
                self.logger.debug("Mesh is a Scene, merging geometries...")
                all_meshes = list(mesh.geometry.values())
                if len(all_meshes) == 0:
                    self.logger.warning(
                        f"Cannot process mesh file {mesh_path}: Scene has no geometry"
                    )
                    return None
                else:
                    mesh = trimesh.util.concatenate(all_meshes)
                    self.logger.debug(f"Merged {len(all_meshes)} geometries")

            if not isinstance(mesh, trimesh.Trimesh):
                self.logger.warning(
                    f"Cannot process mesh file {mesh_path}: Not a valid Trimesh"
                )
                return None

            # Apply scaling
            if scale != 1.0:
                self.logger.debug(f"Applying scale {scale} to mesh")
                mesh.apply_scale(scale)

            # Set density
            if density is not None:
                mesh.density = density
                self.logger.debug(f"Set density to {density} kg/m³")
            elif hasattr(mesh, "volume") and mesh.volume > 0:
                # Calculate density from volume if available
                mesh.density = mass / mesh.volume
                self.logger.debug(
                    f"Calculated density {mesh.density:.2f} kg/m³ from mass {mass:.6f} and volume {mesh.volume:.6e}"
                )
            else:
                # Default density
                mesh.density = 1000  # kg/m^3
                self.logger.debug(f"Using default density {mesh.density} kg/m³")

            # Get mass properties
            mass_properties = mesh.mass_properties
            self.logger.debug(
                f"Mesh volume: {mesh.volume:.6e}, Calculated mass: {mass_properties['mass']:.6e}"
            )

            # Recalculate inertia matrix for target mass
            # Inertia matrix is proportional to mass, so scale accordingly
            calculated_mass = mass_properties["mass"]
            if calculated_mass > 0:
                # Scale inertia matrix to target mass
                scale_factor = mass / calculated_mass
                scaled_inertia = mass_properties["inertia"] * scale_factor
                self.logger.debug(f"Scaled inertia by factor {scale_factor:.6f}")
            else:
                # If mesh has no volume, use default
                scaled_inertia = np.eye(3) * mass * 1e-3  # Default inertia matrix
                self.logger.warning(
                    f"Mesh has zero volume, using default inertia matrix"
                )

            self.logger.info(f"Successfully calculated inertia for {mesh_path}")
            return {
                "mass": mass,  # Keep original mass
                "center_mass": mass_properties["center_mass"],
                "inertia": scaled_inertia,  # Scaled inertia matrix
            }

        except Exception as e:
            self.logger.error(f"Failed to process mesh file {mesh_path}: {e}")
            import traceback

            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None

    @staticmethod
    def inertia_matrix_to_urdf_params(inertia_matrix: np.ndarray) -> Dict[str, float]:
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
    def regularize_inertia_matrix(
        inertia_matrix: np.ndarray,
        rel_tol: float = 1e-6,
        abs_tol: float = 1e-8,
        min_eig: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Ensure inertia matrix is symmetric positive definite by clamping eigenvalues.

        Args:
            inertia_matrix: 3x3 symmetric inertia matrix
            rel_tol: relative tolerance for minimum eigenvalue as fraction of trace
            abs_tol: absolute minimum eigenvalue

        Returns:
            (regularized_matrix, min_eig_used)
        """
        # Symmetrize
        I = 0.5 * (inertia_matrix + inertia_matrix.T)

        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(I)

        trace = np.trace(I)
        min_allowed = max(abs_tol, trace * rel_tol)

        # If caller provided absolute min_eig, use the greater of computed and provided
        if min_eig is not None:
            min_allowed = max(min_allowed, float(min_eig))

        # If enforce_min_eig provided on the instance, use it as a hard minimum
        # Note: we cannot access self here because this is staticmethod; caller may pass desired min_allowed

        # If eigenvalues already OK, return original
        if np.all(eigvals > min_allowed):
            return I, min_allowed

        # Clamp eigenvalues
        eigvals_clamped = np.maximum(eigvals, min_allowed)

        # Reconstruct
        I_reg = (eigvecs * eigvals_clamped) @ eigvecs.T

        return I_reg, min_allowed

    @staticmethod
    def is_valid_inertia_matrix(
        ixx: float, iyy: float, izz: float, ixy: float, ixz: float, iyz: float
    ) -> Tuple[bool, str]:
        """Check if inertia matrix is physically valid"""
        # Build inertia matrix
        I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

        # Check if positive definite (all eigenvalues positive)
        eigenvals = np.linalg.eigvals(I)
        if np.any(eigenvals <= 0):
            return False, "Inertia matrix is not positive definite"

        # Check triangle inequalities
        if not (ixx + iyy >= izz and ixx + izz >= iyy and iyy + izz >= ixx):
            return False, "Triangle inequality violation"

        return True, "Valid"

    def find_mesh_file_for_link(
        self, link_element
    ) -> Tuple[Optional[Path], Optional[np.ndarray]]:
        """
        Extract mesh file path from URDF link element

        Args:
            link_element: URDF link element

        Returns:
            Path: Mesh file path, or None if not found
        """
        geometry_elements = []

        # Find geometry elements based on preference
        if self.geometry_preference == "collision":
            # Prefer collision geometry
            for collision in link_element.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is not None:
                    geometry_elements.append(geometry)

            # Fallback to visual geometry if no collision found
            if not geometry_elements:
                for visual in link_element.findall("visual"):
                    geometry = visual.find("geometry")
                    if geometry is not None:
                        geometry_elements.append(geometry)

        elif self.geometry_preference == "visual":
            # Prefer visual geometry
            for visual in link_element.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is not None:
                    geometry_elements.append(geometry)

            # Fallback to collision geometry if no visual found
            if not geometry_elements:
                for collision in link_element.findall("collision"):
                    geometry = collision.find("geometry")
                    if geometry is not None:
                        geometry_elements.append(geometry)

        # Extract mesh file path and origin
        for geometry in geometry_elements:
            mesh = geometry.find("mesh")
            if mesh is not None:
                filename = mesh.get("filename")
                # geometry origin (optional)
                origin = geometry.find("origin")
                origin_xyz = None
                if origin is not None:
                    xyz = origin.get("xyz")
                    if xyz:
                        try:
                            parts = [float(x) for x in xyz.strip().split()]
                            if len(parts) == 3:
                                origin_xyz = np.array(parts)
                        except Exception:
                            origin_xyz = None

                if filename:
                    # Build full path
                    mesh_path = self.urdf_dir / filename
                    if mesh_path.exists():
                        self.logger.debug(f"Found mesh file: {mesh_path}")
                        return mesh_path, origin_xyz
                    else:
                        # Try to find in subdirectories
                        for subdir in self.urdf_dir.rglob("*"):
                            if subdir.is_dir():
                                potential_path = subdir / Path(filename).name
                                if potential_path.exists():
                                    self.logger.info(
                                        f"Found mesh file in subdirectory: {subdir}"
                                    )
                                    return potential_path, origin_xyz

                        # Try to find with relative path
                        mesh_path_relative = self.urdf_dir / Path(filename)
                        if mesh_path_relative.exists():
                            self.logger.debug(
                                f"Found mesh file with relative path: {mesh_path_relative}"
                            )
                            return mesh_path_relative, origin_xyz

                        self.logger.warning(f"Mesh file not found: {mesh_path}")

        self.logger.warning("No mesh file found for link")
        return None, None

    def update_inertia(self) -> bool:
        """Update inertia matrix in URDF"""

        if not self.urdf_path.exists():
            self.logger.error(f"URDF file not found: {self.urdf_path}")
            return False

        # Parse URDF
        try:
            tree = ET.parse(self.urdf_path)
            self.logger.info(f"Successfully parsed URDF file: {self.urdf_path}")
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse URDF file: {e}")
            return False

        root = tree.getroot()

        self.logger.info("Analyzing links and recalculating inertia matrices...")
        self.logger.info(
            "Note: Only modifying inertia matrices, keeping masses unchanged"
        )

        updated_count = 0
        total_links = 0
        skipped_links = []
        failed_links = []

        for link in root.findall(".//link"):
            link_name = link.get("name")
            total_links += 1

            # Skip certain links (optional)
            skip_keywords = ["world", "base", "ground", "origin"]
            if any(keyword in link_name.lower() for keyword in skip_keywords):
                self.logger.debug(f"Skipping world/base link: {link_name}")
                continue

            self.logger.info(f"Processing link: {link_name}")

            inertial = link.find("inertial")
            if inertial is None:
                self.logger.warning(
                    f"Link {link_name} has no inertial element, skipping"
                )
                skipped_links.append(link_name)
                continue

            # Find corresponding mesh file and geometry origin
            mesh_file, geom_origin = self.find_mesh_file_for_link(link)

            if mesh_file and TRIMESH_AVAILABLE:
                # Determine geometry type used
                geom_type = self.geometry_preference
                self.logger.info(f"Found mesh file: {mesh_file} (source: {geom_type})")

                # Get current mass
                mass_elem = inertial.find("mass")
                current_mass = (
                    float(mass_elem.get("value", 1.0)) if mass_elem is not None else 1.0
                )
                self.logger.info(f"Current mass: {current_mass:.6f} kg")

                # Calculate inertia from mesh
                properties = self.calculate_inertia_from_mesh(
                    mesh_file, mass=current_mass, density=self.density, scale=self.scale
                )

                # If properties include a center of mass, update inertial/origin xyz
                if (
                    properties
                    and "center_mass" in properties
                    and properties["center_mass"] is not None
                ):
                    com = np.array(properties["center_mass"])
                    # If geometry had an origin offset, the mesh COM is in mesh frame; adjust to link origin
                    if geom_origin is not None:
                        # The COM in link frame = geom_origin + com
                        com_link = geom_origin + com
                    else:
                        com_link = com

                    # Find or create inertial/origin element
                    origin_elem = inertial.find("origin")
                    if origin_elem is None:
                        origin_elem = ET.SubElement(inertial, "origin")

                    # Set xyz as the COM offset relative to link origin
                    origin_elem.set(
                        "xyz", f"{com_link[0]:.6e} {com_link[1]:.6e} {com_link[2]:.6e}"
                    )
                    self.logger.info(
                        f"Set inertial origin xyz for {link_name} to {origin_elem.get('xyz')}"
                    )

                if properties:
                    # Update inertia parameters
                    inertia_elem = inertial.find("inertia")
                    if inertia_elem is not None:
                        urdf_params = self.inertia_matrix_to_urdf_params(
                            properties["inertia"]
                        )

                        # Check if calculation result is valid
                        valid, msg = self.is_valid_inertia_matrix(**urdf_params)

                        if valid:
                            # Show comparison between original and new values
                            orig_ixx = float(inertia_elem.get("ixx", 0))
                            orig_iyy = float(inertia_elem.get("iyy", 0))
                            orig_izz = float(inertia_elem.get("izz", 0))

                            self.logger.info(
                                f"Original inertia: ixx={orig_ixx:.3e}, iyy={orig_iyy:.3e}, izz={orig_izz:.3e}"
                            )
                            self.logger.info(
                                f"New inertia: ixx={urdf_params['ixx']:.3e}, iyy={urdf_params['iyy']:.3e}, izz={urdf_params['izz']:.3e}"
                            )
                            self.logger.info(f"Mass unchanged: {current_mass:.6f} kg")

                            # Regularize inertia matrix to ensure positive definiteness
                            I_reg, min_allowed = self.regularize_inertia_matrix(
                                properties["inertia"],
                                rel_tol=self.regularize_rel_tol,
                                abs_tol=self.regularize_abs_tol,
                                min_eig=self.enforce_min_eig,
                            )
                            # Convert regularized matrix to URDF params
                            urdf_params_reg = self.inertia_matrix_to_urdf_params(I_reg)

                            # Log eigenvalue information
                            try:
                                eigs = np.linalg.eigvalsh(I_reg)
                                self.logger.info(
                                    f"Post-regularize min eigenvalue: {eigs[0]:.6e}, threshold used: {min_allowed:.6e}"
                                )
                            except Exception:
                                pass

                            # Update URDF with regularized params
                            for param, value in urdf_params_reg.items():
                                inertia_elem.set(param, f"{value:.6e}")

                            # Clean up incorrect attributes
                            for attr in ["iyx", "izx", "izy"]:
                                if attr in inertia_elem.attrib:
                                    del inertia_elem.attrib[attr]

                            updated_count += 1
                            self.logger.info(
                                f"Successfully updated inertia parameters for {link_name}"
                            )
                        else:
                            self.logger.error(
                                f"Invalid inertia calculation for {link_name}: {msg}"
                            )
                            failed_links.append(link_name)
                    else:
                        self.logger.warning(
                            f"No inertia matrix element found for {link_name}"
                        )
                        skipped_links.append(link_name)
                else:
                    self.logger.error(
                        f"Failed to calculate inertia from mesh for {link_name}"
                    )
                    failed_links.append(link_name)
            else:
                if not mesh_file:
                    self.logger.warning(f"No mesh file found for {link_name}")
                    skipped_links.append(link_name)
                if not TRIMESH_AVAILABLE:
                    self.logger.error(
                        f"Trimesh not available, cannot process {link_name}"
                    )
                    failed_links.append(link_name)

        # Save updated URDF
        if updated_count > 0:
            # Create backup
            backup_path = self.urdf_path.with_suffix(".urdf.backup")
            if not backup_path.exists():
                shutil.copy2(self.urdf_path, backup_path)
                self.logger.info(f"Created backup: {backup_path}")

            # Save updated file
            try:
                tree.write(self.urdf_path, encoding="utf-8", xml_declaration=True)
                self.logger.info(f"\nProcessing Summary:")
                self.logger.info(f"  Total links: {total_links}")
                self.logger.info(f"  Updated links: {updated_count}")
                self.logger.info(f"  Skipped links: {len(skipped_links)}")
                self.logger.info(f"  Failed links: {len(failed_links)}")
                self.logger.info(f"  Saved to: {self.urdf_path}")

                # Log details if there are skipped or failed links
                if skipped_links:
                    self.logger.debug(f"Skipped links: {', '.join(skipped_links)}")
                if failed_links:
                    self.logger.warning(f"Failed links: {', '.join(failed_links)}")

                return True
            except Exception as e:
                self.logger.error(f"Failed to save URDF file: {e}")
                return False
        else:
            self.logger.warning("No updates performed")
            if skipped_links:
                self.logger.info(
                    f"Skipped {len(skipped_links)} links without mesh files"
                )
            return True  # No error even if no updates

    def batch_process(self, urdf_list: list) -> Dict[str, bool]:
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
        self.logger.info(f"\nBatch Processing Summary:")
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
        "--geometry",
        choices=["visual", "collision"],
        default="visual",
        help="Geometry source used for inertia calculation. Defaults to visual.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="Optional material density in kg/m^3.",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Uniform mesh scale factor."
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
        return 0 if calculator.update_inertia() else 1
    except Exception as exc:
        logger.error(f"Inertia recalculation failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
