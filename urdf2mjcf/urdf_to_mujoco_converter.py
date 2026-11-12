#!/usr/bin/env python3
"""Main entry point for the URDF/mesh asset to MuJoCo MJCF conversion pipeline."""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

from urdf2mjcf.asset_bundle import bundle_mjcf_assets, resolve_mjcf_asset_paths
from urdf2mjcf.dae_to_obj_converter import DAE2OBJConverter
from urdf2mjcf.glb_to_obj_converter import GLB2OBJConverter
from urdf2mjcf.inspection import inspect_urdf
from urdf2mjcf.logging_utils import URDF2MJCFLogger
from urdf2mjcf.mesh_to_obj_converter import MeshToOBJConverter
from urdf2mjcf.mjcf_editor import MJCFEditor
from urdf2mjcf.native_mujoco_converter import NativeMujocoConverter
from urdf2mjcf.obj_to_mjcf_converter import OBJ2MJCFImporter
from urdf2mjcf.physics import validate_compiled_physics
from urdf2mjcf.referenced_meshes import preprocess_referenced_meshes
from urdf2mjcf.reporting import new_report, validate_report_path, write_report
from urdf2mjcf.urdf_inertia_calculator import URDFInertiaCalculator
from urdf2mjcf.urdf_to_mjcf_converter import URDF2MJCFConverter, URDFConverterBase


class MujocoConversionManager:
    """Manage end-to-end URDF and mesh asset conversion to MuJoCo MJCF."""

    def __init__(
        self,
        urdf_path: str | Path,
        output_dir: str | Path | None = None,
        fixed_base: bool = True,
        export_collision: bool = False,
        auto_recalculate_inertia: bool = True,
        backend: str = "urdf2mjcf",
    ):
        if backend not in {"urdf2mjcf", "mujoco", "auto"}:
            raise ValueError(f"Unknown conversion backend: {backend}")
        self.backend = backend
        self.last_backend: str | None = None
        self.last_report: dict | None = None
        self._staged_asset_paths: dict[Path, Path] = {}
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.logger = URDF2MJCFLogger.get_logger("MujocoConversionManager")
        self.fixed_base = fixed_base
        self.export_collision = export_collision
        self.auto_recalculate_inertia = auto_recalculate_inertia

        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF file does not exist: {self.urdf_path}")
        if self.urdf_path.suffix.lower() != ".urdf":
            raise ValueError(f"Expected a .urdf input file: {self.urdf_path}")

        if output_dir is None:
            parent = self.urdf_path.parent
            self.output_dir = parent.parent / f"{parent.name}_mjcf"
        else:
            self.output_dir = Path(output_dir).expanduser().resolve()

        self.visual_dir = self._resolve_asset_dir("visual")
        self.collision_dir = self._resolve_asset_dir("collision")

        self.meshes_dir = self.output_dir / "meshes"
        self.visual_output_dir = self.meshes_dir / "visual"
        self.collision_output_dir = self.meshes_dir / "collision"

        if self.urdf_path.is_relative_to(self.meshes_dir.resolve()):
            raise ValueError("Output asset directory contains the input URDF")
        for asset_dir in (self.visual_dir, self.collision_dir):
            if asset_dir.is_dir() and (
                self.output_dir.is_relative_to(asset_dir.resolve())
                or asset_dir.resolve().is_relative_to(self.output_dir)
            ):
                raise ValueError(f"Output directory overlaps input assets: {asset_dir}")
        if (self.output_dir / f"{self.urdf_path.stem}.xml").is_dir():
            raise IsADirectoryError("The output XML path is an existing directory")

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.visual_output_dir.mkdir(exist_ok=True, parents=True)
        if self.export_collision:
            self.collision_output_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Initialized conversion for '{self.urdf_path.name}'")
        self.logger.info(f"Output will be saved to '{self.output_dir}'")

    def _resolve_asset_dir(self, name: str) -> Path:
        """Resolve an asset directory path while tolerating common capitalization variants."""
        roots = [self.urdf_path.parent, self.urdf_path.parent / "meshes"]
        if self.urdf_path.parent.name.lower() == "urdf":
            roots.append(self.urdf_path.parent.parent / "meshes")
        candidates = [
            root / variant
            for root in roots
            for variant in (name, name.capitalize(), name.upper())
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                self.logger.debug(f"Found asset directory: '{candidate}'")
                return candidate
        return self.urdf_path.parent / name

    def _copy_tree(self, src: Path, dst: Path) -> None:
        """Recursively copy a directory while preserving relative layout."""
        if not src.exists():
            self.logger.warning(f"Source directory '{src}' not found, skipping copy.")
            return

        file_list = [path for path in src.rglob("*") if path.is_file()]
        self.logger.info(f"Copying {len(file_list)} assets from '{src}' to '{dst}'...")

        with tqdm(
            total=len(file_list),
            desc=f"Copying {src.name}",
            disable=not sys.stderr.isatty(),
        ) as progress:
            for src_path in file_list:
                dst_path = dst / src_path.relative_to(src)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                self._staged_asset_paths[src_path.resolve()] = dst_path.resolve()
                progress.update(1)

    def _convert_urdf_with_recovery(self, urdf_converter: URDFConverterBase) -> bool:
        """Convert URDF to MJCF, retrying once after inertia recalculation when enabled."""
        self.logger.info("Starting URDF to MJCF conversion...")
        if urdf_converter.convert():
            output_xml = urdf_converter.get_output_xml()
            if output_xml and Path(output_xml).exists():
                self.logger.info("URDF to MJCF conversion successful.")
                return True

        detail = getattr(urdf_converter, "last_error", None)
        codes = getattr(urdf_converter, "last_error_codes", ())
        if (codes and set(codes) != {"invalid_inertia"}) or (
            not codes
            and detail
            and not any(word in detail.lower() for word in ("inertia", "mass"))
        ):
            self.logger.debug(
                "Skipping inertia recovery for a non-inertia conversion error"
            )
            return False

        if not self.auto_recalculate_inertia:
            self.logger.error(
                "URDF conversion failed and inertia recalculation is disabled."
            )
            return False

        self.logger.warning(
            "URDF conversion failed. Recalculating invalid inertia definitions only."
        )
        recovery = {"status": "running", "links": []}
        if self.last_report is not None:
            self.last_report["inertia_recovery_attempts"] += 1
            self.last_report["inertia_recoveries"].append(recovery)
            if self.last_report["attempts"]:
                recovery["backend"] = self.last_report["attempts"][-1]["backend"]
        original_converter_path = urdf_converter.urdf_path
        with tempfile.TemporaryDirectory(
            prefix="urdf2mjcf-recovery-", dir=self.output_dir
        ) as recovery_dir:
            recovery_urdf = Path(recovery_dir) / self.urdf_path.name
            shutil.copy2(self.urdf_path, recovery_urdf)
            inertia_calculator = URDFInertiaCalculator(
                recovery_urdf, mesh_search_dir=self.urdf_path.parent
            )
            if not inertia_calculator.update_inertia(only_invalid=True):
                recovery.update(status="failed", error=inertia_calculator.last_error)
                urdf_converter.last_error = (
                    f"{detail or 'Conversion failed'}; inertia recovery failed: "
                    f"{inertia_calculator.last_error}"
                )
                self.logger.error("Inertia recalculation failed. Aborting.")
                return False
            recovery["links"] = inertia_calculator.updated_links
            if not recovery["links"]:
                recovery["status"] = "unchanged"
                self.logger.warning(
                    "No invalid inertia definitions found; keeping source parameters."
                )
                return False
            recovery["status"] = "recalculated"

            self.logger.info(
                "Inertia recalculation complete. Retrying URDF to MJCF conversion."
            )
            try:
                urdf_converter.urdf_path = recovery_urdf
                if not urdf_converter.convert():
                    self.logger.error("URDF conversion failed again after retry.")
                    return False
            finally:
                urdf_converter.urdf_path = original_converter_path

        output_xml = urdf_converter.get_output_xml()
        if not (output_xml and Path(output_xml).exists()):
            self.logger.error(
                "Conversion succeeded, but the output XML was not generated."
            )
            return False

        self.logger.info("URDF to MJCF conversion successful after retry.")
        return True

    def _convert_mesh_directory(self, directory: Path) -> None:
        """Convert one asset tree and stop if any required mesh failed."""
        inputs = {}
        formats = {".obj", ".dae", ".glb", ".stl", ".ply", ".off", ".3mf"}
        for source in sorted(directory.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in formats:
                continue
            key = (source.parent, source.stem)
            if key in inputs:
                raise ValueError(
                    f"Source meshes share a conversion name: {inputs[key]} and {source}; use distinct filenames"
                )
            inputs[key] = source
        dae_results = DAE2OBJConverter().convert_directory(
            directory, directory, recursive=True, preserve_textures=True
        )
        glb_results = GLB2OBJConverter(str(directory)).convert_directory(
            directory, directory, recursive=True
        )
        mesh_results = MeshToOBJConverter(overwrite=True).convert_directory(directory)
        failed = [
            source
            for source, (success, outputs) in dae_results.items()
            if not success or not outputs
        ]
        failed.extend(source for source, success in glb_results.items() if not success)
        failed.extend(
            str(source) for source, output in mesh_results.items() if output is None
        )
        if failed:
            raise RuntimeError("Mesh conversion failed for: " + ", ".join(failed))

    def run(self) -> Path:
        """Convert and validate the model, returning the final XML path."""
        self.last_backend = None
        self._staged_asset_paths = {}
        started = time.perf_counter()
        self.last_report = new_report("conversion", self.urdf_path, self.backend)
        self.last_report.update(
            used_backend=None,
            output=None,
            attempts=[],
            inertia_recovery_attempts=0,
            inertia_recoveries=[],
            mesh_preprocessing=[],
            options={
                "fixed_base": self.fixed_base,
                "export_collision": self.export_collision,
                "auto_recalculate_inertia": self.auto_recalculate_inertia,
            },
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix="urdf2mjcf-work-",
                dir=self.output_dir,
                ignore_cleanup_errors=True,
            ) as working:
                published_dirs = (
                    self.meshes_dir,
                    self.visual_output_dir,
                    self.collision_output_dir,
                )
                self.meshes_dir = Path(working) / "meshes"
                self.visual_output_dir = self.meshes_dir / "visual"
                self.collision_output_dir = self.meshes_dir / "collision"
                self.visual_output_dir.mkdir(parents=True)
                if self.export_collision:
                    self.collision_output_dir.mkdir(parents=True)
                try:
                    result = self._run_pipeline(Path(working))
                finally:
                    (
                        self.meshes_dir,
                        self.visual_output_dir,
                        self.collision_output_dir,
                    ) = published_dirs
            self.last_report.update(
                status="success",
                stage="completed",
                used_backend=self.last_backend,
                output=str(result),
            )
            return result
        except Exception as exc:
            self.last_report.update(status="failed", error=str(exc))
            raise
        finally:
            self.last_report["duration_seconds"] = round(
                time.perf_counter() - started, 6
            )

    def _set_stage(self, stage: str) -> None:
        if self.last_report is not None:
            self.last_report["stage"] = stage

    def _publish(self, pending_xml: Path, final_xml: Path) -> None:
        """Publish validated assets and XML together, rolling back on I/O errors."""
        destination = self.output_dir / "meshes"
        backup = self.meshes_dir.parent / "previous-meshes"
        had_previous = destination.exists()
        moved_assets = False
        if had_previous:
            destination.rename(backup)
        try:
            self.meshes_dir.rename(destination)
            moved_assets = True
            pending_xml.replace(final_xml)
        except Exception:
            if moved_assets:
                destination.rename(self.meshes_dir)
            if had_previous:
                backup.rename(destination)
            raise

    def _run_pipeline(self, working_dir: Path) -> Path:
        """Keep intermediate XML separate from the last successful result."""
        self.logger.info("Starting full conversion pipeline...")

        self._set_stage("copy_assets")
        self._copy_tree(self.visual_dir, self.visual_output_dir)
        if self.export_collision:
            self._copy_tree(self.collision_dir, self.collision_output_dir)

        self._set_stage("mesh_conversion")
        self.logger.info("Converting DAE/GLB meshes to OBJ format...")
        self._convert_mesh_directory(self.visual_output_dir)
        if self.export_collision:
            self._convert_mesh_directory(self.collision_output_dir)
        self._set_stage("referenced_meshes")
        preprocess_referenced_meshes(
            self.urdf_path,
            self.meshes_dir,
            self.output_dir,
            self._staged_asset_paths,
            self.last_report["mesh_preprocessing"],
        )
        self.logger.info("Mesh format conversion complete.")

        backends = (
            ("urdf2mjcf", "mujoco") if self.backend == "auto" else (self.backend,)
        )
        failures = []
        for backend in backends:
            started = time.perf_counter()
            attempt = {"backend": backend, "status": "running"}
            if self.last_report is not None:
                self.last_report["attempts"].append(attempt)
            self._set_stage("urdf_conversion")
            try:
                result = self._run_backend(working_dir / backend, backend)
            except Exception as exc:
                attempt.update(status="failed", error=str(exc))
                if self.last_report is not None:
                    attempt["stage"] = self.last_report["stage"]
                if self.backend != "auto":
                    raise
                failures.append(f"{backend}: {exc}")
                self.logger.warning("Backend '%s' failed: %s", backend, exc)
            else:
                attempt.update(status="success", stage="completed")
                self.last_backend = backend
                return result
            finally:
                attempt["duration_seconds"] = round(time.perf_counter() - started, 6)
        raise RuntimeError("All conversion backends failed: " + "; ".join(failures))

    def _run_backend(self, working_dir: Path, backend: str) -> Path:
        working_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Using conversion backend: %s", backend)
        converter_type = (
            NativeMujocoConverter if backend == "mujoco" else URDF2MJCFConverter
        )
        urdf_converter = converter_type(self.urdf_path, working_dir, self.meshes_dir)
        urdf_converter.mesh_source_map = self._staged_asset_paths
        if not self._convert_urdf_with_recovery(urdf_converter):
            detail = urdf_converter.last_error or "No output XML was generated"
            raise RuntimeError(f"URDF to MJCF conversion failed: {detail}")

        obj_variant_map = {}
        if backend == "urdf2mjcf":
            self._set_stage("obj_assets")
            self.logger.info("Generating MJCF assets from OBJ files...")
            obj_converter_visual = OBJ2MJCFImporter(str(self.visual_output_dir))
            obj_converter_visual.run(recursive=True, strict=True)
            obj_variant_map = obj_converter_visual.get_obj_variant_map()
            self.logger.info("MJCF asset generation complete.")

        self._set_stage("postprocessing")
        self.logger.info("Post-processing the main MJCF file...")
        output_xml = urdf_converter.get_output_xml()
        if not output_xml or not Path(output_xml).exists():
            raise FileNotFoundError(f"Main MJCF file not found: {output_xml}")

        editor = MJCFEditor(output_xml)
        resolve_mjcf_asset_paths(editor.root, Path(output_xml))
        editor.normalize_mesh_names()

        def relative_mesh_path(filename: str) -> str:
            path = Path(filename)
            if path.is_absolute() and path.is_relative_to(self.meshes_dir):
                return path.relative_to(self.meshes_dir).as_posix()
            return filename

        editor.edit_mesh_file_paths(relative_mesh_path)
        editor.ensure_compiler_meshdir("meshes")
        editor.ensure_compiler_texturedir("meshes")
        editor.ensure_visual_collision_default_classes()
        editor.remove_duplicate_visual_collision_defaults()
        editor.add_default_ground_plane()
        if backend == "mujoco":
            for geom in editor.root.findall(".//body/geom"):
                is_visual = (
                    geom.get("contype") == "0" and geom.get("conaffinity") == "0"
                )
                geom.set("class", "visual" if is_visual else "collision")
        else:
            editor.flatten_visual_collision(self.visual_output_dir)
            editor.update_mesh_and_texture_paths(
                self.visual_output_dir, variant_map=obj_variant_map
            )
            editor.auto_add_mesh_variants(obj_variant_map, prefix="visual/")
            editor.remove_meshes_prefix()
            editor.remap_materials_and_textures_for_mesh_variants_with_xml(
                obj_variant_map, self.visual_output_dir
            )

            try:
                added_textures = (
                    editor.add_textures_and_link_materials_for_visual_meshes(
                        self.meshes_dir
                    )
                )
                self.logger.info(
                    f"Added {added_textures} new textures from visual meshes."
                )
            except Exception as exc:
                self.logger.warning(
                    f"Failed to add textures from visual meshes: {exc}", exc_info=True
                )

            editor.ensure_visual_prefix_for_all_textures()
            editor.deduplicate_mesh_names()
            editor.auto_fix_texture_paths(search_dirs=[self.meshes_dir])
            editor.remove_missing_textures(search_dirs=[self.meshes_dir])
            editor.remove_invalid_material_textures()
            editor.remove_invalid_mesh_geoms()
            editor.add_variant_geoms_and_materials(obj_variant_map)
            editor.remove_invalid_mesh_geoms()

            if self.export_collision:
                self.logger.info("Adding collision geometries...")
                try:
                    added_count = editor.add_collision_meshes_and_geoms(
                        self.collision_output_dir
                    )
                    self.logger.info(f"Added {added_count} collision geoms.")
                except Exception as exc:
                    self.logger.warning(
                        f"Failed to add collision geoms: {exc}", exc_info=True
                    )

        editor.move_geom_scales_to_meshes()

        if self.fixed_base:
            if editor.fix_base(root_body_name="root"):
                self.logger.info("Fixed-base enabled: removed <freejoint> from root.")
            else:
                self.logger.info("Root body is already fixed.")
        else:
            if editor.ensure_freejoint(root_body_name="root"):
                self.logger.info(
                    "Floating-base enabled: inserted <freejoint> under root body."
                )
            else:
                self.logger.info(
                    "Floating-base enabled: existing <freejoint> was kept."
                )

        final_path = self.output_dir / f"{self.urdf_path.stem}.xml"
        import mujoco

        self._set_stage("bundle_assets")
        assets = bundle_mjcf_assets(editor.root, self.meshes_dir)
        editor.root.find("compiler").attrib.update(
            meshdir="meshes", texturedir="meshes", strippath="false", usethread="false"
        )
        # Compile the exact published XML using its final relative directories.
        # The staging root mirrors the final XML + meshes/ layout.
        with tempfile.NamedTemporaryFile(
            dir=self.meshes_dir.parent, suffix=".xml", delete=False
        ) as pending:
            pending_path = Path(pending.name)
        try:
            self._set_stage("validation")
            editor.save(pending_path)
            model = mujoco.MjModel.from_xml_path(str(pending_path))
            validate_compiled_physics(model)
            self._set_stage("save")
            self._publish(pending_path, final_path)
            if self.last_report is not None:
                self.last_report["assets"] = assets
                self.last_report["statistics"] = {
                    "bodies": model.nbody - 1,
                    "joints": model.njnt,
                    "position_coordinates": model.nq,
                    "degrees_of_freedom": model.nv,
                    "actuators": model.nu,
                    "geometries": model.ngeom,
                    "meshes": model.nmesh,
                    "textures": model.ntex,
                    "sensors": model.nsensor,
                    "total_mass": float(model.body_mass.sum()),
                }
        finally:
            pending_path.unlink(missing_ok=True)
        self.logger.info(f"Successfully saved final MJCF file to '{final_path}'")

        self.logger.info("Conversion pipeline completed successfully.")
        return final_path


def add_common_arguments(parser) -> None:
    """Add options shared by single-model and batch commands."""
    parser.add_argument(
        "--backend",
        choices=("urdf2mjcf", "mujoco", "auto"),
        default="urdf2mjcf",
        help="Conversion backend. 'auto' falls back from urdf2mjcf to native MuJoCo.",
    )
    parser.add_argument(
        "--floating-base",
        action="store_true",
        help="Use a floating base by adding a <freejoint> to the root body.\nDefault is fixed base.",
    )
    parser.add_argument(
        "--export-collision",
        action="store_true",
        help="Export and process collision meshes.",
    )
    parser.add_argument(
        "--no-inertia-recalc",
        action="store_true",
        help="Disable automatic inertia recalculation on conversion failure.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check URDF topology, physical parameters, and mesh references without conversion.",
    )
    parser.add_argument(
        "--report", metavar="PATH.json", help="Write a JSON check/conversion report."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )


def main() -> int:
    """Run the conversion pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert URDF to MuJoCo MJCF with mesh processing (DAE/GLB->OBJ, OBJ->MJCF assets).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("urdf_path", help="Path to the input URDF file.")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory.")
    add_common_arguments(parser)
    args = parser.parse_args()

    if args.verbose:
        URDF2MJCFLogger.set_level("DEBUG")

    logger = URDF2MJCFLogger.get_logger()
    try:
        report_path = (
            validate_report_path(args.report, args.urdf_path) if args.report else None
        )
    except Exception as exc:
        logger.error("Invalid report path: %s", exc)
        return 1

    started = time.perf_counter()
    report = new_report(
        "check" if args.check else "conversion", args.urdf_path, args.backend
    )
    manager = None
    try:
        if args.check:
            report = inspect_urdf(args.urdf_path, backend=args.backend)
            for issue in report["issues"]:
                log = logger.warning if issue["severity"] == "warning" else logger.error
                log("[%s] %s", issue["code"], issue["message"])
            logger.info(
                "URDF check %s: %s links, %s joints, %s mesh references. "
                "This checks structure, physical parameters, and file references; it does not compile the model.",
                report["status"],
                report["statistics"].get("links", 0),
                report["statistics"].get("joints", 0),
                report["statistics"].get("mesh_references", 0),
            )
        else:
            manager = MujocoConversionManager(
                urdf_path=args.urdf_path,
                output_dir=args.output_dir,
                fixed_base=not args.floating_base,
                export_collision=args.export_collision,
                auto_recalculate_inertia=not args.no_inertia_recalc,
                backend=args.backend,
            )
            manager.run()
            report = manager.last_report
    except Exception as exc:
        if manager is not None and manager.last_report is not None:
            report = manager.last_report
        report.update(status="failed", error=str(exc))
        report.setdefault("duration_seconds", round(time.perf_counter() - started, 6))
        logger.error(
            "%s failed: %s",
            "Check" if args.check else "Conversion",
            exc,
            exc_info=args.verbose,
        )
    if report_path is not None:
        try:
            write_report(report, report_path)
            logger.info("JSON report saved to '%s'", report_path)
        except Exception as exc:
            logger.error("Cannot write JSON report: %s", exc, exc_info=args.verbose)
            return 1
    return 0 if report["status"] in {"success", "passed"} else 1


if __name__ == "__main__":
    sys.exit(main())
