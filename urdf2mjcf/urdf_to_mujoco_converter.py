#!/usr/bin/env python3
"""Main entry point for the URDF/mesh asset to MuJoCo MJCF conversion pipeline."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

from urdf2mjcf.dae_to_obj_converter import DAE2OBJConverter
from urdf2mjcf.glb_to_obj_converter import GLB2OBJConverter
from urdf2mjcf.logging_utils import URDF2MJCFLogger
from urdf2mjcf.mjcf_editor import MJCFEditor
from urdf2mjcf.obj_to_mjcf_converter import OBJ2MJCFImporter
from urdf2mjcf.urdf_inertia_calculator import URDFInertiaCalculator
from urdf2mjcf.urdf_to_mjcf_converter import URDF2MJCFConverter


class MujocoConversionManager:
    """Manage end-to-end URDF and mesh asset conversion to MuJoCo MJCF."""

    def __init__(
        self,
        urdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        fixed_base: bool = True,
        export_collision: bool = False,
        auto_recalculate_inertia: bool = True,
    ):
        self.urdf_path = Path(urdf_path)
        self.logger = URDF2MJCFLogger.get_logger("MujocoConversionManager")
        self.fixed_base = fixed_base
        self.export_collision = export_collision
        self.auto_recalculate_inertia = auto_recalculate_inertia

        if output_dir is None:
            parent = self.urdf_path.parent
            self.output_dir = parent.parent / f"{parent.name}_mjcf"
        else:
            self.output_dir = Path(output_dir)

        self.visual_dir = self._resolve_asset_dir("visual")
        self.collision_dir = self._resolve_asset_dir("collision")

        self.meshes_dir = self.output_dir / "meshes"
        self.visual_output_dir = self.meshes_dir / "visual"
        self.collision_output_dir = self.meshes_dir / "collision"

        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.meshes_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Initialized conversion for '{self.urdf_path.name}'")
        self.logger.info(f"Output will be saved to '{self.output_dir}'")

    def _resolve_asset_dir(self, name: str) -> Path:
        """Resolve an asset directory path while tolerating common capitalization variants."""
        candidates = [
            self.urdf_path.parent / name,
            self.urdf_path.parent / name.capitalize(),
            self.urdf_path.parent / name.upper(),
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

        with tqdm(total=len(file_list), desc=f"Copying {src.name}") as progress:
            for src_path in file_list:
                dst_path = dst / src_path.relative_to(src)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                progress.update(1)

    def _convert_urdf_with_recovery(self, urdf_converter: URDF2MJCFConverter) -> bool:
        """Convert URDF to MJCF, retrying once after inertia recalculation when enabled."""
        self.logger.info("Starting URDF to MJCF conversion...")
        if urdf_converter.convert():
            output_xml = urdf_converter.get_output_xml()
            if output_xml and Path(output_xml).exists():
                self.logger.info("URDF to MJCF conversion successful.")
                return True

        if not self.auto_recalculate_inertia:
            self.logger.error(
                "URDF conversion failed and inertia recalculation is disabled."
            )
            return False

        self.logger.warning(
            "URDF conversion failed. Recalculating inertia and retrying."
        )
        inertia_calculator = URDFInertiaCalculator(self.urdf_path)
        if not inertia_calculator.update_inertia():
            self.logger.error("Inertia recalculation failed. Aborting.")
            return False

        self.logger.info(
            "Inertia recalculation complete. Retrying URDF to MJCF conversion."
        )
        if not urdf_converter.convert():
            self.logger.error("URDF conversion failed again after retry.")
            return False

        output_xml = urdf_converter.get_output_xml()
        if not (output_xml and Path(output_xml).exists()):
            self.logger.error(
                "Conversion succeeded, but the output XML was not generated."
            )
            return False

        self.logger.info("URDF to MJCF conversion successful after retry.")
        return True

    def run(self) -> None:
        """Execute the full conversion pipeline."""
        self.logger.info("Starting full conversion pipeline...")

        self._copy_tree(self.visual_dir, self.visual_output_dir)
        if self.export_collision:
            self._copy_tree(self.collision_dir, self.collision_output_dir)

        self.logger.info("Converting DAE/GLB meshes to OBJ format...")
        dae_converter = DAE2OBJConverter()
        dae_converter.convert_directory(
            self.visual_output_dir,
            self.visual_output_dir,
            recursive=True,
            preserve_textures=True,
        )

        glb_converter = GLB2OBJConverter(str(self.visual_output_dir))
        glb_converter.convert_directory(
            self.visual_output_dir,
            self.visual_output_dir,
            recursive=True,
        )
        self.logger.info("Mesh format conversion complete.")

        urdf_converter = URDF2MJCFConverter(
            self.urdf_path, self.output_dir, self.meshes_dir
        )
        if not self._convert_urdf_with_recovery(urdf_converter):
            raise RuntimeError("URDF to MJCF conversion failed, aborting pipeline.")

        self.logger.info("Generating MJCF assets from OBJ files...")
        obj_converter_visual = OBJ2MJCFImporter(str(self.visual_output_dir))
        obj_converter_visual.run(recursive=True)
        obj_variant_map = obj_converter_visual.get_obj_variant_map()
        self.logger.info("MJCF asset generation complete.")

        self.logger.info("Post-processing the main MJCF file...")
        output_xml = urdf_converter.get_output_xml()
        if not output_xml or not Path(output_xml).exists():
            raise FileNotFoundError(f"Main MJCF file not found: {output_xml}")

        editor = MJCFEditor(output_xml)
        editor.ensure_compiler_meshdir("meshes")
        editor.ensure_compiler_texturedir("meshes")
        editor.ensure_visual_collision_default_classes()
        editor.remove_duplicate_visual_collision_defaults()
        editor.add_default_ground_plane()
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
            added_textures = editor.add_textures_and_link_materials_for_visual_meshes(
                self.meshes_dir
            )
            self.logger.info(f"Added {added_textures} new textures from visual meshes.")
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

        if self.fixed_base:
            if editor.fix_base(root_body_name="root"):
                self.logger.info("Fixed-base enabled: removed <freejoint> from root.")
            else:
                self.logger.warning(
                    "Fixed-base enabled, but no <freejoint> was found under the root body."
                )
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
        editor.save(final_path)
        self.logger.info(f"Successfully saved final MJCF file to '{final_path}'")

        if Path(output_xml) != final_path and Path(output_xml).exists():
            Path(output_xml).unlink()
            self.logger.debug(f"Removed intermediate file: {output_xml}")

        self.logger.info("Conversion pipeline completed successfully.")


def main() -> int:
    """Run the conversion pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert URDF to MuJoCo MJCF with mesh processing (DAE/GLB->OBJ, OBJ->MJCF assets).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("urdf_path", help="Path to the input URDF file.")
    parser.add_argument("output_dir", nargs="?", default=None, help="Output directory.")
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
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    if args.verbose:
        URDF2MJCFLogger.set_level("DEBUG")

    try:
        manager = MujocoConversionManager(
            urdf_path=args.urdf_path,
            output_dir=args.output_dir,
            fixed_base=not args.floating_base,
            export_collision=args.export_collision,
            auto_recalculate_inertia=not args.no_inertia_recalc,
        )
        manager.run()
        return 0
    except Exception as exc:
        URDF2MJCFLogger.get_logger().error(
            f"A critical error occurred: {exc}", exc_info=True
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
