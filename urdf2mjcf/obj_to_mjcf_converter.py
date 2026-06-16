import os
import subprocess
from pathlib import Path
from typing import Optional, List

from urdf2mjcf.logging_utils import URDF2MJCFLogger


class OBJ2MJCFImporter:
    """
    Scans the root directory for subdirectories containing .obj files, and runs obj2mjcf in each such subdirectory:
      obj2mjcf --obj-dir . --overwrite --save-mjcf

    Usage: After DAE→OBJ conversion, generate MJCF fragments for each asset subdirectory for later reference or validation.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.logger = URDF2MJCFLogger.get_logger("OBJ2MJCFImporter")
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    def get_obj_variant_map(self) -> dict:
        """
        Returns {mesh name: [all obj paths]}, collecting all meshes, regardless of whether there are variants.
        For example: {'base_link': ['chassis/base_link/base_link.obj'], 'head2': ['head/head2/head2_0.obj', ...]}
        """
        from collections import defaultdict

        mapping = defaultdict(list)
        visual_dir = self.root_dir
        for obj_file in visual_dir.rglob("*.obj"):
            stem = obj_file.stem
            # Variant names like foo_0, foo_1, main name is foo
            if "_" in stem and stem.split("_")[-1].isdigit():
                mesh_name = "_".join(stem.split("_")[:-1])
            else:
                mesh_name = stem
            rel_path = obj_file.relative_to(visual_dir).as_posix()
            mapping[mesh_name].append(rel_path)
        return dict(mapping)

    def _has_obj_files(self, dir_path: Path) -> bool:
        """Check if the directory contains any .obj files."""
        try:
            for f in dir_path.iterdir():
                if f.is_file() and f.suffix.lower() == ".obj":
                    return True
        except Exception as e:
            self.logger.warning(f"Error reading directory {dir_path}: {e}")
            return False
        return False

    def handle_dir(self, d: Path, obj2mjcf_path: str = "obj2mjcf") -> bool:
        """
        Process a single directory. Returns True if successful, False otherwise.
        """
        if d.is_dir() and self._has_obj_files(d):
            self.logger.info(f"Processing obj2mjcf: {d}")
            try:
                result = subprocess.run(
                    [obj2mjcf_path, "--obj-dir", ".", "--overwrite", "--save-mjcf"],
                    cwd=str(d),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.logger.debug(result.stdout)
                return True
            except FileNotFoundError:
                self.logger.error(
                    f"obj2mjcf executable not found: {obj2mjcf_path}. Skipping {d}."
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"obj2mjcf failed in {d}: {e.output}")
            except Exception as e:
                self.logger.error(f"Unknown error in {d}: {e}")
        return False

    def run(
        self,
        only_subdir: Optional[str] = None,
        recursive: bool = False,
        obj2mjcf_path: str = "obj2mjcf",
    ) -> List[Path]:
        """
        Run obj2mjcf in subdirectories containing .obj files. Returns a list of successfully processed directories.

        Args:
            only_subdir: If provided, only process this subtree (e.g. 'Visual'), avoiding others like 'Collision'.
            recursive: If True, recursively scan all subdirectories; otherwise, only scan direct subdirectories.
            obj2mjcf_path: Path to the obj2mjcf executable (default: 'obj2mjcf').
        """
        processed: List[Path] = []
        scan_root = self.root_dir
        if only_subdir:
            candidate = self.root_dir / only_subdir
            if candidate.exists():
                scan_root = candidate
            else:
                self.logger.warning(
                    f"Specified subdirectory does not exist: {candidate}"
                )
                return processed

        def process_dir(d: Path):
            if self.handle_dir(d, obj2mjcf_path):
                processed.append(d)

        if recursive:
            for root, dirs, files in os.walk(scan_root):
                dpath = Path(root)
                if any(f.lower().endswith(".obj") for f in files):
                    try:
                        process_dir(dpath)
                    except Exception as e:
                        self.logger.error(f"Error processing {dpath}: {e}")
        else:
            for name in os.listdir(scan_root):
                subdir = scan_root / name
                try:
                    process_dir(subdir)
                except Exception as e:
                    self.logger.error(f"Error processing {subdir}: {e}")
        self.logger.info(f"Total processed directories: {len(processed)}")
        return processed


def main() -> int:
    """Run obj2mjcf over OBJ asset directories from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate MJCF snippets for directories that contain OBJ files."
    )
    parser.add_argument("root_dir", help="Root directory to scan for OBJ files.")
    parser.add_argument(
        "--only-subdir",
        default=None,
        help="Optional subtree under root_dir to process, such as 'visual'.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan all subdirectories. By default only direct children are scanned.",
    )
    parser.add_argument(
        "--obj2mjcf-path",
        default="obj2mjcf",
        help="Path to the obj2mjcf executable. Defaults to 'obj2mjcf'.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    importer = OBJ2MJCFImporter(args.root_dir)
    if args.verbose:
        importer.logger.setLevel("DEBUG")
    processed = importer.run(
        only_subdir=args.only_subdir,
        recursive=args.recursive,
        obj2mjcf_path=args.obj2mjcf_path,
    )
    return 0 if processed else 1


if __name__ == "__main__":
    raise SystemExit(main())
