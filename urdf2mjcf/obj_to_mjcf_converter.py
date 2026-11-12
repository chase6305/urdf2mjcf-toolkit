import os
import shutil
import stat
import subprocess
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

from urdf2mjcf.logging_utils import URDF2MJCFLogger


def _output_signature(path: Path) -> tuple[int, int, int, int] | None:
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _expected_models(directory: Path) -> set[Path]:
    parent = directory.resolve()
    return {
        parent / path.stem / f"{path.stem}.xml"
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".obj"
    }


@contextmanager
def _obj_input_aliases(directory: Path):
    """Expose mixed-case extensions to obj2mjcf's lowercase-only glob.

    Keep aliases beside their source so relative MTL and texture references
    retain their meaning. Never overwrite an existing file or keep an alias
    after the subprocess finishes, including on failure.
    """
    sources = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".obj"
    )
    owners = {}
    for source in sources:
        if source.stem in owners:
            raise ValueError(
                f"Ambiguous OBJ inputs share output stem {source.stem!r}: "
                f"{owners[source.stem]} and {source}"
            )
        owners[source.stem] = source
    aliases = []
    try:
        for source in sources:
            if source.suffix == ".obj":
                continue
            alias = source.with_suffix(".obj")
            if alias.exists() and alias.samefile(source):
                continue  # The filesystem already resolves extensions without case.
            try:
                os.link(source, alias)
            except OSError:
                # Some filesystems do not support hard links. Exclusive creation
                # still protects files created after the input scan.
                with alias.open("xb") as target:
                    aliases.append(alias)
                    with source.open("rb") as original:
                        shutil.copyfileobj(original, target)
            else:
                aliases.append(alias)
        yield
    finally:
        for alias in aliases:
            alias.unlink(missing_ok=True)


class OBJ2MJCFImporter:
    """
    Scans the root directory for subdirectories containing .obj files, and runs obj2mjcf in each such subdirectory:
      obj2mjcf --obj-dir . --overwrite --save-mjcf

    Usage: After DAE→OBJ conversion, generate MJCF fragments for each asset subdirectory for later reference or validation.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.logger = URDF2MJCFLogger.get_logger("OBJ2MJCFImporter")
        self.failed_directories: list[Path] = []
        self._current_models: set[Path] | None = None
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    def get_obj_variant_map(self) -> dict[str, list[str]]:
        """Use generated MJCF asset declarations to identify material parts.

        Numeric suffixes in input filenames are ordinary names. Generated XML
        determines part order and excludes stale files left by earlier runs.
        After run(), only that run's successful input/output pairs are used.
        Before run(), pre-existing generated trees can be inspected directly.
        Ambiguous input basenames cannot be represented by this map and fail
        explicitly instead of silently merging unrelated meshes.
        """
        root = self.root_dir.resolve()
        mapping: dict[str, list[str]] = {}
        owners: dict[str, Path] = {}
        originals = []
        generated_sources = set()

        def register(name: str, paths: list[Path], owner: Path) -> None:
            if name in mapping:
                raise ValueError(
                    f"Ambiguous mesh name '{name}': {owners[name]} and {owner}"
                )
            mapping[name] = [path.relative_to(root).as_posix() for path in paths]
            owners[name] = owner

        for current, children, files in os.walk(root):
            children.sort()
            directory = Path(current)
            subxml = directory / f"{directory.name}.xml"
            current_output = (
                self._current_models is None or subxml in self._current_models
            )
            if current_output and subxml.is_file():
                children.clear()
                try:
                    model = ET.parse(subxml).getroot()
                except ET.ParseError as exc:
                    raise ValueError(
                        f"Invalid generated MJCF: {subxml}: {exc}"
                    ) from exc
                if model.tag != "mujoco":
                    raise ValueError(f"Expected <mujoco> in generated MJCF: {subxml}")
                visual_meshes = {
                    geom.get("mesh")
                    for geom in model.iter("geom")
                    if geom.get("class") not in {"collision", "collisiongeom"}
                }
                compiler = model.find("compiler")
                meshdir = compiler.get("meshdir", "") if compiler is not None else ""
                parts = []
                for mesh in model.findall("asset/mesh"):
                    filename = mesh.get("file", "")
                    if mesh.get("name", Path(filename).stem) not in visual_meshes:
                        continue
                    path = (directory / meshdir / filename).resolve()
                    if not path.is_relative_to(directory):
                        raise ValueError(
                            f"Generated mesh escapes its asset directory: {path}"
                        )
                    if path.suffix.lower() != ".obj" or not path.is_file():
                        raise ValueError(
                            f"Generated OBJ mesh is missing or invalid: {path}"
                        )
                    if path not in parts:
                        parts.append(path)
                if not parts:
                    raise ValueError(
                        f"Generated MJCF contains no visual OBJ assets: {subxml}"
                    )
                register(directory.name, parts, subxml)
                generated_sources.add((directory.parent, directory.name))
            else:
                originals.extend(
                    directory / name
                    for name in sorted(files)
                    if Path(name).suffix.lower() == ".obj"
                )
        for obj_file in originals:
            if (obj_file.parent, obj_file.stem) not in generated_sources:
                register(obj_file.stem, [obj_file], obj_file)
        return {name: mapping[name] for name in sorted(mapping)}

    def _has_obj_files(self, dir_path: Path) -> bool:
        """Check if the directory contains any .obj files."""
        try:
            return any(
                path.is_file() and path.suffix.lower() == ".obj"
                for path in dir_path.iterdir()
            )
        except OSError as exc:
            self.logger.warning(f"Error reading directory {dir_path}: {exc}")
            return False

    def handle_dir(self, d: Path, obj2mjcf_path: str = "obj2mjcf") -> bool:
        """
        Process a single directory. Returns True if successful, False otherwise.
        """
        if d.is_dir() and self._has_obj_files(d):
            self.logger.info(f"Processing obj2mjcf: {d}")
            try:
                expected = _expected_models(d)
                previous = {path: _output_signature(path) for path in expected}
                with _obj_input_aliases(d):
                    result = subprocess.run(
                        [obj2mjcf_path, "--obj-dir", ".", "--overwrite", "--save-mjcf"],
                        cwd=str(d),
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                self.logger.debug(result.stdout)
                stale = [
                    path
                    for path in sorted(expected)
                    if (current := _output_signature(path)) is None
                    or current == previous[path]
                ]
                if stale:
                    self.logger.error(
                        "OBJ conversion did not update expected models: %s",
                        ", ".join(map(str, stale)),
                    )
                    return False
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
        only_subdir: str | None = None,
        recursive: bool = False,
        obj2mjcf_path: str = "obj2mjcf",
        strict: bool = False,
    ) -> list[Path]:
        """
        Run obj2mjcf in subdirectories containing .obj files. Returns a list of successfully processed directories.

        Args:
            only_subdir: If provided, only process this subtree (e.g. 'Visual'), avoiding others like 'Collision'.
            recursive: If True, recursively scan all subdirectories; otherwise, only scan direct subdirectories.
            obj2mjcf_path: Path to the obj2mjcf executable (default: 'obj2mjcf').
            strict: Raise if any OBJ directory fails to convert.
        """
        processed: list[Path] = []
        self.failed_directories = []
        self._current_models = set()
        scan_root = self.root_dir
        if only_subdir:
            candidate = self.root_dir / only_subdir
            if candidate.is_dir():
                scan_root = candidate
            else:
                self.logger.warning(
                    f"Specified subdirectory does not exist: {candidate}"
                )
                raise FileNotFoundError(
                    f"Specified subdirectory does not exist: {candidate}"
                )

        def process_dir(directory: Path) -> None:
            if not self._has_obj_files(directory):
                return
            expected = _expected_models(directory)
            if self.handle_dir(directory, obj2mjcf_path):
                processed.append(directory)
                self._current_models.update(expected)
            else:
                self.failed_directories.append(directory)

        directories = []
        for current, children, files in os.walk(scan_root):
            directory = Path(current)
            input_stems = {
                Path(name).stem for name in files if Path(name).suffix.lower() == ".obj"
            }
            # Generated subtrees are outputs, and must not be reconverted on reruns.
            children[:] = sorted(
                name
                for name in children
                if not (
                    name in input_stems and (directory / name / f"{name}.xml").is_file()
                )
            )
            directories.append(directory)
            if not recursive and directory != scan_root:
                children.clear()

        for directory in sorted(directories):
            process_dir(directory)
        self.logger.info(f"Total processed directories: {len(processed)}")
        if strict and self.failed_directories:
            raise RuntimeError(
                "OBJ to MJCF conversion failed in: "
                + ", ".join(str(path) for path in self.failed_directories)
            )
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

    try:
        importer = OBJ2MJCFImporter(args.root_dir)
        if args.verbose:
            importer.logger.setLevel("DEBUG")
        processed = importer.run(
            only_subdir=args.only_subdir,
            recursive=args.recursive,
            obj2mjcf_path=args.obj2mjcf_path,
            strict=True,
        )
        return 0 if processed else 1
    except (OSError, RuntimeError) as exc:
        URDF2MJCFLogger.get_logger("OBJ2MJCFImporter").error(
            "%s", exc, exc_info=args.verbose
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
