"""Protect OBJ source dependencies and preserve outputs while converting."""

import json
import shlex
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

BACKUP_PREFIX = ".obj2mjcf-backup-"


def _reference_paths(base: Path, text: str) -> set[Path]:
    # Keep the literal spelling as well as quoted/multiple MTL filenames.
    # The pinned converter also accepts an entire map_Kd value as one path.
    candidates = {text.strip()}
    try:
        candidates.update(shlex.split(text, comments=True))
    except ValueError:
        candidates.update(text.split())
    paths = set()
    for name in candidates - {""}:
        path = base / name
        try:
            regular = stat.S_ISREG(path.stat().st_mode)
        except (FileNotFoundError, NotADirectoryError):
            continue
        if regular:
            paths.add(path.resolve())
    return paths


def protect_obj_inputs(sources: list[Path], outputs: list[Path]) -> None:
    """Reject OBJ/MTL/diffuse-texture inputs inside any replaced output tree."""
    roots = {output.resolve() for output in outputs}

    def protect(path: Path):
        if path in roots or any(parent in roots for parent in path.parents):
            raise ValueError(f"OBJ output would replace an input asset: {path}")

    inspected = set()
    for source in sources:
        protect(source.resolve())
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                words = line.split(maxsplit=1)
                if len(words) != 2 or words[0] != "mtllib":
                    continue
                for material in _reference_paths(source.parent, words[1]):
                    protect(material)
                    key = (material, source.parent)
                    if key in inspected:
                        continue
                    inspected.add(key)
                    with material.open(encoding="utf-8") as material_stream:
                        for definition in material_stream:
                            fields = definition.split(maxsplit=1)
                            if len(fields) != 2 or fields[0] != "map_Kd":
                                continue
                            # Protect both standard MTL-relative paths and the
                            # OBJ-relative interpretation used by obj2mjcf.
                            for base in {material.parent, source.parent}:
                                for texture in _reference_paths(base, fields[1]):
                                    protect(texture)


def generated_obj_paths(subxml: Path, directory: Path) -> list[Path]:
    """Validate a generated fragment and return its ordered visual OBJ assets."""
    try:
        model = ET.parse(subxml).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid generated MJCF: {subxml}: {exc}") from exc
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
            raise ValueError(f"Generated mesh escapes its asset directory: {path}")
        if path.suffix.lower() != ".obj" or not path.is_file():
            raise ValueError(f"Generated OBJ mesh is missing or invalid: {path}")
        if path not in parts:
            parts.append(path)
    if not parts:
        raise ValueError(f"Generated MJCF contains no visual OBJ assets: {subxml}")
    return parts


def _remove_output(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


@contextmanager
def preserve_obj_outputs(outputs: list[Path], logger):
    """Restore previous model directories when conversion or validation fails.

    Backups stay on the same filesystem. If rollback itself fails, retain the
    remaining backups and their manifest rather than discard recoverable data.
    """
    for output in outputs:
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise ValueError(f"OBJ output must be a regular directory: {output}")
    backup = Path(tempfile.mkdtemp(prefix=BACKUP_PREFIX, dir=outputs[0].parent))
    moved = {}
    prepared = False
    try:
        plan = [
            {"output": str(output), "backup": str(index)}
            for index, output in enumerate(outputs)
        ]
        (backup / "manifest.json").write_text(json.dumps(plan, indent=2))
        for index, output in enumerate(outputs):
            if output.exists():
                previous = backup / str(index)
                output.rename(previous)
                moved[output] = previous
        prepared = True
        yield
    except BaseException as original:
        failures = []
        for output in reversed(outputs):
            if not prepared and output not in moved:
                continue
            try:
                if prepared:
                    _remove_output(output)
                if output in moved:
                    moved[output].rename(output)
            except OSError as exc:
                failures.append(f"{output}: {exc}")
        if failures:
            raise RuntimeError(
                f"OBJ rollback failed; backups retained in {backup}: "
                + "; ".join(failures)
            ) from original
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            logger.warning("Cannot remove restored OBJ backup %s: %s", backup, exc)
        raise
    else:
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            logger.warning("Cannot remove old OBJ backup %s: %s", backup, exc)
