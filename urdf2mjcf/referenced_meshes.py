"""Preprocess referenced meshes that fall outside the conventional asset trees."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.asset_paths import MeshPathResolver
from urdf2mjcf.dae_to_obj_converter import DAE2OBJConverter
from urdf2mjcf.glb_to_obj_converter import GLB2OBJConverter
from urdf2mjcf.mesh_to_obj_converter import MeshToOBJConverter

PREPROCESS_FORMATS = frozenset({".dae", ".glb", ".ply", ".off", ".3mf"})


def preprocess_referenced_meshes(
    urdf_path: Path,
    assets_dir: Path,
    output_dir: Path,
    source_map: dict[Path, Path],
    report: list[dict],
) -> None:
    """Convert each previously unstaged source once, preserving reference identity.

    Read from the original location so relative material/texture dependencies
    remain resolvable. Write conversions only to the per-run staging tree. OBJ
    and STL can be imported directly and are collected by final asset bundling.
    """
    try:
        robot = ET.parse(urdf_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise RuntimeError(f"Cannot inspect URDF mesh references: {exc}") from exc
    resolver = MeshPathResolver(urdf_path.parent, assets_dir, prefer_converted=False)
    original_sources = {
        staged.resolve(): source for source, staged in source_map.items()
    }
    jobs = {}
    for link in robot.findall("link"):
        for kind in ("visual", "collision"):
            for mesh in link.findall(f"{kind}/geometry/mesh"):
                reference = mesh.get("filename", mesh.get("file", ""))
                try:
                    resolved = resolver.resolve(reference)
                    source = original_sources.get(resolved, resolved)
                except (OSError, ValueError) as exc:
                    raise RuntimeError(
                        f"Link {link.get('name')!r}, {kind} mesh {reference!r}: {exc}"
                    ) from exc
                if source.is_relative_to((output_dir / "meshes").resolve()):
                    raise ValueError(
                        f"Output asset directory contains a referenced source mesh: {source}"
                    )
                if (
                    source in source_map
                    or source.suffix.lower() not in PREPROCESS_FORMATS
                ):
                    continue
                jobs.setdefault(source, []).append(
                    {"link": link.get("name"), "kind": kind, "reference": reference}
                )

    # Generated object names must also avoid the names in the existing trees.
    occupied = {path.stem for path in assets_dir.rglob("*")}
    for source, references in sorted(jobs.items()):
        base = "ref_" + hashlib.sha256(source.as_posix().encode()).hexdigest()[:16]
        name = base
        index = 0
        while name in occupied:
            index += 1
            name = f"{base}_{index}"
        occupied.add(name)
        kind = (
            "visual"
            if any(item["kind"] == "visual" for item in references)
            else "collision"
        )
        directory = assets_dir / kind / "referenced" / name
        output = directory / f"{name}.obj"
        entry = {
            "source": str(source),
            "format": source.suffix.lower(),
            "references": references,
            "status": "running",
            "output": None,
        }
        report.append(entry)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() == ".dae":
                success, outputs = DAE2OBJConverter().convert_single_dae(
                    source, directory
                )
                if not success or len(outputs) != 1:
                    raise RuntimeError(
                        "DAE conversion must produce exactly one complete OBJ"
                    )
                converted = Path(outputs[0])
                if not converted.is_file() or not converted.resolve().is_relative_to(
                    directory
                ):
                    raise RuntimeError("DAE conversion returned an invalid output path")
                if converted != output:
                    converted.replace(output)
            elif source.suffix.lower() == ".glb":
                if not GLB2OBJConverter(str(directory)).convert_glb_to_obj(
                    source, directory
                ):
                    raise RuntimeError("GLB conversion failed")
            else:
                MeshToOBJConverter(overwrite=True).convert_file(source, output)
            if not output.is_file():
                raise RuntimeError("Mesh conversion produced no OBJ file")
            source_map[source] = output.resolve()
            entry.update(
                status="converted",
                output="meshes/" + output.relative_to(assets_dir).as_posix(),
            )
        except Exception as exc:
            entry.update(status="failed", error=str(exc))
            raise RuntimeError(
                f"Referenced mesh conversion failed for {source}: {exc}"
            ) from exc
