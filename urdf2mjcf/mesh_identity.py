"""Keep source mesh files distinct across upstream basename-based importers."""

import copy
import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from urdf2mjcf.physics import numbers, vector


def _mesh_name(filename: Path, assets_dir: Path, scale, occupied: set[str]) -> str:
    path = (
        filename.relative_to(assets_dir)
        if filename.is_relative_to(assets_dir)
        else filename
    )
    identity = path.as_posix() + ":" + numbers(scale)
    base = "mesh_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    name, index = base, 0
    while name in occupied:
        index += 1
        name = f"{base}_{index}"
    occupied.add(name)
    return name


def restore_external_mesh_references(
    mjcf: ET.Element, urdf: ET.Element, assets_dir: Path
) -> None:
    """Bind each external-backend geom to its exact source file.

    The external importer keys assets by filename, and the material editor keys
    processed visuals by stem. Only staged visual instances may use those
    material-part names. Raw external meshes and complete collision meshes get
    independent names so an unrelated visual cannot replace their geometry.
    """
    if not urdf.findall("link/visual/geometry/mesh") and not urdf.findall(
        "link/collision/geometry/mesh"
    ):
        return
    assets_dir = Path(assets_dir).resolve()
    visual_dir = assets_dir / "visual"
    occupied = set()
    for mesh in urdf.findall("link/visual/geometry/mesh"):
        filename = Path(mesh.get("filename"))
        if filename.is_relative_to(visual_dir):
            occupied.update((filename.name, filename.stem))
    bodies = {body.get("name"): body for body in mjcf.findall(".//body")}
    registry = {}
    owners = {}
    for link in urdf.findall("link"):
        for kind in ("visual", "collision"):
            sources = link.findall(kind)
            if not any(source.find("geometry/mesh") is not None for source in sources):
                continue
            body = bodies.get(link.get("name"))
            if body is None:
                raise ValueError(
                    f"Converted model is missing mesh link: {link.get('name')}"
                )
            targets = [
                geom for geom in body.findall("geom") if geom.get("class") == kind
            ]
            if len(sources) != len(targets):
                raise ValueError(
                    f"Converted {kind} count differs for link {link.get('name')!r}: "
                    f"expected {len(sources)}, found {len(targets)}"
                )
            for source, target in zip(sources, targets, strict=True):
                mesh = source.find("geometry/mesh")
                if mesh is None:
                    continue
                filename = Path(mesh.get("filename"))
                processed = kind == "visual" and filename.is_relative_to(visual_dir)
                key = (filename, processed)
                if key not in registry:
                    name = (
                        filename.name
                        if processed
                        else _mesh_name(filename, assets_dir, [1, 1, 1], occupied)
                    )
                    if name in owners and owners[name] != key:
                        raise ValueError(f"Ambiguous staged visual mesh name: {name}")
                    owners[name] = key
                    registry[key] = ET.Element(
                        "mesh", name=name, file=filename.as_posix()
                    )
                target.set("type", "mesh")
                target.set("mesh", registry[key].get("name"))
                scale = vector(mesh.get("scale", "1 1 1"))
                if np.any(scale != 1):
                    target.set("scale", numbers(scale))
                else:
                    target.attrib.pop("scale", None)
    asset = mjcf.find("asset")
    if asset is None:
        asset = ET.SubElement(mjcf, "asset")
    for mesh in asset.findall("mesh"):
        asset.remove(mesh)
    asset.extend(registry.values())


def prepare_native_mesh_aliases(
    tree: ET.ElementTree, alias_dir: Path, assets_dir: Path
) -> tuple[ET.ElementTree, dict[str, str]]:
    """Give every (source file, scale) pair a unique native importer basename.

    The importer can reject scale-generated names during parsing, before a spec
    is available for repair. Scoped aliases prevent both those collisions and
    basename-based reuse. Hardlinks avoid copying large meshes when possible;
    the fallback copies bytes without loading the whole file into memory.
    The spec's filenames are restored to the original files before serialization.
    """
    aliased = ET.ElementTree(copy.deepcopy(tree.getroot()))
    assets_dir = Path(assets_dir).resolve()
    registry = {}
    filenames = {}
    occupied = set()
    for mesh in aliased.findall("link/visual/geometry/mesh") + aliased.findall(
        "link/collision/geometry/mesh"
    ):
        filename = Path(mesh.get("filename"))
        scale = vector(mesh.get("scale", "1 1 1"))
        key = (filename, tuple(scale))
        if key not in registry:
            name = _mesh_name(filename, assets_dir, scale, occupied)
            alias = alias_dir / f"{name}{filename.suffix.lower()}"
            alias_dir.mkdir(parents=True, exist_ok=True)
            try:
                alias.hardlink_to(filename)
            except OSError:
                shutil.copy2(filename, alias)
            registry[key] = alias
            filenames[name] = filename.as_posix()
        mesh.set("filename", registry[key].as_posix())
    return aliased, filenames


def remove_unused_mesh_assets(mjcf: ET.Element) -> None:
    """Remove importer leftovers after all geom references have been repaired."""
    used = {geom.get("mesh") for geom in mjcf.iter("geom")}
    for asset in mjcf.findall("asset"):
        for mesh in asset.findall("mesh"):
            if mesh.get("name") not in used:
                asset.remove(mesh)
