"""Collect final MJCF file assets into a relocatable output tree."""

import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

TEXTURE_FILES = (
    "file",
    "fileright",
    "fileleft",
    "fileup",
    "filedown",
    "filefront",
    "fileback",
)


def resolve_mjcf_asset_paths(root: ET.Element, xml_path: Path) -> None:
    """Resolve existing upstream asset paths before replacing compiler directories.

    Upstream serializers may use their own meshdir/texturedir/assetdir. Preserve
    those exact files before the editor applies the output-directory convention.
    Missing paths are left for the pipeline's diagnostics and final validation.
    """
    compiler = root.find("compiler")
    settings = compiler.attrib if compiler is not None else {}
    xml_dir = Path(xml_path).resolve().parent
    for asset in root.findall("asset/*"):
        if asset.tag not in {"mesh", "hfield", "texture"}:
            continue
        directory = "texturedir" if asset.tag == "texture" else "meshdir"
        base = xml_dir / settings.get(directory, settings.get("assetdir", ""))
        attributes = TEXTURE_FILES if asset.tag == "texture" else ("file",)
        for attribute in attributes:
            filename = asset.get(attribute)
            if not filename:
                continue
            path = Path(filename.replace("\\", "/"))
            if settings.get("strippath") == "true":
                path = Path(path.name)
            resolved = (base / path).resolve()
            if resolved.is_file():
                asset.set(attribute, resolved.as_posix())


def _digest(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest(), path.stat().st_size


def _copy_external(source: Path, assets_dir: Path) -> tuple[Path, str, int]:
    """Hash the actual copied bytes, using bounded memory for large meshes."""
    destination_dir = assets_dir / "bundled"
    destination_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination_dir, delete=False) as pending:
        pending_path = Path(pending.name)
        digest = hashlib.sha256()
        size = 0
        try:
            with source.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    pending.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            pending.close()
            checksum = digest.hexdigest()
            destination = destination_dir / f"{checksum}{source.suffix.lower()}"
            if destination.exists():
                if _digest(destination) != (checksum, size):
                    raise ValueError(
                        f"Bundled asset filename conflicts with existing data: {destination}"
                    )
            else:
                pending_path.replace(destination)
            return destination, checksum, size
        finally:
            pending_path.unlink(missing_ok=True)


def bundle_mjcf_assets(root: ET.Element, assets_dir: Path) -> list[dict]:
    """Localize mesh, heightfield, and texture references and return a manifest.

    Relative input references are resolved against assets_dir, matching the
    pipeline's common meshdir/texturedir. Call only in the per-run staging tree;
    ordinary failure cleanup and publication remain the manager's responsibility.
    The manifest covers runtime MJCF dependencies, not unused source sidecars.
    """
    assets_dir = Path(assets_dir).resolve()
    cached = {}
    manifest = {}
    for asset in root.findall("asset/*"):
        attributes = TEXTURE_FILES if asset.tag == "texture" else ("file",)
        if not any(attribute in asset.attrib for attribute in attributes):
            continue
        if asset.tag not in {"mesh", "hfield", "texture"}:
            raise ValueError(
                f"Cannot bundle file dependencies of asset type: {asset.tag}"
            )
        # A file-derived implicit name must survive the filename rewrite.
        if "name" not in asset.attrib and asset.get("file"):
            asset.set("name", Path(asset.get("file").replace("\\", "/")).stem)
        for attribute in attributes:
            if attribute not in asset.attrib:
                continue
            filename = asset.get(attribute).replace("\\", "/")
            if not filename:
                raise ValueError(
                    f"Empty {attribute} on {asset.tag} asset {asset.get('name')!r}"
                )
            source = (assets_dir / filename).resolve()
            if source not in cached:
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Missing {asset.tag} asset {asset.get('name')!r}: {source}"
                    )
                if source.is_relative_to(assets_dir):
                    checksum, size = _digest(source)
                    destination = source
                else:
                    destination, checksum, size = _copy_external(source, assets_dir)
                cached[source] = (destination, checksum, size)
            destination, checksum, size = cached[source]
            relative = destination.relative_to(assets_dir).as_posix()
            asset.set(attribute, relative)
            path = f"meshes/{relative}"
            entry = manifest.setdefault(
                path,
                {
                    "path": path,
                    "sha256": checksum,
                    "size_bytes": size,
                    "references": [],
                },
            )
            entry["references"].append(
                {"type": asset.tag, "name": asset.get("name"), "attribute": attribute}
            )
    return [manifest[path] for path in sorted(manifest)]
