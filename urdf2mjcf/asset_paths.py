"""Resolve local URDF mesh references and their staged OBJ replacements."""

from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit


class MeshPathResolver:
    """Resolve exact paths first; index staged assets once for basename fallbacks."""

    def __init__(
        self,
        source_dir: Path,
        assets_dir: Path,
        *,
        prefer_converted: bool = True,
        source_map: dict[Path, Path] | None = None,
    ):
        self.source_dir = Path(source_dir).resolve()
        self.assets_dir = Path(assets_dir).resolve()
        self.prefer_converted = prefer_converted
        self.source_map = source_map
        self._by_name: dict[str, list[Path]] | None = None

    def _prefer_obj(self, path: Path) -> Path:
        if self.prefer_converted and path.suffix.lower() != ".obj":
            for converted in (
                path.with_suffix(".obj"),
                path.parent / path.stem / f"{path.stem}.obj",
            ):
                if converted.is_file():
                    return converted.resolve()
        return path.resolve()

    def resolve(self, reference: str) -> Path:
        """Return an existing mesh, rejecting missing and ambiguous references."""
        try:
            return self._resolve_reference(reference)
        except RuntimeError as exc:
            # Python 3.11/3.12 Path.resolve raises RuntimeError for symlink
            # loops. Treat it as an input error so inspection can retain its
            # unresolved_mesh diagnostic and unrelated batch jobs can continue.
            raise ValueError(
                f"Cannot resolve mesh reference {reference!r}: {exc}"
            ) from exc

    def _resolve_reference(self, reference: str) -> Path:
        reference = reference.replace("\\", "/")
        if not reference or reference.isspace():
            raise ValueError("Mesh filename is empty")
        package = None
        if "://" in reference:
            uri = urlsplit(reference)
            if uri.scheme == "file" and uri.netloc in ("", "localhost"):
                reference = unquote(uri.path)
            elif uri.scheme == "package":
                package = uri.netloc
                if not package:
                    raise ValueError(
                        f"Package mesh URI is missing a package name: {reference}"
                    )
                reference = unquote(uri.path).lstrip("/")
            else:
                raise ValueError(f"Unsupported mesh URI: {reference}")

        path = Path(reference)
        source_candidates = (
            []
            if package
            else ([path] if path.is_absolute() else [self.source_dir / path])
        )
        if package:
            for parent in (self.source_dir, *self.source_dir.parents):
                if parent.name == package:
                    source_candidates.append(parent / path)
                source_candidates.append(parent / package / path)
        # Explicit absolute/file-URI and parent-directory references identify a
        # specific source. A missing one must not select a same-named local
        # mesh through the compatibility basename fallback below.
        if (
            not package
            and (path.is_absolute() or ".." in path.parts)
            and not any(candidate.is_file() for candidate in source_candidates)
        ):
            raise FileNotFoundError(
                f"Mesh file not found at explicit path: {reference}"
            )
        if self.source_map is not None or not self.prefer_converted:
            for candidate in source_candidates:
                if candidate.is_file():
                    resolved = candidate.resolve()
                    if self.source_map is not None and resolved in self.source_map:
                        return self._prefer_obj(self.source_map[resolved])
                    return resolved
            if package:
                raise FileNotFoundError(
                    f"Mesh file not found in package '{package}': {reference}"
                )
        relative_paths = [path] if not path.is_absolute() else []
        if path.is_absolute():
            for root in (self.source_dir, self.source_dir.parent):
                if path.is_relative_to(root):
                    relative_paths.append(path.relative_to(root))
        staged_candidates = []
        for relative in relative_paths:
            parts = list(relative.parts)
            if "meshes" in parts:
                parts = parts[parts.index("meshes") + 1 :]
            if parts and parts[0].lower() in ("visual", "collision"):
                parts[0] = parts[0].lower()
            candidate = (self.assets_dir / Path(*parts)).resolve()
            if candidate.is_relative_to(self.assets_dir):
                staged_candidates.append(candidate)
        for candidate in staged_candidates:
            converted = self._prefer_obj(candidate)
            if converted.is_file():
                return converted

        for candidate in source_candidates:
            if candidate.is_file():
                return candidate.resolve()

        # The same basename may belong to different links. Never select one at random.
        if self._by_name is None:
            self._by_name = defaultdict(list)
            for asset in sorted(self.assets_dir.rglob("*")):
                if asset.is_file():
                    self._by_name[asset.name].append(asset)
        matches = self._by_name.get(path.name, [])
        if len(matches) == 1:
            return self._prefer_obj(matches[0])
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous mesh reference '{reference}': "
                + ", ".join(str(match) for match in matches)
            )
        raise FileNotFoundError(f"Mesh file not found: {reference}")
