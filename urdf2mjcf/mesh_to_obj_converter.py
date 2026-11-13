"""Convert common mesh formats to OBJ using trimesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from urdf2mjcf.logging_utils import URDF2MJCFLogger
from urdf2mjcf.mesh_units import normalize_mesh_asset

SUPPORTED_EXTENSIONS = frozenset({".stl", ".ply", ".off", ".3mf"})


class MeshToOBJConverter:
    """Convert trimesh-supported mesh files into single-mesh OBJ files."""

    def __init__(self, overwrite: bool = False):
        self.overwrite = overwrite
        self.logger = URDF2MJCFLogger.get_logger("MeshToOBJConverter")

    def find_mesh_files(
        self, input_path: str | Path, recursive: bool = True
    ) -> list[Path]:
        """Return supported mesh files under a file or directory path."""
        path = Path(input_path)
        if path.is_file():
            return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
        if not path.is_dir():
            return []

        iterator = path.rglob("*") if recursive else path.glob("*")
        return sorted(
            candidate
            for candidate in iterator
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def _as_single_mesh(loaded):
        """Apply scene transforms and return one Trimesh instance."""
        if isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        elif isinstance(loaded, trimesh.Scene):
            mesh = loaded.to_mesh()
        else:
            raise TypeError(f"Unsupported trimesh result: {type(loaded)}")
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            raise ValueError("Mesh contains no triangle geometry")
        if not np.all(np.isfinite(mesh.vertices)):
            raise ValueError("Mesh vertices must contain finite coordinates")
        return mesh

    def convert_file(
        self, source: str | Path, output: str | Path | None = None
    ) -> Path:
        """Convert one supported mesh file and return the OBJ path."""
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Mesh file does not exist: {source_path}")
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported mesh format: {source_path.suffix}; supported: {supported}"
            )

        output_path = Path(output) if output else source_path.with_suffix(".obj")
        if output_path.resolve() == source_path.resolve():
            raise ValueError("Output must differ from the source mesh")
        if output_path.is_dir():
            raise IsADirectoryError(f"OBJ output path is a directory: {output_path}")
        if output_path.exists() and not self.overwrite:
            self.logger.info(f"Keeping existing OBJ: {output_path}")
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)
        loaded = normalize_mesh_asset(
            trimesh.load(source_path, process=False), source_path
        )
        mesh = self._as_single_mesh(loaded)
        mesh.export(output_path, file_type="obj")
        self.logger.info(f"Converted {source_path} -> {output_path}")
        return output_path

    def convert_directory(
        self, input_path: str | Path, recursive: bool = True
    ) -> dict[Path, Path | None]:
        """Convert all supported meshes, retaining a result for every input."""
        results: dict[Path, Path | None] = {}
        for source in self.find_mesh_files(input_path, recursive=recursive):
            try:
                results[source] = self.convert_file(source)
            except Exception as exc:
                self.logger.error(f"Failed to convert {source}: {exc}")
                results[source] = None
        return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert STL, PLY, OFF, and 3MF meshes to OBJ."
    )
    parser.add_argument("input", help="Mesh file or directory to convert.")
    parser.add_argument(
        "--no-recursive", action="store_true", help="Do not scan subdirectories."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing OBJ files."
    )
    args = parser.parse_args()

    converter = MeshToOBJConverter(overwrite=args.overwrite)
    results = converter.convert_directory(args.input, recursive=not args.no_recursive)
    if not results:
        converter.logger.error("No supported mesh files found.")
        return 1
    return 0 if all(output is not None for output in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
