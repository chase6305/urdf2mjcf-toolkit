"""Interpret declared mesh units before applying URDF scale and origins."""

from pathlib import Path

import numpy as np

from urdf2mjcf.collada_assets import normalize_collada_asset


def normalize_mesh_asset(geometry, source: str | Path):
    """Normalize DAE asset frames and 3MF length units in place.

    Other supported formats keep their imported numeric coordinates. Applying
    at the scene root includes instance translations without copying meshes.
    """
    suffix = Path(source).suffix.lower()
    if suffix == ".dae":
        return normalize_collada_asset(geometry, Path(source))
    if suffix == ".3mf":
        unit = geometry.units or geometry.metadata.get("units", "millimeters")
        factors = {
            "micron": 1e-6,
            "millimeter": 0.001,
            "millimeters": 0.001,  # Trimesh's tag when the model omits unit.
            "centimeter": 0.01,
            "inch": 0.0254,
            "foot": 0.3048,
            "meter": 1.0,
            "meters": 1.0,
        }
        if unit not in factors:
            raise ValueError(f"Unsupported 3MF unit: {unit!r}")
        factor = factors[unit]
        geometry.apply_transform(np.diag([factor, factor, factor, 1.0]))
        geometry.units = "meters"
        geometry.metadata["units"] = "meters"
    return geometry
