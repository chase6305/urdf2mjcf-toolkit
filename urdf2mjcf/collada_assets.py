"""Apply COLLADA asset conventions consistently to conversion and inertia."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _asset_conventions(source: Path) -> tuple[float, str]:
    """Read the document asset header without parsing large geometry arrays again."""
    asset = None
    prefix = ""
    depth = 0
    with source.open("rb") as stream:
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start":
                depth += 1
                if depth == 1:
                    if element.tag.rsplit("}", 1)[-1] != "COLLADA":
                        raise ValueError("Expected a COLLADA document")
                    if element.tag.startswith("{"):
                        prefix = element.tag.rsplit("}", 1)[0] + "}"
                elif depth == 2 and element.tag != prefix + "asset":
                    break  # No document asset header; use COLLADA defaults.
            else:
                if depth == 2 and element.tag == prefix + "asset":
                    asset = element
                    break
                depth -= 1
    unit, up_axis = 1.0, "Y_UP"
    if asset is not None:
        units = asset.findall(prefix + "unit")
        axes = asset.findall(prefix + "up_axis")
        if len(units) > 1 or len(axes) > 1:
            raise ValueError("Ambiguous COLLADA asset unit or up_axis")
        if units:
            unit = float(units[0].get("meter", "1"))
        if axes:
            up_axis = (axes[0].text or "").strip()
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError("COLLADA asset unit meter must be finite and positive")
    if up_axis not in {"X_UP", "Y_UP", "Z_UP"}:
        raise ValueError(f"Unsupported COLLADA asset up_axis: {up_axis!r}")
    return unit, up_axis


def normalize_collada_asset(geometry, source: Path):
    """Map the document's units/up axis into meter-based Z-up coordinates.

    Apply this at the scene root so node translations are scaled and rotated
    together with vertices, as in Blender's COLLADA importer. URDF mesh scale
    and visual/collision origins are applied later in the URDF link frame.
    """
    unit, up_axis = _asset_conventions(source)
    rotations = {
        "X_UP": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        "Y_UP": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "Z_UP": np.eye(3),
    }
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotations[up_axis]) * unit
    geometry.apply_transform(transform)
    return geometry
