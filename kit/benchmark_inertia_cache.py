"""Measure the inertia cache on a bounded, reproducible shared-mesh case."""

import argparse
import json
import os
import platform
import statistics
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import trimesh

from urdf2mjcf.logging_utils import URDF2MJCFLogger
from urdf2mjcf.urdf_inertia_calculator import URDFInertiaCalculator


class MeasuredCalculator(URDFInertiaCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh_loads = 0

    def _mesh_properties(self, *args, **kwargs):
        self.mesh_loads += 1
        return super()._mesh_properties(*args, **kwargs)


class UncachedCalculator(MeasuredCalculator):
    def _geometry_properties(self, element, resolver, mesh_cache=None):
        return super()._geometry_properties(element, resolver, None)


def benchmark() -> dict:
    """Compare the same calculation with and without cache reuse.

    Timings exclude fixture creation and imports. Alternate order over three
    samples and require byte-identical URDF outputs; no timing threshold gates
    correctness. Each update starts with a new calculator and an empty cache.
    """
    samples = {"cached": [], "uncached": []}
    loads = {}
    with tempfile.TemporaryDirectory(prefix="inertia-cache-benchmark-") as temporary:
        directory = Path(temporary)
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.1)
        mesh.export(directory / "solid.obj")
        robot = ET.Element("robot", name="shared_mesh_benchmark")
        for index in range(24):
            link = ET.SubElement(robot, "link", name=f"link{index}")
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value=str(index + 1))
            ET.SubElement(inertial, "inertia", ixx="0", iyy="0", izz="0")
            visual = ET.SubElement(link, "visual")
            ET.SubElement(visual, "origin", xyz=f"{index * 0.01} 0 0")
            ET.SubElement(
                ET.SubElement(visual, "geometry"),
                "mesh",
                filename="solid.obj",
                scale="1 2 3",
            )
        template = ET.tostring(robot)
        for repetition in range(3):
            outputs = {}
            order = (
                ("uncached", "cached")
                if repetition % 2 == 0
                else ("cached", "uncached")
            )
            for label in order:
                source = directory / f"{label}.urdf"
                source.write_bytes(template)
                calculator = (
                    MeasuredCalculator if label == "cached" else UncachedCalculator
                )(source)
                started = time.perf_counter()
                if not calculator.update_inertia():
                    raise RuntimeError(calculator.last_error)
                samples[label].append(time.perf_counter() - started)
                loads[label] = calculator.mesh_loads
                outputs[label] = source.read_bytes()
            if outputs["cached"] != outputs["uncached"]:
                raise RuntimeError("Cache changed the calculated URDF bytes")
        medians = {key: statistics.median(values) for key, values in samples.items()}
        return {
            "measured_at_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "versions": {
                name: version(name)
                for name in ("numpy", "trimesh", "urdf2mjcf-toolkit")
            },
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "instances": 24,
            "faces_per_mesh": len(mesh.faces),
            "samples_seconds": samples,
            "median_seconds": medians,
            "mesh_loads_per_update": loads,
            "speedup": medians["uncached"] / medians["cached"],
            "outputs_identical": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Also save the measured JSON report."
    )
    args = parser.parse_args()
    URDF2MJCFLogger.get_logger("InertiaCalculator").setLevel("WARNING")
    report = json.dumps(benchmark(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
