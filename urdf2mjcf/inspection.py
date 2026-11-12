"""Read-only URDF topology, physical parameter, and mesh-reference checks."""

import time
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.asset_paths import MeshPathResolver
from urdf2mjcf.reporting import new_report
from urdf2mjcf.urdf_validation import physical_issues, topology_issues


def inspect_urdf(urdf_path: str | Path, backend: str = "urdf2mjcf") -> dict:
    """Return input diagnostics; this does not compile the model."""
    started = time.perf_counter()
    report = new_report("check", urdf_path, backend)
    source = Path(report["input"])
    report.update(stage="inspection", issues=[], meshes=[], statistics={})

    def issue(code, message, **context):
        report["issues"].append(
            {"severity": "error", "code": code, "message": message, **context}
        )

    def finish():
        report["status"] = (
            "failed"
            if any(item["severity"] == "error" for item in report["issues"])
            else "passed"
        )
        report["duration_seconds"] = round(time.perf_counter() - started, 6)
        return report

    try:
        if source.suffix.lower() != ".urdf":
            raise ValueError(f"Expected a .urdf input file: {source}")
        robot = ET.parse(source).getroot()
        if robot.tag != "robot":
            raise ValueError("URDF root element must be <robot>")
    except (OSError, ET.ParseError, ValueError) as exc:
        issue("invalid_input", str(exc))
        return finish()

    report["robot_name"] = robot.get("name")
    links, joints = robot.findall("link"), robot.findall("joint")
    issues, roots = topology_issues(robot)
    report["issues"].extend(issues)
    report["issues"].extend(physical_issues(robot))
    assets = source.parent / "meshes"
    if not assets.is_dir():
        assets = source.parent
    resolver = MeshPathResolver(source.parent, assets, prefer_converted=False)
    resolved_files = set()
    for link in links:
        for kind in ("visual", "collision"):
            for mesh in link.findall(f"{kind}/geometry/mesh"):
                reference = mesh.get("filename", mesh.get("file", ""))
                entry = {
                    "link": link.get("name"),
                    "kind": kind,
                    "reference": reference,
                    "resolved_path": None,
                    "scale": mesh.get("scale", "1 1 1"),
                }
                try:
                    resolved = resolver.resolve(reference)
                    entry.update(
                        resolved_path=str(resolved), format=resolved.suffix.lower()
                    )
                    resolved_files.add(resolved)
                except (OSError, ValueError) as exc:
                    issue(
                        "unresolved_mesh",
                        str(exc),
                        link=link.get("name"),
                        reference=reference,
                    )
                report["meshes"].append(entry)
    report["statistics"] = {
        "links": len(links),
        "joints": len(joints),
        "root_links": roots,
        "mesh_references": len(report["meshes"]),
        "unique_mesh_files": len(resolved_files),
    }
    return finish()
