"""Read-only URDF topology, physical parameter, and mesh-reference checks."""

import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from pathlib import Path

from urdf2mjcf.asset_paths import MeshPathResolver
from urdf2mjcf.reporting import new_report
from urdf2mjcf.urdf_validation import physical_issues


def inspect_urdf(urdf_path: str | Path, backend: str = "urdf2mjcf") -> dict:
    """Return input diagnostics; this does not compile the model."""
    started = time.perf_counter()
    source = Path(urdf_path).expanduser().resolve()
    report = new_report("check", source, backend)
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
    for kind, elements in (("link", links), ("joint", joints)):
        names = Counter(element.get("name", "") for element in elements)
        for name, count in names.items():
            if not name.strip():
                issue("missing_name", f"{kind} is missing a name", element=kind)
            elif count > 1:
                issue("duplicate_name", f"Duplicate {kind} name: {name}", element=kind)
    link_names = {link.get("name") for link in links if link.get("name")}
    if not links:
        issue("missing_links", "URDF contains no links")
    children = defaultdict(list)
    indegree = dict.fromkeys(link_names, 0)
    for joint in joints:
        parent, child = joint.find("parent"), joint.find("child")
        parent_name = parent.get("link") if parent is not None else None
        child_name = child.get("link") if child is not None else None
        for role, name in (("parent", parent_name), ("child", child_name)):
            if name not in link_names:
                issue(
                    "invalid_joint_link",
                    f"Joint {joint.get('name')!r} has unknown {role} link: {name!r}",
                    joint=joint.get("name"),
                )
        if parent_name in link_names and child_name in link_names:
            children[parent_name].append(child_name)
            indegree[child_name] += 1
    for name, count in indegree.items():
        if count > 1:
            issue("multiple_parents", f"Link {name!r} has {count} parent joints")
    roots = sorted(name for name, count in indegree.items() if count == 0)
    if links and len(roots) != 1:
        issue("invalid_roots", f"Expected one root link, found {len(roots)}: {roots}")
    queue = deque(roots)
    visited = 0
    while queue:
        visited += 1
        for child in children[queue.popleft()]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited < len(link_names):
        issue("joint_cycle", "Joint graph contains a cycle")

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
