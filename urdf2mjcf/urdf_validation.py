"""Shared, read-only diagnostics for URDF structure and physical parameters."""

import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque

import numpy as np

from urdf2mjcf.physics import origin_transform, validate_inertial, vector


class URDFValidationError(ValueError):
    """Input diagnostics that retain their codes across converter boundaries."""

    def __init__(self, issues: list[dict]):
        self.issues = issues
        super().__init__(
            "; ".join(f"[{issue['code']}] {issue['message']}" for issue in issues)
        )


def declaration_issues(robot: ET.Element) -> list[dict]:
    """Reject repeated singleton declarations instead of interpreting the first."""
    issues = []

    def unique(parent, tags, context):
        for tag in tags:
            if len(parent.findall(tag)) > 1:
                location = ", ".join(
                    f"{key}={value!r}" for key, value in context.items()
                )
                issues.append(
                    {
                        "severity": "error",
                        "code": "ambiguous_definition",
                        "message": f"{location}: Multiple <{tag}> elements in <{parent.tag}>",
                        "element": tag,
                        **context,
                    }
                )

    for joint in robot.findall("joint"):
        unique(
            joint,
            ("origin", "axis", "limit", "dynamics", "mimic"),
            {"joint": joint.get("name")},
        )
    for link in robot.findall("link"):
        context = {"link": link.get("name")}
        unique(link, ("inertial",), context)
        for inertial in link.findall("inertial"):
            unique(inertial, ("origin", "mass", "inertia"), context)
        for kind in ("visual", "collision"):
            for index, instance in enumerate(link.findall(kind)):
                unique(
                    instance,
                    ("origin", "geometry"),
                    {**context, "kind": kind, "index": index},
                )
    return issues


def topology_issues(robot: ET.Element) -> tuple[list[dict], list[str]]:
    """Return structural errors and sorted root names without recursive traversal.

    Importers differ in whether they reject malformed graphs, accept multiple
    roots, or silently discard disconnected links. Enforce a single URDF tree
    before either importer can interpret it.
    """
    issues = []

    def issue(code, message, **context):
        issues.append(
            {"severity": "error", "code": code, "message": message, **context}
        )

    links, joints = robot.findall("link"), robot.findall("joint")
    for kind, elements in (("link", links), ("joint", joints)):
        names = Counter(element.get("name", "") for element in elements)
        for name, count in names.items():
            if not name.strip():
                issue("missing_name", f"{kind} is missing a name", element=kind)
            elif count > 1:
                issue("duplicate_name", f"Duplicate {kind} name: {name}", element=kind)
    link_names = {link.get("name") for link in links if link.get("name", "").strip()}
    if not links:
        issue("missing_links", "URDF contains no links")
    children = defaultdict(list)
    indegree = dict.fromkeys(link_names, 0)
    for joint in joints:
        endpoints = {}
        for role in ("parent", "child"):
            elements = joint.findall(role)
            if len(elements) != 1:
                issue(
                    "invalid_joint_link",
                    f"Joint {joint.get('name')!r} must contain exactly one {role} element",
                    joint=joint.get("name"),
                    element=role,
                )
            name = elements[0].get("link") if elements else None
            endpoints[role] = name
            if name not in link_names:
                issue(
                    "invalid_joint_link",
                    f"Joint {joint.get('name')!r} has unknown {role} link: {name!r}",
                    joint=joint.get("name"),
                    element=role,
                )
        parent, child = endpoints["parent"], endpoints["child"]
        if parent in link_names and child in link_names:
            children[parent].append(child)
            indegree[child] += 1
    for name, count in sorted(indegree.items()):
        if count > 1:
            issue(
                "multiple_parents",
                f"Link {name!r} has {count} parent joints",
                link=name,
            )
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
    return issues, roots


def validate_urdf_structure(robot: ET.Element) -> None:
    """Reject malformed graphs consistently with read-only inspection."""
    issues, _ = topology_issues(robot)
    if issues:
        raise URDFValidationError(issues)


def physical_issues(robot: ET.Element) -> list[dict]:
    """Collect physical input errors and known conversion limitations.

    This validates declarations, not mesh contents or the compiled dynamics.
    Context identifies the affected joint, link, and visual/collision instance.
    """
    issues = []

    def check(code, context, callback, *args):
        try:
            callback(*args)
        except (ValueError, np.linalg.LinAlgError) as exc:
            issues.append(
                {"severity": "error", "code": code, "message": str(exc), **context}
            )

    def nonnegative(value, label):
        if vector(value, 1)[0] < 0:
            raise ValueError(f"{label} must be nonnegative: {value}")

    def joint_axis(joint):
        axis = joint.find("axis")
        value = vector(axis.get("xyz", "1 0 0") if axis is not None else "1 0 0")
        if not np.any(value):
            raise ValueError("Joint axis must be nonzero")

    def joint_range(joint):
        limit = joint.find("limit")
        if limit is None or any(key not in limit.attrib for key in ("lower", "upper")):
            raise ValueError("Missing joint limits: lower and upper are required")
        lower, upper = vector(f"{limit.get('lower')} {limit.get('upper')}", 2)
        if lower >= upper:
            raise ValueError("Joint lower limit must be smaller than upper limit")

    def geometry_shape(geometry):
        if geometry is None or len(geometry) != 1:
            raise ValueError("Each geometry must contain exactly one shape")
        shape = geometry[0]
        if shape.tag == "mesh":
            return  # Mesh scales are checked separately for precise diagnostics.
        if shape.tag == "box":
            dimensions = vector(shape.get("size", ""))
        elif shape.tag == "sphere":
            dimensions = vector(shape.get("radius", ""), 1)
        elif shape.tag == "cylinder":
            dimensions = vector(
                f"{shape.get('radius', '')} {shape.get('length', '')}", 2
            )
        else:
            raise ValueError(f"Unsupported URDF geometry: {shape.tag}")
        if np.any(dimensions <= 0):
            raise ValueError(f"{shape.tag} dimensions must be positive")

    def mesh_scale(mesh):
        if np.any(vector(mesh.get("scale", "1 1 1")) == 0):
            raise ValueError("Mesh scale must have three finite, nonzero values")

    for joint in robot.findall("joint"):
        context = {"joint": joint.get("name")}
        if joint.get("type") not in {"fixed", "revolute", "continuous", "prismatic"}:
            issues.append(
                {
                    "severity": "error",
                    "code": "unsupported_joint",
                    "message": f"Unsupported joint type: {joint.get('type')!r}",
                    **context,
                }
            )
        if joint.find("mimic") is not None:
            issues.append(
                {
                    "severity": "error",
                    "code": "unsupported_mimic",
                    "message": "Mimic joint constraints are not supported",
                    **context,
                }
            )
        check("invalid_origin", context, origin_transform, joint.find("origin"))
        if joint.get("type") not in {"revolute", "continuous", "prismatic"}:
            continue
        check("invalid_joint_axis", context, joint_axis, joint)
        if joint.get("type") != "continuous":
            check("invalid_joint_limits", context, joint_range, joint)
        dynamics = joint.find("dynamics")
        if dynamics is not None:
            for key in ("damping", "friction"):
                check(
                    "invalid_joint_dynamics",
                    {**context, "parameter": key},
                    nonnegative,
                    dynamics.get(key, "0"),
                    key,
                )
        limit = joint.find("limit")
        if limit is not None:
            for key in ("effort", "velocity"):
                if key in limit.attrib:
                    before = len(issues)
                    check(
                        "invalid_joint_limits",
                        {**context, "parameter": key},
                        nonnegative,
                        limit.get(key),
                        key,
                    )
                    if key == "velocity" and len(issues) == before:
                        issues.append(
                            {
                                "severity": "warning",
                                "code": "velocity_limit_unenforced",
                                "message": "URDF velocity limits require controller enforcement; conversion does not create a speed constraint",
                                **context,
                            }
                        )

    for link in robot.findall("link"):
        context = {"link": link.get("name")}
        inertial = link.find("inertial")
        if inertial is not None:
            check("invalid_inertia", context, validate_inertial, inertial)
        for kind in ("visual", "collision"):
            for index, instance in enumerate(link.findall(kind)):
                instance_context = {**context, "kind": kind, "index": index}
                check(
                    "invalid_origin",
                    instance_context,
                    origin_transform,
                    instance.find("origin"),
                )
                geometry = instance.find("geometry")
                check("invalid_geometry", instance_context, geometry_shape, geometry)
                for mesh in instance.findall("geometry/mesh"):
                    check(
                        "invalid_mesh_scale",
                        {
                            **instance_context,
                            "reference": mesh.get("filename", mesh.get("file", "")),
                        },
                        mesh_scale,
                        mesh,
                    )
    for issue in issues:
        context = ", ".join(
            f"{key}={issue[key]!r}"
            for key in ("link", "joint", "kind", "index")
            if key in issue
        )
        issue["message"] = f"{context}: {issue['message']}"
    return declaration_issues(robot) + issues


def validate_urdf_physics(robot: ET.Element) -> None:
    """Use the same input checks for conversion as for the inspection command."""
    errors = [issue for issue in physical_issues(robot) if issue["severity"] == "error"]
    if errors:
        raise URDFValidationError(errors)
