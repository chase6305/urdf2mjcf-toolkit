"""URDF frame conventions and physical properties shared by the converters."""

import xml.etree.ElementTree as ET

import numpy as np
from trimesh.transformations import (
    euler_from_matrix,
    euler_matrix,
    quaternion_from_matrix,
)


def vector(text: str, size: int = 3) -> np.ndarray:
    result = np.array([float(value) for value in text.split()])
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected {size} finite numbers: {text!r}")
    return result


def numbers(values) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def origin_transform(origin: ET.Element | None) -> np.ndarray:
    if origin is None:
        return np.eye(4)
    if "quat_xyzw" in origin.attrib:
        raise ValueError(
            "Quaternion origins (quat_xyzw) are not supported; express the orientation using rpy"
        )
    transform = euler_matrix(*vector(origin.get("rpy", "0 0 0")), axes="sxyz")
    transform[:3, 3] = vector(origin.get("xyz", "0 0 0"))
    return transform


def inertia_matrix(element: ET.Element) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = vector(
        " ".join(
            element.get(key, "0") for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
        6,
    )
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def validate_inertia_tensor(matrix: np.ndarray) -> None:
    """Require finite positive principal moments satisfying the triangle rule."""
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Inertia matrix must contain finite values")
    scale = np.max(np.abs(matrix))
    if scale == 0:
        raise ValueError("Inertia matrix is not positive definite")
    values = np.linalg.eigvalsh(matrix / scale)
    if np.any(values <= 0):
        raise ValueError("Inertia matrix is not positive definite")
    # Scale the comparison to avoid overflow for large finite tensors.
    normalized = values / values[-1]
    if normalized[0] + normalized[1] < 1 - 1e-12:
        raise ValueError("Principal-moment triangle inequality violation")


def validate_inertial(inertial: ET.Element) -> None:
    """Check an explicit inertia definition without inferring or changing it."""
    mass = inertial.find("mass")
    if mass is None:
        raise ValueError("Missing URDF mass element")
    try:
        value = vector(mass.get("value", ""), 1)[0]
        if value < 0:
            raise ValueError("negative mass")
    except ValueError as exc:
        raise ValueError(f"Invalid URDF mass: {mass.get('value')}") from exc
    origin_transform(inertial.find("origin"))
    tensor = inertial.find("inertia")
    if tensor is None or any(key not in tensor.attrib for key in ("ixx", "iyy", "izz")):
        raise ValueError("Missing URDF inertia or diagonal components")
    matrix = inertia_matrix(tensor)
    if value == 0:
        if np.any(matrix != 0):
            raise ValueError("Zero URDF mass requires zero inertia")
        # Static massless frames are allowed; moving-body validity is compiled.
        return
    validate_inertia_tensor(matrix)


def normalize_urdf_physics(robot: ET.Element) -> None:
    """Make implicit axes explicit and express inertia in principal axes.

    The external backend drops off-diagonal entries; native MuJoCo 3.8 ignores
    inertial-origin rotation when reading a full URDF tensor. Diagonalizing the
    staged URDF and rotating its inertial frame preserves the original tensor
    for both importers, without modifying the source file.
    """
    for joint in robot.findall("joint"):
        kind = joint.get("type")
        if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError(
                f"Unsupported URDF joint type {kind!r}: {joint.get('name')}"
            )
        if joint.find("mimic") is not None:
            raise ValueError(
                f"URDF mimic joints are not supported: {joint.get('name')}"
            )
        if kind != "fixed":
            axis = joint.find("axis")
            if axis is None:
                axis = ET.SubElement(joint, "axis")
            value = vector(axis.get("xyz", "1 0 0"))
            if not np.any(value):
                raise ValueError(f"Joint axis must be nonzero: {joint.get('name')}")
            # Scaling first avoids overflow and underflow while normalizing
            # finite, nonzero axes. Axis magnitude has no URDF physical meaning.
            value /= np.max(np.abs(value))
            value /= np.linalg.norm(value)
            axis.set("xyz", numbers(value))
    for inertial in robot.findall("link/inertial"):
        mass = inertial.find("mass")
        if mass is not None:
            try:
                value = vector(mass.get("value", ""), 1)[0]
                if value < 0:
                    raise ValueError("negative mass")
            except ValueError as exc:
                raise ValueError(f"Invalid URDF mass: {mass.get('value')}") from exc
        tensor = inertial.find("inertia")
        if tensor is None:
            continue
        matrix = inertia_matrix(tensor)
        # The native importer requires all six entries even for a diagonal tensor.
        for key in ("ixy", "ixz", "iyz"):
            tensor.attrib.setdefault(key, "0")
        if not np.any(matrix - np.diag(np.diag(matrix))):
            continue
        values, axes = np.linalg.eigh(matrix)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        origin = inertial.find("origin")
        transform = origin_transform(origin)
        transform[:3, :3] = transform[:3, :3] @ axes
        if origin is None:
            origin = ET.SubElement(inertial, "origin")
        origin.set("rpy", numbers(euler_from_matrix(transform, axes="sxyz")))
        for key, value in zip(("ixx", "iyy", "izz"), values, strict=True):
            tensor.set(key, format(float(value), ".17g"))
        for key in ("ixy", "ixz", "iyz"):
            tensor.set(key, "0")


def restore_urdf_physics(mjcf: ET.Element, urdf: ET.Element) -> None:
    """Restore inertial frames and joint semantics after upstream serialization."""
    bodies = {body.get("name"): body for body in mjcf.findall(".//body")}
    for link in urdf.findall("link"):
        source = link.find("inertial")
        if source is None:
            continue
        body = bodies.get(link.get("name"))
        if body is None:
            raise ValueError(
                f"Converted model is missing inertial link: {link.get('name')}"
            )
        tensor = source.find("inertia")
        mass = source.find("mass")
        if tensor is None or mass is None:
            raise ValueError(f"Incomplete URDF inertia: {link.get('name')}")
        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(body, "inertial")
        transform = origin_transform(source.find("origin"))
        inertial.attrib.clear()
        inertial.attrib.update(
            mass=mass.get("value", "0"),
            pos=numbers(transform[:3, 3]),
            quat=numbers(quaternion_from_matrix(transform)),
            diaginertia=numbers(np.diag(inertia_matrix(tensor))),
        )
    compiler = mjcf.find("compiler")
    if compiler is not None:
        compiler.set("balanceinertia", "false")
    converted = {joint.get("name"): joint for joint in mjcf.findall(".//body/joint")}
    for source in urdf.findall("joint"):
        child = source.find("child")
        if child is not None and child.get("link") in bodies:
            transform = origin_transform(source.find("origin"))
            body = bodies[child.get("link")]
            body.set("pos", numbers(transform[:3, 3]))
            body.set("quat", numbers(quaternion_from_matrix(transform)))
        kind = source.get("type")
        if kind == "fixed":
            continue
        joint = converted.get(source.get("name"))
        if joint is None:
            raise ValueError(f"Converted model is missing joint: {source.get('name')}")
        joint.set("axis", source.find("axis").get("xyz"))
        dynamics = source.find("dynamics")
        for source_key, target_key in (
            ("damping", "damping"),
            ("friction", "frictionloss"),
        ):
            value = dynamics.get(source_key, "0") if dynamics is not None else "0"
            scalar = vector(value, 1)[0]
            if scalar < 0:
                raise ValueError(f"Negative joint {source_key}: {source.get('name')}")
            joint.set(target_key, value)
        if kind == "continuous":
            joint.set("limited", "false")
            joint.attrib.pop("range", None)
        else:
            limit = source.find("limit")
            if (
                limit is None
                or "lower" not in limit.attrib
                or "upper" not in limit.attrib
            ):
                raise ValueError(f"Missing joint limits: {source.get('name')}")
            joint.set("limited", "true")
            joint.set("range", f"{limit.get('lower')} {limit.get('upper')}")
        limit = source.find("limit")
        if limit is not None and "effort" in limit.attrib:
            effort = vector(limit.get("effort"), 1)[0]
            if effort < 0:
                raise ValueError(f"Negative joint effort limit: {source.get('name')}")
            if effort > 0:
                joint.set("actuatorfrclimited", "true")
                joint.set("actuatorfrcrange", numbers([-effort, effort]))
            else:
                # MuJoCo 3.8 rejects zero-width joint and actuator force ranges.
                # Retain control slots and transmission kinematics, but make
                # each generated motor a zero-gain, zero-bias actuator.
                joint.set("actuatorfrclimited", "false")
                joint.attrib.pop("actuatorfrcrange", None)
                for actuator in mjcf.findall("actuator/*"):
                    if source.get("name") in (
                        actuator.get("joint"),
                        actuator.get("jointinparent"),
                    ):
                        if actuator.tag != "motor":
                            raise ValueError(
                                f"Cannot disable unsupported actuator type: {actuator.tag}"
                            )
                        actuator.tag = "general"
                        actuator.attrib.update(
                            dyntype="none",
                            gaintype="fixed",
                            gainprm="0",
                            biastype="none",
                            biasprm="0",
                            forcelimited="false",
                        )
                        actuator.attrib.pop("forcerange", None)


def validate_compiled_physics(model) -> None:
    """Compilation alone permits infinity in some physical model arrays."""
    for name in (
        "body_mass",
        "body_subtreemass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "dof_damping",
        "dof_frictionloss",
        "dof_M0",
        "dof_invweight0",
        "body_invweight0",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "mesh_vert",
        "mesh_scale",
    ):
        if not np.all(np.isfinite(getattr(model, name))):
            raise ValueError(f"Compiled model has non-finite physical values: {name}")
