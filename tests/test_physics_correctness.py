"""Analytical checks independent of the URDF importer and MJCF serializer."""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def _rotation(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _pose(xyz, rpy):
    result = np.eye(4)
    result[:3, :3] = _rotation(rpy)
    result[:3, 3] = xyz
    return result


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_axis_direction_is_independent_of_finite_magnitude(
    tmp_path, backend, magnitude, kind
):
    source = tmp_path / "robot.urdf"
    source.write_text(
        f"""<robot name="axis"><link name="base"/>
    <link name="arm"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
    <visual><geometry><box size="1 1 1"/></geometry></visual></link>
    <joint name="joint" type="{kind}"><parent link="base"/><child link="arm"/>
    <axis xyz="{magnitude} {2*magnitude} {-2*magnitude}"/><limit lower="-1" upper="1" effort="10"/></joint></robot>"""
    )
    original = source.read_bytes()
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    expected = np.array([1, 2, -2]) / 3
    np.testing.assert_allclose(model.joint("joint").axis, expected, atol=1e-12)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.25
    mujoco.mj_forward(model, data)
    if kind == "prismatic":
        base = data.body("base")
        np.testing.assert_allclose(
            data.body("arm").xpos - base.xpos,
            base.xmat.reshape(3, 3) @ (0.25 * expected),
            atol=1e-12,
        )
    else:
        cross = np.array(
            [
                [0, -expected[2], expected[1]],
                [expected[2], 0, -expected[0]],
                [-expected[1], expected[0], 0],
            ]
        )
        rotation = (
            np.eye(3) + np.sin(0.25) * cross + (1 - np.cos(0.25)) * (cross @ cross)
        )
        np.testing.assert_allclose(
            data.body("arm").xmat.reshape(3, 3),
            data.body("base").xmat.reshape(3, 3) @ rotation,
            atol=1e-12,
        )
    assert manager.last_report["inertia_recovery_attempts"] == 0
    assert source.read_bytes() == original


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_zero_effort_retains_passive_motion_and_disables_generated_actuation(
    tmp_path, backend
):
    source = tmp_path / "robot.urdf"
    source.write_text("""<robot name="passive"><link name="base"/>
    <link name="arm"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
    <visual><geometry><box size="1 1 1"/></geometry></visual></link>
    <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
    <limit lower="-1" upper="1" effort="0"/></joint></robot>""")
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    assert model.nv == 1
    assert model.nu == (1 if backend == "urdf2mjcf" else 0)
    if model.nu:
        assert bool(model.actuator_forcelimited[0])
        np.testing.assert_array_equal(model.actuator_forcerange[0], [0, 0])
    data = mujoco.MjData(model)
    for command in (-1e6, 1e6):
        data.ctrl[:] = command
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(data.qfrc_actuator, [0])
    data.qpos[0] = 0.3
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        data.body("arm").xmat.reshape(3, 3), _rotation([0.3, 0, 0]), atol=1e-12
    )


@pytest.mark.parametrize("backend", ["urdf2mjcf", "mujoco"])
@pytest.mark.parametrize("floating", [False, True])
def test_joint_kinematics_and_full_inertia_match_analytic_values(
    tmp_path, monkeypatch, backend, floating
):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    root = ET.Element("robot", name="analytic")
    tensor = np.array([[2, 0.4, 0.2], [0.4, 3, 0.1], [0.2, 0.1, 4]])
    inertial_rpy = [0.3, -0.4, 0.5]
    for name in ("base", "arm", "slider", "tool"):
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.1 0.2 0.3", rpy="0.3 -0.4 0.5")
        ET.SubElement(inertial, "mass", value="2")
        ET.SubElement(
            inertial,
            "inertia",
            ixx="2",
            iyy="3",
            izz="4",
            ixy="0.4",
            ixz="0.2",
            iyz="0.1",
        )
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz="0.1 -0.2 0.4", rpy="-0.2 0.3 -0.4")
        ET.SubElement(ET.SubElement(visual, "geometry"), "box", size="0.2 0.3 0.4")
    origins = [
        ([0.2, -0.3, 0.4], [0.2, 0.3, -0.4]),
        ([0.4, 0.1, -0.2], [-0.1, 0.2, 0.3]),
        ([0.1, 0.2, 0.3], [0.4, -0.3, 0.2]),
    ]
    for index, (parent, child, kind) in enumerate(
        (
            ("base", "arm", "revolute"),
            ("arm", "slider", "prismatic"),
            ("slider", "tool", "continuous"),
        )
    ):
        joint = ET.SubElement(root, "joint", name=f"j{index}", type=kind)
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        xyz, rpy = origins[index]
        ET.SubElement(
            joint, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy))
        )
        if index > 0:
            ET.SubElement(joint, "axis", xyz="0 2 0" if index == 1 else "0 0 3")
        ET.SubElement(joint, "limit", lower="-1", upper="1", effort="10", velocity="2")
        ET.SubElement(joint, "dynamics", damping="0.25", friction="0.15")
    source = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(source)
    original = source.read_bytes()
    manager = MujocoConversionManager(
        source,
        tmp_path / "output",
        backend=backend,
        fixed_base=not floating,
        auto_recalculate_inertia=False,
    )
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    assert model.nq == (10 if floating else 3)
    for name in ("base", "arm", "slider", "tool"):
        body = model.body(name)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, body.iquat)
        rotation = rotation.reshape(3, 3)
        actual = rotation @ np.diag(body.inertia) @ rotation.T
        expected_rotation = _rotation(inertial_rpy)
        np.testing.assert_allclose(
            actual, expected_rotation @ tensor @ expected_rotation.T, atol=2e-5
        )
        np.testing.assert_allclose(body.ipos, [0.1, 0.2, 0.3], atol=1e-7)
        assert body.mass[0] == 2
    for index, axis in enumerate(([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        joint = model.joint(f"j{index}")
        dof = int(joint.dofadr[0])
        np.testing.assert_allclose(joint.axis, axis, atol=1e-7)
        assert model.dof_damping[dof] == pytest.approx(0.25)
        assert model.dof_frictionloss[dof] == pytest.approx(0.15)
        assert bool(joint.limited[0]) is (index != 2)
        assert bool(model.jnt_actfrclimited[joint.id])
        np.testing.assert_allclose(model.jnt_actfrcrange[joint.id], [-10, 10])
    data = mujoco.MjData(model)
    for q in ((0.4, -0.2, 0.7), (-0.5, 0.3, -1.2), (0, 0, 0)):
        for index, value in enumerate(q):
            data.qpos[model.joint(f"j{index}").qposadr[0]] = value
        mujoco.mj_forward(model, data)
        # The legacy backend intentionally lifts its root; compare in that root frame.
        expected = np.eye(4)
        expected[:3, :3] = data.body("base").xmat.reshape(3, 3)
        expected[:3, 3] = data.body("base").xpos
        for index, name in enumerate(("arm", "slider", "tool")):
            motion = np.eye(4)
            if index == 1:
                motion[1, 3] = q[index]
            else:
                angles = [0.0, 0.0, 0.0]
                angles[0 if index == 0 else 2] = q[index]
                motion[:3, :3] = _rotation(angles)
            expected = expected @ _pose(*origins[index]) @ motion
            np.testing.assert_allclose(data.body(name).xpos, expected[:3, 3], atol=2e-6)
            np.testing.assert_allclose(
                data.body(name).xmat.reshape(3, 3), expected[:3, :3], atol=2e-6
            )
            geom_id = model.body(name).geomadr[0]
            geom_pose = expected @ _pose([0.1, -0.2, 0.4], [-0.2, 0.3, -0.4])
            np.testing.assert_allclose(
                data.geom_xpos[geom_id], geom_pose[:3, 3], atol=2e-6
            )
            np.testing.assert_allclose(
                data.geom_xmat[geom_id].reshape(3, 3), geom_pose[:3, :3], atol=2e-6
            )
    assert source.read_bytes() == original


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
@pytest.mark.parametrize("feature", ["mimic", "planar", "floating"])
def test_unsupported_joint_semantics_fail_without_silent_loss(
    tmp_path, backend, feature
):
    source = tmp_path / "robot.urdf"
    source.write_text(
        f"""<robot><link name="a"/><link name="b"/>
      <joint name="j" type="{'revolute' if feature == 'mimic' else feature}">
      <parent link="a"/><child link="b"/>{'<mimic joint="x"/>' if feature == 'mimic' else ''}</joint></robot>"""
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    with pytest.raises(RuntimeError, match="Unsupported|not supported"):
        manager.run()


@pytest.mark.parametrize("backend", ["urdf2mjcf", "mujoco"])
def test_infinite_mass_is_rejected_even_when_mujoco_would_compile_it(tmp_path, backend):
    source = tmp_path / "robot.urdf"
    source.write_text("""<robot><link name="arm"><inertial><mass value="inf"/>
      <inertia ixx="1" iyy="1" izz="1"/></inertial></link></robot>""")
    manager = MujocoConversionManager(
        source, tmp_path / "output", backend=backend, auto_recalculate_inertia=False
    )
    with pytest.raises(RuntimeError, match="mass"):
        manager.run()
    assert not (manager.output_dir / "robot.xml").exists()


@pytest.mark.parametrize("backend", ["urdf2mjcf", "mujoco"])
def test_nonfinite_geometry_cannot_be_published(tmp_path, monkeypatch, backend):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    source = tmp_path / "robot.urdf"
    source.write_text(
        """<robot><link name="arm"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
      <visual><origin xyz="inf 0 0"/><geometry><box size="1 1 1"/></geometry></visual></link></robot>"""
    )
    manager = MujocoConversionManager(
        source, tmp_path / "output", backend=backend, auto_recalculate_inertia=False
    )
    with pytest.raises((RuntimeError, ValueError)):
        manager.run()
    assert not (manager.output_dir / "robot.xml").exists()


@pytest.mark.parametrize("backend", ["urdf2mjcf", "mujoco"])
@pytest.mark.parametrize(
    "bad_tensor",
    ['ixx="0" iyy="0" izz="0"', 'ixx="2" iyy="2" izz="2" ixy="1" ixz="1" iyz="1"'],
)
def test_automatic_recovery_only_changes_invalid_link(
    tmp_path, monkeypatch, backend, bad_tensor
):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    source = tmp_path / "robot.urdf"
    source.write_text(f"""<robot name="selective">
    <link name="calibrated"><inertial><origin xyz="0.1 0.2 0.3" rpy="0.3 -0.4 0.5"/>
    <mass value="2"/><inertia ixx="2" iyy="3" izz="4" ixy="0.1" ixz="0.2" iyz="0.3"/></inertial>
    <visual><geometry><box size="1 1 1"/></geometry></visual></link>
    <link name="invalid"><inertial><mass value="12"/><inertia {bad_tensor}/></inertial>
    <visual><geometry><box size="1 2 3"/></geometry></visual></link>
    <joint name="hinge" type="revolute"><parent link="calibrated"/><child link="invalid"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/></joint></robot>""")
    original = source.read_bytes()
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    body = model.body("calibrated")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, body.iquat)
    rotation = rotation.reshape(3, 3)
    tensor = np.array([[2, 0.1, 0.2], [0.1, 3, 0.3], [0.2, 0.3, 4]])
    expected = _rotation([0.3, -0.4, 0.5])
    np.testing.assert_allclose(
        rotation @ np.diag(body.inertia) @ rotation.T,
        expected @ tensor @ expected.T,
        atol=1e-10,
    )
    np.testing.assert_allclose(body.ipos, [0.1, 0.2, 0.3], atol=1e-12)
    np.testing.assert_allclose(model.body("invalid").inertia, [13, 10, 5], atol=1e-10)
    assert model.body_mass.sum() == 14
    assert manager.last_report["inertia_recovery_attempts"] == 1
    assert manager.last_report["inertia_recoveries"] == [
        {"status": "recalculated", "links": ["invalid"], "backend": backend}
    ]
    assert source.read_bytes() == original


@pytest.mark.parametrize("backend", ["urdf2mjcf", "mujoco"])
def test_diagonal_inertia_with_omitted_zero_terms_needs_no_recalculation(
    tmp_path, monkeypatch, backend
):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    source = tmp_path / "robot.urdf"
    source.write_text(
        """<robot name="diagonal"><link name="arm"><inertial><origin xyz="0.1 0.2 0.3"/>
    <mass value="2"/><inertia ixx="2" iyy="3" izz="4"/></inertial>
    <visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>"""
    )
    manager = MujocoConversionManager(
        source, tmp_path / "output", backend=backend, auto_recalculate_inertia=False
    )
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    np.testing.assert_allclose(model.body("arm").inertia, [2, 3, 4], atol=1e-12)
    np.testing.assert_allclose(model.body("arm").ipos, [0.1, 0.2, 0.3], atol=1e-12)
    assert manager.last_report["inertia_recovery_attempts"] == 0
