"""Invalid physical inputs must be diagnosed before import or publication."""

import sys
import xml.etree.ElementTree as ET

import pytest

from urdf2mjcf.batch_conversion import BatchConversionManager
from urdf2mjcf.inspection import inspect_urdf
from urdf2mjcf.native_mujoco_converter import NativeMujocoConverter
from urdf2mjcf.urdf_to_mujoco_converter import main

ROBOT = """<robot name="validation">
<link name="base"/>
<link name="arm"><inertial><mass value="2"/><inertia ixx="2" iyy="3" izz="4"/></inertial>
<visual><geometry><box size="1 2 3"/></geometry></visual></link>
<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
<axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10"/>
<dynamics damping="0.2" friction="0.1"/></joint></robot>"""


@pytest.mark.parametrize(
    "path, attributes, code",
    [
        ("joint/axis", {"xyz": "0 0 0"}, "invalid_joint_axis"),
        ("joint/axis", {"xyz": "nan 0 0"}, "invalid_joint_axis"),
        ("joint/limit", {"lower": "2"}, "invalid_joint_limits"),
        ("joint/limit", {"lower": "1"}, "invalid_joint_limits"),
        ("joint/limit", {"upper": "inf"}, "invalid_joint_limits"),
        ("joint/limit", {"effort": "-2"}, "invalid_joint_limits"),
        ("joint/limit", {"velocity": "-1"}, "invalid_joint_limits"),
        ("joint/dynamics", {"damping": "-1"}, "invalid_joint_dynamics"),
        ("joint/dynamics", {"friction": "inf"}, "invalid_joint_dynamics"),
        ("link/inertial/mass", {"value": "nan"}, "invalid_inertia"),
        ("link/inertial/mass", {"value": "-1"}, "invalid_inertia"),
        ("link/inertial/mass", {"value": "0"}, "invalid_inertia"),
        ("link/inertial/inertia", {"ixx": "0"}, "invalid_inertia"),
        ("link/inertial/inertia", {"izz": "10"}, "invalid_inertia"),
        ("link/visual/geometry/box", {"size": "1 -2 3"}, "invalid_geometry"),
        ("link/visual/geometry/box", {"size": "1 2 inf"}, "invalid_geometry"),
    ],
)
def test_check_and_converter_reject_same_physical_errors(
    tmp_path, path, attributes, code
):
    root = ET.fromstring(ROBOT)
    root.find(path).attrib.update(attributes)
    source = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(source)
    original = source.read_bytes()
    report = inspect_urdf(source)
    assert report["status"] == "failed"
    errors = [issue for issue in report["issues"] if issue["severity"] == "error"]
    assert [issue["code"] for issue in errors] == [code]
    assert "arm" in errors[0]["message"] or "hinge" in errors[0]["message"]
    converter = NativeMujocoConverter(source, tmp_path / "output", tmp_path)
    with pytest.raises(ValueError, match=code):
        converter.prepare_urdf_tree()
    assert source.read_bytes() == original
    assert not converter.output_dir.exists()


def test_check_collects_inertia_origins_and_shapes_without_changing_files(tmp_path):
    root = ET.fromstring(ROBOT)
    tensor = root.find("link/inertial/inertia")
    # Coordinate diagonals pass the triangle rule, but eigenvalues are 1, 1, 4.
    tensor.attrib.update(ixx="2", iyy="2", izz="2", ixy="1", ixz="1", iyz="1")
    ET.SubElement(root.find("joint"), "origin", xyz="1 2")
    ET.SubElement(root.find("link/visual"), "origin", rpy="nan 0 0")
    ET.SubElement(root.find("link/visual/geometry"), "sphere", radius="1")
    collision = ET.SubElement(root.find("link[@name='arm']"), "collision")
    ET.SubElement(
        ET.SubElement(collision, "geometry"), "cylinder", radius="1", length="0"
    )
    source = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(source)
    original = source.read_bytes()
    report = inspect_urdf(source)
    assert report["status"] == "failed"
    assert len(report["issues"]) == 5
    inertia = next(
        issue for issue in report["issues"] if issue["code"] == "invalid_inertia"
    )
    assert "triangle" in inertia["message"]
    assert inertia["link"] == "arm"
    assert {issue.get("kind") for issue in report["issues"]} >= {"visual", "collision"}
    assert list(tmp_path.iterdir()) == [source]
    assert source.read_bytes() == original


def test_velocity_warning_keeps_single_and_batch_checks_successful(
    tmp_path, monkeypatch
):
    source = tmp_path / "robot.urdf"
    source.write_text(ROBOT.replace('effort="10"', 'effort="10" velocity="2"'))
    report = inspect_urdf(source)
    assert report["status"] == "passed"
    assert [(issue["severity"], issue["code"]) for issue in report["issues"]] == [
        ("warning", "velocity_limit_unenforced")
    ]
    monkeypatch.setattr(sys, "argv", ["toolkit", str(source), "--check"])
    assert main() == 0
    batch = BatchConversionManager(tmp_path, tmp_path / "output", check_only=True)
    assert batch.run()["summary"]["succeeded"] == 1
    assert not batch.output_dir.exists()


def test_massless_static_frame_and_nonunit_axis_are_valid_inputs(tmp_path):
    root = ET.fromstring(ROBOT)
    inertial = ET.SubElement(root.find("link[@name='base']"), "inertial")
    ET.SubElement(inertial, "mass", value="0")
    ET.SubElement(inertial, "inertia", ixx="0", iyy="0", izz="0")
    root.find("joint/axis").set("xyz", "0 20 0")
    source = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(source)
    assert inspect_urdf(source)["status"] == "passed"


@pytest.mark.parametrize(
    "parent_path, child_tag",
    [
        ("joint", "axis"),
        ("joint", "origin"),
        ("joint", "limit"),
        ("joint", "dynamics"),
        ("link[@name='arm']", "inertial"),
        ("link/inertial", "mass"),
        ("link/inertial", "inertia"),
        ("link/inertial", "origin"),
        ("link/visual", "geometry"),
        ("link/visual", "origin"),
    ],
)
def test_repeated_singleton_definitions_are_rejected(tmp_path, parent_path, child_tag):
    import copy

    from urdf2mjcf.urdf_inertia_calculator import URDFInertiaCalculator

    root = ET.fromstring(ROBOT)
    parent = root.find(parent_path)
    child = parent.find(child_tag)
    if child is None:
        child = ET.SubElement(parent, child_tag)
    parent.append(copy.deepcopy(child))
    source = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(source)
    original = source.read_bytes()
    report = inspect_urdf(source)
    assert report["status"] == "failed"
    assert any(
        issue["code"] == "ambiguous_definition" and issue["element"] == child_tag
        for issue in report["issues"]
    )
    converter = NativeMujocoConverter(source, tmp_path / "output", tmp_path)
    with pytest.raises(ValueError, match="ambiguous_definition"):
        converter.prepare_urdf_tree()
    calculator = URDFInertiaCalculator(source)
    assert not calculator.update_inertia()
    assert "ambiguous_definition" in calculator.last_error
    assert source.read_bytes() == original
    assert not source.with_suffix(".urdf.backup").exists()


@pytest.mark.parametrize("origin_path", ["joint", "link/inertial", "link/visual"])
def test_quaternion_origins_cannot_be_silently_ignored(tmp_path, origin_path):
    root = ET.fromstring(ROBOT)
    root.set("version", "1.1")
    ET.SubElement(root.find(origin_path), "origin", quat_xyzw="0 0 1 0")
    source = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(source)
    report = inspect_urdf(source)
    assert report["status"] == "failed"
    assert any("quat_xyzw" in issue["message"] for issue in report["issues"])
    converter = NativeMujocoConverter(source, tmp_path / "output", tmp_path)
    with pytest.raises(ValueError, match="quat_xyzw"):
        converter.prepare_urdf_tree()
