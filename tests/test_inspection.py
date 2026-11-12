import xml.etree.ElementTree as ET

import pytest

from urdf2mjcf.inspection import inspect_urdf


def test_inspection_resolves_original_meshes_without_writing(tmp_path):
    source = tmp_path / "example/urdf/robot.urdf"
    source.parent.mkdir(parents=True)
    mesh = tmp_path / "example/meshes/arm.stl"
    mesh.parent.mkdir()
    mesh.touch()
    mesh.with_suffix(".obj").write_text("stale conversion")
    source.write_text("""<robot name="test"><link name="arm">
      <visual><geometry><mesh filename="package://example/meshes/arm.stl"/></geometry></visual>
      <collision><geometry><mesh filename="../meshes/arm.stl"/></geometry></collision>
    </link></robot>""")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    report = inspect_urdf(source)
    assert report["status"] == "passed"
    assert report["statistics"] == {
        "links": 1,
        "joints": 0,
        "root_links": ["arm"],
        "mesh_references": 2,
        "unique_mesh_files": 1,
    }
    assert {entry["resolved_path"] for entry in report["meshes"]} == {str(mesh)}
    assert before == {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize(
    "xml, codes",
    [
        ('<robot><link name="a"/><link name="a"/></robot>', {"duplicate_name"}),
        ('<robot><link name="a"/><link name="b"/></robot>', {"invalid_roots"}),
        (
            '<robot><link name="a"/><joint name="j"><parent link="a"/><child link="b"/></joint></robot>',
            {"invalid_joint_link"},
        ),
        (
            '<robot><link name="a"/><joint name="j"><parent link="a"/><child link="a"/></joint></robot>',
            {"invalid_roots", "joint_cycle"},
        ),
        ("<robot><link/></robot>", {"missing_name"}),
        ("<robot/>", {"missing_links"}),
        ("<mujoco/>", {"invalid_input"}),
        ("malformed XML", {"invalid_input"}),
    ],
)
def test_inspection_reports_invalid_topology(tmp_path, xml, codes):
    source = tmp_path / "robot.urdf"
    source.write_text(xml)
    report = inspect_urdf(source)
    assert report["status"] == "failed"
    assert codes <= {issue["code"] for issue in report["issues"]}


def test_inspection_collects_missing_ambiguous_meshes_and_bad_scales(tmp_path):
    for directory in ("left", "right"):
        folder = tmp_path / directory
        folder.mkdir()
        (folder / "arm.obj").touch()
    source = tmp_path / "robot.urdf"
    source.write_text("""<robot><link name="arm">
      <visual><geometry><mesh filename="arm.obj" scale="nan 1 1"/></geometry></visual>
      <visual><geometry><mesh filename="absent.stl"/></geometry></visual>
    </link></robot>""")
    report = inspect_urdf(source)
    messages = [issue["message"] for issue in report["issues"]]
    assert report["status"] == "failed"
    assert len(messages) == 3
    assert any("Ambiguous" in message for message in messages)
    assert any("not found" in message for message in messages)
    assert report["statistics"]["unique_mesh_files"] == 0


def test_inspection_handles_long_joint_chains_without_recursion(tmp_path):
    robot = ET.Element("robot", name="chain")
    for index in range(1500):
        ET.SubElement(robot, "link", name=f"link{index}")
        if index:
            joint = ET.SubElement(robot, "joint", name=f"joint{index}", type="fixed")
            ET.SubElement(joint, "parent", link=f"link{index-1}")
            ET.SubElement(joint, "child", link=f"link{index}")
    source = tmp_path / "robot.urdf"
    ET.ElementTree(robot).write(source)
    assert inspect_urdf(source)["status"] == "passed"


def test_inspection_reports_mesh_symlink_loops_without_raising(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot><link name="arm"><visual><geometry><mesh filename="loop.obj"/>'
        "</geometry></visual></link></robot>"
    )
    (tmp_path / "loop.obj").symlink_to("loop.obj")
    original = source.read_bytes()
    report = inspect_urdf(source)
    assert report["status"] == "failed"
    assert [issue["code"] for issue in report["issues"]] == ["unresolved_mesh"]
    assert report["meshes"][0]["resolved_path"] is None
    assert "loop.obj" in report["issues"][0]["message"]
    assert source.read_bytes() == original
    assert (tmp_path / "loop.obj").is_symlink()
