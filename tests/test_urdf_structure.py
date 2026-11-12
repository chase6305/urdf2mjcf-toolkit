"""Both importers must enforce the same single-tree topology as inspection."""

import xml.etree.ElementTree as ET

import pytest

from urdf2mjcf.inspection import inspect_urdf
from urdf2mjcf.native_mujoco_converter import NativeMujocoConverter
from urdf2mjcf.urdf_to_mjcf_converter import URDF2MJCFConverter
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def _joint(name, parent, child):
    return (
        f'<joint name="{name}" type="fixed">'
        f'<parent link="{parent}"/><child link="{child}"/></joint>'
    )


@pytest.mark.parametrize("converter_type", [NativeMujocoConverter, URDF2MJCFConverter])
@pytest.mark.parametrize(
    "body, code",
    [
        ("", "missing_links"),
        ('<link name="a"/><link name="b"/>', "invalid_roots"),
        ('<link name="a"/><link name="a"/>', "duplicate_name"),
        ('<link name=" "/>', "missing_name"),
        ('<link name="a"/>' + _joint("j", "a", "a"), "joint_cycle"),
        ('<link name="a"/>' + _joint("j", "missing", "a"), "invalid_joint_link"),
        (
            '<link name="a"/><link name="b"/><link name="c"/>'
            + _joint("j1", "a", "c")
            + _joint("j2", "b", "c"),
            "multiple_parents",
        ),
        (
            '<link name="a"/><link name="b"/><link name="c"/>'
            + _joint("same", "a", "b")
            + _joint("same", "b", "c"),
            "duplicate_name",
        ),
        (
            '<link name="a"/><link name="b"/><link name="c"/>'
            + _joint("j1", "b", "c")
            + _joint("j2", "c", "b"),
            "joint_cycle",
        ),
        (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="fixed"><child link="b"/></joint>',
            "invalid_joint_link",
        ),
        (
            '<link name="a"/><link name="b"/>'
            '<joint name="j" type="fixed"><parent link="a"/>'
            '<parent link="b"/><child link="b"/></joint>',
            "invalid_joint_link",
        ),
    ],
)
def test_inspection_and_importers_reject_malformed_graphs(
    tmp_path, converter_type, body, code
):
    source = tmp_path / "robot.urdf"
    source.write_text(f'<robot name="invalid">{body}</robot>')
    original = source.read_bytes()
    report = inspect_urdf(source)
    assert report["status"] == "failed"
    assert code in {issue["code"] for issue in report["issues"]}
    converter = converter_type(source, tmp_path / "output", tmp_path / "assets")
    with pytest.raises(ValueError, match=code):
        converter.prepare_urdf_tree()
    assert source.read_bytes() == original
    assert not converter.output_dir.exists()


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf", "auto"])
def test_invalid_topology_does_not_replace_previous_model(tmp_path, backend):
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="valid"><link name="mass_mount">'
        '<visual><geometry><box size="1 1 1"/></geometry></visual>'
        "</link></robot>"
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    output = manager.run()
    before = {
        p.relative_to(manager.output_dir): p.read_bytes()
        for p in manager.output_dir.rglob("*")
        if p.is_file()
    }
    root = ET.parse(source).getroot()
    ET.SubElement(root, "link", name="disconnected")
    ET.ElementTree(root).write(source)
    original = source.read_bytes()
    manager.backend = backend
    with pytest.raises(RuntimeError, match="invalid_roots"):
        manager.run()
    assert manager.last_report["status"] == "failed"
    assert manager.last_report["inertia_recovery_attempts"] == 0
    assert source.read_bytes() == original
    assert output.is_file()
    assert before == {
        p.relative_to(manager.output_dir): p.read_bytes()
        for p in manager.output_dir.rglob("*")
        if p.is_file()
    }
