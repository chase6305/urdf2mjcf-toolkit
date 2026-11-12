"""Material errors must agree between inspection and every conversion backend."""

import xml.etree.ElementTree as ET

import mujoco
import pytest

from urdf2mjcf.inspection import inspect_urdf
from urdf2mjcf.physics import validate_compiled_physics
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def _robot(material, *, global_material=False):
    root = ET.fromstring(
        '<robot name="material_check"><link name="body"><inertial>'
        '<mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>'
        '<visual><geometry><box size="1 1 1"/></geometry></visual></link></robot>'
    )
    visual = root.find("link/visual")
    if global_material:
        root.append(material)
        ET.SubElement(visual, "material", name=material.get("name", ""))
    else:
        visual.append(material)
    return root


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf", "auto"])
@pytest.mark.parametrize(
    "problem,global_material,code",
    [
        ("nan 0 0 1", False, "invalid_material"),
        ("0 inf 0 1", True, "invalid_material"),
        ("2 0 0 1", False, "invalid_material"),
        ("0 0 0 -0.1", True, "invalid_material"),
        ("1 0 0", False, "invalid_material"),
        ("", True, "invalid_material"),
        ("duplicate_color", False, "ambiguous_definition"),
        ("duplicate_texture", True, "ambiguous_definition"),
        ("duplicate_visual_material", False, "ambiguous_definition"),
        ("duplicate_global_material", True, "duplicate_material"),
    ],
)
def test_material_errors_fail_before_assets_and_preserve_outputs(
    tmp_path, monkeypatch, backend, problem, global_material, code
):
    material = ET.fromstring(
        '<material name="paint"><color rgba="1 0 0 1"/></material>'
    )
    root = _robot(material, global_material=global_material)
    if problem == "duplicate_color":
        ET.SubElement(material, "color", rgba="0 0 1 1")
    elif problem == "duplicate_texture":
        for name in ("first.png", "second.png"):
            ET.SubElement(material, "texture", filename=name)
    elif problem == "duplicate_visual_material":
        root.find("link/visual").append(
            ET.fromstring('<material name="other"><color rgba="0 0 1 1"/></material>')
        )
    elif problem == "duplicate_global_material":
        root.append(
            ET.fromstring('<material name="paint"><color rgba="0 0 1 1"/></material>')
        )
    else:
        material.find("color").set("rgba", problem)
    source = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(source)
    original = source.read_bytes()
    output = tmp_path / "output"
    meshes = output / "meshes"
    meshes.mkdir(parents=True)
    (output / "robot.xml").write_bytes(b"previous model")
    (meshes / "keep.obj").write_bytes(b"previous assets")

    check = inspect_urdf(source)
    assert check["status"] == "failed"
    assert [issue["code"] for issue in check["issues"]] == [code]
    assert "paint" in check["issues"][0]["message"]
    manager = MujocoConversionManager(source, output, backend=backend)

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid material reached asset copying")

    monkeypatch.setattr(manager, "_copy_tree", unexpected)
    with pytest.raises(RuntimeError, match=code):
        manager.run()
    assert manager.last_report["stage"] == "input_validation"
    assert manager.last_report["attempts"] == []
    assert manager.last_report["issues"] == check["issues"]
    assert source.read_bytes() == original
    assert (output / "robot.xml").read_bytes() == b"previous model"
    assert (meshes / "keep.obj").read_bytes() == b"previous assets"


def test_inline_shared_material_names_and_color_boundaries_are_valid(tmp_path):
    material = ET.fromstring(
        '<material name="paint"><color rgba="0 1 0 0"/></material>'
    )
    robot = _robot(material)
    visual = ET.fromstring(ET.tostring(robot.find("link/visual")))
    robot.find("link").append(visual)
    source = tmp_path / "robot.urdf"
    ET.ElementTree(robot).write(source)

    assert inspect_urdf(source)["status"] == "passed"


@pytest.mark.parametrize("target", ["geom", "material"])
def test_compiler_acceptance_does_not_hide_nonfinite_visual_values(
    tmp_path, monkeypatch, target
):
    monkeypatch.chdir(tmp_path)  # Keep MuJoCo's expected NaN warning log temporary.
    if target == "geom":
        xml = (
            '<mujoco><worldbody><geom size="1" rgba="nan 0 0 1"/></worldbody></mujoco>'
        )
        array = "geom_rgba"
    else:
        xml = '<mujoco><asset><material name="paint" rgba="nan 0 0 1"/></asset><worldbody><geom size="1" material="paint"/></worldbody></mujoco>'
        array = "mat_rgba"
    model = mujoco.MjModel.from_xml_string(xml)

    with pytest.raises(ValueError, match=array):
        validate_compiled_physics(model)
