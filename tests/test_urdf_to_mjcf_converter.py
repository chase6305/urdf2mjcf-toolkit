import os
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

import urdf2mjcf.urdf_to_mjcf_converter as converter_module
from urdf2mjcf.physics import rename_reserved_body_names
from urdf2mjcf.urdf_to_mjcf_converter import URDF2MJCFConverter


def test_world_alias_updates_body_references_without_renaming_joints_or_geoms():
    root = ET.fromstring(
        '<mujoco><worldbody><body name="world" pos="1 2 3"><joint name="world"/>'
        '<geom name="world"/><body name="world_urdf"/></body>'
        '<camera target="world"/><light target="world"/></worldbody>'
        '<contact><exclude body1="world" body2="world_urdf"/></contact>'
        '<sensor><subtreecom body="world"/><framepos objtype="body" objname="world"/>'
        '<framequat objtype="xbody" objname="world" reftype="body" refname="world"/>'
        '<framepos objtype="geom" objname="world"/></sensor>'
        '<actuator><motor joint="world"/></actuator></mujoco>'
    )
    aliases = rename_reserved_body_names(root)
    assert aliases == {"world": "world_urdf_1"}
    body = root.find("worldbody/body")
    assert body.attrib == {"name": "world_urdf_1", "pos": "1 2 3"}
    assert body.find("joint").get("name") == "world"
    assert body.find("geom").get("name") == "world"
    assert root.find("actuator/motor").get("joint") == "world"
    assert root.find("contact/exclude").attrib == {
        "body1": "world_urdf_1",
        "body2": "world_urdf",
    }
    assert root.find("sensor/subtreecom").get("body") == "world_urdf_1"
    assert root.find("sensor/framepos").get("objname") == "world_urdf_1"
    quaternion = root.find("sensor/framequat")
    assert quaternion.get("objname") == quaternion.get("refname") == "world_urdf_1"
    assert root.findall("sensor/framepos")[1].get("objname") == "world"
    assert root.find("worldbody/camera").get("target") == "world_urdf_1"
    assert root.find("worldbody/light").get("target") == "world_urdf_1"
    assert rename_reserved_body_names(root) == {}


def _converter(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="test"><link name="base"/></robot>', encoding="utf-8"
    )
    return URDF2MJCFConverter(source, tmp_path / "output", tmp_path / "assets")


@pytest.mark.parametrize("kind", ["checkout", "relative_checkout", "combined_install"])
def test_child_pythonpath_does_not_shadow_the_upstream_converter(
    tmp_path, monkeypatch, kind
):
    converter = _converter(tmp_path)
    checkout = converter.assets_dir / "checkout"
    package = checkout / "urdf2mjcf"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "urdf_to_mjcf_converter.py").touch()
    helpers = tmp_path / "user_helpers"
    helpers.mkdir()
    if kind == "combined_install":
        (package / "convert.py").touch()
    entry = "checkout" if kind == "relative_checkout" else str(checkout)
    original = os.pathsep.join([entry, str(helpers)])
    monkeypatch.setenv("PYTHONPATH", original)
    monkeypatch.setenv("URDF2MJCF_TEST_SETTING", "preserved")

    def run(*args, **kwargs):
        expected = original if kind == "combined_install" else str(helpers)
        assert kwargs["env"]["PYTHONPATH"] == expected
        assert kwargs["env"]["URDF2MJCF_TEST_SETTING"] == "preserved"
        converter.output_xml.write_text('<mujoco model="fresh"/>')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", run)
    assert converter.convert()
    assert os.environ["PYTHONPATH"] == original


def test_success_without_fresh_output_is_rejected(tmp_path, monkeypatch):
    converter = _converter(tmp_path)
    converter.output_dir.mkdir()
    stale_output = converter.output_dir / "robot.xml"
    stale_output.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        converter_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    assert not converter.convert()
    assert stale_output.read_text(encoding="utf-8") == "stale"


def test_fresh_output_is_accepted(tmp_path, monkeypatch):
    converter = _converter(tmp_path)

    def fake_run(*args, **kwargs):
        converter.output_xml.write_text('<mujoco model="fresh" />', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", fake_run)

    assert converter.convert()
    assert ET.parse(converter.get_output_xml()).getroot().get("model") == "fresh"


def test_staged_urdf_prefers_converted_obj(tmp_path, monkeypatch):
    converter = _converter(tmp_path)
    source_mesh = converter.assets_dir / "visual" / "arm.stl"
    source_mesh.parent.mkdir(parents=True)
    source_mesh.touch()
    source_mesh.with_suffix(".obj").touch()
    converter.urdf_path.write_text(
        '<robot name="test"><link name="arm"><visual><geometry>'
        '<mesh filename="meshes/visual/arm.stl" />'
        "</geometry></visual></link></robot>",
        encoding="utf-8",
    )

    def fake_run(command, **kwargs):
        staged_urdf = Path(command[1])
        mesh = ET.parse(staged_urdf).getroot().find(".//mesh")
        assert mesh.get("filename").endswith("/visual/arm.obj")
        converter.output_xml.write_text(
            '<mujoco model="fresh"><worldbody><body name="arm">'
            '<geom type="mesh" mesh="arm.obj" class="visual"/>'
            "</body></worldbody></mujoco>",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", fake_run)

    assert converter.convert()


def test_subprocess_paths_are_absolute_and_staging_is_cleaned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = '<robot name="test"><link name="base"/></robot>'
    Path("robot.urdf").write_text(original)
    converter = URDF2MJCFConverter(
        Path("robot.urdf"), Path("output"), Path("output/assets")
    )
    staged = []

    def fake_run(command, **kwargs):
        assert Path(command[1]).is_absolute()
        assert Path(command[-1]).is_absolute()
        assert Path(kwargs["cwd"]).is_absolute()
        staged.append(Path(command[1]))
        converter.output_xml.write_text('<mujoco model="fresh" />')
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", fake_run)
    assert converter.convert()
    assert not staged[0].exists()
    assert Path("robot.urdf").read_text() == original


def test_missing_mesh_fails_before_external_command(tmp_path, monkeypatch):
    converter = _converter(tmp_path)
    converter.urdf_path.write_text(
        '<robot><link name="base"><visual><geometry><mesh filename="missing.obj" /></geometry></visual></link></robot>'
    )
    calls = []
    monkeypatch.setattr(
        converter_module.subprocess, "run", lambda *a, **k: calls.append(a)
    )
    assert not converter.convert()
    assert calls == []
