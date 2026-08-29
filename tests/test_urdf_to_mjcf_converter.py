import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import urdf2mjcf.urdf_to_mjcf_converter as converter_module
from urdf2mjcf.urdf_to_mjcf_converter import URDF2MJCFConverter


def _converter(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text('<robot name="test" />', encoding="utf-8")
    return URDF2MJCFConverter(source, tmp_path / "output", tmp_path / "assets")


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
        converter.output_xml.write_text("fresh", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", fake_run)

    assert converter.convert()
    assert converter.get_output_xml().read_text(encoding="utf-8") == "fresh"


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
        converter.output_xml.write_text("fresh", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(converter_module.subprocess, "run", fake_run)

    assert converter.convert()
