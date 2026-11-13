"""The standalone OBJ step must preserve source assets and failed rerun outputs."""

import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

import urdf2mjcf.obj_to_mjcf_converter as obj_module
from urdf2mjcf.obj_assets import BACKUP_PREFIX
from urdf2mjcf.obj_to_mjcf_converter import OBJ2MJCFImporter


def _model(directory, part="old.obj"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / part).write_text(part)
    root = ET.Element("mujoco")
    ET.SubElement(ET.SubElement(root, "asset"), "mesh", file=part)
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    ET.SubElement(body, "geom", mesh=Path(part).stem)
    ET.ElementTree(root).write(directory / f"{directory.name}.xml")


def _contents(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "kind",
    [
        "mesh_symlink",
        "material",
        "texture",
        "quoted_mtl",
        "literal_quote_mtl",
        "texture_symlink",
        "mtl_relative_texture",
        "other_model_output",
    ],
)
def test_obj_dependencies_inside_output_trees_fail_before_running_converter(
    tmp_path, monkeypatch, kind
):
    output = tmp_path / "arm"
    output.mkdir()
    source = tmp_path / "arm.obj"
    if kind == "mesh_symlink":
        (output / "source.obj").write_bytes(b"original mesh")
        source.symlink_to(output / "source.obj")
    elif kind == "material":
        (output / "colors.mtl").write_text("newmtl paint\nKd 1 0 0\n")
        source.write_text("mtllib arm/colors.mtl\n")
    elif kind == "quoted_mtl":
        (output / "colors with spaces.mtl").write_text("newmtl paint\nKd 1 0 0\n")
        source.write_text('mtllib "arm/colors with spaces.mtl"\n')
    elif kind == "literal_quote_mtl":
        (output / "color's.mtl").write_text("newmtl paint\nKd 1 0 0\n")
        source.write_text("mtllib arm/color's.mtl\n")
    elif kind == "other_model_output":
        other = tmp_path / "leg"
        other.mkdir()
        (other / "source.obj").write_bytes(b"other model's source")
        (tmp_path / "leg.obj").touch()
        source.symlink_to(other / "source.obj")
    elif kind == "texture_symlink":
        (output / "texture.png").write_bytes(b"original texture")
        (tmp_path / "texture.png").symlink_to(output / "texture.png")
        (tmp_path / "colors.mtl").write_text("newmtl paint\nmap_Kd texture.png\n")
        source.write_text("mtllib colors.mtl\n")
    elif kind == "mtl_relative_texture":
        (output / "texture.png").write_bytes(b"original texture")
        material = tmp_path / "materials/colors.mtl"
        material.parent.mkdir()
        material.write_text("newmtl paint\nmap_Kd ../arm/texture.png\n")
        source.write_text("mtllib materials/colors.mtl\n")
    else:
        (output / "texture.png").write_bytes(b"original texture")
        (tmp_path / "colors.mtl").write_text(
            "newmtl paint\nKd 1 1 1\nmap_Kd arm/texture.png\n"
        )
        source.write_text("mtllib colors.mtl\n")
    original = _contents(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("A referenced input would be removed by obj2mjcf")

    monkeypatch.setattr(obj_module.subprocess, "run", unexpected)
    assert not OBJ2MJCFImporter(tmp_path).handle_dir(tmp_path)
    assert _contents(tmp_path) == original


@pytest.mark.parametrize(
    "failure", ["process", "missing_model", "malformed_xml", "missing_part", "escape"]
)
def test_failed_obj_directory_restores_every_previous_model(
    tmp_path, monkeypatch, failure
):
    for name in ("arm", "leg"):
        (tmp_path / f"{name}.obj").touch()
        _model(tmp_path / name)
    original = _contents(tmp_path)

    def run(*args, **kwargs):
        _model(tmp_path / "arm", "fresh.obj")
        if failure == "process":
            raise subprocess.CalledProcessError(
                1, "obj2mjcf", output="failed after arm"
            )
        if failure != "missing_model":
            _model(tmp_path / "leg", "fresh.obj")
        model = tmp_path / "arm/arm.xml"
        if failure == "malformed_xml":
            model.write_text("broken XML")
        elif failure == "missing_part":
            (tmp_path / "arm/fresh.obj").unlink()
        elif failure == "escape":
            tree = ET.parse(model)
            tree.find("asset/mesh").set("file", "../arm.obj")
            tree.write(model)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(obj_module.subprocess, "run", run)
    assert not OBJ2MJCFImporter(tmp_path).handle_dir(tmp_path)
    assert _contents(tmp_path) == original
    assert not list(tmp_path.glob(BACKUP_PREFIX + "*"))


def test_successful_obj_rerun_replaces_models_and_removes_old_parts(
    tmp_path, monkeypatch
):
    source = tmp_path / "arm.OBJ"
    source.write_bytes(b"source OBJ")
    _model(tmp_path / "arm")

    def run(*args, **kwargs):
        assert source.with_suffix(".obj").read_bytes() == source.read_bytes()
        _model(tmp_path / "arm", "fresh.obj")
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(obj_module.subprocess, "run", run)
    importer = OBJ2MJCFImporter(tmp_path)
    assert importer.run(strict=True) == [tmp_path]
    assert importer.get_obj_variant_map() == {"arm": ["arm/fresh.obj"]}
    assert source.read_bytes() == b"source OBJ"
    assert not source.with_suffix(".obj").exists()
    assert not (tmp_path / "arm/old.obj").exists()
    assert not list(tmp_path.glob(BACKUP_PREFIX + "*"))


def test_backup_preparation_failure_restores_already_moved_models(
    tmp_path, monkeypatch
):
    for name in ("arm", "leg"):
        (tmp_path / f"{name}.obj").touch()
        _model(tmp_path / name)
    original = _contents(tmp_path)
    rename = Path.rename

    def fail_second(self, target):
        if self == tmp_path / "leg":
            raise OSError("simulated second backup failure")
        return rename(self, target)

    def unexpected(*args, **kwargs):
        pytest.fail("Conversion must not start after incomplete backup preparation")

    monkeypatch.setattr(Path, "rename", fail_second)
    monkeypatch.setattr(obj_module.subprocess, "run", unexpected)
    assert not OBJ2MJCFImporter(tmp_path).handle_dir(tmp_path)
    assert _contents(tmp_path) == original
    assert not list(tmp_path.glob(BACKUP_PREFIX + "*"))


def test_failed_rollback_retains_original_files_and_recovery_manifest(
    tmp_path, monkeypatch
):
    (tmp_path / "arm.obj").touch()
    _model(tmp_path / "arm")
    original = _contents(tmp_path / "arm")
    rename = Path.rename

    def cannot_restore(self, target):
        if self.name == "0" and self.parent.name.startswith(BACKUP_PREFIX):
            raise OSError("simulated restore failure")
        return rename(self, target)

    def fail_conversion(*args, **kwargs):
        _model(tmp_path / "arm", "fresh.obj")
        raise subprocess.CalledProcessError(1, "obj2mjcf", output="invalid OBJ")

    monkeypatch.setattr(Path, "rename", cannot_restore)
    monkeypatch.setattr(obj_module.subprocess, "run", fail_conversion)
    importer = OBJ2MJCFImporter(tmp_path)
    assert not importer.handle_dir(tmp_path)
    backups = list(tmp_path.glob(BACKUP_PREFIX + "*"))
    assert len(backups) == 1
    assert _contents(backups[0] / "0") == original
    assert json.loads((backups[0] / "manifest.json").read_text()) == [
        {"output": str(tmp_path / "arm"), "backup": "0"}
    ]
    assert importer.get_obj_variant_map() == {"arm": ["arm.obj"]}
