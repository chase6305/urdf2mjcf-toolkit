import pytest

import urdf2mjcf.urdf_to_mujoco_converter as manager_module
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def test_manager_rejects_missing_input_without_creating_output(tmp_path):
    output = tmp_path / "output"

    with pytest.raises(FileNotFoundError, match="URDF file does not exist"):
        MujocoConversionManager(tmp_path / "missing.urdf", output)

    assert not output.exists()


def test_manager_rejects_non_urdf_input(tmp_path):
    source = tmp_path / "robot.xml"
    source.touch()

    with pytest.raises(ValueError, match=r"Expected a \.urdf"):
        MujocoConversionManager(source, tmp_path / "output")


@pytest.mark.parametrize("through_symlink", [False, True])
def test_input_inside_output_assets_is_preserved(tmp_path, through_symlink):
    output = tmp_path / "output"
    source = output / "meshes/robot.urdf"
    source.parent.mkdir(parents=True)
    original = b'<robot name="test"><link name="base"/></robot>'
    source.write_bytes(original)
    input_path = source
    if through_symlink:
        input_path = tmp_path / "alias.urdf"
        input_path.symlink_to(source)

    with pytest.raises(ValueError, match="contains the input URDF"):
        MujocoConversionManager(input_path, output, backend="mujoco").run()

    assert source.read_bytes() == original
    assert list(source.parent.iterdir()) == [source]
    assert not (output / "robot.xml").exists()


def test_manager_resolves_capitalized_asset_directories(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text('<robot name="test" />', encoding="utf-8")
    visual = tmp_path / "Visual"
    visual.mkdir()

    manager = MujocoConversionManager(source, tmp_path / "output")

    assert manager.visual_dir == visual


def test_inertia_recovery_does_not_modify_source_urdf(tmp_path, monkeypatch):
    source = tmp_path / "robot.urdf"
    original = '<robot name="original" />'
    source.write_text(original, encoding="utf-8")
    manager = MujocoConversionManager(source, tmp_path / "output")

    class FakeCalculator:
        def __init__(self, urdf_path, mesh_search_dir):
            assert urdf_path != source
            assert mesh_search_dir == source.parent
            self.urdf_path = urdf_path

        def update_inertia(self, *, only_invalid=False):
            assert only_invalid
            self.updated_links = ["repaired"]
            self.urdf_path.write_text('<robot name="repaired" />', encoding="utf-8")
            return True

    class FakeConverter:
        def __init__(self):
            self.urdf_path = source
            self.calls = 0
            self.output = tmp_path / "output" / "robot.xml"

        def convert(self):
            self.calls += 1
            if self.calls == 2:
                assert self.urdf_path.read_text(encoding="utf-8") == (
                    '<robot name="repaired" />'
                )
                self.output.touch()
                return True
            return False

        def get_output_xml(self):
            return self.output

    monkeypatch.setattr(manager_module, "URDFInertiaCalculator", FakeCalculator)
    converter = FakeConverter()

    assert manager._convert_urdf_with_recovery(converter)
    assert source.read_text(encoding="utf-8") == original
    assert converter.urdf_path == source


def test_relative_input_has_named_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "robot.urdf"
    source.write_text('<robot name="test" />')
    manager = MujocoConversionManager("robot.urdf")
    assert manager.urdf_path == source
    assert manager.output_dir == tmp_path.parent / f"{tmp_path.name}_mjcf"


def test_nested_asset_layout_is_recognized(tmp_path):
    source = tmp_path / "urdf/robot.urdf"
    source.parent.mkdir()
    source.write_text('<robot name="test" />')
    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    manager = MujocoConversionManager(source, tmp_path / "output")
    assert manager.visual_dir == visual


def test_output_inside_source_assets_is_rejected_before_creation(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text('<robot name="test" />')
    visual = tmp_path / "visual"
    visual.mkdir()
    output = visual / "converted"
    with pytest.raises(ValueError, match="overlaps input assets"):
        MujocoConversionManager(source, output)
    assert not output.exists()


@pytest.mark.parametrize("format_name", ["DAE", "GLB", "STL"])
def test_failed_mesh_stage_aborts_pipeline(tmp_path, monkeypatch, format_name):
    source = tmp_path / "robot.urdf"
    source.write_text('<robot name="test" />')
    manager = MujocoConversionManager(source, tmp_path / "output")
    monkeypatch.setattr(
        manager_module.DAE2OBJConverter,
        "convert_directory",
        lambda *a, **k: {"arm.dae": (False, [])} if format_name == "DAE" else {},
    )
    monkeypatch.setattr(
        manager_module.GLB2OBJConverter,
        "convert_directory",
        lambda *a, **k: {"arm.glb": False} if format_name == "GLB" else {},
    )
    monkeypatch.setattr(
        manager_module.MeshToOBJConverter,
        "convert_directory",
        lambda *a, **k: (
            {source.with_suffix(".stl"): None} if format_name == "STL" else {}
        ),
    )
    with pytest.raises(RuntimeError, match="Mesh conversion failed"):
        manager.run()


def test_invalid_final_model_does_not_replace_previous_xml(tmp_path, monkeypatch):
    source = tmp_path / "robot.urdf"
    source.write_text('<robot name="test" />')
    manager = MujocoConversionManager(source, tmp_path / "output")
    output = manager.output_dir / "robot.xml"
    output.write_text("previous successful model")

    def fake_conversion(converter):
        converter.output_xml = converter.output_dir / "robot.xml"
        converter.output_xml.write_text(
            '<mujoco><asset /><worldbody><body name="root"><geom type="invalid" /></body></worldbody></mujoco>'
        )
        return True

    monkeypatch.setattr(manager, "_convert_urdf_with_recovery", fake_conversion)
    with pytest.raises(ValueError):
        manager.run()
    assert output.read_text() == "previous successful model"
    assert sorted(path.name for path in manager.output_dir.iterdir()) == [
        "meshes",
        "robot.xml",
    ]


@pytest.mark.parametrize("failure", ["conversion", "publication"])
def test_failed_rerun_preserves_previous_assets_and_xml(tmp_path, monkeypatch, failure):
    from pathlib import Path

    import mujoco
    import trimesh

    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    mesh = visual / "box.obj"
    trimesh.creation.box().export(mesh)
    source = tmp_path / "robot.urdf"
    source.write_text(
        """<robot name="test"><link name="arm"><inertial><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
      <visual><geometry><mesh filename="meshes/visual/box.obj"/></geometry></visual></link></robot>"""
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    output = manager.run()
    previous = {
        path.relative_to(manager.output_dir): path.read_bytes()
        for path in manager.output_dir.rglob("*")
        if path.is_file()
    }
    trimesh.creation.box(extents=[3, 4, 5]).export(mesh)
    if failure == "conversion":
        source.write_text("invalid XML")
    else:
        replace = Path.replace

        def failed_replace(path, target):
            if Path(target) == output:
                raise OSError("publication failed")
            return replace(path, target)

        monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises((RuntimeError, OSError)):
        manager.run()
    assert previous == {
        path.relative_to(manager.output_dir): path.read_bytes()
        for path in manager.output_dir.rglob("*")
        if path.is_file()
    }
    assert manager.meshes_dir == manager.output_dir / "meshes"
    assert mujoco.MjModel.from_xml_path(str(output)).ngeom == 2


def test_same_named_source_formats_cannot_silently_replace_each_other(tmp_path):
    import trimesh

    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    trimesh.creation.box().export(visual / "arm.obj")
    trimesh.creation.box(extents=[3, 4, 5]).export(visual / "arm.stl")
    source = tmp_path / "robot.urdf"
    source.write_text('<robot name="test"><link name="arm"/></robot>')
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    with pytest.raises(ValueError, match="share a conversion name"):
        manager.run()
    assert not list(manager.visual_output_dir.iterdir())


def test_failed_selective_recovery_reports_unrepairable_link(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text(
        """<robot name="unrepairable"><link name="broken"><inertial><mass value="1"/>
    <inertia ixx="0" iyy="0" izz="0"/></inertial></link></robot>"""
    )
    original = source.read_bytes()
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    with pytest.raises(RuntimeError, match="broken.*without geometry"):
        manager.run()
    recovery = manager.last_report["inertia_recoveries"][0]
    assert recovery["status"] == "failed"
    assert recovery["links"] == []
    assert "broken" in recovery["error"]
    assert manager.last_report["status"] == "failed"
    assert source.read_bytes() == original
    assert not (manager.output_dir / "robot.xml").exists()


def test_disabled_recovery_rejects_invalid_inertia_before_import(tmp_path, monkeypatch):
    source = tmp_path / "robot.urdf"
    source.write_text(
        """<robot name="invalid"><link name="broken"><inertial><mass value="1"/>
    <inertia ixx="0" iyy="0" izz="0"/></inertial>
    <visual><geometry><box size="1 2 3"/></geometry></visual></link></robot>"""
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Disabled recovery must not instantiate the calculator")

    monkeypatch.setattr(manager_module, "URDFInertiaCalculator", unexpected)
    manager = MujocoConversionManager(
        source, tmp_path / "output", backend="mujoco", auto_recalculate_inertia=False
    )
    with pytest.raises(RuntimeError, match="positive definite"):
        manager.run()
    assert manager.last_report["inertia_recovery_attempts"] == 0
    assert manager.last_report["inertia_recoveries"] == []
