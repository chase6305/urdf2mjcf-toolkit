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

        def update_inertia(self):
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
