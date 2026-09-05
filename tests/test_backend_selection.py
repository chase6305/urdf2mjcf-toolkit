import xml.etree.ElementTree as ET

import mujoco
import pytest

import urdf2mjcf.urdf_to_mujoco_converter as manager_module
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager

ROBOT = """<robot name="test"><link name="arm">
  <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><geometry><box size="1 1 1"/></geometry></visual>
  <collision><geometry><box size="1 1 1"/></geometry></collision>
</link></robot>"""


def _manager(tmp_path, backend):
    source = tmp_path / "robot.urdf"
    source.write_text(ROBOT)
    return MujocoConversionManager(source, tmp_path / "output", backend=backend)


def test_native_backend_does_not_invoke_external_converters(tmp_path, monkeypatch):
    import subprocess

    def forbidden(*args, **kwargs):
        pytest.fail("Native conversion invoked an external command")

    monkeypatch.setattr(subprocess, "run", forbidden)
    manager = _manager(tmp_path, "mujoco")
    output = manager.run()
    assert manager.last_backend == "mujoco"
    assert mujoco.MjModel.from_xml_path(str(output)).ngeom == 3


def test_auto_backend_recovers_from_missing_cli_without_recalculating_inertia(
    tmp_path, monkeypatch
):
    import subprocess

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("urdf2mjcf executable not found")

    def no_recalculation(*args, **kwargs):
        pytest.fail("Missing CLI should not trigger inertia recalculation")

    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.setattr(manager_module, "URDFInertiaCalculator", no_recalculation)
    manager = _manager(tmp_path, "auto")
    output = manager.run()
    assert manager.last_backend == "mujoco"
    assert mujoco.MjModel.from_xml_path(str(output)).ngeom == 3
    assert manager.urdf_path.read_text() == ROBOT


@pytest.mark.parametrize("stage", ["assets", "validation"])
def test_auto_backend_recovers_from_postprocessing_failure(
    tmp_path, monkeypatch, stage
):
    def fake_convert(converter):
        converter.output_xml = converter.output_dir / "robot.xml"
        converter.output_xml.write_text(
            '<mujoco><worldbody><body name="arm"><geom type="invalid" /></body></worldbody></mujoco>'
        )
        return True

    monkeypatch.setattr(manager_module.URDF2MJCFConverter, "convert", fake_convert)
    if stage == "assets":

        def failed_assets(*args, **kwargs):
            raise RuntimeError("OBJ conversion failed")

        monkeypatch.setattr(manager_module.OBJ2MJCFImporter, "run", failed_assets)
    manager = _manager(tmp_path, "auto")
    output = manager.run()
    assert manager.last_backend == "mujoco"
    assert mujoco.MjModel.from_xml_path(str(output)).ngeom == 3
    assert not list(manager.output_dir.glob("urdf2mjcf-work-*"))


def test_explicit_backend_does_not_switch_on_error(tmp_path, monkeypatch):
    def failed_convert(converter):
        converter.last_error = "external conversion failed"
        return False

    monkeypatch.setattr(manager_module.URDF2MJCFConverter, "convert", failed_convert)
    manager = _manager(tmp_path, "urdf2mjcf")
    with pytest.raises(RuntimeError, match="external conversion failed"):
        manager.run()
    assert manager.last_backend is None
    assert not (manager.output_dir / "robot.xml").exists()


def test_all_backend_failures_preserve_previous_result(tmp_path):
    manager = _manager(tmp_path, "auto")
    # Valid XML reaches backend selection; malformed XML now fails discovery first.
    manager.urdf_path.write_text("<invalid_model/>")
    output = manager.output_dir / "robot.xml"
    output.write_text("previous result")
    with pytest.raises(
        RuntimeError, match="All conversion backends failed: urdf2mjcf:.*mujoco:"
    ):
        manager.run()
    assert output.read_text() == "previous result"
    assert manager.last_backend is None


def test_unknown_backend_is_rejected_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match="Unknown conversion backend"):
        _manager(tmp_path, "unknown")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_backends_preserve_multiple_instances_and_scales(
    tmp_path, monkeypatch, backend
):
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import trimesh

    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    trimesh.creation.box().export(visual / "box.obj")
    manager = _manager(tmp_path, backend)
    tree = ET.parse(manager.urdf_path)
    link = tree.getroot().find("link")
    link.remove(link.find("visual"))
    for index, scale in enumerate(("1 1 1", "2 1 1", "1 1 1")):
        geom = ET.SubElement(link, "visual", name=f"instance_{index}")
        ET.SubElement(geom, "origin", xyz=f"{index * 2} 0 0")
        shape = ET.SubElement(geom, "geometry")
        ET.SubElement(shape, "mesh", filename="meshes/visual/box.obj", scale=scale)
    tree.write(manager.urdf_path)
    output = manager.run()
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.ngeom == 5
    for index, scale in enumerate(((1, 1, 1), (2, 1, 1), (1, 1, 1))):
        geom = model.geom(
            f"instance_{index}" if backend == "mujoco" else f"arm_visual_{index}"
        )
        np.testing.assert_allclose(geom.pos, [index * 2, 0, 0], atol=1e-12)
        np.testing.assert_allclose(model.mesh_scale[geom.dataid[0]], scale)
