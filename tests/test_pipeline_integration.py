"""Exercise the external converters and compile the resulting MJCF with MuJoCo."""

import os
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


@pytest.mark.parametrize("backend", ["urdf2mjcf", "mujoco"])
@pytest.mark.parametrize("mesh_format", [None, "stl", "glb", "dae"])
@pytest.mark.parametrize("fixed_base", [True, False])
def test_pipeline_produces_loadable_model(
    tmp_path, monkeypatch, mesh_format, fixed_base, backend
):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    if backend == "urdf2mjcf" and (
        not shutil.which("urdf2mjcf") or not shutil.which("obj2mjcf")
    ):
        pytest.skip("External converters are not installed")
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "robot"
    project.mkdir()
    visual_geometry = '<box size="0.1 0.1 0.1" />'
    collision_geometry = visual_geometry
    if mesh_format:
        for kind, name in (("visual", "arm"), ("collision", "arm_collision")):
            directory = project / "meshes" / kind
            directory.mkdir(parents=True)
            trimesh.creation.box(extents=[0.1, 0.1, 0.1]).export(
                directory / f"{name}.{mesh_format}"
            )
        visual_geometry = (
            f'<mesh filename="meshes/visual/arm.{mesh_format}" scale="2 1 1" />'
        )
        collision_geometry = f'<mesh filename="meshes/collision/arm_collision.{mesh_format}" scale="2 1 1" />'
    source = project / "robot.urdf"
    source.write_text(f"""<robot name="test"><link name="base">
      <inertial><mass value="1" /><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0" /></inertial>
      <visual><geometry>{visual_geometry}</geometry></visual>
      <collision><geometry>{collision_geometry}</geometry></collision>
    </link></robot>""")
    original = source.read_bytes()
    manager = MujocoConversionManager(
        "robot/robot.urdf",
        "output",
        fixed_base=fixed_base,
        export_collision=True,
        auto_recalculate_inertia=False,
        backend=backend,
    )
    output = manager.run()
    assert manager.last_backend == backend
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.nq == (0 if fixed_base else 7)
    assert model.ngeom == 3  # One visual, one collision, and the ground plane.
    if mesh_format:
        assert model.nmesh == 2
        np.testing.assert_allclose(model.mesh_scale, [[2, 1, 1], [2, 1, 1]])
    first_files = sorted(
        path.relative_to(manager.output_dir)
        for path in manager.output_dir.rglob("*")
        if path.is_file()
    )
    manager.run()
    repeated = mujoco.MjModel.from_xml_path(str(output))
    assert repeated.ngeom == model.ngeom
    assert (
        sorted(
            path.relative_to(manager.output_dir)
            for path in manager.output_dir.rglob("*")
            if path.is_file()
        )
        == first_files
    )
    assert source.read_bytes() == original


def test_numbered_meshes_and_multiple_materials_are_preserved(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET

    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    # Two closed meshes in one OBJ, with distinct materials and positions.
    lines = ["mtllib colors.mtl"]
    vertex_offset = 1
    for index, color in enumerate(("red", "green")):
        mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        mesh.apply_translation([index * 0.3, 0, 0])
        lines.append(f"o part_{index}")
        lines.extend("v " + " ".join(map(str, vertex)) for vertex in mesh.vertices)
        lines.append(f"usemtl {color}")
        lines.extend(
            "f " + " ".join(str(int(value) + vertex_offset) for value in face)
            for face in mesh.faces
        )
        vertex_offset += len(mesh.vertices)
    (visual / "link_1.obj").write_text("\n".join(lines) + "\n")
    (visual / "colors.mtl").write_text("newmtl red\nKd 1 0 0\nnewmtl green\nKd 0 1 0\n")
    trimesh.creation.box(extents=[0.2, 0.1, 0.1]).export(visual / "link_2.obj")
    source = tmp_path / "robot.urdf"
    source.write_text("""<robot name="parts"><link name="arm">
      <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
      <visual><geometry><mesh filename="meshes/visual/link_1.obj"/></geometry></visual>
      <visual><origin xyz="1 0 0"/><geometry><mesh filename="meshes/visual/link_1.obj" scale="2 1 1"/></geometry></visual>
      <visual><origin xyz="2 0 0"/><geometry><mesh filename="meshes/visual/link_2.obj"/></geometry></visual>
      <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
    </link></robot>""")
    manager = MujocoConversionManager(source, tmp_path / "output", backend="urdf2mjcf")
    output = manager.run()
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.ngeom == 7  # Four material parts, one other visual, collision, floor.
    root = ET.parse(output).getroot()
    colors = {
        tuple(
            float(v)
            for v in root.find(f"asset/material[@name='{name}']").get("rgba").split()
        )
        for name in ("link_1", "link_1_1")
    }
    assert colors == {(1, 0, 0, 1), (0, 1, 0, 1)}
    assert root.find("asset/mesh[@name='link_2']") is not None
    first_xml = output.read_bytes()
    manager.run()
    assert output.read_bytes() == first_xml
