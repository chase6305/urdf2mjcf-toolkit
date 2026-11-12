"""Exercise the external converters and compile the resulting MJCF with MuJoCo."""

import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def _scene_colors(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=100)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return {
        geom.objid: geom.rgba.copy()
        for geom in scene.geoms[: scene.ngeom]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
    }


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
@pytest.mark.parametrize("mesh_format", ["obj", "stl"])
def test_shared_mesh_instances_keep_explicit_global_and_inline_urdf_colors(
    tmp_path, backend, mesh_format
):
    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(visual / f"arm.{mesh_format}")
    source = tmp_path / "robot.urdf"
    robot = ET.Element("robot", name="instance_colors")
    expected = ([1, 0, 0, 0.4], [0, 0, 1, 1], [0.5, 0.5, 0.5, 1])
    for name, rgba in (("red", expected[0]), ("arm", expected[2])):
        ET.SubElement(
            ET.SubElement(robot, "material", name=name),
            "color",
            rgba=" ".join(map(str, rgba)),
        )
    link = ET.SubElement(robot, "link", name="arm")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value="2")
    ET.SubElement(inertial, "inertia", ixx="2", iyy="3", izz="4")
    for index, name in enumerate(("red", "blue", "arm")):
        instance = ET.SubElement(link, "visual")
        ET.SubElement(instance, "origin", xyz=f"{index} 0 0")
        ET.SubElement(
            ET.SubElement(instance, "geometry"),
            "mesh",
            filename=f"meshes/visual/arm.{mesh_format}",
        )
        material = ET.SubElement(instance, "material", name=name)
        if name == "blue":
            ET.SubElement(material, "color", rgba="0 0 1 1")
    ET.ElementTree(robot).write(source)
    original = source.read_bytes()
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    output = manager.run()
    model = mujoco.MjModel.from_xml_path(str(output))
    colors = _scene_colors(model)
    visual_ids = sorted(
        (
            gid
            for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == model.body("arm").id
        ),
        key=lambda gid: model.geom_pos[gid, 0],
    )
    assert len(visual_ids) == 3
    for gid, rgba in zip(visual_ids, expected, strict=True):
        np.testing.assert_allclose(colors[gid], rgba, atol=1e-7)
    manager.run()
    repeated = mujoco.MjModel.from_xml_path(str(output))
    repeated_colors = _scene_colors(repeated)
    for gid, rgba in zip(visual_ids, expected, strict=True):
        np.testing.assert_allclose(repeated_colors[gid], rgba, atol=1e-7)
    assert source.read_bytes() == original


def _painted_obj(path):
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
    path.write_text("\n".join(lines) + "\n")
    path.with_name("colors.mtl").write_text(
        "newmtl red\nKd 1 0 0\nnewmtl green\nKd 0 1 0\n"
    )


def test_explicit_urdf_color_covers_all_parts_without_recoloring_other_instances(
    tmp_path,
):
    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    _painted_obj(visual / "arm.obj")
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="part_colors"><material name="arm"><color rgba="0 0 1 0.6"/></material>'
        '<link name="arm"><inertial><mass value="2"/><inertia ixx="2" iyy="3" izz="4"/></inertial>'
        '<visual><geometry><mesh filename="meshes/visual/arm.obj"/></geometry><material name="arm"/></visual>'
        '<visual><origin xyz="2 0 0"/><geometry><mesh filename="meshes/visual/arm.obj"/></geometry></visual>'
        "</link></robot>"
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend="urdf2mjcf")
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    colors = _scene_colors(model)
    visual_ids = sorted(
        (
            gid
            for gid in range(model.ngeom)
            if model.geom_bodyid[gid] == model.body("arm").id
        ),
        key=lambda gid: model.geom_pos[gid, 0],
    )
    assert len(visual_ids) == 4
    expected = ([0, 0, 1, 0.6], [0, 0, 1, 0.6], [1, 0, 0, 1], [0, 1, 0, 1])
    for gid, rgba in zip(visual_ids, expected, strict=True):
        np.testing.assert_allclose(colors[gid], rgba, atol=1e-7)


def test_native_visual_named_floor_keeps_its_geometry_and_a_distinct_ground(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="floor_name"><link name="arm"><inertial><mass value="2"/>'
        '<inertia ixx="2" iyy="3" izz="4"/></inertial><visual name="floor">'
        '<geometry><box size="0.1 0.2 0.3"/></geometry></visual></link></robot>'
    )
    original = source.read_bytes()
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    assert model.ngeom == 2
    assert model.geom("floor").type[0] == mujoco.mjtGeom.mjGEOM_BOX
    assert np.count_nonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1
    np.testing.assert_allclose(model.geom("floor").size, [0.05, 0.1, 0.15])
    assert source.read_bytes() == original
    assert manager.last_report["body_aliases"] == {}


@pytest.mark.parametrize("world_position", ["root", "child"])
def test_external_world_link_gets_a_distinct_body_without_losing_physics(
    tmp_path, world_position
):
    root_name, child_name = (
        ("world", "world_urdf") if world_position == "root" else ("world_urdf", "world")
    )
    source = tmp_path / "robot.urdf"
    source.write_text(
        f'<robot name="world_body"><link name="{root_name}"/><link name="{child_name}">'
        '<inertial><origin xyz="0.1 0.2 0.3"/><mass value="2"/>'
        '<inertia ixx="2" iyy="3" izz="4"/></inertial>'
        '<visual><geometry><box size="0.1 0.2 0.3"/></geometry></visual></link>'
        f'<joint name="world" type="revolute"><parent link="{root_name}"/>'
        f'<child link="{child_name}"/><origin xyz="0.1 0.2 0.3"/>'
        '<limit lower="-1" upper="1" effort="10"/></joint></robot>'
    )
    original = source.read_bytes()
    manager = MujocoConversionManager(
        source, tmp_path / "output", backend="urdf2mjcf", auto_recalculate_inertia=False
    )
    output = manager.run()
    aliases = manager.last_report["body_aliases"]
    assert aliases == {"world": "world_urdf_1"}
    model = mujoco.MjModel.from_xml_path(str(output))
    moving = model.body(aliases.get(child_name, child_name))
    np.testing.assert_allclose(moving.mass, [2])
    np.testing.assert_allclose(moving.ipos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(np.sort(moving.inertia), [2, 3, 4])
    np.testing.assert_allclose(moving.pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(model.joint("world").range, [-1, 1])
    assert model.actuator_trnid[0, 0] == model.joint("world").id
    assert model.body("world").id == 0
    assert model.body("world_urdf_1").id > 0
    manager.run()
    assert manager.last_report["body_aliases"] == aliases
    assert source.read_bytes() == original


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


@pytest.mark.parametrize("shared_collision_mesh", [False, True])
def test_numbered_meshes_and_multiple_materials_are_preserved(
    tmp_path, monkeypatch, shared_collision_mesh
):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    _painted_obj(visual / "link_1.obj")
    trimesh.creation.box(extents=[0.2, 0.1, 0.1]).export(visual / "link_2.obj")
    source = tmp_path / "robot.urdf"
    collision = (
        '<mesh filename="meshes/visual/link_1.obj" scale="0.5 2 1"/>'
        if shared_collision_mesh
        else '<box size="0.1 0.1 0.1"/>'
    )
    source.write_text(f"""<robot name="parts"><link name="arm">
      <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
      <visual><geometry><mesh filename="meshes/visual/link_1.obj"/></geometry></visual>
      <visual><origin xyz="1 0 0"/><geometry><mesh filename="meshes/visual/link_1.obj" scale="2 1 1"/></geometry></visual>
      <visual><origin xyz="2 0 0"/><geometry><mesh filename="meshes/visual/link_2.obj"/></geometry></visual>
      <collision><geometry>{collision}</geometry></collision>
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
    if shared_collision_mesh:
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        gid = next(
            i
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == model.body("arm").id
            and model.geom_contype[i] != 0
        )
        mesh_id = model.geom_dataid[gid]
        start = model.mesh_vertadr[mesh_id]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
        world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        body = data.body("arm")
        local = (world - body.xpos) @ body.xmat.reshape(3, 3)
        np.testing.assert_allclose(
            [local.min(axis=0), local.max(axis=0)],
            [[-0.025, -0.1, -0.05], [0.175, 0.1, 0.05]],
            atol=2e-6,
        )
    first_xml = output.read_bytes()
    manager.run()
    assert output.read_bytes() == first_xml
