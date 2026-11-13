"""URDF conversion should process its referenced visual meshes independently."""

import hashlib
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from urdf2mjcf.glb_to_obj_converter import GLB2OBJConverter
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def _source(path, filename):
    path.write_text(
        '<robot name="selected"><link name="arm"><inertial><mass value="1"/>'
        '<inertia ixx="1" iyy="1" izz="1"/></inertial>'
        f'<visual><geometry><mesh filename="{filename}"/></geometry></visual>'
        "</link></robot>"
    )
    return path


def _extents(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = next(
        i
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_contype[i] == 0
        and model.geom_conaffinity[i] == 0
    )
    mesh = model.geom_dataid[gid]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    body = data.body("arm")
    return np.ptp((world - body.xpos) @ body.xmat.reshape(3, 3), axis=0)


@pytest.fixture(autouse=True)
def _path(monkeypatch):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
@pytest.mark.parametrize("unused", ["duplicate_names", "malformed"])
def test_unrelated_visual_meshes_are_not_converted_or_imported(
    tmp_path, monkeypatch, backend, unused
):
    visual = tmp_path / "meshes/visual"
    used = visual / "used/arm.glb"
    used.parent.mkdir(parents=True)
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(used)
    if unused == "duplicate_names":
        for side in ("left", "right"):
            extra = visual / side / "hand.glb"
            extra.parent.mkdir()
            trimesh.creation.box(extents=[2, 3, 4]).export(extra)
    else:
        (visual / "broken.glb").write_bytes(b"not a GLB")
    source = _source(tmp_path / "robot.urdf", "meshes/visual/used/arm.glb")
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in visual.rglob("*")
        if p.is_file()
    }
    conversions = []
    convert = GLB2OBJConverter.convert_glb_to_obj

    def counted(self, path, *args, **kwargs):
        conversions.append(Path(path).name)
        return convert(self, path, *args, **kwargs)

    monkeypatch.setattr(GLB2OBJConverter, "convert_glb_to_obj", counted)
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)

    model = mujoco.MjModel.from_xml_path(str(manager.run()))

    assert conversions == ["arm.glb"]
    assert model.nmesh == 1
    np.testing.assert_allclose(_extents(model), [0.1, 0.2, 0.3], atol=1e-6)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert not list(manager.visual_output_dir.rglob("hand.glb"))
    assert not list(manager.visual_output_dir.rglob("broken.glb"))


def test_selected_visual_sources_refresh_when_the_same_manager_is_reused(tmp_path):
    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    for name, size in (("a", [0.1, 0.2, 0.3]), ("b", [0.4, 0.5, 0.6])):
        trimesh.creation.box(extents=size).export(visual / f"{name}.glb")
    source = _source(tmp_path / "robot.urdf", "meshes/visual/a.glb")
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    first = mujoco.MjModel.from_xml_path(str(manager.run()))
    np.testing.assert_allclose(_extents(first), [0.1, 0.2, 0.3], atol=1e-6)
    _source(source, "meshes/visual/b.glb")

    second = mujoco.MjModel.from_xml_path(str(manager.run()))

    np.testing.assert_allclose(_extents(second), [0.4, 0.5, 0.6], atol=1e-6)
    assert not list(manager.visual_output_dir.rglob("a.glb"))


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_legacy_generated_obj_reference_still_refreshes_from_source_glb(
    tmp_path, backend
):
    visual = tmp_path / "meshes/visual"
    old = visual / "arm/arm.obj"
    old.parent.mkdir(parents=True)
    trimesh.creation.box(extents=[4, 5, 6]).export(old)
    original = old.read_bytes()
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(visual / "arm.glb")
    source = _source(tmp_path / "robot.urdf", "arm.obj")
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)

    model = mujoco.MjModel.from_xml_path(str(manager.run()))

    np.testing.assert_allclose(_extents(model), [0.1, 0.2, 0.3], atol=1e-6)
    assert old.read_bytes() == original


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_collision_supplementation_keeps_unreferenced_collision_files(
    tmp_path, backend
):
    visual = tmp_path / "meshes/visual"
    collision = tmp_path / "meshes/collision"
    visual.mkdir(parents=True)
    collision.mkdir()
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(visual / "arm.glb")
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(collision / "arm.obj")
    source = _source(tmp_path / "robot.urdf", "meshes/visual/arm.glb")
    manager = MujocoConversionManager(
        source, tmp_path / "output", backend=backend, export_collision=True
    )

    model = mujoco.MjModel.from_xml_path(str(manager.run()))

    assert model.ngeom == 3
    colliders = [
        gid
        for gid in range(model.ngeom)
        if model.geom_bodyid[gid] == model.body("arm").id
        and model.geom_contype[gid] != 0
        and model.geom_conaffinity[gid] != 0
    ]
    assert len(colliders) == 1
    assert (manager.collision_output_dir / "arm.obj").is_file()
