"""Source file identity must survive importer and material-part naming rules."""

import hashlib
import shutil

import mujoco
import numpy as np
import pytest
import trimesh

from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def _robot(path, geometries):
    path.write_text(
        '<robot name="identities"><link name="arm"><inertial><mass value="2"/>'
        '<inertia ixx="2" iyy="3" izz="4"/></inertial>' + geometries + "</link></robot>"
    )
    return path


def _bounds(model, data, gid):
    mesh = model.geom_dataid[gid]
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    world = vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    body = data.body("arm")
    local = (world - body.xpos) @ body.xmat.reshape(3, 3)
    return np.array([local.min(axis=0), local.max(axis=0)])


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
@pytest.mark.parametrize("mesh_format", ["obj", "stl"])
def test_distinct_same_named_files_keep_each_geometry_and_scale(
    tmp_path, backend, mesh_format
):
    inputs = tmp_path / "inputs"
    geometries = []
    for index in range(2):
        mesh = inputs / f"shapes{index}/arm.{mesh_format}"
        mesh.parent.mkdir(parents=True)
        trimesh.creation.box(extents=[0.1 * (index + 1), 0.2, 0.3]).export(mesh)
        geometries.append(
            f'<visual><origin xyz="{index} 0 0"/><geometry>'
            f'<mesh filename="{mesh}" scale="1 2 1"/></geometry></visual>'
        )
    # A third instance shares one source but requires a distinct effective scale.
    geometries.append(
        '<collision><origin xyz="2 0 0"/><geometry>'
        f'<mesh filename="{inputs / f"shapes0/arm.{mesh_format}"}" scale="3 1 1"/>'
        "</geometry></collision>"
    )
    source = _robot(inputs / "robot.urdf", "".join(geometries))
    original = {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    output = manager.run()
    model = mujoco.MjModel.from_xml_path(str(output))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = sorted(
        (
            _bounds(model, data, gid)
            for gid in range(model.ngeom)
            if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH
        ),
        key=lambda bound: bound.mean(axis=0)[0],
    )
    assert len(bounds) == 3
    for index, (bound, size) in enumerate(
        zip(bounds, ([0.1, 0.4, 0.3], [0.2, 0.4, 0.3], [0.3, 0.2, 0.3]), strict=True)
    ):
        np.testing.assert_allclose(bound.mean(axis=0), [index, 0, 0], atol=2e-6)
        np.testing.assert_allclose(bound[1] - bound[0], size, atol=2e-6)
    manifest = manager.last_report["assets"]
    manager.run()
    assert manager.last_report["assets"] == manifest
    assert original == {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    moved = tmp_path / "relocated"
    shutil.copytree(manager.output_dir, moved)
    inputs.rename(tmp_path / "hidden-inputs")
    manager.output_dir.rename(tmp_path / "hidden-output")
    relocated = mujoco.MjModel.from_xml_path(str(moved / output.name))
    np.testing.assert_array_equal(relocated.mesh_vert, model.mesh_vert)
    for asset in manifest:
        assert (
            hashlib.sha256((moved / asset["path"]).read_bytes()).hexdigest()
            == asset["sha256"]
        )


def test_unreferenced_visual_mesh_cannot_replace_an_external_mesh(tmp_path):
    inputs = tmp_path / "inputs"
    external = inputs / "custom/arm.obj"
    external.parent.mkdir(parents=True)
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(external)
    unrelated = inputs / "meshes/visual/arm.obj"
    unrelated.parent.mkdir(parents=True)
    trimesh.creation.box(extents=[4, 5, 6]).export(unrelated)
    source = _robot(
        inputs / "robot.urdf",
        f'<visual><geometry><mesh filename="{external}"/></geometry></visual>',
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend="urdf2mjcf")
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = next(
        i
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
    )
    bound = _bounds(model, data, gid)
    np.testing.assert_allclose(bound[1] - bound[0], [0.1, 0.2, 0.3], atol=2e-6)


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_mesh_file_alias_is_normalized_for_both_importers(tmp_path, backend):
    mesh = tmp_path / "shape.obj"
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(mesh)
    source = _robot(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh file="shape.obj"/></geometry></visual>',
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    assert model.nmesh == 1
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gid = next(
        i
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
    )
    bound = _bounds(model, data, gid)
    np.testing.assert_allclose(bound[1] - bound[0], [0.1, 0.2, 0.3], atol=2e-6)


def test_native_scale_suffix_cannot_collide_with_another_source_filename(tmp_path):
    for name, size in (("arm", 0.1), ("arm1", 0.4)):
        trimesh.creation.box(extents=[size, 0.2, 0.3]).export(tmp_path / f"{name}.obj")
    source = _robot(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh filename="arm.obj"/></geometry></visual>'
        '<visual><origin xyz="1 0 0"/><geometry><mesh filename="arm.obj" scale="2 1 1"/></geometry></visual>'
        '<visual><origin xyz="2 0 0"/><geometry><mesh filename="arm1.obj"/></geometry></visual>'
        '<visual><origin xyz="3 0 0"/><geometry><mesh filename="arm.obj" scale="2 1 1"/></geometry></visual>',
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = sorted(
        (
            _bounds(model, data, i)
            for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        ),
        key=lambda bound: bound.mean(axis=0)[0],
    )
    assert len(bounds) == 4
    for index, (bound, width) in enumerate(
        zip(bounds, [0.1, 0.2, 0.4, 0.2], strict=True)
    ):
        np.testing.assert_allclose(bound.mean(axis=0), [index, 0, 0], atol=2e-6)
        np.testing.assert_allclose(bound[1] - bound[0], [width, 0.2, 0.3], atol=2e-6)


def test_native_aliases_fall_back_to_copying_and_leave_no_temporary_files(
    tmp_path, monkeypatch
):
    from pathlib import Path

    mesh = tmp_path / "shape.obj"
    trimesh.creation.box().export(mesh)
    original = mesh.read_bytes()
    source = _robot(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh filename="shape.obj"/></geometry></visual>',
    )

    def unsupported(*args, **kwargs):
        raise OSError("Hardlinks unavailable across filesystems")

    monkeypatch.setattr(Path, "hardlink_to", unsupported)
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    output = manager.run()
    assert mujoco.MjModel.from_xml_path(str(output)).nmesh == 1
    assert mesh.read_bytes() == original
    assert {p.name for p in manager.output_dir.iterdir()} == {"meshes", "robot.xml"}
    assert not list(manager.meshes_dir.rglob("mesh-aliases"))


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_missing_absolute_mesh_cannot_use_an_unrelated_visual_on_rerun(
    tmp_path, backend
):
    visual = tmp_path / "meshes/visual/arm.obj"
    visual.parent.mkdir(parents=True)
    trimesh.creation.box().export(visual)
    source = _robot(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh filename="meshes/visual/arm.obj"/></geometry></visual>',
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    manager.run()
    previous = {
        p.relative_to(manager.output_dir): p.read_bytes()
        for p in manager.output_dir.rglob("*")
        if p.is_file()
    }
    _robot(
        source,
        f'<visual><geometry><mesh filename="{tmp_path / "missing/arm.obj"}"/></geometry></visual>',
    )
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match="explicit path"):
        manager.run()
    assert source.read_bytes() == original
    assert previous == {
        p.relative_to(manager.output_dir): p.read_bytes()
        for p in manager.output_dir.rglob("*")
        if p.is_file()
    }
