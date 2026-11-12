import hashlib
import os
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from PIL import Image

from urdf2mjcf.dae_to_obj_converter import DAE2OBJConverter
from urdf2mjcf.glb_to_obj_converter import GLB2OBJConverter
from urdf2mjcf.urdf_to_mjcf_converter import URDF2MJCFConverter
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


@pytest.fixture(autouse=True)
def converters(monkeypatch):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )

    def unavailable():
        raise RuntimeError("Exercise the supported Collada fallback")

    monkeypatch.setattr(
        DAE2OBJConverter, "check_blender_available", staticmethod(unavailable)
    )


def _source(path, geometry):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<robot name="references"><link name="arm"><inertial><mass value="2"/>
    <inertia ixx="2" iyy="3" izz="4"/></inertial>{geometry}</link></robot>"""
    )
    return path


def _bounds_in_body(model, data, geom_id):
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
    world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    body = data.body("arm")
    local = (world - body.xpos) @ body.xmat.reshape(3, 3)
    return np.array([local.min(axis=0), local.max(axis=0)])


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
@pytest.mark.parametrize("mesh_format", ["glb", "dae", "ply", "off", "3mf"])
def test_external_formats_convert_once_and_preserve_visual_collision_scale(
    tmp_path, backend, mesh_format
):
    inputs = tmp_path / "inputs"
    mesh = inputs / "shared_shapes/custom assets" / f"arm.{mesh_format}"
    mesh.parent.mkdir(parents=True)
    shape = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    shape.apply_translation([0.02, 0.03, 0.04])
    shape.export(mesh)
    source = _source(
        inputs / "robot_description/urdf/robot.urdf",
        f"""
      <visual><origin xyz="0.3 0.4 0.5"/><geometry><mesh filename="{mesh.as_uri()}" scale="2 1 1"/></geometry></visual>
      <collision><geometry><mesh filename="package://shared_shapes/custom assets/arm.{mesh_format}"/></geometry></collision>""",
    )
    original = {
        p.relative_to(inputs): p.read_bytes() for p in inputs.rglob("*") if p.is_file()
    }
    manager = MujocoConversionManager(
        source, tmp_path / "output", backend=backend, auto_recalculate_inertia=False
    )
    output = manager.run()
    report = manager.last_report["mesh_preprocessing"]
    assert len(report) == 1
    assert report[0]["status"] == "converted"
    assert report[0]["source"] == str(mesh)
    assert {ref["kind"] for ref in report[0]["references"]} == {"visual", "collision"}
    assert (manager.output_dir / report[0]["output"]).is_file()
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.ngeom == 3
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        visual = (
            model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0
        )
        center = np.array([0.02, 0.03, 0.04])
        half = np.array([0.05, 0.1, 0.15])
        if mesh_format == "dae":
            # Trimesh's DAE exporter declares Y_UP; Blender maps it into Z-up.
            center = np.array([0.02, -0.04, 0.03])
            half = np.array([0.05, 0.15, 0.1])
        elif mesh_format == "3mf":
            # Its 3MF exporter declares millimeters for these numeric vertices.
            center *= 0.001
            half *= 0.001
        if visual:
            center = center * [2, 1, 1] + [0.3, 0.4, 0.5]
            half *= [2, 1, 1]
        np.testing.assert_allclose(
            _bounds_in_body(model, data, geom_id),
            [center - half, center + half],
            atol=2e-6,
        )
    manifest = manager.last_report["assets"]
    manager.run()
    assert manager.last_report["mesh_preprocessing"] == report
    assert manager.last_report["assets"] == manifest
    assert original == {
        p.relative_to(inputs): p.read_bytes() for p in inputs.rglob("*") if p.is_file()
    }
    moved = tmp_path / "moved"
    shutil.copytree(manager.output_dir, moved)
    inputs.rename(tmp_path / "hidden-inputs")
    manager.output_dir.rename(tmp_path / "hidden-output")
    loaded = mujoco.MjModel.from_xml_path(str(moved / output.name))
    np.testing.assert_array_equal(loaded.mesh_vert, model.mesh_vert)
    for entry in manifest:
        assert (
            hashlib.sha256((moved / entry["path"]).read_bytes()).hexdigest()
            == entry["sha256"]
        )


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_same_named_external_meshes_keep_distinct_geometry(tmp_path, backend):
    geometries = []
    for index in range(2):
        mesh = tmp_path / f"shapes{index}/custom/arm.glb"
        mesh.parent.mkdir(parents=True)
        trimesh.creation.box(extents=[0.1 * (index + 1), 0.2, 0.3]).export(mesh)
        geometries.append(
            f'<visual><origin xyz="{index} 0 0"/><geometry><mesh filename="package://shapes{index}/custom/arm.glb"/></geometry></visual>'
        )
    source = _source(tmp_path / "description/robot.urdf", "".join(geometries))
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    entries = manager.last_report["mesh_preprocessing"]
    assert len({entry["output"] for entry in entries}) == 2
    assert model.nmesh == 2
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = sorted(
        (
            _bounds_in_body(model, data, i)
            for i in range(model.ngeom)
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        ),
        key=lambda bound: bound[0, 0],
    )
    for index, bound in enumerate(bounds):
        np.testing.assert_allclose(bound.mean(axis=0), [index, 0, 0], atol=1e-7)
        np.testing.assert_allclose(
            bound[1] - bound[0], [0.1 * (index + 1), 0.2, 0.3], atol=1e-7
        )


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_collision_only_glb_is_converted_without_export_collision(tmp_path, backend):
    mesh = tmp_path / "meshes/collision/arm.glb"
    mesh.parent.mkdir(parents=True)
    trimesh.creation.box().export(mesh)
    source = _source(
        tmp_path / "robot.urdf",
        '<collision><geometry><mesh filename="meshes/collision/arm.glb"/></geometry></collision>',
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    assert model.ngeom == 2
    assert manager.last_report["mesh_preprocessing"][0]["output"].startswith(
        "meshes/collision/referenced/"
    )


def test_failed_referenced_conversion_keeps_previous_output_and_reports_source(
    tmp_path,
):
    mesh = tmp_path / "custom/arm.glb"
    mesh.parent.mkdir()
    trimesh.creation.box().export(mesh)
    source = _source(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh filename="custom/arm.glb"/></geometry></visual>',
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    manager.run()
    original = {
        p.relative_to(manager.output_dir): p.read_bytes()
        for p in manager.output_dir.rglob("*")
        if p.is_file()
    }
    mesh.write_bytes(b"broken GLB")
    with pytest.raises(RuntimeError, match="Referenced mesh conversion failed"):
        manager.run()
    assert manager.last_report["stage"] == "referenced_meshes"
    entry = manager.last_report["mesh_preprocessing"][0]
    assert entry["status"] == "failed" and entry["source"] == str(mesh)
    assert entry["output"] is None
    assert entry["references"][0]["link"] == "arm"
    assert original == {
        p.relative_to(manager.output_dir): p.read_bytes()
        for p in manager.output_dir.rglob("*")
        if p.is_file()
    }


def test_staged_legacy_basename_reference_is_not_converted_twice(tmp_path, monkeypatch):
    mesh = tmp_path / "visual/arm.glb"
    mesh.parent.mkdir()
    trimesh.creation.box().export(mesh)
    source = _source(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh filename="old-layout/arm.glb"/></geometry></visual>',
    )
    calls = []
    convert = GLB2OBJConverter.convert_glb_to_obj

    def counted(self, path, *args, **kwargs):
        calls.append(path)
        return convert(self, path, *args, **kwargs)

    monkeypatch.setattr(GLB2OBJConverter, "convert_glb_to_obj", counted)
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    manager.run()
    assert len(calls) == 1
    assert manager.last_report["mesh_preprocessing"] == []


def test_referenced_input_inside_published_meshes_is_not_overwritten(tmp_path):
    mesh = tmp_path / "output/meshes/original.obj"
    mesh.parent.mkdir(parents=True)
    trimesh.creation.box().export(mesh)
    source = _source(
        tmp_path / "robot.urdf",
        f'<visual><geometry><mesh filename="{mesh}"/></geometry></visual>',
    )
    original = mesh.read_bytes()
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    with pytest.raises(ValueError, match="contains a referenced source"):
        manager.run()
    assert mesh.read_bytes() == original


def test_external_textured_glb_keeps_its_pixels_in_legacy_output(tmp_path):
    mesh = tmp_path / "custom/painted.glb"
    mesh.parent.mkdir()
    shape = trimesh.creation.box()
    shape.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(shape.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            name="paint", baseColorTexture=Image.new("RGB", (4, 4), (23, 47, 89))
        ),
    )
    shape.export(mesh)
    source = _source(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh filename="custom/painted.glb"/></geometry></visual>',
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend="urdf2mjcf")
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    referenced = {
        ref["name"]
        for entry in manager.last_report["assets"]
        for ref in entry["references"]
        if ref["type"] == "texture"
    }
    assert referenced
    for name in referenced:
        index = model.texture(name).id
        size = (
            model.tex_width[index] * model.tex_height[index] * model.tex_nchannel[index]
        )
        pixels = model.tex_data[
            model.tex_adr[index] : model.tex_adr[index] + size
        ].reshape(-1, model.tex_nchannel[index])
        np.testing.assert_array_equal(
            pixels[:, :3], np.tile([23, 47, 89], (len(pixels), 1))
        )


def test_backend_fallback_reuses_the_referenced_conversion(tmp_path, monkeypatch):
    mesh = tmp_path / "custom/arm.glb"
    mesh.parent.mkdir()
    trimesh.creation.box().export(mesh)
    source = _source(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh filename="custom/arm.glb"/></geometry></visual>',
    )
    calls = []
    convert = GLB2OBJConverter.convert_glb_to_obj

    def counted(self, path, *args, **kwargs):
        calls.append(path)
        return convert(self, path, *args, **kwargs)

    def unavailable(self):
        self.last_error = "External converter unavailable"
        return False

    monkeypatch.setattr(GLB2OBJConverter, "convert_glb_to_obj", counted)
    monkeypatch.setattr(URDF2MJCFConverter, "convert", unavailable)
    manager = MujocoConversionManager(source, tmp_path / "output", backend="auto")
    manager.run()
    assert manager.last_backend == "mujoco"
    assert calls == [mesh]
    assert len(manager.last_report["mesh_preprocessing"]) == 1
    assert [attempt["status"] for attempt in manager.last_report["attempts"]] == [
        "failed",
        "success",
    ]


def test_generated_reference_directory_does_not_reuse_existing_asset_folder(tmp_path):
    mesh = tmp_path / "custom/arm.glb"
    mesh.parent.mkdir()
    trimesh.creation.box().export(mesh)
    base = "ref_" + hashlib.sha256(mesh.as_posix().encode()).hexdigest()[:16]
    existing = tmp_path / "visual/referenced" / base / "keep.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("existing asset directory")
    source = _source(
        tmp_path / "robot.urdf",
        '<visual><geometry><mesh filename="custom/arm.glb"/></geometry></visual>',
    )
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    manager.run()
    converted = Path(manager.last_report["mesh_preprocessing"][0]["output"])
    assert converted.parent.name != base
    assert (
        manager.visual_output_dir / "referenced" / base / "keep.txt"
    ).read_text() == "existing asset directory"
