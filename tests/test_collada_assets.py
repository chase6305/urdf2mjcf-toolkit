"""DAE fallback geometry and inertia must use the same declared asset frame."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh

from urdf2mjcf.collada_assets import normalize_collada_asset
from urdf2mjcf.dae_to_obj_converter import DAE2OBJConverter
from urdf2mjcf.urdf_inertia_calculator import URDFInertiaCalculator
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager

NS = "{http://www.collada.org/2005/11/COLLADASchema}"


def _dae(directory, *, unit=0.001, up_axis="Z_UP"):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "body.dae"
    box = trimesh.creation.box(extents=[100, 200, 300])
    box.export(source)
    tree = ET.parse(source)
    asset = tree.find(f"{NS}asset")
    for element in asset.findall(f"{NS}unit"):
        asset.remove(element)
    ET.SubElement(asset, f"{NS}unit", name="source_unit", meter=str(unit))
    asset.find(f"{NS}up_axis").text = up_axis
    node = tree.find(f"{NS}library_visual_scenes/{NS}visual_scene/{NS}node")
    translation = ET.Element(f"{NS}translate")
    translation.text = "1000 2000 3000"
    node.insert(0, translation)
    tree.write(source)
    return source, box.vertices + [1000, 2000, 3000]


def _expected_vertices(vertices, unit, up_axis):
    x, y, z = vertices.T
    if up_axis == "Y_UP":
        vertices = np.column_stack((x, -z, y))
    elif up_axis == "X_UP":
        vertices = np.column_stack((-z, y, x))
    return vertices * unit


def _calculator(directory, mesh):
    source = directory / "robot.urdf"
    source.write_text(
        '<robot name="dae_units"><link name="body"><inertial><mass value="12"/>'
        '<inertia ixx="0" iyy="0" izz="0"/></inertial>'
        f'<visual><geometry><mesh filename="{mesh.relative_to(directory).as_posix()}"/>'
        "</geometry></visual></link></robot>"
    )
    return URDFInertiaCalculator(source)


@pytest.mark.parametrize("unit", [0.001, 1])
@pytest.mark.parametrize("up_axis", ["X_UP", "Y_UP", "Z_UP"])
def test_fallback_applies_asset_units_and_frame_after_node_transform(
    tmp_path, unit, up_axis
):
    source, vertices = _dae(tmp_path, unit=unit, up_axis=up_axis)
    original = source.read_bytes()

    success, outputs = DAE2OBJConverter()._convert_with_trimesh(
        source, tmp_path / "output", preserve_textures=True
    )

    assert success
    assert len(outputs) == 1
    mesh = trimesh.load(outputs[0], force="mesh", process=False)
    expected = _expected_vertices(vertices, unit, up_axis)
    np.testing.assert_allclose(
        mesh.bounds, [expected.min(axis=0), expected.max(axis=0)]
    )
    np.testing.assert_allclose(mesh.center_mass, expected.mean(axis=0))
    assert source.read_bytes() == original


@pytest.mark.parametrize("unit", [0.001, 1])
@pytest.mark.parametrize("up_axis", ["X_UP", "Y_UP", "Z_UP"])
def test_direct_dae_inertia_uses_declared_units_and_frame(tmp_path, unit, up_axis):
    source, vertices = _dae(tmp_path, unit=unit, up_axis=up_axis)
    calculator = _calculator(tmp_path, source)

    properties = calculator.calculate_inertia_from_mesh(source, mass=12)

    assert properties is not None, calculator.last_error
    expected = _expected_vertices(vertices, unit, up_axis)
    dimensions = np.ptp(expected, axis=0)
    x, y, z = dimensions
    np.testing.assert_allclose(properties["center_mass"], expected.mean(axis=0))
    np.testing.assert_allclose(
        properties["inertia"],
        np.diag([y * y + z * z, x * x + z * z, x * x + y * y]),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_recovered_dae_inertia_matches_converted_mesh(tmp_path, monkeypatch, backend):
    source, vertices = _dae(tmp_path / "meshes/visual", up_axis="Y_UP")
    calculator = _calculator(tmp_path, source)
    original = calculator.urdf_path.read_bytes()
    original_mesh = source.read_bytes()

    def unavailable(*args):
        raise RuntimeError("Use the Collada fallback in this regression")

    monkeypatch.setattr(DAE2OBJConverter, "check_blender_available", unavailable)
    manager = MujocoConversionManager(
        calculator.urdf_path, tmp_path / "output", backend=backend
    )
    model = mujoco.MjModel.from_xml_path(str(manager.run()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    expected = _expected_vertices(vertices, 0.001, "Y_UP")
    body = model.body("body")
    np.testing.assert_allclose(body.mass, [12])
    np.testing.assert_allclose(body.ipos, expected.mean(axis=0), atol=1e-10)
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, body.iquat)
    rotation = rotation.reshape(3, 3)
    x, y, z = np.ptp(expected, axis=0)
    tensor = np.diag([y * y + z * z, x * x + z * z, x * x + y * y])
    np.testing.assert_allclose(
        rotation @ np.diag(body.inertia) @ rotation.T,
        tensor,
        rtol=1e-10,
        atol=1e-12,
    )
    gid = next(
        i
        for i in range(model.ngeom)
        if model.geom_bodyid[i] == body.id
        and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
    )
    mesh_id = model.geom_dataid[gid]
    start = model.mesh_vertadr[mesh_id]
    actual = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
    world = actual @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    frame = data.body("body")
    local = (world - frame.xpos) @ frame.xmat.reshape(3, 3)
    np.testing.assert_allclose(
        [local.min(axis=0), local.max(axis=0)],
        [expected.min(axis=0), expected.max(axis=0)],
        atol=1e-6,
    )
    assert manager.last_report["inertia_recovery_attempts"] == 1
    assert manager.last_report["inertia_recoveries"][0]["links"] == ["body"]
    assert calculator.urdf_path.read_bytes() == original
    assert source.read_bytes() == original_mesh


@pytest.mark.parametrize("unit", [0, -1, float("nan"), float("inf")])
def test_invalid_dae_units_do_not_replace_an_existing_obj(tmp_path, unit):
    source, _ = _dae(tmp_path, unit=unit)
    output = tmp_path / "output"
    output.mkdir()
    obj = output / "body.obj"
    obj.write_bytes(b"previous valid OBJ")

    success, outputs = DAE2OBJConverter()._convert_with_trimesh(
        source, output, preserve_textures=True
    )

    assert not success
    assert outputs == []
    assert obj.read_bytes() == b"previous valid OBJ"


def test_missing_unit_and_up_axis_use_collada_defaults(tmp_path):
    source, vertices = _dae(tmp_path, unit=1, up_axis="Y_UP")
    tree = ET.parse(source)
    asset = tree.find(f"{NS}asset")
    for tag in ("unit", "up_axis"):
        asset.remove(asset.find(NS + tag))
    tree.write(source)

    success, outputs = DAE2OBJConverter()._convert_with_trimesh(
        source, tmp_path / "output", preserve_textures=True
    )

    assert success
    actual = trimesh.load(outputs[0], force="mesh", process=False)
    expected = _expected_vertices(vertices, 1, "Y_UP")
    np.testing.assert_allclose(
        actual.bounds, [expected.min(axis=0), expected.max(axis=0)]
    )


@pytest.mark.parametrize("tag", ["unit", "up_axis"])
def test_ambiguous_asset_conventions_preserve_existing_obj(tmp_path, tag):
    source, _ = _dae(tmp_path)
    tree = ET.parse(source)
    asset = tree.find(f"{NS}asset")
    asset.append(ET.fromstring(ET.tostring(asset.find(NS + tag))))
    tree.write(source)
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "body.obj"
    existing.write_bytes(b"previous valid OBJ")

    success, outputs = DAE2OBJConverter()._convert_with_trimesh(
        source, output, preserve_textures=True
    )

    assert not success
    assert outputs == []
    assert existing.read_bytes() == b"previous valid OBJ"


def test_asset_header_reader_does_not_read_large_geometry_tail(tmp_path, monkeypatch):
    source = tmp_path / "large.dae"
    with source.open("wb") as stream:
        stream.write(
            b'<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
            b"<asset><created>2026-11-13T00:00:00Z</created><modified>2026-11-13T00:00:00Z</modified>"
            b'<unit meter="0.001"/><up_axis>Z_UP</up_axis></asset><!--'
        )
        for _ in range(128):
            stream.write(b" " * (64 * 1024))
        stream.write(b"--></COLLADA>")
    consumed = []
    original_open = Path.open

    class Counted:
        def __init__(self, stream):
            self.stream = stream

        def read(self, size=-1):
            chunk = self.stream.read(size)
            consumed.append(len(chunk))
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

    def opened(self, *args, **kwargs):
        stream = original_open(self, *args, **kwargs)
        return Counted(stream) if self == source else stream

    monkeypatch.setattr(Path, "open", opened)
    mesh = trimesh.creation.box(extents=[100, 200, 300])

    normalize_collada_asset(mesh, source)

    np.testing.assert_allclose(mesh.extents, [0.1, 0.2, 0.3])
    assert source.stat().st_size > 8 * 1024**2
    assert 0 < sum(consumed) <= 64 * 1024
