"""3MF model units apply to mesh vertices and build-item translations."""

import xml.etree.ElementTree as ET
import zipfile

import mujoco
import numpy as np
import pytest
import trimesh

from urdf2mjcf.mesh_to_obj_converter import MeshToOBJConverter
from urdf2mjcf.urdf_inertia_calculator import URDFInertiaCalculator
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager

UNITS = [
    (None, 0.001),
    ("micron", 1e-6),
    ("millimeter", 0.001),
    ("centimeter", 0.01),
    ("inch", 0.0254),
    ("foot", 0.3048),
    ("meter", 1),
]


def _source(directory, unit):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "body.3mf"
    shape = trimesh.creation.box(extents=[100, 200, 300])
    root = ET.Element(
        "model", xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
    )
    if unit is not None:
        root.set("unit", unit)
    resource = ET.SubElement(
        ET.SubElement(root, "resources"), "object", id="1", type="model"
    )
    mesh = ET.SubElement(resource, "mesh")
    vertices = ET.SubElement(mesh, "vertices")
    for x, y, z in shape.vertices:
        ET.SubElement(vertices, "vertex", x=str(x), y=str(y), z=str(z))
    triangles = ET.SubElement(mesh, "triangles")
    for v1, v2, v3 in shape.faces:
        ET.SubElement(triangles, "triangle", v1=str(v1), v2=str(v2), v3=str(v3))
    ET.SubElement(
        ET.SubElement(root, "build"),
        "item",
        objectid="1",
        transform="1 0 0 0 1 0 0 0 1 1000 2000 3000",
    )
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("3D/3dmodel.model", ET.tostring(root))
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>',
        )
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>',
        )
    return source, shape.vertices + [1000, 2000, 3000]


def _calculator(directory, source):
    urdf = directory / "robot.urdf"
    urdf.write_text(
        '<robot name="3mf_units"><link name="body"><inertial><mass value="12"/>'
        '<inertia ixx="0" iyy="0" izz="0"/></inertial><visual><geometry>'
        f'<mesh filename="{source.relative_to(directory).as_posix()}"/>'
        "</geometry></visual></link></robot>"
    )
    return URDFInertiaCalculator(urdf)


@pytest.mark.parametrize("unit,factor", UNITS)
def test_3mf_obj_conversion_scales_vertices_and_build_translations(
    tmp_path, unit, factor
):
    source, vertices = _source(tmp_path, unit)
    original = source.read_bytes()

    output = MeshToOBJConverter().convert_file(source)

    actual = trimesh.load(output, force="mesh", process=False)
    expected = vertices * factor
    np.testing.assert_allclose(
        actual.bounds, [expected.min(axis=0), expected.max(axis=0)]
    )
    assert source.read_bytes() == original


@pytest.mark.parametrize("unit,factor", UNITS)
def test_3mf_direct_inertia_uses_declared_units(tmp_path, unit, factor):
    source, vertices = _source(tmp_path, unit)
    calculator = _calculator(tmp_path, source)

    properties = calculator.calculate_inertia_from_mesh(str(source), mass=12)

    assert properties is not None, calculator.last_error
    np.testing.assert_allclose(
        properties["center_mass"], (vertices * factor).mean(axis=0)
    )
    x, y, z = np.array([100, 200, 300]) * factor
    tensor = np.diag([y * y + z * z, x * x + z * z, x * x + y * y])
    np.testing.assert_allclose(
        properties["inertia"], tensor, rtol=1e-10, atol=np.max(tensor) * 1e-10
    )


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_3mf_inertia_recovery_matches_meter_based_output(tmp_path, backend):
    source, _ = _source(tmp_path / "shapes", "millimeter")
    calculator = _calculator(tmp_path, source)
    original = calculator.urdf_path.read_bytes()
    manager = MujocoConversionManager(
        calculator.urdf_path, tmp_path / "output", backend=backend
    )

    model = mujoco.MjModel.from_xml_path(str(manager.run()))

    body = model.body("body")
    np.testing.assert_allclose(body.mass, [12])
    np.testing.assert_allclose(body.ipos, [1, 2, 3])
    np.testing.assert_allclose(np.sort(body.inertia), [0.05, 0.1, 0.13])
    assert manager.last_report["inertia_recovery_attempts"] == 1
    assert calculator.urdf_path.read_bytes() == original


@pytest.mark.parametrize("unit", ["furlong", ""])
def test_unknown_3mf_unit_does_not_overwrite_existing_output(tmp_path, unit):
    source, _ = _source(tmp_path, unit)
    output = tmp_path / "existing.obj"
    output.write_bytes(b"previous valid OBJ")

    with pytest.raises(ValueError, match="3MF unit"):
        MeshToOBJConverter(overwrite=True).convert_file(source, output)

    assert output.read_bytes() == b"previous valid OBJ"
