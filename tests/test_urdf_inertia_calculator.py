import xml.etree.ElementTree as ET

import numpy as np

from urdf2mjcf.urdf_inertia_calculator import URDFInertiaCalculator


def _calculator(tmp_path, body='<link name="arm" />'):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(f'<robot name="test">{body}</robot>', encoding="utf-8")
    return URDFInertiaCalculator(urdf)


def test_mesh_origin_is_read_from_visual_element(tmp_path):
    mesh = tmp_path / "meshes" / "arm.obj"
    mesh.parent.mkdir()
    mesh.touch()
    calculator = _calculator(tmp_path)
    link = ET.fromstring(
        '<link name="arm"><visual><origin xyz="1 2 3" />'
        '<geometry><mesh filename="meshes/arm.obj" /></geometry></visual></link>'
    )

    mesh_path, origin = calculator.find_mesh_file_for_link(link)

    assert mesh_path == mesh
    np.testing.assert_array_equal(origin, [1.0, 2.0, 3.0])


def test_file_uri_mesh_path_is_supported(tmp_path):
    mesh = tmp_path / "arm.obj"
    mesh.touch()
    calculator = _calculator(tmp_path)
    link = ET.fromstring(
        f'<link name="arm"><visual><geometry><mesh filename="file://{mesh}" />'
        "</geometry></visual></link>"
    )

    mesh_path, _ = calculator.find_mesh_file_for_link(link)

    assert mesh_path == mesh


def test_regularization_makes_singular_matrix_positive_definite():
    matrix = np.diag([1.0, 0.0, -1.0])

    regularized, threshold = URDFInertiaCalculator.regularize_inertia_matrix(matrix)

    assert threshold > 0
    assert np.all(np.linalg.eigvalsh(regularized) >= threshold)
