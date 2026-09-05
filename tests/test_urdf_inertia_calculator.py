import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

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


@pytest.mark.parametrize(
    "parameter", ["regularize_rel_tol", "regularize_abs_tol", "enforce_min_eig"]
)
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_regularization_options_leave_source_unchanged(
    tmp_path, parameter, value
):
    source = _calculator(tmp_path).urdf_path
    original = source.read_bytes()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        URDFInertiaCalculator(source, **{parameter: value})
    assert source.read_bytes() == original
    assert not source.with_suffix(".urdf.backup").exists()


def test_regularization_helper_rejects_infinite_minimum():
    with pytest.raises(ValueError, match="min_eig must be finite"):
        URDFInertiaCalculator.regularize_inertia_matrix(np.eye(3), min_eig=float("inf"))


def test_finite_regularization_overflow_does_not_write_source(tmp_path):
    calculator = _calculator(
        tmp_path,
        '<link name="arm"><inertial><mass value="10"/>'
        '<inertia ixx="0" iyy="0" izz="0"/></inertial>'
        '<visual><geometry><box size="1 1 1"/></geometry></visual></link>',
    )
    calculator.regularize_rel_tol = 1e308
    original = calculator.urdf_path.read_bytes()

    assert not calculator.update_inertia(only_invalid=True)
    assert "threshold is not finite" in calculator.last_error
    assert calculator.updated_links == []
    assert calculator.urdf_path.read_bytes() == original
    assert not calculator.urdf_path.with_suffix(".urdf.backup").exists()


def _read_tensor(inertial):
    element = inertial.find("inertia")
    xx, yy, zz, xy, xz, yz = [
        float(element.get(key)) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    ]
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def test_recalculation_applies_mesh_scale_and_both_coordinate_frames(tmp_path):
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    mesh.apply_translation([0.2, 0.3, 0.4])
    mesh.export(tmp_path / "box.obj")
    calculator = _calculator(
        tmp_path,
        """<link name="base_link"><inertial>
      <origin rpy="0.3 -0.4 0.5"/><mass value="12"/><inertia ixx="0" iyy="0" izz="0"/></inertial>
      <visual><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
      <geometry><mesh filename="box.obj" scale="2 3 4"/></geometry></visual></link>""",
    )
    original = calculator.urdf_path.read_bytes()
    assert calculator.update_inertia()
    inertial = ET.parse(calculator.urdf_path).find("link/inertial")
    # Scale first, then rotate the mesh about Z, then apply the origin translation.
    np.testing.assert_allclose(
        np.fromstring(inertial.find("origin").get("xyz"), sep=" "),
        [0.1, 2.4, 4.6],
        atol=1e-8,
    )
    from test_physics_correctness import _rotation

    rotation = _rotation([0.3, -0.4, 0.5])
    # Dimensions become 2, 6, 12; a quarter-turn swaps the X and Y moments.
    expected_link_tensor = np.diag([2**2 + 12**2, 6**2 + 12**2, 2**2 + 6**2])
    np.testing.assert_allclose(
        rotation @ _read_tensor(inertial) @ rotation.T, expected_link_tensor, atol=1e-7
    )
    assert inertial.find("mass").get("value") == "12"
    assert calculator.urdf_path.with_suffix(".urdf.backup").read_bytes() == original


def test_compound_geometry_includes_parallel_axis_terms(tmp_path):
    calculator = _calculator(
        tmp_path,
        """<link name="origin_base"><inertial><mass value="2"/>
      <inertia ixx="0" iyy="0" izz="0"/></inertial>
      <visual><origin xyz="-1 0 0"/><geometry><box size="1 1 1"/></geometry></visual>
      <visual><origin xyz="1 0 0"/><geometry><box size="1 1 1"/></geometry></visual></link>""",
    )
    assert calculator.update_inertia()
    inertial = ET.parse(calculator.urdf_path).find("link/inertial")
    np.testing.assert_allclose(_read_tensor(inertial), np.diag([1 / 3, 7 / 3, 7 / 3]))
    np.testing.assert_allclose(
        np.fromstring(inertial.find("origin").get("xyz"), sep=" "), [0, 0, 0]
    )


def test_nonphysical_principal_moments_are_rejected_even_with_valid_diagonal():
    matrix = np.full((3, 3), 1.0) + np.eye(3)
    valid, reason = URDFInertiaCalculator.is_valid_inertia_matrix(
        **URDFInertiaCalculator.inertia_matrix_to_urdf_params(matrix)
    )
    assert not valid
    assert "triangle" in reason


def test_failed_link_does_not_publish_partial_inertia_changes(tmp_path):
    trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]]
    ).export(tmp_path / "open.obj")
    calculator = _calculator(
        tmp_path,
        """<link name="good"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
      <visual><geometry><box size="1 2 3"/></geometry></visual></link>
      <link name="bad"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>
      <visual><geometry><mesh filename="open.obj"/></geometry></visual></link>""",
    )
    original = calculator.urdf_path.read_bytes()
    assert not calculator.update_inertia()
    assert "closed" in calculator.last_error
    assert calculator.urdf_path.read_bytes() == original


@pytest.mark.parametrize(
    "shape, expected",
    [
        ('<sphere radius="2"/>', [3.2, 3.2, 3.2]),
        ('<cylinder radius="2" length="6"/>', [8, 8, 4]),
    ],
)
def test_primitive_inertia_uses_analytic_solid_formulas(tmp_path, shape, expected):
    calculator = _calculator(
        tmp_path,
        f"""<link name="body"><inertial><mass value="2"/></inertial>
      <collision><geometry>{shape}</geometry></collision></link>""",
    )
    assert calculator.update_inertia()
    np.testing.assert_allclose(
        _read_tensor(ET.parse(calculator.urdf_path).find("link/inertial")),
        np.diag(expected),
    )


def test_selective_recalculation_preserves_valid_inertia_and_skips_its_open_mesh(
    tmp_path,
):
    calculator = _calculator(
        tmp_path,
        """<link name="calibrated"><inertial><origin xyz="0.1 0.2 0.3" rpy="0.3 -0.4 0.5"/>
        <mass value="12"/><inertia ixx="2" iyy="3" izz="4" ixy="0.1" ixz="0.2" iyz="0.3"/></inertial>
        <visual><geometry><mesh filename="not-needed.obj"/></geometry></visual></link>
        <link name="invalid"><inertial><mass value="12"/><inertia ixx="0" iyy="0" izz="0"/></inertial>
        <visual><geometry><box size="1 2 3"/></geometry></visual></link>""",
    )
    original = ET.tostring(
        ET.parse(calculator.urdf_path).find("link[@name='calibrated']")
    )
    assert calculator.update_inertia(only_invalid=True)
    assert calculator.updated_links == ["invalid"]
    root = ET.parse(calculator.urdf_path)
    assert ET.tostring(root.find("link[@name='calibrated']")) == original
    np.testing.assert_allclose(
        _read_tensor(root.find("link[@name='invalid']/inertial")), np.diag([13, 10, 5])
    )
    current = calculator.urdf_path.read_bytes()
    assert calculator.update_inertia(only_invalid=True)
    assert calculator.updated_links == []
    assert calculator.urdf_path.read_bytes() == current


def test_selective_recalculation_fails_if_invalid_link_has_no_geometry(tmp_path):
    calculator = _calculator(
        tmp_path,
        '<link name="broken"><inertial><mass value="1"/><inertia ixx="0" iyy="0" izz="0"/></inertial></link>',
    )
    original = calculator.urdf_path.read_bytes()
    assert not calculator.update_inertia(only_invalid=True)
    assert (
        "broken" in calculator.last_error
        and "without geometry" in calculator.last_error
    )
    assert calculator.updated_links == []
    assert calculator.urdf_path.read_bytes() == original
    assert not calculator.urdf_path.with_suffix(".urdf.backup").exists()


def test_only_invalid_cli_leaves_valid_source_and_creates_no_backup(
    tmp_path, monkeypatch
):
    import sys

    from urdf2mjcf.urdf_inertia_calculator import main

    calculator = _calculator(
        tmp_path,
        """<link name="calibrated"><inertial><mass value="1"/>
    <inertia ixx="2" iyy="3" izz="4"/></inertial>
    <visual><geometry><box size="1 1 1"/></geometry></visual></link>""",
    )
    original = calculator.urdf_path.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["urdf-inertia", str(calculator.urdf_path), "--only-invalid"]
    )
    assert main() == 0
    assert calculator.urdf_path.read_bytes() == original
    assert not calculator.urdf_path.with_suffix(".urdf.backup").exists()
