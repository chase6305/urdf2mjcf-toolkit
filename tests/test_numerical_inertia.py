"""Analytic checks for finite mass properties at floating-point boundaries."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from urdf2mjcf.urdf_inertia_calculator import URDFInertiaCalculator


def _calculator(tmp_path, *, mass=1, size=1, position="0 0 0"):
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="numeric"><link name="body"><inertial>'
        f'<mass value="{mass}"/><inertia ixx="0" iyy="0" izz="0"/></inertial>'
        f'<visual><origin xyz="{position}"/><geometry><box size="{size} {size} {size}"/>'
        "</geometry></visual></link></robot>"
    )
    return URDFInertiaCalculator(source)


@pytest.mark.parametrize("mass,size", [(1e308, 0.1), (1e-300, 1e10)])
def test_mesh_mass_normalization_keeps_representable_inertia(tmp_path, mass, size):
    calculator = _calculator(tmp_path)
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=[size] * 3).export(mesh)

    result = calculator.calculate_inertia_from_mesh(mesh, mass=mass)

    assert result is not None, calculator.last_error
    expected = np.eye(3) * (mass * (size**2 / 6))
    assert np.all(np.isfinite(result["inertia"]))
    np.testing.assert_allclose(result["inertia"], expected, rtol=1e-12, atol=0)


def test_unrepresentable_mesh_inertia_fails_and_next_success_clears_error(tmp_path):
    calculator = _calculator(tmp_path)
    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=[10] * 3).export(mesh)

    assert calculator.calculate_inertia_from_mesh(mesh, mass=1e308) is None
    assert "finite" in calculator.last_error.lower()
    assert calculator.calculate_inertia_from_mesh(mesh, mass=1) is not None
    assert calculator.last_error is None


def test_large_mass_can_update_a_small_primitive_without_overflow(tmp_path):
    calculator = _calculator(tmp_path, mass=1e308, size=0.1)
    original = calculator.urdf_path.read_bytes()

    assert calculator.update_inertia(), calculator.last_error

    inertia = ET.parse(calculator.urdf_path).find("link/inertial/inertia")
    diagonal = [float(inertia.get(key)) for key in ("ixx", "iyy", "izz")]
    np.testing.assert_allclose(diagonal, [1e308 * (0.1**2 / 6)] * 3, rtol=1e-12)
    assert calculator.urdf_path.with_suffix(".urdf.backup").read_bytes() == original


def test_compound_center_does_not_multiply_volume_by_large_position(tmp_path):
    calculator = _calculator(tmp_path, size=2, position="1e308 0 0")

    assert calculator.update_inertia(), calculator.last_error

    inertial = ET.parse(calculator.urdf_path).find("link/inertial")
    center = np.fromstring(inertial.find("origin").get("xyz"), sep=" ")
    np.testing.assert_array_equal(center, [1e308, 0, 0])
    diagonal = [
        float(inertial.find("inertia").get(key)) for key in ("ixx", "iyy", "izz")
    ]
    np.testing.assert_allclose(diagonal, [2 / 3] * 3, rtol=1e-12)


@pytest.mark.parametrize("relative_tolerance", [0, 1e-6])
def test_regularization_scales_trace_before_it_can_overflow(relative_tolerance):
    matrix = np.eye(3) * 1e308

    regularized, threshold = URDFInertiaCalculator.regularize_inertia_matrix(
        matrix, rel_tol=relative_tolerance
    )

    np.testing.assert_array_equal(regularized, matrix)
    assert threshold == pytest.approx(max(1e-8, 3 * (1e308 * relative_tolerance)))
