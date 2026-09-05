import sys

import pytest

import urdf2mjcf.glb_to_obj_converter as glb_module


@pytest.mark.parametrize("successes", [0, 1, 2])
def test_cli_fails_if_any_glb_conversion_failed(tmp_path, monkeypatch, successes):
    (tmp_path / "arm.Glb").touch()
    (tmp_path / "head.glb").touch()
    monkeypatch.setattr(
        sys, "argv", ["glb-to-obj", str(tmp_path), "-o", str(tmp_path / "output")]
    )
    monkeypatch.setattr(
        glb_module.GLB2OBJConverter, "convert_batch", lambda *args: successes
    )
    assert glb_module.main() == (0 if successes == 2 else 1)


@pytest.mark.parametrize("merge", [False, True])
def test_glb_instances_and_transforms_are_preserved(tmp_path, merge):
    import numpy as np
    import trimesh

    scene = trimesh.Scene()
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    scene.add_geometry(mesh, geom_name="box", node_name="first")
    transform = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    transform[:3, 3] = [4, 5, 6]
    scene.graph.update(frame_to="second", geometry="box", matrix=transform)
    source = tmp_path / "scene.glb"
    scene.export(source)
    converter = glb_module.GLB2OBJConverter(
        str(tmp_path / "output"), merge_meshes=merge
    )
    assert converter.convert_glb_to_obj(source)
    output = trimesh.load(tmp_path / "output/scene/scene.obj", force="mesh")
    expected = scene.to_mesh()
    assert len(output.faces) == len(expected.faces)
    np.testing.assert_allclose(output.bounds, expected.bounds, atol=1e-7)
    np.testing.assert_allclose(output.center_mass, expected.center_mass, atol=1e-7)


def test_glb_export_does_not_round_small_mesh_to_zero(tmp_path):
    import numpy as np
    import trimesh

    source = tmp_path / "tiny.glb"
    trimesh.creation.box(extents=[1e-7, 2e-7, 3e-7]).export(source)
    converter = glb_module.GLB2OBJConverter(str(tmp_path / "output"))
    assert converter.convert_glb_to_obj(source)
    output = trimesh.load(
        tmp_path / "output/tiny/tiny.obj", force="mesh", process=False
    )
    np.testing.assert_allclose(output.extents, [1e-7, 2e-7, 3e-7], rtol=1e-6, atol=0)
