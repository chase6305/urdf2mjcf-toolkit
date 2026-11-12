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


@pytest.mark.parametrize("merge", [False, True])
def test_named_materials_keep_their_own_textures_and_colors(tmp_path, merge):
    import numpy as np
    import trimesh
    from PIL import Image

    scene = trimesh.Scene()
    for index, (name, rgb) in enumerate(
        (("red_paint", (255, 0, 0)), ("blue_paint", (0, 0, 255)))
    ):
        mesh = trimesh.creation.box()
        mesh.apply_translation([2 * index, 0, 0])
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.full((len(mesh.vertices), 2), 0.5),
            material=trimesh.visual.material.PBRMaterial(
                name=name,
                baseColorFactor=[255, 255, 255, 255],
                baseColorTexture=Image.new("RGB", (4, 4), rgb),
            ),
        )
        scene.add_geometry(mesh, geom_name=name)
    source = tmp_path / "paint.glb"
    scene.export(source)
    converter = glb_module.GLB2OBJConverter(
        str(tmp_path / "output"), merge_meshes=merge
    )
    assert converter.convert_glb_to_obj(source)
    directory = tmp_path / "output/paint"
    definitions = {
        line.removeprefix("newmtl ")
        for line in (directory / "paint.mtl").read_text().splitlines()
        if line.startswith("newmtl ")
    }
    references = {
        line.removeprefix("usemtl ")
        for line in (directory / "paint.obj").read_text().splitlines()
        if line.startswith("usemtl ")
    }
    assert references <= definitions
    result = trimesh.load(directory / "paint.obj", force="scene", process=False)
    assert sum(len(mesh.faces) for mesh in result.geometry.values()) == 24
    seen = set()
    for mesh in result.geometry.values():
        colors = mesh.visual.to_color().vertex_colors[:, :3]
        for vertex, rgb in zip(mesh.vertices, colors, strict=True):
            expected = (255, 0, 0) if vertex[0] < 1 else (0, 0, 255)
            np.testing.assert_array_equal(rgb, expected)
            seen.add(tuple(rgb))
        np.testing.assert_array_equal(mesh.visual.material.diffuse[:3], [255, 255, 255])
    assert seen == {(255, 0, 0), (0, 0, 255)}


@pytest.mark.parametrize("merge", [False, True])
def test_untextured_named_materials_export_their_diffuse_colors(tmp_path, merge):
    import numpy as np
    import trimesh

    scene = trimesh.Scene()
    expected = ([128, 0, 0, 255], [0, 0, 192, 255])
    for index, rgba in enumerate(expected):
        mesh = trimesh.creation.box()
        mesh.apply_translation([2 * index, 0, 0])
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.full((len(mesh.vertices), 2), 0.5),
            material=trimesh.visual.material.PBRMaterial(
                name=f"paint_{index}", baseColorFactor=rgba
            ),
        )
        scene.add_geometry(mesh)
    source = tmp_path / "solid.glb"
    scene.export(source)
    converter = glb_module.GLB2OBJConverter(
        str(tmp_path / "output"), merge_meshes=merge
    )
    assert converter.convert_glb_to_obj(source)
    result = trimesh.load(
        tmp_path / "output/solid/solid.obj", force="scene", process=False
    )
    assert len(result.geometry) == 2
    for mesh in result.geometry.values():
        index = int(mesh.vertices[:, 0].mean() > 1)
        np.testing.assert_allclose(
            mesh.visual.material.diffuse, expected[index], atol=1
        )
        assert mesh.visual.material.image is None


def test_embedded_image_names_cannot_escape_the_output_directory(tmp_path):
    import json
    import struct

    import numpy as np
    import trimesh
    from PIL import Image

    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.full((len(mesh.vertices), 2), 0.5),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (4, 4), "red")
        ),
    )
    original = mesh.export(file_type="glb")
    size, kind = struct.unpack_from("<I4s", original, 12)
    assert kind == b"JSON"
    metadata = json.loads(original[20 : 20 + size])
    # Unreferenced embedded images are legal glTF. Their display names are not paths.
    metadata["images"][0]["name"] = "../../victim"
    metadata["materials"] = []
    for shape in metadata["meshes"]:
        for primitive in shape["primitives"]:
            primitive.pop("material", None)
    payload = json.dumps(metadata).encode()
    payload += b" " * (-len(payload) % 4)
    tail = original[20 + size :]
    source = tmp_path / "source.glb"
    source.write_bytes(
        struct.pack("<4sII", b"glTF", 2, 20 + len(payload) + len(tail))
        + struct.pack("<I4s", len(payload), b"JSON")
        + payload
        + tail
    )
    victim = tmp_path / "victim.png"
    victim.write_bytes(b"original source data")
    destination = tmp_path / "output/nested"
    destination.mkdir(parents=True)
    converter = glb_module.GLB2OBJConverter(str(destination))
    mapping, _ = converter.extract_textures_from_glb(source, destination)
    assert victim.read_bytes() == b"original source data"
    assert len(mapping) == 1
    for filename in mapping.values():
        assert (destination / filename).resolve().is_relative_to(destination)
        assert (destination / filename).is_file()
