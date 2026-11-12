import base64
import builtins
import io
import json
import struct
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


def _glb_bytes(metadata, chunks=()):
    payload = json.dumps(metadata).encode()
    payload += b" " * (-len(payload) % 4)
    body = struct.pack("<I4s", len(payload), b"JSON") + payload
    for kind, data in chunks:
        body += struct.pack("<I4s", len(data), kind) + data
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _track_source_reads(monkeypatch, source):
    requests = []
    source_size = source.stat().st_size

    class TrackedReader:
        def __init__(self, file_obj):
            self.file_obj = file_obj

        def __getattr__(self, name):
            return getattr(self.file_obj, name)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.file_obj.__exit__(*args)

        def read(self, size=-1):
            requests.append(size)
            # Fail before a malformed chunk can trigger a large allocation.
            assert 0 <= size <= source_size
            return self.file_obj.read(size)

    def tracked_open(path, mode="r", *args, **kwargs):
        file_obj = builtins.open(path, mode, *args, **kwargs)  # noqa: SIM115
        return TrackedReader(file_obj) if path == source and mode == "rb" else file_obj

    monkeypatch.setattr(glb_module, "open", tracked_open, raising=False)
    return requests


@pytest.mark.parametrize(
    "corruption",
    [
        "short_header",
        "wrong_magic",
        "old_version",
        "wrong_total_length",
        "huge_json_length",
        "unaligned_length",
        "wrong_first_chunk",
        "duplicate_json",
        "duplicate_bin",
        "late_bin",
        "short_chunk_header",
        "huge_bin_length",
        "json_array",
    ],
)
def test_invalid_glb_is_rejected_before_mesh_loading(tmp_path, monkeypatch, corruption):
    data = bytearray(_glb_bytes({"asset": {"version": "2.0"}}))
    if corruption == "short_header":
        data = data[:7]
    elif corruption == "wrong_magic":
        data[:4] = b"FAIL"
    elif corruption == "old_version":
        struct.pack_into("<I", data, 4, 1)
    elif corruption == "wrong_total_length":
        struct.pack_into("<I", data, 8, len(data) + 4)
    elif corruption == "huge_json_length":
        struct.pack_into("<I", data, 12, 0xFFFFFFFC)
    elif corruption == "unaligned_length":
        struct.pack_into("<I", data, 12, len(data) - 21)
    elif corruption == "wrong_first_chunk":
        data[16:20] = b"BIN\0"
    elif corruption == "duplicate_json":
        data = _glb_bytes({}, [(b"JSON", b"{}  ")])
    elif corruption == "duplicate_bin":
        data = _glb_bytes({}, [(b"BIN\0", b"\0" * 4)] * 2)
    elif corruption == "late_bin":
        data = _glb_bytes({}, [(b"EXTN", b"\0" * 4), (b"BIN\0", b"\0" * 4)])
    elif corruption == "short_chunk_header":
        data += b"abcd"
        struct.pack_into("<I", data, 8, len(data))
    elif corruption == "huge_bin_length":
        data += struct.pack("<I4s", 0xFFFFFFFC, b"BIN\0")
        struct.pack_into("<I", data, 8, len(data))
    elif corruption == "json_array":
        data = _glb_bytes([])
    source = tmp_path / "bad.glb"
    source.write_bytes(data)
    requests = _track_source_reads(monkeypatch, source)
    converter = glb_module.GLB2OBJConverter(str(tmp_path / "output"))
    loads = []
    monkeypatch.setattr(converter.trimesh, "load", lambda *args: loads.append(args))
    assert not converter.convert_glb_to_obj(source)
    assert loads == []
    assert all(0 <= size <= len(data) for size in requests)
    assert not list((tmp_path / "output").rglob("*.obj"))


def test_texture_extraction_reads_only_the_image_range(tmp_path, monkeypatch):
    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(image, format="PNG")
    image_bytes = image.getvalue()
    bin_length = 8 * 1024 * 1024
    image_offset = bin_length - 1024
    metadata = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": bin_length}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": image_offset, "byteLength": len(image_bytes)}
        ],
        "images": [{"bufferView": 0, "mimeType": "image/png"}],
    }
    prefix = bytearray(_glb_bytes(metadata))
    total_length = len(prefix) + 8 + bin_length + 12
    struct.pack_into("<I", prefix, 8, total_length)
    source = tmp_path / "sparse.glb"
    with source.open("wb") as file_obj:
        file_obj.write(prefix)
        file_obj.write(struct.pack("<I4s", bin_length, b"BIN\0"))
        file_obj.seek(image_offset, 1)
        file_obj.write(image_bytes)
        file_obj.seek(len(prefix) + 8 + bin_length)
        # Unknown extension chunks after the BIN chunk are legal and are skipped.
        file_obj.write(struct.pack("<I4s", 4, b"EXTN") + b"test")
    requests = _track_source_reads(monkeypatch, source)
    destination = tmp_path / "output"
    converter = glb_module.GLB2OBJConverter(str(destination))
    mapping, parsed = converter.extract_textures_from_glb(source, destination)
    assert parsed == metadata
    assert (destination / mapping["texture_0"]).read_bytes() == image_bytes
    assert sum(requests) < 2048  # Not the 8 MiB BIN payload.
    assert max(requests) < 1024


@pytest.mark.parametrize("invalid_range", ["negative", "padding", "overflow", "index"])
def test_embedded_image_ranges_must_stay_within_the_buffer(tmp_path, invalid_range):
    view = {"buffer": 0, "byteLength": 3}
    image = {"bufferView": 0, "mimeType": "image/png"}
    if invalid_range == "negative":
        view["byteOffset"] = -1
    elif invalid_range == "padding":
        view["byteLength"] = 4
    elif invalid_range == "overflow":
        view["byteLength"] = 1 << 32
    else:
        image["bufferView"] = -1
    source = tmp_path / "bad.glb"
    source.write_bytes(
        _glb_bytes(
            {"buffers": [{"byteLength": 3}], "bufferViews": [view], "images": [image]},
            [(b"BIN\0", b"abcd")],
        )
    )
    destination = tmp_path / "output"
    converter = glb_module.GLB2OBJConverter(str(destination))
    assert converter.extract_textures_from_glb(source, destination) == ({}, None)
    assert list(destination.iterdir()) == []


def test_data_uri_and_external_buffer_images_do_not_read_the_bin_chunk(tmp_path):
    image_bytes = b"embedded PNG bytes"
    metadata = {
        "buffers": [{"byteLength": 4}, {"uri": "external.bin", "byteLength": 4}],
        "bufferViews": [{"buffer": 1, "byteLength": 4}],
        "images": [
            {"bufferView": 0, "mimeType": "image/png"},
            {"uri": "data:image/png;base64," + base64.b64encode(image_bytes).decode()},
        ],
    }
    source = tmp_path / "source.glb"
    # Use an aligned BIN chunk; the external buffer's same range must be ignored.
    source.write_bytes(_glb_bytes(metadata, [(b"BIN\0", b"data")]))
    destination = tmp_path / "output"
    converter = glb_module.GLB2OBJConverter(str(destination))
    mapping, parsed = converter.extract_textures_from_glb(source, destination)
    assert parsed == metadata
    assert mapping == {"texture_1": "texture_001.png"}
    assert (destination / mapping["texture_1"]).read_bytes() == image_bytes
