import hashlib
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from PIL import Image

import urdf2mjcf.asset_bundle as bundle_module
from urdf2mjcf.asset_bundle import TEXTURE_FILES, bundle_mjcf_assets
from urdf2mjcf.mjcf_editor import MJCFEditor
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def test_bundle_preserves_implicit_names_textures_and_mesh_instances(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    mesh = source / "arm.obj"
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(mesh)
    texture = source / "color.png"
    Image.new("RGB", (4, 4), (20, 40, 80)).save(texture)
    root = ET.fromstring(
        f"""<mujoco><compiler meshdir="meshes" texturedir="meshes"/>
      <asset><mesh file="{mesh}"/><mesh name="large" file="{mesh}" scale="2 1 1"/>
      <texture type="2d" file="{texture}"/><material name="paint" texture="color"/></asset>
      <worldbody><body><geom type="mesh" mesh="arm" material="paint"/>
      <geom type="mesh" mesh="large" pos="1 0 0" material="paint"/></body></worldbody></mujoco>"""
    )
    expected = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    output = tmp_path / "output"
    assets = output / "meshes"
    assets.mkdir(parents=True)
    manifest = bundle_mjcf_assets(root, assets)
    assert len(manifest) == 2
    assert root.find("asset/mesh").get("name") == "arm"
    assert root.find("asset/texture").get("name") == "color"
    assert root.find("asset/mesh").get("file") == root.find(
        "asset/mesh[@name='large']"
    ).get("file")
    for item in manifest:
        contents = (output / item["path"]).read_bytes()
        assert item["sha256"] == hashlib.sha256(contents).hexdigest()
        assert item["size_bytes"] == len(contents)
        assert not Path(item["path"]).is_absolute()
    ET.ElementTree(root).write(output / "robot.xml")
    moved = tmp_path / "moved"
    shutil.copytree(output, moved)
    source.rename(tmp_path / "hidden-source")
    output.rename(tmp_path / "hidden-output")
    actual = mujoco.MjModel.from_xml_path(str(moved / "robot.xml"))
    for attribute in (
        "mesh_vert",
        "mesh_face",
        "mesh_scale",
        "geom_pos",
        "geom_quat",
        "tex_data",
    ):
        np.testing.assert_array_equal(
            getattr(actual, attribute), getattr(expected, attribute)
        )


def test_bundle_keeps_different_same_named_images_and_deduplicates_cubemap_faces(
    tmp_path,
):
    sources = []
    for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
        path = tmp_path / f"source{index}" / "color.png"
        path.parent.mkdir()
        Image.new("RGB", (4, 4), color).save(path)
        sources.append(path)
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="red", type="2d", file=str(sources[0]))
    ET.SubElement(asset, "texture", name="blue", type="2d", file=str(sources[1]))
    cube = ET.SubElement(asset, "texture", name="cube", type="cube")
    for attribute in TEXTURE_FILES[1:]:
        cube.set(attribute, str(sources[0]))
    assets = tmp_path / "output/meshes"
    manifest = bundle_mjcf_assets(root, assets)
    assert len(manifest) == 2
    assert sorted(len(item["references"]) for item in manifest) == [1, 7]
    assert root.find("asset/texture[@name='red']").get("file") != root.find(
        "asset/texture[@name='blue']"
    ).get("file")
    assert {cube.get(attribute) for attribute in TEXTURE_FILES[1:]} == {
        root.find("asset/texture").get("file")
    }
    first = ET.tostring(root)
    assert bundle_mjcf_assets(root, assets) == manifest
    assert ET.tostring(root) == first


def test_absolute_texture_and_mesh_paths_are_not_replaced_by_basename_guesses(tmp_path):
    mesh = tmp_path / "outside/arm.obj"
    texture = tmp_path / "outside/arm.png"
    path = tmp_path / "robot.xml"
    path.write_text(f"""<mujoco><asset><mesh name="arm" file="{mesh}"/>
      <texture name="paint" type="2d" file="{texture}"/></asset></mujoco>""")
    editor = MJCFEditor(path)
    editor.update_mesh_and_texture_paths(variant_map={"arm": ["unrelated.obj"]})
    editor.ensure_visual_prefix_for_all_textures()
    assert editor.root.find("asset/mesh").get("file") == str(mesh)
    assert editor.root.find("asset/texture").get("file") == str(texture)


@pytest.mark.parametrize("backend", ["urdf2mjcf", "mujoco"])
@pytest.mark.parametrize("reference_kind", ["absolute", "file_uri", "package_uri"])
def test_external_visual_and_collision_meshes_survive_output_relocation(
    tmp_path, monkeypatch, backend, reference_kind
):
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    inputs = tmp_path / "inputs"
    source_dir = inputs / "robot_description/urdf"
    source_dir.mkdir(parents=True)
    mesh_dir = inputs / "shapes/models with spaces"
    mesh_dir.mkdir(parents=True)
    visual, collision = mesh_dir / "arm.obj", mesh_dir / "arm_collision.stl"
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(visual)
    trimesh.creation.box(extents=[0.2, 0.3, 0.4]).export(collision)

    def reference(path):
        if reference_kind == "file_uri":
            return path.as_uri()
        if reference_kind == "package_uri":
            return "package://shapes/" + path.relative_to(inputs / "shapes").as_posix()
        return str(path)

    source = source_dir / "robot.urdf"
    source.write_text(
        f"""<robot name="portable"><link name="arm"><inertial><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1"/></inertial>
      <visual><origin xyz="0.1 0.2 0.3"/><geometry><mesh filename="{reference(visual)}" scale="2 1 1"/></geometry></visual>
      <collision><geometry><mesh filename="{reference(collision)}"/></geometry></collision></link></robot>"""
    )
    before = {
        path.relative_to(inputs): path.read_bytes()
        for path in inputs.rglob("*")
        if path.is_file()
    }
    manager = MujocoConversionManager(
        source, tmp_path / "output", backend=backend, auto_recalculate_inertia=False
    )
    output = manager.run()
    expected = mujoco.MjModel.from_xml_path(str(output))
    manifest = manager.last_report["assets"]
    assert len(manifest) == 2
    assert {item["sha256"] for item in manifest} == {
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (visual, collision)
    }
    first_files = [
        asset.get("file") for asset in ET.parse(output).findall("asset/mesh")
    ]
    manager.run()
    # MuJoCo's cache can omit redundant content_type attributes on a later run.
    # Dependency paths and the compiled arrays below must stay unchanged.
    assert [
        asset.get("file") for asset in ET.parse(output).findall("asset/mesh")
    ] == first_files
    assert manager.last_report["assets"] == manifest
    assert before == {
        path.relative_to(inputs): path.read_bytes()
        for path in inputs.rglob("*")
        if path.is_file()
    }
    moved = tmp_path / "moved output"
    shutil.copytree(manager.output_dir, moved)
    inputs.rename(tmp_path / "hidden-inputs")
    manager.output_dir.rename(tmp_path / "hidden-output")
    actual = mujoco.MjModel.from_xml_path(str(moved / output.name))
    assert actual.ngeom == 3
    assert actual.nmesh == 2
    for attribute in (
        "mesh_vert",
        "mesh_face",
        "mesh_scale",
        "geom_pos",
        "geom_quat",
        "body_mass",
        "body_inertia",
    ):
        np.testing.assert_array_equal(
            getattr(actual, attribute), getattr(expected, attribute)
        )
    for item in manifest:
        assert (
            hashlib.sha256((moved / item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


def test_failed_asset_copy_preserves_last_published_model(tmp_path, monkeypatch):
    mesh = tmp_path / "arm.obj"
    trimesh.creation.box().export(mesh)
    source = tmp_path / "robot.urdf"
    source.write_text(f"""<robot name="test"><link name="arm"><visual><geometry>
      <mesh filename="{mesh}"/></geometry></visual></link></robot>""")
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    manager.run()
    previous = {
        path.relative_to(manager.output_dir): path.read_bytes()
        for path in manager.output_dir.rglob("*")
        if path.is_file()
    }

    def fail_copy(*args):
        raise OSError("asset copy failed")

    monkeypatch.setattr(bundle_module, "_copy_external", fail_copy)
    with pytest.raises(OSError, match="asset copy failed"):
        manager.run()
    assert manager.last_report["stage"] == "bundle_assets"
    assert manager.last_report["status"] == "failed"
    assert "assets" not in manager.last_report
    assert previous == {
        path.relative_to(manager.output_dir): path.read_bytes()
        for path in manager.output_dir.rglob("*")
        if path.is_file()
    }


def test_published_xml_is_identical_to_the_compiled_xml(tmp_path, monkeypatch):
    mesh = tmp_path / "arm.obj"
    trimesh.creation.box().export(mesh)
    source = tmp_path / "robot.urdf"
    source.write_text(f"""<robot name="test"><link name="arm"><visual><geometry>
      <mesh filename="{mesh}"/></geometry></visual></link></robot>""")
    compiled = []
    original = mujoco.MjModel.from_xml_path

    def record(path, *args, **kwargs):
        compiled.append(Path(path).read_bytes())
        root = ET.fromstring(compiled[-1])
        assert root.find("compiler").get("meshdir") == "meshes"
        for asset in root.findall("asset/mesh"):
            assert not Path(asset.get("file")).is_absolute()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", record)
    manager = MujocoConversionManager(source, tmp_path / "output", backend="mujoco")
    output = manager.run()
    assert compiled == [output.read_bytes()]


def test_missing_final_asset_is_reported_with_its_name(tmp_path):
    root = ET.fromstring(
        '<mujoco><asset><mesh name="missing_arm" file="absent.obj"/></asset></mujoco>'
    )
    with pytest.raises(FileNotFoundError, match="missing_arm.*absent.obj"):
        bundle_mjcf_assets(root, tmp_path / "meshes")


def test_internal_assets_are_kept_and_manifest_hashes_match_their_bytes(tmp_path):
    assets = tmp_path / "meshes"
    path = assets / "visual/arm.obj"
    path.parent.mkdir(parents=True)
    trimesh.creation.box().export(path)
    root = ET.fromstring(
        '<mujoco><asset><mesh name="arm" file="visual/arm.obj"/></asset></mujoco>'
    )
    manifest = bundle_mjcf_assets(root, assets)
    assert manifest[0]["path"] == "meshes/visual/arm.obj"
    assert manifest[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (assets / "bundled").exists()


def test_external_symlink_and_parent_reference_are_copied_inside_output(tmp_path):
    source = tmp_path / "outside.obj"
    trimesh.creation.box().export(source)
    assets = tmp_path / "output/meshes"
    assets.mkdir(parents=True)
    (assets / "linked.obj").symlink_to(source)
    root = ET.fromstring(
        '<mujoco><asset><mesh name="linked" file="linked.obj"/><mesh name="parent" file="../../outside.obj"/></asset></mujoco>'
    )
    manifest = bundle_mjcf_assets(root, assets)
    assert len(manifest) == 1
    assert len(manifest[0]["references"]) == 2
    destination = assets.parent / manifest[0]["path"]
    assert not destination.is_symlink()
    assert destination.read_bytes() == source.read_bytes()
    assert destination.resolve().is_relative_to(assets)


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf"])
def test_upstream_texture_directory_is_preserved_and_then_bundled(
    tmp_path, monkeypatch, backend
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    texture_dir = inputs / "shared-textures"
    texture_dir.mkdir()
    Image.new("RGB", (4, 4), (20, 40, 80)).save(texture_dir / "paint.png")
    # A same-named local file must not replace the explicit upstream reference.
    visual_dir = inputs / "visual"
    visual_dir.mkdir()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(visual_dir / "paint.png")
    source = inputs / "robot.urdf"
    source.write_text('<robot name="test"><link name="arm"/></robot>')
    manager = MujocoConversionManager(source, tmp_path / "output", backend=backend)

    def upstream(converter):
        converter.output_xml = converter.output_dir / "robot.xml"
        converter.output_xml.write_text(
            f"""<mujoco><compiler texturedir="{texture_dir}"/>
          <asset><texture name="paint" type="2d" file="paint.png"/>
          <material name="surface" texture="paint"/></asset>
          <worldbody><body name="arm"><geom type="box" size="0.1 0.2 0.3" material="surface"/></body></worldbody></mujoco>"""
        )
        return True

    monkeypatch.setattr(manager, "_convert_urdf_with_recovery", upstream)
    output = manager.run()
    assert len(manager.last_report["assets"]) == 1
    root = ET.parse(output)
    assert root.find("compiler").get("texturedir") == "meshes"
    moved = tmp_path / "moved"
    shutil.copytree(manager.output_dir, moved)
    inputs.rename(tmp_path / "hidden-inputs")
    model = mujoco.MjModel.from_xml_path(str(moved / output.name))
    texture_id = model.texture("paint").id
    offset = model.tex_adr[texture_id]
    np.testing.assert_array_equal(
        model.tex_data[offset : offset + 48].reshape(-1, 3),
        np.tile([20, 40, 80], (16, 1)),
    )
