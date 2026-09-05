import xml.etree.ElementTree as ET

import pytest

from urdf2mjcf.obj_to_mjcf_converter import OBJ2MJCFImporter


def _generated_model(directory, meshes):
    directory.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujoco")
    asset = ET.SubElement(root, "asset")
    body = ET.SubElement(ET.SubElement(root, "worldbody"), "body")
    for mesh in meshes:
        (directory / mesh).touch()
        ET.SubElement(asset, "mesh", file=mesh)
        ET.SubElement(
            body, "geom", mesh=mesh.removesuffix(".obj"), **{"class": "visual"}
        )
    ET.ElementTree(root).write(directory / f"{directory.name}.xml")


def test_variant_map_is_grouped_and_deterministic(tmp_path):
    _generated_model(tmp_path / "arm/mesh", ["mesh_0.obj", "mesh_1.obj", "mesh_10.obj"])
    (tmp_path / "base.obj").touch()

    importer = OBJ2MJCFImporter(tmp_path)

    assert importer.get_obj_variant_map() == {
        "base": ["base.obj"],
        "mesh": ["arm/mesh/mesh_0.obj", "arm/mesh/mesh_1.obj", "arm/mesh/mesh_10.obj"],
    }


def test_non_recursive_run_processes_root_and_direct_children(tmp_path, monkeypatch):
    (tmp_path / "root.obj").touch()
    child = tmp_path / "child"
    child.mkdir()
    (child / "child.obj").touch()
    nested = child / "nested"
    nested.mkdir()
    (nested / "nested.obj").touch()
    importer = OBJ2MJCFImporter(tmp_path)
    seen = []

    monkeypatch.setattr(
        importer, "handle_dir", lambda path, _command: seen.append(path) or True
    )

    assert importer.run() == [tmp_path, child]
    assert seen == [tmp_path, child]


def test_strict_run_rejects_partial_failure(tmp_path, monkeypatch):
    import pytest

    (tmp_path / "root.obj").touch()
    child = tmp_path / "child"
    child.mkdir()
    (child / "child.obj").touch()
    importer = OBJ2MJCFImporter(tmp_path)
    monkeypatch.setattr(importer, "handle_dir", lambda path, _command: path == tmp_path)
    with pytest.raises(RuntimeError, match="OBJ to MJCF conversion failed"):
        importer.run(strict=True)
    assert importer.failed_directories == [child]


def test_generated_objs_replace_sources_in_variant_map(tmp_path):
    (tmp_path / "arm.obj").touch()
    generated = tmp_path / "arm"
    _generated_model(generated, ["arm.obj"])
    (generated / "arm_1.obj").touch()  # Stale part from a previous conversion.
    assert OBJ2MJCFImporter(tmp_path).get_obj_variant_map() == {"arm": ["arm/arm.obj"]}


def test_numeric_suffixes_are_independent_input_meshes(tmp_path):
    (tmp_path / "link_1.obj").touch()
    (tmp_path / "link_2.obj").touch()
    assert OBJ2MJCFImporter(tmp_path).get_obj_variant_map() == {
        "link_1": ["link_1.obj"],
        "link_2": ["link_2.obj"],
    }
    _generated_model(tmp_path / "link_1", ["link_1_0.obj", "link_1_1.obj"])
    assert OBJ2MJCFImporter(tmp_path).get_obj_variant_map() == {
        "link_1": ["link_1/link_1_0.obj", "link_1/link_1_1.obj"],
        "link_2": ["link_2.obj"],
    }


def test_unrelated_duplicate_basenames_are_rejected(tmp_path):
    for side in ("left", "right"):
        _generated_model(tmp_path / side / "arm", ["arm.obj"])
    with pytest.raises(ValueError, match="Ambiguous mesh name 'arm'"):
        OBJ2MJCFImporter(tmp_path).get_obj_variant_map()


def test_variant_order_comes_from_xml_and_skips_stale_parts(tmp_path):
    _generated_model(tmp_path / "arm", [f"arm_{i}.obj" for i in range(12)])
    (tmp_path / "arm/arm_old.obj").touch()
    assert OBJ2MJCFImporter(tmp_path).get_obj_variant_map()["arm"] == [
        f"arm/arm_{i}.obj" for i in range(12)
    ]


@pytest.mark.parametrize("problem", ["missing", "escape", "malformed"])
def test_invalid_generated_assets_fail_explicitly(tmp_path, problem):
    generated = tmp_path / "arm"
    _generated_model(generated, ["arm.obj"])
    if problem == "missing":
        (generated / "arm.obj").unlink()
    elif problem == "escape":
        tree = ET.parse(generated / "arm.xml")
        tree.getroot().find("asset/mesh").set("file", "../arm.obj")
        (tmp_path / "arm.obj").touch()
        tree.write(generated / "arm.xml")
    else:
        (generated / "arm.xml").write_text("broken XML")
    with pytest.raises(ValueError, match="Generated|generated"):
        OBJ2MJCFImporter(tmp_path).get_obj_variant_map()
