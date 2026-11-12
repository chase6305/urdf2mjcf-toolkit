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


@pytest.mark.parametrize("existing_output", [False, True])
def test_successful_process_without_fresh_models_is_rejected(
    tmp_path, monkeypatch, existing_output
):
    from types import SimpleNamespace

    import urdf2mjcf.obj_to_mjcf_converter as obj_module

    (tmp_path / "arm.obj").touch()
    if existing_output:
        _generated_model(tmp_path / "arm", ["arm.obj"])
    monkeypatch.setattr(
        obj_module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout="", returncode=0),
    )
    assert not OBJ2MJCFImporter(tmp_path).handle_dir(tmp_path)


@pytest.mark.parametrize("complete", [False, True])
def test_success_requires_each_input_to_produce_a_model(
    tmp_path, monkeypatch, complete
):
    from types import SimpleNamespace

    import urdf2mjcf.obj_to_mjcf_converter as obj_module

    for name in ("arm", "leg"):
        (tmp_path / f"{name}.obj").touch()

    def run(*args, **kwargs):
        for name in (("arm", "leg") if complete else ("arm",)):
            _generated_model(tmp_path / name, [f"{name}.obj"])
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(obj_module.subprocess, "run", run)
    assert OBJ2MJCFImporter(tmp_path).handle_dir(tmp_path) is complete


@pytest.mark.parametrize("successful", [False, True])
def test_mixed_case_obj_reaches_converter_and_temporary_alias_is_cleaned(
    tmp_path, monkeypatch, successful
):
    import subprocess
    from types import SimpleNamespace

    import urdf2mjcf.obj_to_mjcf_converter as obj_module

    source = tmp_path / "arm.ObJ"
    source.write_bytes(b"original OBJ and relative material references")
    alias = source.with_suffix(".obj")

    def run(*args, **kwargs):
        assert alias.read_bytes() == source.read_bytes()
        if not successful:
            raise subprocess.CalledProcessError(1, "obj2mjcf", output="invalid OBJ")
        _generated_model(tmp_path / "arm", ["arm.obj"])
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(obj_module.subprocess, "run", run)
    assert OBJ2MJCFImporter(tmp_path).handle_dir(tmp_path) is successful
    assert not alias.exists()
    assert source.read_bytes() == b"original OBJ and relative material references"


def test_obj_inputs_with_the_same_stem_fail_before_overwriting_outputs(
    tmp_path, monkeypatch
):
    import urdf2mjcf.obj_to_mjcf_converter as obj_module

    lower = tmp_path / "arm.obj"
    upper = tmp_path / "arm.OBJ"
    lower.write_bytes(b"first mesh")
    upper.write_bytes(b"second mesh")
    _generated_model(tmp_path / "arm", ["arm.obj"])
    previous = (tmp_path / "arm/arm.xml").read_bytes()

    def unexpected(*args, **kwargs):
        pytest.fail("Ambiguous inputs must be rejected before running obj2mjcf")

    monkeypatch.setattr(obj_module.subprocess, "run", unexpected)
    assert not OBJ2MJCFImporter(tmp_path).handle_dir(tmp_path)
    assert lower.read_bytes() == b"first mesh"
    assert upper.read_bytes() == b"second mesh"
    assert (tmp_path / "arm/arm.xml").read_bytes() == previous


def test_run_does_not_let_old_parent_xml_hide_current_outputs(tmp_path, monkeypatch):
    parent = tmp_path / "paint"
    _generated_model(parent, ["old.obj"])
    (parent / "paint.obj").touch()
    previous = (parent / "paint.xml").read_bytes()
    (tmp_path / "root.obj").touch()
    importer = OBJ2MJCFImporter(tmp_path)

    def convert(directory, command):
        if directory == tmp_path:
            _generated_model(tmp_path / "root", ["root.obj"])
            return True
        assert directory == parent
        _generated_model(parent / "paint", ["fresh.obj"])
        _generated_model(parent / "old", ["old.obj"])
        return True

    monkeypatch.setattr(importer, "handle_dir", convert)
    importer.run(recursive=True, strict=True)
    assert importer.get_obj_variant_map()["paint"] == ["paint/paint/fresh.obj"]
    assert (parent / "paint.xml").read_bytes() == previous


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
