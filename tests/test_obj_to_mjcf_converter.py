from urdf2mjcf.obj_to_mjcf_converter import OBJ2MJCFImporter


def test_variant_map_is_grouped_and_deterministic(tmp_path):
    for relative_path in ("arm/mesh_1.obj", "base.obj", "arm/mesh_0.obj"):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    importer = OBJ2MJCFImporter(tmp_path)

    assert importer.get_obj_variant_map() == {
        "base": ["base.obj"],
        "mesh": ["arm/mesh_0.obj", "arm/mesh_1.obj"],
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
