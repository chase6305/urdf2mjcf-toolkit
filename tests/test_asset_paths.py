from pathlib import Path

import pytest

from urdf2mjcf.asset_paths import MeshPathResolver


@pytest.mark.parametrize(
    "reference",
    [
        "meshes/visual/arm.glb",
        "Visual/arm.glb",
        "package://robot_description/meshes/visual/arm.glb",
    ],
)
def test_nested_converted_obj_is_preferred(tmp_path, reference):
    assets = tmp_path / "assets"
    obj = assets / "visual/arm/arm.obj"
    obj.parent.mkdir(parents=True)
    obj.touch()
    (assets / "visual/arm.glb").touch()

    assert MeshPathResolver(tmp_path, assets).resolve(reference) == obj


def test_absolute_source_path_uses_staged_obj(tmp_path):
    original = tmp_path / "robot/meshes/visual/arm.stl"
    original.parent.mkdir(parents=True)
    original.touch()
    assets = tmp_path / "assets"
    obj = assets / "visual/arm.obj"
    obj.parent.mkdir(parents=True)
    obj.touch()

    assert MeshPathResolver(tmp_path / "robot", assets).resolve(str(original)) == obj


def test_file_uri_with_spaces_is_resolved(tmp_path):
    original = tmp_path / "arm mesh.obj"
    original.touch()

    assert (
        MeshPathResolver(tmp_path, tmp_path / "assets").resolve(original.as_uri())
        == original
    )


def test_package_uri_resolves_from_urdf_subdirectory(tmp_path):
    package = tmp_path / "robot_description"
    source_dir = package / "urdf"
    source_dir.mkdir(parents=True)
    mesh = package / "meshes/arm.obj"
    mesh.parent.mkdir()
    mesh.touch()

    assert (
        MeshPathResolver(source_dir, tmp_path / "assets").resolve(
            "package://robot_description/meshes/arm.obj"
        )
        == mesh
    )


def test_exact_source_is_not_replaced_by_unrelated_basename(tmp_path):
    source = tmp_path / "collision/arm.obj"
    source.parent.mkdir()
    source.touch()
    visual = tmp_path / "assets/visual/arm.obj"
    visual.parent.mkdir(parents=True)
    visual.touch()

    assert (
        MeshPathResolver(tmp_path, tmp_path / "assets").resolve("collision/arm.obj")
        == source
    )


def test_ambiguous_basename_is_rejected(tmp_path):
    for directory in ("left", "right"):
        mesh = tmp_path / "assets" / directory / "arm.obj"
        mesh.parent.mkdir(parents=True)
        mesh.touch()

    with pytest.raises(ValueError, match="Ambiguous mesh reference"):
        MeshPathResolver(tmp_path, tmp_path / "assets").resolve("arm.obj")


def test_basename_fallback_indexes_assets_only_once(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    for name in ("arm.stl", "head.stl"):
        (assets / name).touch()
        (assets / name).with_suffix(".obj").touch()
    original_rglob = Path.rglob
    scans = []

    def counted_rglob(path, pattern):
        scans.append(path)
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", counted_rglob)
    resolver = MeshPathResolver(tmp_path, assets)
    assert resolver.resolve("legacy/arm.stl") == assets / "arm.obj"
    assert resolver.resolve("legacy/head.stl") == assets / "head.obj"
    assert scans == [assets]


def test_package_identity_is_not_replaced_by_local_same_named_mesh(tmp_path):
    source_dir = tmp_path / "package_a/urdf"
    source_dir.mkdir(parents=True)
    own = tmp_path / "package_a/meshes/visual/arm.obj"
    other = tmp_path / "package_b/meshes/visual/arm.obj"
    staged = tmp_path / "assets/visual/arm.obj"
    for path in (own, other, staged):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    resolver = MeshPathResolver(
        source_dir, tmp_path / "assets", source_map={own: staged}
    )
    assert resolver.resolve("package://package_a/meshes/visual/arm.obj") == staged
    assert resolver.resolve("package://package_b/meshes/visual/arm.obj") == other
    with pytest.raises(FileNotFoundError, match="package_c"):
        resolver.resolve("package://package_c/meshes/visual/arm.obj")
