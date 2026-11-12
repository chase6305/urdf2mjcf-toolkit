"""Publication must preserve OBJ side files across single and batch jobs."""

import pytest

from urdf2mjcf.batch_conversion import BatchConversionManager
from urdf2mjcf.inspection import inspect_urdf
from urdf2mjcf.obj_assets import iter_obj_dependencies
from urdf2mjcf.urdf_to_mujoco_converter import MujocoConversionManager


def _robot(path, *, mesh=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    geometry = (
        '<mesh filename="meshes/visual/arm.obj"/>' if mesh else '<box size="1 1 1"/>'
    )
    path.write_text(
        '<robot name="sidecars"><link name="arm"><inertial><mass value="1"/>'
        '<inertia ixx="1" iyy="1" izz="1"/></inertial>'
        f"<visual><geometry>{geometry}</geometry></visual></link></robot>"
    )
    return path


def _sidecar(visual, protected, kind):
    visual.mkdir(parents=True, exist_ok=True)
    protected.parent.mkdir(parents=True, exist_ok=True)
    if kind in {"texture", "copied_texture_alias"}:
        protected.write_bytes(b"original texture")
        material = visual / "colors.mtl"
        if kind == "copied_texture_alias":
            (visual / "texture.png").symlink_to(protected)
            texture = "texture.png"
        else:
            texture = str(protected)
        material.write_text(f"newmtl paint\nKd 1 1 1\nmap_Kd {texture}\n")
    else:
        protected.write_text("newmtl paint\nKd 1 0 0\n")
        material = protected
        if kind == "copied_alias":
            material = visual / "colors.mtl"
            material.symlink_to(protected)
    (visual / "arm.obj").write_text(f"mtllib {material}\nusemtl paint\n")


def _contents(directory):
    return {
        path.relative_to(directory): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf", "auto"])
@pytest.mark.parametrize(
    "kind",
    [
        "material",
        "texture",
        "copied_alias",
        "copied_texture_alias",
        "xml_target_material",
        "unreferenced_obj",
    ],
)
def test_single_conversion_protects_sidecars_before_mesh_processing(
    tmp_path, monkeypatch, backend, kind
):
    inputs = tmp_path / "inputs"
    output = tmp_path / "output"
    protected = output / (
        "robot.xml" if kind == "xml_target_material" else "meshes/source.asset"
    )
    _sidecar(inputs / "meshes/visual", protected, kind)
    source = _robot(inputs / "robot.urdf", mesh=kind != "unreferenced_obj")
    if kind != "xml_target_material":
        (output / "robot.xml").write_bytes(b"previous model XML")
    original = _contents(tmp_path)
    assert inspect_urdf(source)["status"] == "passed"
    manager = MujocoConversionManager(source, output, backend=backend)

    def unexpected(*args, **kwargs):
        pytest.fail("Source overlap must be rejected before mesh conversion")

    monkeypatch.setattr(manager, "_convert_mesh_directory", unexpected)
    with pytest.raises(ValueError, match="referenced source asset"):
        manager.run()
    assert manager.last_report["status"] == "failed"
    assert manager.last_report["attempts"] == []
    assert str(protected) in manager.last_report["error"]
    assert _contents(tmp_path) == original


@pytest.mark.parametrize("backend", ["mujoco", "urdf2mjcf", "auto"])
@pytest.mark.parametrize("writer,reader", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize(
    "kind", ["material", "texture", "copied_alias", "unreferenced_obj"]
)
def test_batch_protects_every_jobs_sidecars_before_any_job_runs(
    tmp_path, monkeypatch, backend, writer, reader, kind
):
    inputs = tmp_path / "inputs"
    output = tmp_path / "converted"
    protected = output / writer / "robot/meshes/source.asset"
    _sidecar(inputs / reader / "meshes/visual", protected, kind)
    _robot(inputs / reader / "robot.urdf", mesh=kind != "unreferenced_obj")
    _robot(inputs / writer / "robot.urdf", mesh=False)
    original = _contents(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("Batch dependency protection must precede the first model")

    monkeypatch.setattr(MujocoConversionManager, "run", unexpected)
    report = BatchConversionManager(inputs, output, backend=backend).run()
    assert report["status"] == "failed"
    assert report["stage"] == "discovery"
    assert "would replace an input asset" in report["error"]
    assert str(protected) in report["error"]
    assert report["results"] == []
    assert _contents(tmp_path) == original
    assert (
        BatchConversionManager(inputs, output, check_only=True).run()["status"]
        == "success"
    )


def test_obj_sidecar_can_also_be_another_jobs_xml_target(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    output = tmp_path / "converted"
    protected = output / "a/robot/robot.xml"
    _sidecar(inputs / "b/meshes/visual", protected, "material")
    _robot(inputs / "a/robot.urdf", mesh=False)
    _robot(inputs / "b/robot.urdf")
    original = _contents(tmp_path)

    def unexpected(*args, **kwargs):
        pytest.fail("An XML target is also protected when used as MTL input")

    monkeypatch.setattr(MujocoConversionManager, "run", unexpected)
    report = BatchConversionManager(inputs, output).run()
    assert report["stage"] == "discovery"
    assert report["status"] == "failed"
    assert str(protected) in report["error"]
    assert _contents(tmp_path) == original


def test_ordinary_external_sidecars_do_not_block_batch_planning(tmp_path):
    inputs = tmp_path / "inputs"
    output = tmp_path / "converted"
    material = output / "shared/colors.mtl"
    _sidecar(inputs / "b/meshes/visual", material, "material")
    _robot(inputs / "b/robot.urdf")
    _robot(inputs / "a/robot.urdf", mesh=False)
    original = _contents(tmp_path)

    jobs = BatchConversionManager(inputs, output)._discover()

    assert len(jobs) == 2
    assert _contents(tmp_path) == original


@pytest.mark.parametrize(
    "header",
    [b"# caf\xe9\nmtllib colors.mtl\n", b"mtllib colors.mtl # " + b"x" * 5000 + b"\n"],
)
def test_dependency_scan_handles_comment_encoding_and_long_literal_alternatives(
    tmp_path, header
):
    source = tmp_path / "arm.obj"
    material = tmp_path / "colors.mtl"
    source.write_bytes(header)
    material.write_bytes(b"newmtl paint\n# caf\xe9\nKd 1 0 0\n")

    assert set(iter_obj_dependencies([source])) == {source, material}
