import json
import sys
from pathlib import Path

import mujoco
import pytest

import urdf2mjcf.batch_conversion as batch_module
from urdf2mjcf.batch_conversion import BatchConversionManager

ROBOT = """<robot name="test"><link name="arm">
  <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><geometry><box size="1 1 1"/></geometry></visual>
</link></robot>"""


def _source(root, relative, content=ROBOT):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.mark.parametrize("check_only", [False, True])
def test_mesh_symlink_loop_does_not_abort_other_batch_jobs(tmp_path, check_only):
    inputs = tmp_path / "models"
    broken = _source(
        inputs,
        "a.urdf",
        ROBOT.replace('<box size="1 1 1"/>', '<mesh filename="loop.obj"/>'),
    )
    original = broken.read_bytes()
    (inputs / "loop.obj").symlink_to("loop.obj")
    _source(inputs, "b.urdf")
    output = tmp_path / "converted"
    report = BatchConversionManager(
        inputs, output, backend="mujoco", check_only=check_only
    ).run()
    assert report["summary"] == {"total": 2, "succeeded": 1, "failed": 1, "skipped": 0}
    assert report["results"][0]["status"] == "failed"
    assert report["results"][1]["status"] == ("passed" if check_only else "success")
    assert broken.read_bytes() == original
    assert (inputs / "loop.obj").is_symlink()
    if check_only:
        assert report["results"][0]["issues"][0]["code"] == "unresolved_mesh"
        assert not output.exists()
    else:
        assert mujoco.MjModel.from_xml_path(str(output / "b/b.xml")).nbody == 2


@pytest.mark.parametrize("check_only", [False, True])
def test_file_alias_into_nested_output_is_protected_as_an_input(tmp_path, check_only):
    inputs = tmp_path / "models"
    output = inputs / "converted"
    _source(inputs, "a.urdf")
    original_source = _source(output, "a/meshes/original.urdf")
    alias = inputs / "b.urdf"
    alias.symlink_to(original_source)
    before = {path: path.read_bytes() for path in inputs.rglob("*") if path.is_file()}
    report = BatchConversionManager(
        inputs, output, backend="mujoco", check_only=check_only
    ).run()
    if check_only:
        assert report["status"] == "success"
        assert report["summary"] == {
            "total": 2,
            "succeeded": 2,
            "failed": 0,
            "skipped": 0,
        }
        assert [result["relative_input"] for result in report["results"]] == [
            "a.urdf",
            "b.urdf",
        ]
    else:
        assert report["status"] == "failed"
        assert report["stage"] == "discovery"
        assert "would replace an input URDF" in report["error"]
        assert report["results"] == []
    assert alias.is_symlink() and alias.read_bytes() == original_source.read_bytes()
    assert before == {
        path: path.read_bytes() for path in inputs.rglob("*") if path.is_file()
    }


def test_batch_converts_same_named_models_and_continues_after_failure(tmp_path):
    inputs = tmp_path / "models"
    first = _source(inputs, "a/robot.urdf")
    _source(inputs, "b/broken.urdf", "invalid XML")
    _source(inputs, "c/robot.urdf", ROBOT.replace('value="1"', 'value="2"'))
    output = tmp_path / "converted"
    manager = BatchConversionManager(inputs, output, backend="mujoco", fixed_base=False)
    report = manager.run()
    assert report["status"] == "failed"
    assert report["summary"] == {"total": 3, "succeeded": 2, "failed": 1, "skipped": 0}
    assert [result["relative_input"] for result in report["results"]] == [
        "a/robot.urdf",
        "b/broken.urdf",
        "c/robot.urdf",
    ]
    for result, mass in ((report["results"][0], 1), (report["results"][2], 2)):
        model = mujoco.MjModel.from_xml_path(result["output"])
        assert model.nq == 7
        assert result["statistics"]["total_mass"] == pytest.approx(mass)
        assert result["used_backend"] == "mujoco"
    first_output = output / "a/robot/robot.xml"
    previous = first_output.read_bytes()
    first.write_text("broken input on rerun")
    rerun = manager.run()
    assert rerun is manager.last_report
    assert rerun["summary"] == {"total": 3, "succeeded": 1, "failed": 2, "skipped": 0}
    assert report["summary"]["succeeded"] == 2
    assert first_output.read_bytes() == previous


@pytest.mark.parametrize("fail_fast", [True, False])
def test_batch_fail_fast_marks_remaining_models_skipped(tmp_path, fail_fast):
    inputs = tmp_path / "models"
    _source(inputs, "a.urdf", "invalid XML")
    _source(inputs, "b.urdf")
    output = tmp_path / "converted"
    report = BatchConversionManager(
        inputs, output, backend="mujoco", fail_fast=fail_fast
    ).run()
    assert report["summary"] == {
        "total": 2,
        "failed": 1,
        "skipped": int(fail_fast),
        "succeeded": int(not fail_fast),
    }
    assert (output / "b/b.xml").exists() is not fail_fast


def test_batch_check_ignores_outputs_and_directory_symlinks(tmp_path, monkeypatch):
    inputs = tmp_path / "models"
    output = inputs / "converted"
    _source(inputs, "a.urdf")
    _source(inputs, "nested/b.URDF")
    _source(output, "old.urdf", "invalid XML")
    outside = tmp_path / "outside"
    _source(outside, "unrelated.urdf", "invalid XML")
    (inputs / "alias").symlink_to(outside, target_is_directory=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Batch check invoked conversion")

    monkeypatch.setattr(batch_module, "MujocoConversionManager", forbidden)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    report = BatchConversionManager(inputs, output, check_only=True).run()
    assert report["kind"] == "batch_check"
    assert report["summary"] == {"total": 2, "succeeded": 2, "failed": 0, "skipped": 0}
    assert report["output_directory"] is None
    assert before == sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    direct = BatchConversionManager(
        inputs, output, check_only=True, recursive=False
    ).run()
    assert direct["summary"]["total"] == 1


@pytest.mark.parametrize("mode", ["same", "nested", "case"])
def test_batch_rejects_output_conflicts_before_writing(tmp_path, mode):
    inputs = tmp_path / "models"
    _source(inputs, "a.urdf")
    output = tmp_path / "converted"
    if mode == "same":
        output = inputs
    elif mode == "nested":
        _source(inputs, "a/b.urdf")
    else:
        _source(inputs, "a.URDF")
    before = sorted(tmp_path.rglob("*"))
    report = BatchConversionManager(inputs, output, backend="mujoco").run()
    assert report["status"] == "failed"
    assert report["stage"] == "discovery"
    assert "error" in report
    assert sorted(tmp_path.rglob("*")) == before
    # No output layout is needed for a read-only check.
    check = BatchConversionManager(inputs, output, check_only=True).run()
    assert check["status"] == "success"


def test_batch_rejects_output_symlink_escape(tmp_path):
    inputs = tmp_path / "models"
    _source(inputs, "robot.urdf")
    output = tmp_path / "converted"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "robot").symlink_to(outside, target_is_directory=True)
    report = BatchConversionManager(inputs, output).run()
    assert report["status"] == "failed"
    assert "escapes" in report["error"]
    assert not list(outside.iterdir())


@pytest.mark.parametrize("missing", [True, False])
def test_empty_or_missing_input_has_failure_report(tmp_path, monkeypatch, missing):
    inputs = tmp_path / "models"
    if not missing:
        inputs.mkdir()
    output = tmp_path / "converted"
    report_path = tmp_path / "batch.json"
    monkeypatch.setattr(
        sys, "argv", ["batch", str(inputs), str(output), "--report", str(report_path)]
    )
    assert batch_module.main() == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["stage"] == "discovery"
    assert report["summary"]["total"] == 0
    assert not output.exists()


def test_batch_cli_writes_aggregate_check_report_without_assets(tmp_path, monkeypatch):
    inputs = tmp_path / "models with spaces"
    _source(inputs, "valid.urdf")
    _source(inputs, "bad.urdf", "invalid XML")
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys, "argv", ["batch", str(inputs), "--check", "--report", str(report_path)]
    )
    assert batch_module.main() == 1
    report = json.loads(report_path.read_text())
    assert report["summary"]["succeeded"] == 1
    assert report["summary"]["failed"] == 1
    assert report["results"][0]["issues"][0]["code"] == "invalid_input"
    assert not inputs.with_name(f"{inputs.name}_mjcf").exists()


def test_batch_cli_auto_fallback_and_shared_options(tmp_path, monkeypatch):
    import subprocess

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("external converter unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    inputs = tmp_path / "models"
    _source(inputs, "robot.urdf")
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch",
            str(inputs),
            "--backend",
            "auto",
            "--floating-base",
            "--export-collision",
            "--no-inertia-recalc",
            "--report",
            str(report_path),
        ],
    )
    assert batch_module.main() == 0
    report = json.loads(report_path.read_text())
    result = report["results"][0]
    assert report["summary"]["succeeded"] == 1
    assert result["used_backend"] == "mujoco"
    assert result["statistics"]["degrees_of_freedom"] == 6
    assert result["options"]["export_collision"] is True
    assert result["options"]["auto_recalculate_inertia"] is False
    assert Path(result["output"]).is_file()


@pytest.mark.parametrize("reader, writer", [("a", "b"), ("b", "a")])
@pytest.mark.parametrize("reference", ["absolute", "relative", "symlink"])
def test_batch_protects_mesh_inputs_from_other_jobs(
    tmp_path, reader, writer, reference
):
    import trimesh

    inputs = tmp_path / "models"
    output = inputs / "converted"
    mesh = output / writer / "meshes/source.obj"
    mesh.parent.mkdir(parents=True)
    trimesh.creation.box(extents=[1, 2, 3]).export(mesh)
    if reference == "absolute":
        filename = str(mesh)
    elif reference == "relative":
        filename = mesh.relative_to(inputs).as_posix()
    else:
        alias = inputs / "source.obj"
        alias.symlink_to(mesh)
        filename = alias.name
    _source(inputs, f"{writer}.urdf")
    _source(
        inputs,
        f"{reader}.urdf",
        ROBOT.replace('<box size="1 1 1"/>', f'<mesh filename="{filename}"/>'),
    )
    before = {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    report = BatchConversionManager(inputs, output, backend="mujoco").run()
    assert report["status"] == "failed"
    assert report["stage"] == "discovery"
    assert "would replace an input mesh" in report["error"]
    assert str(mesh) in report["error"]
    assert report["results"] == []
    assert before == {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    assert not (output / reader).exists()
    check = BatchConversionManager(inputs, output, check_only=True).run()
    assert check["status"] == "success"


def test_batch_protects_symlinked_urdf_inside_another_jobs_assets(tmp_path):
    inputs = tmp_path / "models"
    output = tmp_path / "converted"
    original = _source(output, "a/meshes/original.urdf")
    _source(inputs, "a.urdf")
    (inputs / "b.urdf").symlink_to(original)
    before = original.read_bytes()
    report = BatchConversionManager(inputs, output, backend="mujoco").run()
    assert report["status"] == "failed"
    assert report["stage"] == "discovery"
    assert "would replace an input URDF" in report["error"]
    assert original.read_bytes() == before
    assert report["results"] == []


def test_batch_can_read_previous_assets_that_no_job_replaces(tmp_path):
    import trimesh

    inputs = tmp_path / "models"
    output = tmp_path / "converted"
    mesh = output / "shared/meshes/source.obj"
    mesh.parent.mkdir(parents=True)
    trimesh.creation.box().export(mesh)
    original = mesh.read_bytes()
    _source(
        inputs,
        "robot.urdf",
        ROBOT.replace('<box size="1 1 1"/>', f'<mesh filename="{mesh}"/>'),
    )
    report = BatchConversionManager(inputs, output, backend="mujoco").run()
    assert report["status"] == "success"
    assert mesh.read_bytes() == original
    assert mujoco.MjModel.from_xml_path(report["results"][0]["output"]).nmesh == 1


@pytest.mark.parametrize("nested_output", [False, True])
@pytest.mark.parametrize("check_only", [False, True])
def test_unreadable_source_symlink_does_not_abort_other_batch_jobs(
    tmp_path, nested_output, check_only
):
    inputs = tmp_path / "models"
    _source(inputs, "valid.urdf")
    (inputs / "loop.urdf").symlink_to("loop.urdf")
    output = (inputs if nested_output else tmp_path) / "converted"
    report = BatchConversionManager(
        inputs, output, backend="mujoco", check_only=check_only
    ).run()
    assert report["status"] == "failed"
    assert report["stage"] == "completed"
    assert report["summary"] == {"total": 2, "succeeded": 1, "failed": 1, "skipped": 0}
    assert (output / "valid/valid.xml").is_file() is not check_only
