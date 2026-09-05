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
