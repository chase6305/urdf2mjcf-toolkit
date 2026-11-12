import json
import sys

import mujoco
import pytest

import urdf2mjcf.urdf_to_mujoco_converter as manager_module
from urdf2mjcf.reporting import write_report

ROBOT = """<robot name="test"><link name="arm">
  <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
  <visual><geometry><box size="1 1 1"/></geometry></visual>
</link></robot>"""


def test_check_cli_does_not_start_conversion_or_create_output(tmp_path, monkeypatch):
    source = tmp_path / "robot.urdf"
    source.write_text(ROBOT)
    output = tmp_path / "output"
    report_path = tmp_path / "diagnostics/check.json"

    def forbidden(*args, **kwargs):
        pytest.fail("Check mode constructed a conversion manager")

    monkeypatch.setattr(manager_module, "MujocoConversionManager", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["toolkit", str(source), str(output), "--check", "--report", str(report_path)],
    )
    assert manager_module.main() == 0
    report = json.loads(report_path.read_text())
    assert report["kind"] == "check"
    assert report["status"] == "passed"
    assert not output.exists()
    assert source.read_text() == ROBOT


@pytest.mark.parametrize("check", [True, False])
def test_cli_writes_failed_report_for_missing_input(tmp_path, monkeypatch, check):
    report_path = tmp_path / "failure.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["toolkit", str(tmp_path / "missing.urdf"), "--report", str(report_path)]
        + (["--check"] if check else []),
    )
    assert manager_module.main() == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["duration_seconds"] >= 0
    assert list(tmp_path.iterdir()) == [report_path]


def test_cli_report_records_compiled_model_and_backend_fallback(tmp_path, monkeypatch):
    import subprocess

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("urdf2mjcf unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    source = tmp_path / "robot.urdf"
    source.write_text(ROBOT)
    output = tmp_path / "output"
    report_path = tmp_path / "conversion.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "toolkit",
            str(source),
            str(output),
            "--backend",
            "auto",
            "--floating-base",
            "--report",
            str(report_path),
        ],
    )
    assert manager_module.main() == 0
    report = json.loads(report_path.read_text())
    model = mujoco.MjModel.from_xml_path(report["output"])
    assert report["status"] == "success"
    assert report["used_backend"] == "mujoco"
    assert report["requested_backend"] == "auto"
    assert report["statistics"]["degrees_of_freedom"] == model.nv == 6
    assert report["statistics"]["total_mass"] == pytest.approx(1)
    assert report["statistics"]["bodies"] == model.nbody - 1
    assert report["options"]["fixed_base"] is False
    failed, success = report["attempts"]
    assert failed["stage"] == "urdf_conversion"
    assert "unavailable" in failed["error"]
    assert success["status"] == "success"
    assert report["inertia_recovery_attempts"] == 0


def test_conversion_failure_retains_stage_and_previous_output(tmp_path, monkeypatch):
    source = tmp_path / "robot.urdf"
    source.write_text(ROBOT)
    manager = manager_module.MujocoConversionManager(
        source, tmp_path / "output", backend="mujoco"
    )
    output = manager.run()
    previous = output.read_bytes()

    def invalid(*args, **kwargs):
        raise ValueError("invalid model")

    monkeypatch.setattr(mujoco.MjModel, "from_xml_path", invalid)
    with pytest.raises(ValueError, match="invalid model"):
        manager.run()
    report = manager.last_report
    assert report["status"] == "failed"
    assert report["stage"] == "validation"
    assert report["attempts"][0]["stage"] == "validation"
    assert report["output"] is None
    assert report["used_backend"] is None
    assert "statistics" not in report
    assert output.read_bytes() == previous


def test_report_path_cannot_overwrite_input(tmp_path, monkeypatch):
    source = tmp_path / "robot.urdf"
    source.write_text(ROBOT)
    report_path = tmp_path / "report.json"
    report_path.symlink_to(source)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys, "argv", ["toolkit", str(source), str(output), "--report", str(report_path)]
    )
    assert manager_module.main() == 1
    assert source.read_text() == ROBOT
    assert not output.exists()


def test_failed_report_write_preserves_previous_json(tmp_path, monkeypatch):
    from pathlib import Path

    report_path = tmp_path / "report.json"
    report_path.write_text('{"status":"previous"}')

    def fail_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        write_report({"status": "success"}, report_path)
    assert json.loads(report_path.read_text()) == {"status": "previous"}
    assert list(tmp_path.iterdir()) == [report_path]


@pytest.mark.parametrize("check", [True, False])
def test_cli_can_report_an_input_symlink_loop(tmp_path, monkeypatch, check):
    source = tmp_path / "loop.urdf"
    source.symlink_to(source.name)
    report_path = tmp_path / "failure.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["toolkit", str(source), "--report", str(report_path)]
        + (["--check"] if check else []),
    )
    assert manager_module.main() == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["input"] == str(source)
    assert source.is_symlink()
    assert set(tmp_path.iterdir()) == {source, report_path}
