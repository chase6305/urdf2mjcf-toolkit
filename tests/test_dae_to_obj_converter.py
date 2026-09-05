import ast

import pytest

from urdf2mjcf.dae_to_obj_converter import DAE2OBJConverter


def test_missing_blender_reports_failure_for_each_input(tmp_path, monkeypatch):
    source = tmp_path / "arm.DaE"
    source.touch()
    converter = DAE2OBJConverter()

    def unavailable():
        raise RuntimeError("Blender executable not found")

    monkeypatch.setattr(converter, "check_blender_available", unavailable)
    assert converter.convert_directory(tmp_path, tmp_path) == {str(source): (False, [])}


def test_success_without_obj_output_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "arm.dae"
    source.touch()
    converter = DAE2OBJConverter()
    monkeypatch.setattr(converter, "check_blender_available", lambda: True)
    monkeypatch.setattr(converter, "_run_blender_batch_conversion", lambda script: True)
    assert converter.convert_directory(tmp_path, tmp_path) == {str(source): (False, [])}


@pytest.mark.parametrize("batch", [False, True])
def test_generated_scripts_handle_quoted_paths(tmp_path, batch):
    directory = tmp_path / 'robot "quoted" directory'
    source = directory / "arm.dae"
    converter = DAE2OBJConverter()
    if batch:
        script = converter._create_batch_conversion_script(
            [source], directory, directory, True
        )
    else:
        script = converter._create_single_conversion_script(source, directory, True)
    ast.parse(script)
    assert str(directory) in script


def test_failed_blender_process_does_not_accept_stale_obj(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import urdf2mjcf.dae_to_obj_converter as dae_module

    (tmp_path / "arm.obj").touch()
    monkeypatch.setattr(
        dae_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="failed"
        ),
    )
    success, _ = DAE2OBJConverter()._run_blender_conversion("", "arm.dae", tmp_path)
    assert not success
