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


@pytest.mark.parametrize(
    "written_name", [None, "arm1.obj", "arm[1]_old.obj", "arm[1].obj"]
)
def test_successful_blender_process_requires_a_fresh_exact_output(
    tmp_path, monkeypatch, written_name
):
    from types import SimpleNamespace

    import urdf2mjcf.dae_to_obj_converter as dae_module

    expected = tmp_path / "arm[1].obj"
    expected.write_bytes(b"old geometry")
    (tmp_path / "arm1.obj").write_bytes(b"unrelated geometry")
    scripts = []

    def run(command, **kwargs):
        scripts.append(dae_module.Path(command[-1]))
        assert scripts[-1].is_file()
        if written_name:
            (tmp_path / written_name).write_bytes(b"new geometry with different size")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(dae_module.subprocess, "run", run)
    success, outputs = DAE2OBJConverter()._run_blender_conversion(
        "# conversion script", "arm[1].dae", tmp_path
    )
    assert success == (written_name == expected.name)
    assert outputs == ([expected] if success else [])
    assert scripts and all(not script.exists() for script in scripts)
    if not success:
        assert expected.read_bytes() == b"old geometry"


def test_batch_success_reports_stale_outputs_as_failures(tmp_path, monkeypatch):
    converter = DAE2OBJConverter()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    output_root = tmp_path / "output"
    expected = {}
    for name in ("fresh", "stale"):
        source = inputs / f"{name}.dae"
        source.touch()
        output = output_root / name / f"{name}.obj"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"old geometry")
        (output.parent / "unrelated.obj").touch()
        expected[name] = (source, output)

    def run(script):
        expected["fresh"][1].write_bytes(b"freshly converted geometry")
        return True

    monkeypatch.setattr(converter, "check_blender_available", lambda: True)
    monkeypatch.setattr(converter, "_run_blender_batch_conversion", run)
    assert converter.convert_directory(inputs, output_root) == {
        str(expected["fresh"][0]): (True, [expected["fresh"][1]]),
        str(expected["stale"][0]): (False, []),
    }
    assert expected["stale"][1].read_bytes() == b"old geometry"
