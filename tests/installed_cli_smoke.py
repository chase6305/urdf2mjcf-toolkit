"""Run from outside the checkout with an isolated Python and an installed wheel."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import trimesh

import urdf2mjcf.mesh_identity


def _run_cli(*args, expected=0, command="urdf2mjcf-toolkit"):
    executable = Path(sys.executable).parent / command
    result = subprocess.run(
        [str(executable), *map(str, args)], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == expected, result.stdout + result.stderr


def _files(directory):
    return {
        path.relative_to(directory): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


def main():
    module_path = Path(urdf2mjcf.mesh_identity.__file__).resolve()
    assert "site-packages" in module_path.parts, module_path
    # External converter subprocesses must use this environment as well.
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    with tempfile.TemporaryDirectory(prefix="urdf2mjcf-installed-") as temporary:
        root = Path(temporary)
        inputs = root / "inputs"
        geometries = []
        for index, width in enumerate((0.1, 0.2)):
            path = inputs / str(index) / "arm.obj"
            path.parent.mkdir(parents=True)
            trimesh.creation.box(extents=[width, 0.2, 0.3]).export(path)
            geometries.append(
                f'<visual><origin xyz="{index} 0 0"/><geometry><mesh filename="{path}" '
                'scale="1 2 1"/></geometry></visual>'
            )
        source = inputs / "robot.urdf"
        source.write_text(
            '<robot name="installed"><link name="base"/><link name="arm"><inertial>'
            '<origin xyz="0.1 0.2 0.3"/><mass value="2"/>'
            '<inertia ixx="2" iyy="3" izz="4"/></inertial>'
            + "".join(geometries)
            + '</link><joint name="hinge" type="revolute"><parent link="base"/>'
            '<child link="arm"/><axis xyz="1e-300 2e-300 -2e-300"/>'
            '<limit lower="-1" upper="1" effort="0"/></joint></robot>'
        )
        original_source = source.read_bytes()
        original_inputs = _files(inputs)
        check_report = root / "check.json"
        check_output = root / "check-output"
        _run_cli(source, check_output, "--check", "--report", check_report)
        assert json.loads(check_report.read_text())["status"] == "passed"
        assert not check_output.exists()

        results = []
        for backend in ("mujoco", "urdf2mjcf"):
            output = root / backend
            report_path = root / f"{backend}.json"
            arguments = (
                source,
                output,
                "--backend",
                backend,
                "--no-inertia-recalc",
                "--report",
                report_path,
            )
            _run_cli(*arguments)
            report = json.loads(report_path.read_text())
            assert report["status"] == "success"
            assert report["used_backend"] == backend
            assert report["inertia_recovery_attempts"] == 0
            model = mujoco.MjModel.from_xml_path(str(output / "robot.xml"))
            assert model.nv == 1 and model.nmesh == 2
            np.testing.assert_allclose(model.body("arm").mass, [2])
            np.testing.assert_allclose(model.body("arm").ipos, [0.1, 0.2, 0.3])
            np.testing.assert_allclose(np.sort(model.body("arm").inertia), [2, 3, 4])
            np.testing.assert_allclose(
                model.joint("hinge").axis, np.array([1, 2, -2]) / 3
            )
            data = mujoco.MjData(model)
            data.ctrl[:] = 1e6
            mujoco.mj_forward(model, data)
            np.testing.assert_array_equal(data.qfrc_actuator, [0])
            bounds = []
            body = data.body("arm")
            for gid in range(model.ngeom):
                if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_id = model.geom_dataid[gid]
                start = model.mesh_vertadr[mesh_id]
                vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
                world = (
                    vertices @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
                )
                local = (world - body.xpos) @ body.xmat.reshape(3, 3)
                bounds.append(np.array([local.min(axis=0), local.max(axis=0)]))
            bounds.sort(key=lambda box: box.mean(axis=0)[0])
            for index, box in enumerate(bounds):
                np.testing.assert_allclose(box.mean(axis=0), [index, 0, 0], atol=2e-6)
                np.testing.assert_allclose(
                    box[1] - box[0], [0.1 * (index + 1), 0.4, 0.3], atol=2e-6
                )
            assert _files(inputs) == original_inputs
            previous = _files(output)
            source.write_bytes(original_source.replace(b"arm.obj", b"missing.obj"))
            invalid_source = source.read_bytes()
            _run_cli(*arguments, expected=1)
            assert json.loads(report_path.read_text())["status"] == "failed"
            assert source.read_bytes() == invalid_source
            assert _files(output) == previous
            source.write_bytes(original_source)
            relocated = root / f"relocated-{backend}"
            shutil.copytree(output, relocated)
            shutil.rmtree(output)
            results.append((relocated, "robot.xml", model, report["assets"]))

        assert _files(inputs) == original_inputs
        (inputs / "copy.urdf").write_bytes(original_source)
        batch_output = root / "batch"
        batch_report = root / "batch.json"
        batch_arguments = (
            inputs,
            batch_output,
            "--backend",
            "mujoco",
            "--no-inertia-recalc",
            "--report",
            batch_report,
        )
        _run_cli(*batch_arguments, command="urdf2mjcf-batch")
        report = json.loads(batch_report.read_text())
        assert report["summary"] == {
            "total": 2,
            "succeeded": 2,
            "failed": 0,
            "skipped": 0,
        }
        successful = report["results"]
        previous_batch = _files(batch_output)
        (inputs / "broken.urdf").write_text("invalid XML")
        batch_inputs = _files(inputs)
        _run_cli(*batch_arguments, command="urdf2mjcf-batch", expected=1)
        report = json.loads(batch_report.read_text())
        assert report["summary"] == {
            "total": 3,
            "succeeded": 2,
            "failed": 1,
            "skipped": 0,
        }
        assert _files(batch_output) == previous_batch
        assert _files(inputs) == batch_inputs
        for result in successful:
            output_xml = Path(result["output"])
            model = mujoco.MjModel.from_xml_path(str(output_xml))
            assert model.nv == 1 and model.nmesh == 2
            np.testing.assert_allclose(model.body("arm").mass, [2])
            relocated = root / f"relocated-batch-{output_xml.stem}"
            shutil.copytree(output_xml.parent, relocated)
            results.append((relocated, output_xml.name, model, result["assets"]))
        shutil.rmtree(batch_output)
        shutil.rmtree(inputs)
        for relocated, filename, original, assets in results:
            model = mujoco.MjModel.from_xml_path(str(relocated / filename))
            for name in (
                "mesh_vert",
                "mesh_face",
                "body_mass",
                "body_inertia",
                "jnt_axis",
            ):
                np.testing.assert_array_equal(
                    getattr(model, name), getattr(original, name)
                )
            assert assets
            for asset in assets:
                path = relocated / asset["path"]
                assert path.stat().st_size == asset["size_bytes"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
    print(
        "Installed wheel: both single-model backends, batch CLI, physics, failure preservation, and relocation passed."
    )


if __name__ == "__main__":
    main()
