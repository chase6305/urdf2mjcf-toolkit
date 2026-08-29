import xml.etree.ElementTree as ET

from urdf2mjcf.mjcf_editor import MJCFEditor


def _write_model(tmp_path, body='<body name="root"><freejoint /></body>'):
    model = tmp_path / "model.xml"
    model.write_text(
        f'<mujoco><asset /><worldbody>{body}</worldbody></mujoco>', encoding="utf-8"
    )
    return model


def test_fixed_and_floating_base_are_idempotent(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path))

    assert editor.fix_base()
    assert not editor.fix_base()
    assert editor.ensure_freejoint()
    assert not editor.ensure_freejoint()
    assert len(editor.root.findall(".//body[@name='root']/freejoint")) == 1


def test_ground_plane_setup_is_idempotent(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path, '<body name="root" />'))

    editor.add_default_ground_plane()
    editor.add_default_ground_plane()

    assert len(editor.root.findall(".//texture[@name='groundplane']")) == 1
    assert len(editor.root.findall(".//material[@name='groundplane']")) == 1
    assert len(editor.root.findall(".//geom[@name='floor']")) == 1


def test_save_writes_parseable_indented_xml(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path))
    destination = tmp_path / "result.xml"

    editor.save(destination)

    assert ET.parse(destination).getroot().tag == "mujoco"
    assert "\n  <" in destination.read_text(encoding="utf-8")
