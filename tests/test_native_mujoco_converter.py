import xml.etree.ElementTree as ET

import mujoco

from urdf2mjcf.native_mujoco_converter import NativeMujocoConverter

ROBOT = """<robot name="native_test">
  <mujoco><compiler discardvisual="true" fusestatic="true" strippath="true" /></mujoco>
  <link name="base">
    <inertial><mass value="2"/><inertia ixx="1" iyy="2" izz="2" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="1 1 1"/></geometry></visual>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial><origin xyz="0.1 0 0"/><mass value="1"/><inertia ixx="0.01" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><sphere radius="0.1"/></geometry></visual>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 1"/>
    <axis xyz="0 1 0"/><limit lower="-0.5" upper="1.5" effort="10" velocity="2"/>
    <dynamics damping="0.2" friction="0.1"/>
  </joint>
</robot>"""


def test_native_conversion_preserves_links_visuals_and_dynamics(tmp_path):
    import numpy as np

    source = tmp_path / "robot.urdf"
    source.write_text(ROBOT)
    converter = NativeMujocoConverter(source, tmp_path / "output", tmp_path / "assets")
    assert converter.convert()
    output = converter.get_output_xml()
    model = mujoco.MjModel.from_xml_path(str(output))
    assert model.nbody == 3
    assert model.ngeom == 3
    assert model.njnt == 1
    assert model.nu == 0
    np.testing.assert_allclose(model.jnt_range[0], [-0.5, 1.5])
    np.testing.assert_allclose(model.jnt_axis[0], [0, 1, 0])
    np.testing.assert_allclose(model.dof_damping[0], 0.2)
    np.testing.assert_allclose(model.body_mass[1:], [2, 1])
    assert source.read_text() == ROBOT
    assert list(converter.output_dir.iterdir()) == [output]
    assert ET.parse(output).getroot().find(".//body[@name='base']") is not None


def test_native_failure_clears_result_and_preserves_existing_output(tmp_path):
    source = tmp_path / "robot.urdf"
    source.write_text(ROBOT)
    converter = NativeMujocoConverter(source, tmp_path / "output", tmp_path / "assets")
    assert converter.convert()
    output = converter.get_output_xml()
    original = output.read_bytes()
    source.write_text("invalid xml")
    assert not converter.convert()
    assert converter.get_output_xml() is None
    assert output.read_bytes() == original
    assert list(converter.output_dir.iterdir()) == [output]
