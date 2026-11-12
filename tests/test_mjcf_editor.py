import xml.etree.ElementTree as ET

import pytest

from urdf2mjcf.mjcf_editor import MJCFEditor


def _write_model(tmp_path, body='<body name="root"><freejoint /></body>'):
    model = tmp_path / "model.xml"
    model.write_text(
        f"<mujoco><asset /><worldbody>{body}</worldbody></mujoco>", encoding="utf-8"
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


@pytest.mark.parametrize("location", ["body", "world"])
def test_default_scene_preserves_conflicting_geom_texture_and_camera_names(
    tmp_path, location
):
    import mujoco

    editor = MJCFEditor(_write_model(tmp_path, '<body name="root"/>'))
    asset = editor.root.find("asset")
    original_texture = ET.SubElement(
        asset,
        "texture",
        name="groundplane",
        type="2d",
        builtin="flat",
        rgb1="1 0 0",
        width="2",
        height="2",
    )
    ET.SubElement(asset, "material", name="paint", texture="groundplane")
    worldbody = editor.root.find("worldbody")
    body = worldbody.find("body")
    original_geom = ET.SubElement(
        body if location == "body" else worldbody,
        "geom",
        name="floor",
        type="box",
        size="0.1 0.2 0.3",
        material="paint",
    )
    camera = ET.SubElement(body, "camera", name="top_camera", pos="1 2 3")
    originals = [
        (element, element.attrib.copy())
        for element in (original_texture, original_geom, camera)
    ]
    editor.add_default_ground_plane()
    editor.add_default_ground_plane()
    assert all(element.attrib == attributes for element, attributes in originals)
    for tag in ("geom", "texture", "camera"):
        names = [
            element.get("name")
            for element in editor.root.iter(tag)
            if element.get("name")
        ]
        assert len(names) == len(set(names))
    planes = [geom for geom in worldbody.findall("geom") if geom.get("type") == "plane"]
    assert len(planes) == int(location == "body")
    if planes:
        assert planes[0].get("name") != "floor"
    saved = tmp_path / "prepared.xml"
    editor.save(saved)
    editor = MJCFEditor(saved)
    editor.add_default_ground_plane()
    model = mujoco.MjModel.from_xml_string(ET.tostring(editor.root, encoding="unicode"))
    assert model.ngeom == 1 + int(location == "body") and model.ncam == 1
    assert model.geom("floor").type[0] == mujoco.mjtGeom.mjGEOM_BOX


@pytest.mark.parametrize(
    "attributes", [{"rgba": "1 0 0 1"}, {"texture": "exact_source"}]
)
def test_texture_heuristics_preserve_explicit_materials(tmp_path, attributes):
    editor = MJCFEditor(_write_model(tmp_path))
    asset = editor.root.find("asset")
    ET.SubElement(asset, "mesh", name="arm", file="visual/arm.obj")
    material = ET.SubElement(asset, "material", name="arm", **attributes)
    visual = tmp_path / "meshes/visual"
    visual.mkdir(parents=True)
    (visual / "arm.obj").touch()
    (visual / "albedo.png").touch()
    original = material.attrib.copy()
    assert (
        editor.add_textures_and_link_materials_for_visual_meshes(tmp_path / "meshes")
        == 0
    )
    assert material.attrib == original
    assert asset.findall("texture") == []


def test_save_writes_parseable_indented_xml(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path))
    destination = tmp_path / "result.xml"

    editor.save(destination)

    assert ET.parse(destination).getroot().tag == "mujoco"
    assert "\n  <" in destination.read_text(encoding="utf-8")


def test_explicit_free_joint_is_recognized(tmp_path):
    editor = MJCFEditor(
        _write_model(tmp_path, '<body name="base"><joint type="free" /></body>')
    )
    assert not editor.ensure_freejoint()
    assert editor.fix_base()
    assert editor.root.find(".//joint") is None


def test_root_names_with_quotes_are_matched_literally(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path, ""))
    worldbody = editor.root.find("worldbody")
    other = ET.SubElement(worldbody, "body", name="other")
    ET.SubElement(other, "freejoint")
    name = 'base\'s "frame"'
    selected = ET.SubElement(worldbody, "body", name=name)
    ET.SubElement(selected, "freejoint")
    assert editor.fix_base(name)
    assert selected.find("freejoint") is None
    assert other.find("freejoint") is not None
    assert editor.ensure_freejoint(name)
    assert len(selected.findall("freejoint")) == 1


def test_material_fallback_accepts_literal_names_with_quotes_and_dots(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path, "<body/>"))
    name = "arm's.v2"
    ET.SubElement(editor.root.find("worldbody/body"), "geom", mesh=name, material=name)
    ET.SubElement(
        editor.root.find("asset"), "material", name=name, rgba="0.2 0.3 0.4 1"
    )
    editor.remap_materials_and_textures_for_mesh_variants_with_xml(
        {name: [f"{name}.obj"]}, tmp_path
    )
    material = next(
        mat for mat in editor.root.findall("asset/material") if mat.get("name") == name
    )
    assert material.get("rgba") == "0.2 0.3 0.4 1"


@pytest.mark.parametrize(
    "method",
    [
        "add_textures_for_mesh_variants",
        "remap_materials_and_textures_for_mesh_variants_with_xml",
    ],
)
def test_texture_lookup_preserves_dots_in_the_object_stem(tmp_path, method):
    editor = MJCFEditor(_write_model(tmp_path, "<body/>"))
    directory = tmp_path / "parts"
    directory.mkdir()
    (directory / "arm.v2_0.png").touch()
    (directory / "arm.png").touch()
    getattr(editor, method)({"arm.v2": ["parts/arm.v2_0.obj"]}, tmp_path)
    assert editor.root.find("asset/texture").get("file") == "parts/arm.v2_0.png"


def test_material_helpers_import_self_closing_material_elements(tmp_path):
    directory = tmp_path / "parts"
    directory.mkdir()
    source = directory / "mesh.xml"
    source.write_text(
        '<mujoco><asset><material name="paint" rgba="0.2 0.3 0.4 1" specular="0.6"/></asset><worldbody><body><geom mesh="arm" material="paint"/></body></worldbody></mujoco>'
    )
    editor = MJCFEditor(_write_model(tmp_path, "<body/>"))
    expected = {"name": "paint", "rgba": "0.2 0.3 0.4 1", "specular": "0.6"}
    assert editor.get_material_info_from_subxml(source, "arm")["material"] == expected
    assert (
        MJCFEditor.get_material_info_from_subxml(source, "arm")["material"] == expected
    )
    editor.add_material_from_subxml_if_no_texture("arm", ["parts/mesh.obj"], tmp_path)
    material = editor.root.find("asset/material")
    assert material is not None
    assert material.get("name") == "arm"
    assert material.get("rgba") == expected["rgba"]
    assert material.get("specular") == expected["specular"]


def test_floating_base_rejects_existing_hinge(tmp_path):
    import pytest

    editor = MJCFEditor(
        _write_model(tmp_path, '<body name="base"><joint type="hinge" /></body>')
    )
    with pytest.raises(ValueError, match="existing joints"):
        editor.ensure_freejoint()
    assert editor.root.find(".//freejoint") is None


def test_failed_save_preserves_previous_file(tmp_path, monkeypatch):
    import pytest

    model = _write_model(tmp_path)
    original = model.read_bytes()
    editor = MJCFEditor(model)

    def failed_write(destination, **kwargs):
        destination.write_text("partial")
        raise OSError("disk full")

    monkeypatch.setattr(editor.tree, "write", failed_write)
    with pytest.raises(OSError, match="disk full"):
        editor.save(model)
    assert model.read_bytes() == original
    assert list(tmp_path.iterdir()) == [model]


def test_mesh_variants_preserve_unrelated_meshes_and_scale(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path))
    asset = editor.root.find("asset")
    ET.SubElement(asset, "mesh", name="arm", file="arm.obj", scale="2 3 4")
    collision = ET.SubElement(asset, "mesh", name="collision", file="collision.obj")
    editor.auto_add_mesh_variants({"arm": ["arm_0.obj", "arm_1.obj"]})
    assert asset.find("mesh[@name='collision']") is collision
    assert asset.find("mesh[@name='arm']").get("scale") == "2 3 4"
    assert asset.find("mesh[@name='arm_1']").get("scale") == "2 3 4"


def test_variant_names_cannot_overwrite_independent_assets(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path))
    asset = editor.root.find("asset")
    ET.SubElement(asset, "mesh", name="arm", file="arm.obj")
    original = ET.SubElement(asset, "mesh", name="arm_1", file="collision/arm_1.obj")
    with pytest.raises(ValueError, match="conflicts with an existing asset"):
        editor.auto_add_mesh_variants({"arm": ["arm/arm_0.obj", "arm/arm_1.obj"]})
    assert original.get("file") == "collision/arm_1.obj"


def test_material_remapping_uses_actual_submesh_names(tmp_path):
    editor = MJCFEditor(
        _write_model(tmp_path, '<body><geom mesh="arm" material="arm" /></body>')
    )
    generated = tmp_path / "arm"
    generated.mkdir()
    (generated / "arm.xml").write_text("""<mujoco><asset>
      <mesh file="arm_0.obj" name="first"/><mesh file="arm_1.obj"/>
      <material name="red" rgba="1 0 0 1"/><material name="green" rgba="0 1 0 1"/>
      </asset><worldbody><body><geom mesh="first" material="red"/>
      <geom mesh="arm_1" material="green"/></body></worldbody></mujoco>""")
    editor.remap_materials_and_textures_for_mesh_variants_with_xml(
        {"arm": ["arm/arm_0.obj", "arm/arm_1.obj"]}, tmp_path
    )
    asset = editor.root.find("asset")
    assert asset.find("material[@name='arm']").get("rgba") == "1 0 0 1"
    assert asset.find("material[@name='arm_1']").get("rgba") == "0 1 0 1"


def test_material_fallback_preserves_same_name_properties(tmp_path):
    editor = MJCFEditor(
        _write_model(tmp_path, '<body><geom mesh="arm" material="arm" /></body>')
    )
    asset = editor.root.find("asset")
    ET.SubElement(asset, "material", name="arm", rgba="0.2 0.3 0.4 1")
    editor.remap_materials_and_textures_for_mesh_variants_with_xml(
        {"arm": ["arm.obj"]}, tmp_path
    )
    assert asset.find("material[@name='arm']").get("rgba") == "0.2 0.3 0.4 1"


@pytest.mark.parametrize("part_count", [1, 3])
def test_variant_expansion_preserves_every_instance_and_is_idempotent(
    tmp_path, part_count
):
    editor = MJCFEditor(_write_model(tmp_path, "<body />"))
    asset = editor.root.find("asset")
    body = editor.root.find("worldbody/body")
    variant_map = {}
    originals = []
    for mesh_name in ("arm", "hand"):
        variant_map[mesh_name] = [f"{mesh_name}_{i}.obj" for i in range(part_count)]
        for index in range(part_count):
            name = mesh_name if index == 0 else f"{mesh_name}_{index}"
            ET.SubElement(asset, "mesh", name=name, file=f"{name}.obj")
            ET.SubElement(asset, "material", name=name, rgba="1 1 1 1")
        for pos in ("0 0 0", "2 0 0", "2 0 0"):
            originals.append(
                ET.SubElement(
                    body,
                    "geom",
                    mesh=mesh_name,
                    pos=pos,
                    scale="2 3 4",
                    quat="0 1 0 0",
                    rgba="1 0 0 1",
                    group="2",
                )
            )
    collision = ET.SubElement(body, "geom", mesh="arm", **{"class": "collision"})
    # An unrelated existing name must not be overwritten or mistaken for a part.
    unrelated = ET.SubElement(
        body, "geom", type="sphere", size="1", name="arm_visual_1__variant_1"
    )
    editor.add_variant_geoms_and_materials(variant_map)
    assert len(body.findall("geom")) == 6 * part_count + 2
    assert all(geom in list(body) for geom in originals)
    assert collision.attrib == {"mesh": "arm", "class": "collision"}
    assert unrelated.get("type") == "sphere"
    for geom in body.findall("geom"):
        if geom is collision or geom is unrelated:
            continue
        assert geom.get("pos") in {"0 0 0", "2 0 0"}
        assert geom.get("scale") == "2 3 4"
        assert geom.get("quat") == "0 1 0 0"
        assert geom.get("rgba") == "1 0 0 1"
    names = [geom.get("name") for geom in body.findall("geom") if geom.get("name")]
    assert len(names) == len(set(names))
    editor.save()
    before = editor.xml_path.read_bytes()
    reloaded = MJCFEditor(editor.xml_path)
    reloaded.add_variant_geoms_and_materials(variant_map)
    reloaded.save()
    assert editor.xml_path.read_bytes() == before


def test_variant_expansion_does_not_reference_missing_material(tmp_path):
    editor = MJCFEditor(
        _write_model(tmp_path, '<body><geom mesh="arm" rgba="1 0 0 1" /></body>')
    )
    ET.SubElement(editor.root.find("asset"), "mesh", name="arm", file="arm.obj")
    editor.add_variant_geoms_and_materials({"arm": ["arm.obj"]})
    geom = editor.root.find("worldbody/body/geom")
    assert geom.get("material") is None
    assert geom.get("rgba") == "1 0 0 1"


def test_shared_mesh_scales_are_kept_separate(tmp_path):
    editor = MJCFEditor(
        _write_model(
            tmp_path,
            '<body name="root"><geom type="mesh" mesh="arm" scale="2 1 1" /><geom type="mesh" mesh="arm" /></body>',
        )
    )
    asset = editor.root.find("asset")
    ET.SubElement(asset, "mesh", name="arm", file="arm.obj", scale="1 3 1")
    editor.move_geom_scales_to_meshes()
    geoms = editor.root.findall(".//body/geom")
    assert all("scale" not in geom.attrib for geom in geoms)
    assert geoms[0].get("mesh") != geoms[1].get("mesh")
    assert asset.find(f"mesh[@name='{geoms[0].get('mesh')}']").get("scale") == "2 3 1"
    assert asset.find(f"mesh[@name='{geoms[1].get('mesh')}']").get("scale") == "1 3 1"


def test_procedural_textures_survive_cleanup(tmp_path):
    editor = MJCFEditor(_write_model(tmp_path))
    editor.add_default_ground_plane()
    editor.remove_missing_textures([tmp_path])
    assert editor.root.find(".//texture[@builtin='checker']") is not None
    assert editor.root.find(".//texture[@type='skybox']") is not None
