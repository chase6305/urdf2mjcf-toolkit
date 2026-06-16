import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.logging_utils import URDF2MJCFLogger


class MJCFEditor:
    """
    Utility class for loading, editing, and saving MJCF (xml) files.
    """

    def __init__(self, xml_path):
        self.xml_path = Path(xml_path)
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()
        self.logger = URDF2MJCFLogger.get_logger("MJCFEditor")

    def find_elements(self, tag):
        """Find all elements with the specified tag."""
        return self.root.findall(f".//{tag}")

    def edit_element(self, tag, attrib_key, new_value):
        """Batch modify a specific attribute of all elements with the specified tag."""
        for elem in self.find_elements(tag):
            if attrib_key in elem.attrib:
                elem.attrib[attrib_key] = new_value

    def edit_mesh_file_paths(self, new_path_func):
        """
        Batch modify the file attribute of <asset><mesh> nodes.
        new_path_func: a function that takes the original file path string and returns the new path string.
        """
        for mesh in self.root.findall(".//asset/mesh"):
            if "file" in mesh.attrib:
                old_file = mesh.attrib["file"]
                mesh.attrib["file"] = new_path_func(old_file)

    def get_all_mesh_and_texture_files(self):
        """
        Recursively get the file paths of MJCF <asset><mesh> nodes and all <asset><texture> nodes.
        Returns: (mesh_files, texture_files) - two lists, both as relative path strings.
        """
        mesh_files = []
        texture_files = []
        for mesh in self.root.findall(".//asset/mesh"):
            if "file" in mesh.attrib:
                mesh_files.append(mesh.attrib["file"])
        for tex in self.root.findall(".//asset/texture"):
            if "file" in tex.attrib:
                texture_files.append(tex.attrib["file"])
        return mesh_files, texture_files

    def save(self, out_path=None):
        """Save the modified xml file, auto-indented (no extra newlines), default with _edit suffix."""
        if out_path is None:
            out_path = self.xml_path.with_name(self.xml_path.stem + "_edit.xml")
        try:
            ET.indent(self.tree, space="  ")  # Python 3.9+
        except AttributeError:
            pass  # No indent support in lower versions, just write raw format
        self.tree.write(out_path, encoding="utf-8", xml_declaration=True)

    def fix_base(self, root_body_name: str = "root") -> bool:
        """Fix the base by removing `<freejoint>` from the root body.

        MuJoCo's `<freejoint>` makes the body a 6-DoF floating base.
        For a fixed-base model, remove it.

        Returns True if any freejoint was removed.
        """

        worldbody = self.root.find(".//worldbody")
        if worldbody is None:
            return False

        # Prefer the explicit root body name; otherwise fall back to the first body.
        root_body = worldbody.find(f"./body[@name='{root_body_name}']")
        if root_body is None:
            root_body = worldbody.find("./body")
        if root_body is None:
            return False

        removed = False
        for fj in list(root_body.findall("freejoint")):
            try:
                root_body.remove(fj)
                removed = True
            except Exception:
                pass

        return removed

    def ensure_freejoint(self, root_body_name: str = "root") -> bool:
        """Ensure the root body has one `<freejoint>` for floating-base models.

        Returns True if a freejoint was inserted, False if one already existed or root body not found.
        """

        worldbody = self.root.find(".//worldbody")
        if worldbody is None:
            return False

        root_body = worldbody.find(f"./body[@name='{root_body_name}']")
        if root_body is None:
            root_body = worldbody.find("./body")
        if root_body is None:
            return False

        if root_body.find("freejoint") is not None:
            return False

        fj = ET.Element("freejoint")
        root_body.insert(0, fj)
        return True

    def add_default_ground_plane(self):
        """
        Add a default floor, skybox, lighting, camera, and visual settings.
        """
        # Ensure the asset section exists.
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            asset_node = ET.Element("asset")
            # Insert before the first worldbody when possible.
            insert_index = 0
            for i, child in enumerate(list(self.root)):
                if child.tag == "worldbody":
                    insert_index = i
                    break
            self.root.insert(insert_index, asset_node)

        # Add groundplane assets when missing.
        if asset_node.find("material[@name='groundplane']") is None:
            # skybox
            if asset_node.find("texture[@type='skybox']") is None:
                skybox = ET.Element(
                    "texture",
                    attrib={
                        "type": "skybox",
                        "builtin": "gradient",
                        "rgb1": "0.3 0.5 0.7",
                        "rgb2": "0 0 0",
                        "width": "512",
                        "height": "3072",
                    },
                )
                asset_node.append(skybox)
            # groundplane texture
            tex = ET.Element(
                "texture",
                attrib={
                    "type": "2d",
                    "name": "groundplane",
                    "builtin": "checker",
                    "mark": "edge",
                    "rgb1": "0.2 0.3 0.4",
                    "rgb2": "0.1 0.2 0.3",
                    "markrgb": "0.8 0.8 0.8",
                    "width": "300",
                    "height": "300",
                },
            )
            asset_node.append(tex)
            # groundplane material
            mat = ET.Element(
                "material",
                attrib={
                    "name": "groundplane",
                    "texture": "groundplane",
                    "texuniform": "true",
                    "texrepeat": "5 5",
                    "reflectance": "0.2",
                },
            )
            asset_node.append(mat)

        # Ensure the worldbody section exists.
        worldbody = self.root.find(".//worldbody")
        if worldbody is None:
            worldbody = ET.Element("worldbody")
            self.root.append(worldbody)

        # Add a floor only when missing.
        if worldbody.find("geom[@name='floor']") is None:
            floor = ET.Element(
                "geom",
                attrib={
                    "name": "floor",
                    "size": "0 0 0.05",
                    "type": "plane",
                    "material": "groundplane",
                },
            )
            worldbody.append(floor)

        # Add default light and camera when missing.
        if worldbody.find("light") is None:
            light = ET.Element(
                "light",
                attrib={"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
            )
            worldbody.append(light)
        if worldbody.find("camera[@name='top_camera']") is None:
            cam = ET.Element(
                "camera",
                attrib={"name": "top_camera", "pos": "0 0 5", "xyaxes": "1 0 0 0 1 0"},
            )
            worldbody.append(cam)

        # Add visual settings when missing.
        if self.root.find(".//visual") is None:
            visual = ET.Element("visual")
            headlight = ET.Element(
                "headlight",
                attrib={
                    "diffuse": "0.6 0.6 0.6",
                    "ambient": "0.3 0.3 0.3",
                    "specular": "0 0 0",
                },
            )
            visual.append(headlight)
            rgba = ET.Element("rgba", attrib={"haze": "0.15 0.25 0.35 1"})
            visual.append(rgba)
            global_elem = ET.Element(
                "global", attrib={"azimuth": "120", "elevation": "-20"}
            )
            visual.append(global_elem)
            # Insert before asset/worldbody when possible.
            insert_index = 0
            for i, child in enumerate(list(self.root)):
                if child.tag in {"asset", "worldbody"}:
                    insert_index = i
                    break
            self.root.insert(insert_index, visual)

        # Add statistics settings when missing.
        if self.root.find(".//statistic") is None:
            statistic = ET.Element(
                "statistic", attrib={"center": "0 0 .3", "extent": "1.2"}
            )
            # Insert before asset/worldbody when possible.
            insert_index = 0
            for i, child in enumerate(list(self.root)):
                if child.tag in {"asset", "worldbody"}:
                    insert_index = i
                    break
            self.root.insert(insert_index, statistic)

    def remove_duplicate_visual_collision_defaults(self):
        """
        Keep only top-level visual/collision default classes and remove nested duplicates.
        """
        default_node = self.root.find("./default")
        if default_node is None:
            return
        seen = set()
        to_remove = []
        for child in list(default_node):
            if child.tag == "default" and child.attrib.get("class") in {
                "visual",
                "collision",
            }:
                key = child.attrib["class"]
                if key in seen:
                    to_remove.append(child)
                else:
                    seen.add(key)
        for node in to_remove:
            default_node.remove(node)
        # Remove nested visual/collision defaults.
        for child in list(default_node):
            if child.tag == "default":
                for sub in list(child):
                    if sub.tag == "default" and sub.attrib.get("class") in {
                        "visual",
                        "collision",
                    }:
                        child.remove(sub)

    def ensure_compiler_meshdir(self, meshdir: str = "meshes") -> None:
        """Ensure <compiler> has a `meshdir` so mesh file paths can be relative to meshes/.

        In MuJoCo, `meshdir` applies to mesh assets, but textures use `texturedir`.
        The pipeline stores meshes under `meshes/visual/` and `meshes/collision`, so meshdir="meshes".
        """
        compiler = self.root.find(".//compiler")
        if compiler is None:
            # Create <compiler> when missing.
            compiler = ET.Element("compiler")
            # Insert before the first asset/worldbody when possible.
            insert_index = 0
            for i, child in enumerate(list(self.root)):
                if child.tag in {"asset", "worldbody"}:
                    insert_index = i
                    break
            self.root.insert(insert_index, compiler)
        compiler.set("meshdir", meshdir)
        compiler.set("balanceinertia", "true")

    def ensure_compiler_texturedir(self, texturedir: str = "meshes") -> None:
        """Ensure <compiler> has a `texturedir` so texture file paths can be relative to meshes/.

        In MuJoCo, `meshdir` applies to mesh assets, but textures use `texturedir`.
        Our pipeline stores textures under `meshes/visual/...`, so set texturedir="meshes".
        """

        compiler = self.root.find(".//compiler")
        if compiler is None:
            return
        if "texturedir" not in compiler.attrib:
            compiler.set("texturedir", texturedir)

    def ensure_visual_collision_default_classes(self) -> None:
        """Ensure default classes for visual and collision geoms exist.

        Convention used by the converter:
        - `visualgeom`: render-only, no contacts, no mass/inertia contribution.
        - `collisiongeom`: contact-only, no mass/inertia contribution, hidden in group 3 by convention.
        """

        default_node = self.root.find("./default")
        if default_node is None:
            default_node = ET.Element("default")
            # Put default before asset/worldbody if possible.
            insert_index = 0
            for i, child in enumerate(list(self.root)):
                if child.tag in {"asset", "worldbody"}:
                    insert_index = i
                    break
            self.root.insert(insert_index, default_node)

        def _ensure_class(class_name: str, geom_attrib: dict) -> None:
            node = default_node.find(f"./default[@class='{class_name}']")
            if node is None:
                node = ET.Element("default")
                node.set("class", class_name)
                default_node.append(node)
            geom = node.find("geom")
            if geom is None:
                geom = ET.Element("geom")
                node.append(geom)
            # Reset geom attributes before applying the canonical defaults.
            geom.attrib.clear()
            # Apply canonical attributes.
            for k, v in geom_attrib.items():
                geom.set(k, v)

        # visualgeom: render-only physical defaults.
        _ensure_class(
            "visualgeom",
            {
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "1",
            },
        )
        # collisiongeom: contact-only physical defaults.
        _ensure_class(
            "collisiongeom",
            {
                "contype": "1",
                "conaffinity": "15",
                "density": "0",
                "group": "3",
                "condim": "4",
            },
        )

        # visual/collision only provide grouping and rendering hints.
        def _ensure_simple_class(class_name: str, geom_attrib: dict) -> None:
            node = default_node.find(f"./default[@class='{class_name}']")
            if node is None:
                node = ET.Element("default")
                node.set("class", class_name)
                default_node.append(node)
            geom = node.find("geom")
            if geom is None:
                geom = ET.Element("geom")
                node.append(geom)
            # Keep only grouping/render attributes.
            geom.attrib.clear()
            for k, v in geom_attrib.items():
                geom.set(k, v)

        _ensure_simple_class("visual", {"material": "visualgeom", "group": "1"})
        _ensure_simple_class(
            "collision", {"material": "collision_material", "group": "3"}
        )

    def add_textures_and_link_materials_for_visual_meshes(
        self,
        meshes_dir,
        preferred_filenames=("albedo.png", "basecolor.png", "diffuse.png"),
        exts=(".png", ".jpg", ".jpeg"),
    ) -> int:
        """Create <texture> nodes from meshes/visual and link them to same-name materials.

        - For each <asset><mesh name=... file="visual/.../*.obj">, look for an image in the same
          directory under meshes/visual.
        - Prefer common names like albedo.png; otherwise pick the first image.
        - Create <texture name=... file="visual/.../albedo.png"> and set <material name=mesh_name texture=...>.

        Returns number of textures added.
        """

        meshes_dir = Path(meshes_dir)
        visual_root = meshes_dir / "visual"
        if not visual_root.exists():
            return 0

        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return 0

        existing_textures = {
            t.attrib.get("name")
            for t in asset_node.findall("texture")
            if t.attrib.get("name")
        }
        materials_by_name = {
            m.attrib.get("name"): m
            for m in asset_node.findall("material")
            if m.attrib.get("name")
        }

        added = 0

        def _pick_image(dir_path: Path) -> Path | None:
            # preferred names first
            for fname in preferred_filenames:
                cand = dir_path / fname
                if cand.exists():
                    return cand
            # then any supported ext
            for ext in exts:
                hits = sorted(dir_path.glob(f"*{ext}"))
                if hits:
                    return hits[0]
            return None

        for mesh in asset_node.findall("mesh"):
            mesh_name = mesh.attrib.get("name")
            mesh_file = (mesh.attrib.get("file") or "").replace("\\", "/")
            if not mesh_name or not mesh_file:
                continue
            if not mesh_file.startswith("visual/"):
                continue

            obj_abs = meshes_dir / mesh_file
            if not obj_abs.exists():
                continue
            img = _pick_image(obj_abs.parent)
            if img is None:
                continue

            tex_name = mesh_name
            if tex_name in existing_textures:
                # already has a texture node; still ensure material links it
                pass
            else:
                # avoid collision with some unrelated texture names
                if tex_name in existing_textures:
                    tex_name = f"{mesh_name}_tex"
                tex_elem = ET.Element("texture")
                tex_elem.set("type", "2d")
                tex_elem.set("name", tex_name)
                rel = img.relative_to(meshes_dir).as_posix()
                tex_elem.set("file", rel)
                asset_node.append(tex_elem)
                existing_textures.add(tex_name)
                added += 1

            mat = materials_by_name.get(mesh_name)
            if mat is None:
                mat = ET.Element("material")
                mat.set("name", mesh_name)
                mat.set("specular", "0.0")
                mat.set("shininess", "0.25")
                asset_node.append(mat)
                materials_by_name[mesh_name] = mat
            # Link texture
            mat.set("texture", tex_name)

        return added

    def add_collision_meshes_and_geoms(
        self,
        collision_dir,
        mesh_name_suffix: str = "_collision",
    ) -> int:
        """Add collision meshes/geoms from a `meshes/collision` directory.

        Expected directory layout (relative to compiler meshdir='meshes'):
          collision/.../*.obj

        This will:
        - Add `<asset><mesh>` entries named `<stem>_collision` pointing to `collision/.../<stem>.obj`.
        - For each body that already has a visual geom with `mesh=<stem>`, append a collision geom
          with `type='mesh' mesh='<stem>_collision'`.

        Returns the number of collision geoms added.
        """

        collision_dir = Path(collision_dir)
        if not collision_dir.exists():
            return 0

        meshes_root = collision_dir.parent

        # Collect collision obj files (by stem)
        collision_map = {}
        for obj_path in collision_dir.rglob("*.obj"):
            rel = obj_path.relative_to(meshes_root).as_posix()
            collision_map[obj_path.stem] = rel

        if not collision_map:
            return 0

        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return 0

        existing_mesh_names = {
            m.attrib.get("name")
            for m in asset_node.findall("mesh")
            if m.attrib.get("name")
        }

        worldbody = self.root.find(".//worldbody")
        if worldbody is None:
            return 0

        collision_classes = {"collision", "collisiongeom"}

        # Global guard: if URDF/converted MJCF already contains any collision geometry,
        # do not inject auto-generated rough collision meshes for missing links.
        existing_collision_in_model = any(
            (g.attrib.get("class") in collision_classes)
            or (g.attrib.get("contype") not in (None, "0"))
            or (g.attrib.get("conaffinity") not in (None, "0"))
            for g in worldbody.findall(".//geom")
        )
        if existing_collision_in_model:
            return 0

        added = 0
        # For each body: infer its link mesh name from the first geom's mesh attribute
        for body in worldbody.findall(".//body"):
            geoms = body.findall("geom")
            if not geoms:
                continue

            # Find the visual geom mesh name without the collision suffix.
            visual_geom = None
            for g in geoms:
                if g.attrib.get("class") in {"visual", "visualgeom"}:
                    visual_geom = g
                    break
            if visual_geom is None:
                visual_geom = geoms[0]
            link_mesh_name = visual_geom.attrib.get("mesh")
            if not link_mesh_name:
                continue
            # collision_map stores stems without the collision suffix.
            if link_mesh_name not in collision_map:
                continue

            col_mesh_name = f"{link_mesh_name}{mesh_name_suffix}"

            # Ensure mesh asset exists only when we really need to add this collision geom.
            if col_mesh_name not in existing_mesh_names:
                rel_file = collision_map[link_mesh_name]
                mesh_elem = ET.Element("mesh")
                mesh_elem.set("name", col_mesh_name)
                mesh_elem.set("file", rel_file)
                mesh_elem.set("content_type", "model/obj")
                asset_node.append(mesh_elem)
                existing_mesh_names.add(col_mesh_name)

            if any(g.attrib.get("mesh") == col_mesh_name for g in geoms):
                continue

            col_geom = ET.Element("geom")
            col_geom.set("type", "mesh")
            col_geom.set("mesh", col_mesh_name)
            col_geom.set("class", "collisiongeom")
            # Avoid double-counting mass/inertia from collision geometry
            col_geom.set("density", "0")
            body.append(col_geom)
            added += 1

        return added

    def add_textures_for_mesh_variants(
        self, variant_map, mesh_dir, exts=(".png", ".jpg", ".jpeg")
    ):
        """
        Create one texture node for each mesh variant.

        The texture name matches the mesh variant name. The texture file is found
        by replacing the OBJ suffix with a supported image extension. Existing
        texture nodes are cleared before new variant textures are added.

        variant_map: {mesh_name: [list of obj paths]}
        mesh_dir: Path or str pointing to the visual asset root.
        exts: Supported texture suffixes.
        """
        from pathlib import Path

        mesh_dir = Path(mesh_dir)
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return
        # Clear existing texture nodes.
        for tex in list(asset_node.findall("texture")):
            asset_node.remove(tex)
        # Create texture nodes for every mesh variant.
        for mesh_name, obj_list in variant_map.items():
            for idx, obj_path in enumerate(obj_list):
                # Variant name.
                if idx == 0:
                    tex_name = mesh_name
                else:
                    tex_name = f"{mesh_name}_{idx}"
                # Drop the OBJ suffix before trying image extensions.
                obj_path_no_ext = Path(obj_path).with_suffix("")
                # Look for texture files in the same relative directory.
                for ext in exts:
                    tex_path = mesh_dir / obj_path_no_ext.with_suffix(ext)
                    if tex_path.exists():
                        tex_elem = ET.Element("texture")
                        tex_elem.set("type", "2d")
                        tex_elem.set("name", tex_name)
                        rel_path = tex_path.relative_to(mesh_dir)
                        tex_elem.set("file", str(rel_path).replace("\\", "/"))
                        asset_node.append(tex_elem)
                        break

    def get_material_info_from_subxml(subxml_path, variant_name):
        """
        Parse a generated sub-XML file and return material/texture properties for a variant.

        Returns {"material": {...}, "texture": {...}} or None.
        """
        try:
            tree = ET.parse(subxml_path)
            root = tree.getroot()
        except Exception:
            return None
        # Find <geom mesh=... material=...>.
        mat_name = None
        for geom in root.findall(".//geom"):
            mesh = geom.attrib.get("mesh")
            if mesh == variant_name:
                mat_name = geom.attrib.get("material")
                break
        if not mat_name:
            return None
        # Find <material name=mat_name>.
        mat_info = None
        for mat in root.findall(".//asset/material"):
            if mat.attrib.get("name") == mat_name:
                mat_info = mat.attrib.copy()
                break
        if not mat_info:
            return None
        # Find <texture> when the material references one.
        tex_info = None
        tex_name = mat_info.get("texture")
        if tex_name:
            for tex in root.findall(".//asset/texture"):
                if tex.attrib.get("name") == tex_name:
                    tex_info = tex.attrib.copy()
                    break
        return {"material": mat_info, "texture": tex_info}

    def add_material_from_subxml_if_no_texture(self, mesh_name, mesh_dict, mesh_dir):
        """
        Add a material from a generated sub-XML file when a mesh has no texture.
        """
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return

        if not mesh_dict or len(mesh_dict) == 0:
            return

        obj_path = Path(mesh_dict[0])
        subxml_path = Path(mesh_dir) / obj_path.parent / (obj_path.stem + ".xml")
        if not subxml_path.exists():
            self.logger.warning(
                f"Subxml not found for mesh: {mesh_name} at {subxml_path}"
            )
            return
        try:
            tree = ET.parse(subxml_path)
            root = tree.getroot()
        except Exception:
            return

        for body in root.findall(".//body"):
            for geom in body.findall("geom"):
                mesh = geom.attrib.get("mesh")
                material = geom.attrib.get("material")
                self.logger.debug(f"mesh: {mesh}, material: {material}")

        # Inspect asset materials.
        for mat in root.findall(".//asset/material"):
            name = mat.attrib.get("name")
            specular = mat.attrib.get("specular")
            shininess = mat.attrib.get("shininess")
            rgba = mat.attrib.get("rgba")
            self.logger.debug(
                f"material name: {name}, specular: {specular}, shininess: {shininess}, rgba: {rgba}"
            )

        # Find <geom material=... mesh=mesh_name>.
        mat_name = None
        for geom in root.findall(".//geom"):
            if geom.attrib.get("mesh") == mesh_name:
                mat_name = geom.attrib.get("material")
                break
        if not mat_name:
            return
        # Find <material name=mat_name>.
        mat_elem = None
        for mat in root.findall(".//asset/material"):
            if mat.attrib.get("name") == mat_name:
                mat_elem = mat
                break
        if not mat_elem:
            return
        # Create a new material node and normalize shininess.
        new_mat = ET.Element("material")
        new_mat.set("name", mesh_name)
        for k, v in mat_elem.attrib.items():
            if k == "name":
                continue
            if k == "shininess":
                new_mat.set("shininess", "0.25")
            else:
                new_mat.set(k, v)
        asset_node.append(new_mat)

    def remap_materials_and_textures_for_mesh_variants_with_xml(
        self, variant_map, mesh_dir, exts=(".png", ".jpg", ".jpeg")
    ):
        """
        Create material and texture nodes for every mesh variant.

        Material and texture names match the mesh variant name. If a texture is
        found, the material references it. Otherwise, material attributes such as
        rgba, specular, and shininess are copied from the generated sub-XML.

        variant_map: {mesh_name: [list of obj paths]}
        mesh_dir: Path or str pointing to the visual asset root.
        exts: Supported texture suffixes.
        """
        from pathlib import Path

        mesh_dir = Path(mesh_dir)
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return

        # Remove only material and texture nodes related to this variant map.
        variant_names = set()
        for mesh_name, obj_list in variant_map.items():
            for idx in range(len(obj_list)):
                variant_names.add(f"{mesh_name}_{idx}" if idx > 0 else mesh_name)

        for mat in list(asset_node.findall("material")):
            if mat.attrib.get("name") in variant_names:
                asset_node.remove(mat)
        for tex in list(asset_node.findall("texture")):
            if tex.attrib.get("name") in variant_names:
                asset_node.remove(tex)

        # Map mesh names to their original material names from <geom> nodes.
        mesh_to_material = {}
        for body in self.root.findall(".//body"):
            for geom in body.findall("geom"):
                mesh_name = geom.attrib.get("mesh")
                material_name = geom.attrib.get("material")
                if mesh_name and material_name:
                    mesh_to_material[mesh_name] = material_name

        # Cache original material attributes.
        material_props = {}
        for mat in self.root.findall(".//asset/material"):
            name = mat.attrib.get("name")
            if name:
                material_props[name] = mat.attrib.copy()

        # Create new material and texture nodes.
        for mesh_name, obj_list in variant_map.items():
            for idx, obj_path in enumerate(obj_list):
                variant_name = f"{mesh_name}_{idx}" if idx > 0 else mesh_name
                obj_path_no_ext = Path(obj_path).with_suffix("")
                # Look for texture files.
                tex_found = False
                for ext in exts:
                    tex_path = mesh_dir / obj_path_no_ext.with_suffix(ext)
                    if tex_path.exists():
                        # Create texture node.
                        tex_elem = ET.Element("texture")
                        tex_elem.set("type", "2d")
                        tex_elem.set("name", variant_name)
                        rel_path = tex_path.relative_to(mesh_dir)
                        tex_elem.set("file", str(rel_path).replace("\\", "/"))
                        asset_node.append(tex_elem)
                        # Create material node referencing the texture.
                        mat_elem = ET.Element("material")
                        mat_elem.set("name", variant_name)
                        mat_elem.set("texture", variant_name)
                        mat_elem.set("specular", "0.0")
                        mat_elem.set("shininess", "0.25")
                        asset_node.append(mat_elem)
                        tex_found = True
                        break
                if not tex_found:
                    # Without a texture, fall back to material attributes from the sub-XML.
                    # variant_dict = variant_map.get(variant_name, None)
                    obj_path = Path(obj_list[0])
                    subxml_path = mesh_dir / obj_path.parent / (mesh_name + ".xml")

                    if subxml_path.exists():

                        try:
                            tree = ET.parse(subxml_path)
                            root = tree.getroot()
                        except Exception:
                            root = None
                        if root is not None:
                            mat_name = None
                            for geom in root.findall(".//geom"):
                                if geom.attrib.get("mesh") == variant_name:
                                    mat_name = geom.attrib.get("material")
                                    break
                            if mat_name:
                                # find the material element in subxml
                                for mat in root.findall(".//asset/material"):
                                    if mat.attrib.get("name") == mat_name:
                                        rgba = mat.attrib.get("rgba")
                                        texture_ref = mat.attrib.get("texture")
                                        # if material references a texture, locate its file
                                        if texture_ref:
                                            tex_file = None
                                            for tex in root.findall(".//asset/texture"):
                                                if (
                                                    tex.attrib.get("name")
                                                    == texture_ref
                                                ):
                                                    tex_file = tex.attrib.get("file")
                                                    break
                                            # try to compute a path relative to mesh_dir
                                            rel_path = None
                                            if tex_file:
                                                tex_path = Path(tex_file)
                                                if not tex_path.is_absolute():
                                                    tex_abs = (
                                                        subxml_path.parent / tex_file
                                                    ).resolve()
                                                else:
                                                    tex_abs = tex_path
                                                try:
                                                    rel_path = tex_abs.relative_to(
                                                        mesh_dir
                                                    )
                                                    rel_path = str(rel_path).replace(
                                                        "\\", "/"
                                                    )
                                                except Exception:
                                                    rel_path = str(tex_file)
                                            if rel_path:
                                                # create texture node named as variant
                                                tex_elem = ET.Element("texture")
                                                tex_elem.set("type", "2d")
                                                tex_elem.set("name", variant_name)
                                                tex_elem.set("file", rel_path)
                                                asset_node.append(tex_elem)
                                                # create material referencing this texture
                                                mat_elem = ET.Element("material")
                                                mat_elem.set("name", variant_name)
                                                mat_elem.set("texture", variant_name)
                                                mat_elem.set("specular", "0.0")
                                                mat_elem.set("shininess", "0.25")
                                                asset_node.append(mat_elem)
                                            else:
                                                # no texture file available, fall back to rgba
                                                mat_elem = ET.Element("material")
                                                mat_elem.set("name", variant_name)
                                                mat_elem.set("specular", "0.0")
                                                mat_elem.set("shininess", "0.25")
                                                if rgba:
                                                    mat_elem.set("rgba", rgba)
                                                asset_node.append(mat_elem)
                                        else:
                                            # no texture reference, use rgba if present
                                            mat_elem = ET.Element("material")
                                            mat_elem.set("name", variant_name)
                                            mat_elem.set("specular", "0.0")
                                            mat_elem.set("shininess", "0.25")
                                            if rgba:
                                                mat_elem.set("rgba", rgba)
                                            asset_node.append(mat_elem)
                                        break

    def ensure_visual_prefix_for_all_textures(self):
        """
        Ensure all <texture> file attributes are normalized and start with 'visual/' (lowercase).

        Notes:
        - The pipeline uses output folders `meshes/visual` and `meshes/collision`.
        - Some upstream tools emit `meshes/Visual/...` or `Visual/...` (capitalized).
          This function normalizes those to lowercase to avoid missing-file warnings.
        """

        def _norm(p: str) -> str:
            p = p.replace("\\", "/")
            if p.startswith("./"):
                p = p[2:]
            # normalize common casing/prefix variants
            for src, dst in (
                ("meshes/Visual/", "meshes/visual/"),
                ("meshes/Collision/", "meshes/collision/"),
                ("Visual/", "visual/"),
                ("Collision/", "collision/"),
            ):
                if p.startswith(src):
                    p = dst + p[len(src) :]
            # keep relative paths without leading slash
            return p.lstrip("/")

        for tex in self.root.findall(".//asset/texture"):
            file_path = tex.attrib.get("file")
            if not file_path:
                continue
            file_path = _norm(file_path)
            # After `remove_meshes_prefix`, the expected form is `visual/...`.
            if file_path.startswith("meshes/"):
                file_path = file_path[len("meshes/") :]
            if not file_path.startswith(("visual/", "collision/")):
                file_path = "visual/" + file_path
            tex.attrib["file"] = file_path

    def add_texture_for_meshes(self, texture_exts=(".png", ".jpg", ".jpeg")):
        """
        For meshes without a <texture> node, automatically search for texture files in the same directory as the mesh file and add a <texture> node for it.
        Only supports mesh files as relative paths, and texture files must be in the same directory as the mesh file.
        Automatically adds 'visual/' prefix to the texture file path if not present.
        """
        import os

        mesh_nodes = self.root.findall(".//asset/mesh")
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return
        for mesh in mesh_nodes:
            mesh_file = mesh.attrib.get("file", None)
            if not mesh_file:
                continue
            # Check if a corresponding texture already exists
            mesh_name = mesh.attrib.get("name", None)
            has_texture = False
            for tex in self.root.findall(".//asset/texture"):
                if tex.attrib.get("name", None) == mesh_name:
                    has_texture = True
                    break
            if has_texture:
                continue
            # Search for texture in the same directory as the mesh file
            mesh_path = Path(self.xml_path).parent / mesh_file
            if not mesh_path.exists():
                continue
            for ext in texture_exts:
                tex_path = mesh_path.with_suffix(ext)
                if tex_path.exists():
                    # Add <texture> node
                    tex_elem = ET.Element("texture")
                    # Always set texture name to mesh_name for consistency
                    tex_elem.set("name", mesh_name if mesh_name else "")
                    rel_path = str(tex_path.relative_to(self.xml_path.parent))
                    rel_path = rel_path.replace("\\", "/")
                    # Ensure 'visual/' prefix
                    if rel_path.startswith("meshes/"):
                        rel_path = rel_path[len("meshes/") :]
                    if rel_path.startswith("Visual/"):
                        rel_path = "visual/" + rel_path[len("Visual/") :]
                    if not rel_path.startswith(("visual/", "collision/")):
                        rel_path = "visual/" + rel_path
                    tex_elem.set("file", rel_path)
                    asset_node.append(tex_elem)
                    break

    def update_mesh_and_texture_paths(
        self, visual_dir=None, prefix="visual/", variant_map=None
    ):
        """
        Complete the file attribute of all <mesh> and <texture> nodes as prefix+relative path (preserving subdirectory structure).
        visual_dir: Path or str, points to the Visual/Collision directory.
        prefix: str, prefix (such as 'visual/' or 'collision/')
        variant_map: dict, obj variant map (such as {'head2': [...], ...}), if provided, will be used with priority.
        """
        if visual_dir is None and variant_map is None:
            return
        file_map = {}
        if variant_map is not None:
            # Use only the first obj path in the map for completion (if there are multiple variants, use the first by default)
            for k, v in variant_map.items():
                if v:
                    file_map[k] = v[0]
        else:
            visual_dir = Path(visual_dir).resolve()
            for f in visual_dir.rglob("*"):
                if f.is_file():
                    rel_path = f.relative_to(visual_dir).as_posix()
                    file_map[f.name] = rel_path
                    file_map[f.stem] = rel_path

        # Normalize geom mesh references to mesh names without file extensions.
        for geom in self.root.findall(".//geom"):
            mesh_attr = geom.get("mesh")
            if mesh_attr:
                mesh_name = Path(mesh_attr).stem
                if mesh_attr != mesh_name:
                    geom.set("mesh", mesh_name)

        for mesh in self.root.findall(".//asset/mesh"):
            mesh_name = mesh.attrib.get("name")
            if mesh_name and mesh_name in file_map:
                mesh.attrib["file"] = prefix + file_map[mesh_name]
            elif mesh_name and Path(mesh_name).stem in file_map:
                mesh.attrib["file"] = prefix + file_map[Path(mesh_name).stem]
            else:
                self.logger.warning(f"Mesh file not found for asset: {mesh_name}")

        for tex in self.root.findall(".//asset/texture"):
            fname = tex.attrib.get("file")
            if fname:
                fname_only = Path(fname.replace("\\", "/")).name
            else:
                fname_only = ""
            if fname_only and fname_only in file_map:
                tex.attrib["file"] = prefix + file_map[fname_only]
            elif fname:
                base = Path(fname.replace("\\", "/")).stem
                matches = [v for k, v in file_map.items() if k.startswith(base)]
                if len(matches) == 1:
                    tex.attrib["file"] = prefix + matches[0]
                    self.logger.info(f"Texture file fallback: {fname} -> {matches[0]}")
                else:
                    self.logger.warning(f"Texture file not found in asset dir: {fname}")

    def remove_meshes_prefix(self):
        """
        Batch remove the 'meshes/' prefix from the file attribute of <mesh> and <texture> nodes (if present).
        """
        for mesh in self.root.findall(".//asset/mesh"):
            if "file" in mesh.attrib and mesh.attrib["file"].startswith("meshes/"):
                mesh.attrib["file"] = mesh.attrib["file"][7:]
        for tex in self.root.findall(".//asset/texture"):
            if "file" in tex.attrib and tex.attrib["file"].startswith("meshes/"):
                tex.attrib["file"] = tex.attrib["file"][7:]

    def add_texture_and_material_for_meshes(
        self, mesh_dir, texture_exts=(".png", ".jpg", ".jpeg")
    ):
        """
        For each mesh, automatically add <texture> and <material> nodes. The texture name is the same as the mesh name, and the texture path is the image with the same name in the same directory as the mesh.
        If there are multiple textures, they are automatically numbered.
        """
        import os

        mesh_dir = Path(mesh_dir)
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return
        mesh_nodes = self.root.findall(".//asset/mesh")
        for mesh in mesh_nodes:
            mesh_name = mesh.attrib.get("name", None)
            if not mesh_name:
                continue
            found_obj = None
            for obj_path in mesh_dir.rglob(f"{mesh_name}.obj"):
                found_obj = obj_path
                break
            if not found_obj:
                continue
            # Find all textures in the same directory
            tex_candidates = []
            for ext in texture_exts:
                for tex_path in found_obj.parent.glob(f"*{ext}"):
                    tex_candidates.append(tex_path)
            for idx, tex_path in enumerate(tex_candidates):
                tex_name = mesh_name if idx == 0 else f"{mesh_name}_{idx}"
                # Check if this texture already exists
                exists = any(
                    t.attrib.get("name") == tex_name
                    for t in self.root.findall(".//asset/texture")
                )
                if not exists:
                    tex_elem = ET.Element("texture")
                    tex_elem.set("type", "2d")
                    tex_elem.set("name", tex_name)
                    rel_tex = tex_path.relative_to(mesh_dir)
                    tex_elem.set("file", str(rel_tex).replace("\\", "/"))
                    asset_node.append(tex_elem)
                # # Check if this material already exists
                # exists = any(
                #     m.attrib.get("name") == tex_name
                #     for m in self.root.findall(".//asset/material")
                # )
                # if not exists:
                #     mat_elem = ET.Element("material")
                #     mat_elem.set("name", tex_name)
                #     mat_elem.set("texture", tex_name)
                #     mat_elem.set("specular", "0.0")
                #     mat_elem.set("shininess", "0.25")
                #     asset_node.append(mat_elem)

    def flatten_visual_collision(self, visual_dir):
        """
        Recursively move visual_dir/Collision/* to visual_dir, removing the redundant Collision level.
        """
        import shutil

        visual_dir = Path(visual_dir)
        collision_dir = visual_dir / "Collision"
        if not collision_dir.exists():
            return
        for sub in collision_dir.iterdir():
            target = visual_dir / sub.name
            if target.exists():
                # Merge contents
                for f in sub.rglob("*"):
                    rel = f.relative_to(sub)
                    dest = target / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                try:
                    sub.rmdir()
                except Exception:
                    pass
            else:
                shutil.move(str(sub), str(target))
        # Delete empty Collision directory
        try:
            collision_dir.rmdir()
        except Exception:
            pass

    def auto_add_mesh_variants(self, variant_map, prefix="Visual/"):
        """
        Clear existing <mesh> nodes and add globally unique mesh assets.
        variant_map: {mesh_name: [list of obj paths]}
        prefix: path prefix (such as 'Visual/')
        """
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return
        # Clear existing mesh nodes.
        for mesh in list(asset_node.findall("mesh")):
            asset_node.remove(mesh)

        # Deduplicate by file path globally.
        added_files = set()
        mesh_items = []
        for mesh_name, obj_list in variant_map.items():
            # Deduplicate while preserving order.
            seen = set()
            unique_obj_list = []
            for obj_path in obj_list:
                if obj_path not in seen:
                    seen.add(obj_path)
                    unique_obj_list.append(obj_path)
            # Sort by numeric filename suffix so variant indices follow file suffixes.
            try:
                import re

                def _sort_key(p):
                    stem = Path(p).stem
                    m = re.search(r"_(\d+)$", stem)
                    if m:
                        return (0, int(m.group(1)), stem)
                    m2 = re.search(r"(\d+)$", stem)
                    if m2:
                        return (0, int(m2.group(1)), stem)
                    return (1, stem)

                unique_obj_list.sort(key=_sort_key)
            except Exception:
                pass
            for idx, obj_path in enumerate(unique_obj_list):
                file_path = prefix + obj_path
                mesh_items.append((mesh_name, idx, file_path))
        for mesh_name, idx, file_path in mesh_items:
            if file_path in added_files:
                continue
            added_files.add(file_path)
            if idx == 0:
                mjcf_mesh_name = mesh_name
            else:
                mjcf_mesh_name = f"{mesh_name}_{idx}"
            mesh_elem = ET.Element("mesh")
            mesh_elem.set("name", mjcf_mesh_name)
            mesh_elem.set("file", file_path)
            mesh_elem.set("content_type", "model/obj")
            asset_node.append(mesh_elem)

    def deduplicate_mesh_names(self):
        """
        Ensure all <mesh> node names are unique by appending an index to duplicates (e.g., head1, head1_1, head1_2).
        """
        name_count = {}
        for mesh in self.root.findall(".//asset/mesh"):
            name = mesh.attrib.get("name")
            if not name:
                continue
            if name not in name_count:
                name_count[name] = 0
            else:
                name_count[name] += 1
                new_name = f"{name}_{name_count[name]}"
                mesh.attrib["name"] = new_name

    def auto_fix_texture_paths(self, search_dirs=None):
        """
        Automatically fix <texture> file paths:
        - If the file does not exist, search for similar files in the same directory (case-insensitive, underscore/dash variants).
        - If found, update the file attribute; if not, print a warning.
        search_dirs: list of Path or str, directories to search for textures. If None, use xml file's parent.
        """
        import os
        from pathlib import Path
        import difflib

        if search_dirs is None:
            search_dirs = [self.xml_path.parent]
        else:
            search_dirs = [Path(d) for d in search_dirs]

        for tex in self.root.findall(".//asset/texture"):
            tex_file = tex.attrib.get("file")
            if not tex_file:
                continue
            # normalize common casing/prefix variants to reduce false negatives
            tex_file = tex_file.replace("\\", "/")
            if tex_file.startswith("meshes/Visual/"):
                tex_file = "meshes/visual/" + tex_file[len("meshes/Visual/") :]
            if tex_file.startswith("Visual/"):
                tex_file = "visual/" + tex_file[len("Visual/") :]
            if tex_file.startswith("meshes/Collision/"):
                tex_file = "meshes/collision/" + tex_file[len("meshes/Collision/") :]
            if tex_file.startswith("Collision/"):
                tex_file = "collision/" + tex_file[len("Collision/") :]
            tex.attrib["file"] = tex_file
            found = False
            for search_dir in search_dirs:
                abs_path = search_dir / tex_file
                if abs_path.exists():
                    found = True
                    break
                # Try to find similar files in the same directory
                candidates = (
                    list(abs_path.parent.glob("*.png"))
                    + list(abs_path.parent.glob("*.jpg"))
                    + list(abs_path.parent.glob("*.jpeg"))
                )
                # Use difflib to find closest match
                matches = difflib.get_close_matches(
                    abs_path.name, [c.name for c in candidates], n=1, cutoff=0.7
                )
                if matches:
                    best_match = abs_path.parent / matches[0]
                    rel_path = best_match.relative_to(search_dir)
                    tex.attrib["file"] = str(rel_path)
                    self.logger.info(f"Fixed texture path: {tex_file} -> {rel_path}")
                    found = True
                    break
            if not found:
                self.logger.warning(f"Texture file not found: {tex_file}")

    def add_variant_geoms_and_materials(self, variant_map):
        """
        For each entry in variant_map, ensure that under the corresponding
        `<body name="<mesh_name>">` in `./worldbody` there are `<geom>` nodes
        for each variant (class="visualgeom", mesh="<variant_name>") and that
        each such geom references a material named exactly as the variant.

        If the material does not exist in `<asset>`, create it. Preference
        order for material creation: existing texture (same-name texture node),
        else attempt to find texture file under mesh_dir, else create rgba-only
        material using a default appearance.

        variant_map: dict of {mesh_name: [list of obj paths]}
        """
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            self.logger.warning(
                "No <asset> node found, skipping add_variant_geoms_and_materials"
            )
            return

        # collect existing material names for quick checks
        existing_mats = {m.attrib.get("name") for m in asset_node.findall("material")}

        worldbody = self.root.find(".//worldbody")
        if worldbody is None:
            self.logger.warning(
                "No <worldbody> node found, skipping add_variant_geoms_and_materials"
            )
            return

        self.logger.info(
            f"add_variant_geoms_and_materials: processing {len(variant_map)} variants"
        )

        visual_class_name = "visualgeom"
        collision_classes = {"collision", "collisiongeom"}

        # For each body (preserve nesting order), if its name is in variant_map,
        # ensure visual geoms exist and remove inline rgba on geoms (use materials instead).
        for body in worldbody.findall(".//body"):
            body_name = body.attrib.get("name")

            if not body_name:
                continue
            geoms_list = body.findall("geom")
            if not geoms_list:
                continue

            # Prefer a visual mesh in this body; many models place collision geom first.
            mesh_name = None
            for g in geoms_list:
                g_mesh = g.attrib.get("mesh")
                if not g_mesh:
                    continue
                g_class = g.attrib.get("class")
                if g_class in {"visual", "visualgeom"} and g_mesh in variant_map:
                    mesh_name = g_mesh
                    break

            # Fallback: pick any non-collision mesh that exists in variant_map.
            if mesh_name is None:
                for g in geoms_list:
                    g_mesh = g.attrib.get("mesh")
                    if not g_mesh:
                        continue
                    g_class = g.attrib.get("class")
                    if g_class in collision_classes:
                        continue
                    if g_mesh in variant_map:
                        mesh_name = g_mesh
                        break

            if mesh_name not in variant_map:
                continue

            obj_list = variant_map.get(mesh_name, [])
            for idx, obj_path in enumerate(obj_list):
                variant_name = f"{mesh_name}_{idx}" if idx > 0 else mesh_name

                # check whether any geom already references this mesh
                geom_exists = False
                same_mesh_geoms = [
                    g
                    for g in body.findall("geom")
                    if g.attrib.get("mesh") == variant_name
                ]
                chosen = None
                # prefer an already visual geom
                for g in same_mesh_geoms:
                    if g.attrib.get("class") in {visual_class_name, "visual"}:
                        chosen = g
                        break
                # otherwise prefer any non-collision geom to upgrade
                if chosen is None:
                    for g in same_mesh_geoms:
                        if g.attrib.get("class") not in collision_classes:
                            chosen = g
                            break
                if chosen is not None:
                    # ensure it is visual and references the correct material
                    if chosen.attrib.get("class") not in {visual_class_name, "visual"}:
                        chosen.set("class", visual_class_name)
                    if chosen.attrib.get("material") != variant_name:
                        chosen.set("material", variant_name)
                        self.logger.debug(
                            f"Set material='{variant_name}' on existing geom(mesh={variant_name}) in body {mesh_name}"
                        )
                    # remove inline rgba from chosen geom to avoid conflicting appearance
                    if "rgba" in chosen.attrib:
                        del chosen.attrib["rgba"]
                        self.logger.debug(
                            f"Removed inline rgba from geom(mesh={variant_name}) in body {mesh_name}"
                        )
                    # remove other non-collision duplicates that reference same mesh
                    for g in list(same_mesh_geoms):
                        if g is chosen:
                            continue
                        if g.attrib.get("class") not in collision_classes:
                            try:
                                body.remove(g)
                                self.logger.debug(
                                    f"Removed duplicate geom(mesh={variant_name}) in body {mesh_name}"
                                )
                            except Exception:
                                pass
                    geom_exists = True

                # create material if missing (commented out previously) — log status
                if variant_name not in existing_mats:
                    self.logger.debug(f"Material '{variant_name}' not found in <asset>")

                if geom_exists:
                    continue

                if len(obj_list) > 1:
                    # Prefer upgrading an existing non-collision geom that already references this mesh
                    upgraded = False
                    for g in body.findall("geom"):
                        if g.attrib.get("mesh") == variant_name:
                            # If it's not a collision geom, upgrade to visual and set material
                            if g.attrib.get("class") not in collision_classes:
                                g.set("material", variant_name)
                                if "rgba" in g.attrib:
                                    del g.attrib["rgba"]
                                upgraded = True
                                self.logger.debug(
                                    f"Upgraded existing geom(mesh={variant_name}) to visual/material in body {mesh_name}"
                                )
                                break
                    if upgraded:
                        continue
                    # otherwise create visual geom and insert before first collision geom to preserve ordering
                    new_geom = ET.Element("geom")
                    new_geom.set("type", "mesh")
                    new_geom.set("class", visual_class_name)
                    new_geom.set("material", variant_name)
                    new_geom.set("mesh", variant_name)
                else:
                    # single variant: try to reuse or upgrade an existing geom instead of adding duplicates
                    updated = False
                    for g in body.findall("geom"):
                        if g.attrib.get("mesh") == variant_name:
                            # prefer not to convert collision geoms to visual; only upgrade non-collision geoms
                            if g.attrib.get("class") not in collision_classes:
                                g.set("class", visual_class_name)
                                g.set("material", variant_name)
                                if "rgba" in g.attrib:
                                    del g.attrib["rgba"]
                                updated = True
                                self.logger.debug(
                                    f"Updated existing geom(mesh={variant_name}) to visual/material in body {mesh_name}"
                                )
                                break
                    if updated:
                        continue
                    # if no suitable existing geom found, insert a visual geom.
                    # NOTE: for type="mesh", MuJoCo requires a valid `mesh` attribute.
                    new_geom = ET.Element("geom")
                    new_geom.set("type", "mesh")
                    new_geom.set("class", visual_class_name)
                    new_geom.set("material", variant_name)
                    new_geom.set("mesh", variant_name)

                inserted = False
                for i, child in enumerate(list(body)):
                    if (
                        child.tag == "geom"
                        and child.attrib.get("class") in collision_classes
                    ):
                        body.insert(i, new_geom)
                        inserted = True
                        self.logger.debug(
                            f"Inserted visual geom(mesh={variant_name}) before collision geom in body {mesh_name}"
                        )
                        break
                if not inserted:
                    body.append(new_geom)
                    self.logger.debug(
                        f"Appended visual geom(mesh={variant_name}) to body {mesh_name}"
                    )

    def remove_missing_textures(self, search_dirs=None):
        """
        Remove <texture> nodes whose file does not exist in the given search directories.
        search_dirs: list of Path or str, directories to search for textures. If None, use xml file's parent.
        """
        from pathlib import Path

        if search_dirs is None:
            search_dirs = [self.xml_path.parent]
        else:
            search_dirs = [Path(d) for d in search_dirs]
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return
        textures = list(asset_node.findall("texture"))
        for tex in textures:
            tex_file = tex.attrib.get("file")
            if not tex_file:
                asset_node.remove(tex)
                continue
            tex_file = tex_file.replace("\\", "/")
            # normalize common casing variants
            if tex_file.startswith("meshes/Visual/"):
                tex_file_norm = "meshes/visual/" + tex_file[len("meshes/Visual/") :]
            elif tex_file.startswith("Visual/"):
                tex_file_norm = "visual/" + tex_file[len("Visual/") :]
            else:
                tex_file_norm = tex_file
            found = False
            for search_dir in search_dirs:
                if (search_dir / tex_file).exists() or (
                    search_dir / tex_file_norm
                ).exists():
                    found = True
                    break
            if not found:
                asset_node.remove(tex)
                self.logger.info(f"Removed missing texture: {tex_file}")

    def remove_invalid_material_textures(self):
        """
        Remove or fix <material> nodes whose texture attribute references a missing <texture> node.
        If texture is missing, remove the texture attribute from material.
        """
        asset_node = self.root.find(".//asset")
        if asset_node is None:
            return
        # Collect all valid texture names
        valid_textures = set()
        for tex in asset_node.findall("texture"):
            name = tex.attrib.get("name")
            if name:
                valid_textures.add(name)
        # Check all materials
        for mat in asset_node.findall("material"):
            tex_name = mat.attrib.get("texture")
            if tex_name and tex_name not in valid_textures:
                del mat.attrib["texture"]
                self.logger.info(
                    f"[INFO] Removed invalid texture reference from material: {mat.attrib.get('name', '')}"
                )

    def remove_invalid_mesh_geoms(self) -> int:
        """Remove invalid `<geom type='mesh'>` nodes without a `mesh` attribute.

        Returns the number of removed geoms.
        """

        removed = 0
        for body in self.root.findall(".//body"):
            for geom in list(body.findall("geom")):
                if geom.attrib.get("type") != "mesh":
                    continue
                mesh_name = (geom.attrib.get("mesh") or "").strip()
                if mesh_name:
                    continue
                body.remove(geom)
                removed += 1
        if removed:
            self.logger.info(f"Removed invalid mesh geoms without mesh attr: {removed}")
        return removed
