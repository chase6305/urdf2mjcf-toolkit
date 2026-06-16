#!/usr/bin/env python3
"""
GLB to OBJ Converter
Convert GLB files to OBJ files and extract textures.

Usage:
    python -m urdf2mjcf.glb_to_obj_converter input_file_or_directory [-o output_directory] [--no-merge]

Dependencies:
    pip install trimesh[easy] numpy pillow

Examples:
    # Convert a single GLB file.
    python -m urdf2mjcf.glb_to_obj_converter model.glb -o converted_models

    # By default, GLB geometry is merged into one mesh to avoid many small *_0.obj files downstream.
    # Use --no-merge to preserve geometry grouping.

    # Convert all GLB files in a directory.
    python -m urdf2mjcf.glb_to_obj_converter ./models/ -o ./output/
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from urdf2mjcf.logging_utils import URDF2MJCFLogger


def check_dependencies():
    """Check and import required dependencies."""
    missing_deps = []

    try:
        import trimesh  # type: ignore
    except ImportError:
        missing_deps.append("trimesh[easy]")
        trimesh = None

    try:
        import numpy as np  # type: ignore
    except ImportError:
        missing_deps.append("numpy")
        np = None

    try:
        from PIL import Image  # type: ignore
    except ImportError:
        missing_deps.append("pillow")
        Image = None

    if missing_deps:
        raise RuntimeError(
            "Missing required dependencies: "
            + ", ".join(missing_deps)
            + ". Please run: pip install trimesh[easy] numpy pillow"
        )

    return trimesh, np, Image


class GLB2OBJConverter:
    """GLB to OBJ converter."""

    def __init__(self, output_dir: str = "output", merge_meshes: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logger = URDF2MJCFLogger.get_logger("GLB2OBJConverter")
        self.trimesh, self.np, self.Image = check_dependencies()
        self.merge_meshes = merge_meshes

    def _merge_scene_to_single_mesh(self, scene: Any) -> Any:
        """Merge a trimesh.Scene into one Trimesh.

        obj2mjcf can split compound OBJ files that contain multiple groups or objects.
        Merging the GLB scene first keeps downstream output compact by default.
        """

        if isinstance(scene, self.trimesh.Trimesh):
            return scene

        if not isinstance(scene, self.trimesh.Scene):
            raise TypeError(
                f"Expected trimesh.Scene or trimesh.Trimesh, got {type(scene)}"
            )

        # dump(concatenate=True) applies node transforms and concatenates geometry.
        try:
            dumped = scene.dump(concatenate=True)
        except Exception:
            dumped = None

        # Depending on trimesh version and scene layout, dump may return Trimesh or list[Trimesh].
        if dumped is None:
            meshes = [
                geom
                for geom in scene.geometry.values()
                if isinstance(geom, self.trimesh.Trimesh)
            ]
            if not meshes:
                raise ValueError("Scene contains no mesh geometry")
            return self.trimesh.util.concatenate(meshes)

        if isinstance(dumped, list):
            dumped_meshes = [m for m in dumped if isinstance(m, self.trimesh.Trimesh)]
            if not dumped_meshes:
                raise ValueError("Scene dump contains no mesh geometry")
            return self.trimesh.util.concatenate(dumped_meshes)

        return dumped

    def find_glb_files(self, input_path: str) -> List[Path]:
        """Find GLB files under an input path."""
        path = Path(input_path)
        glb_files: List[Path] = []

        if path.is_file() and path.suffix.lower() == ".glb":
            glb_files.append(path)
        elif path.is_dir():
            glb_files.extend(path.rglob("*.glb"))
            glb_files.extend(path.rglob("*.GLB"))

        return sorted(set(glb_files))

    def extract_textures_from_scene(
        self, scene: Any, output_folder: Path
    ) -> Dict[str, str]:
        """Extract textures from a trimesh scene."""
        texture_mapping: Dict[str, str] = {}
        texture_counter = 0

        self.logger.info(f"Extracting textures to: {output_folder}")

        for name, mesh in scene.geometry.items():
            self.logger.debug(f"Processing geometry: {name}")

            if hasattr(mesh, "visual") and mesh.visual:
                try:
                    visual = mesh.visual
                    material = getattr(visual, "material", None)

                    self.logger.debug(f"Material attributes: {dir(material)}")

                    if material is not None:
                        found_texture = False
                        for attr_name, alias in [
                            ("image", "diffuse"),
                            ("baseColorTexture", "albedo"),
                            ("diffuse", "diffuse"),
                            ("albedo", "albedo"),
                            ("baseColor", "baseColor"),
                            ("map_Kd", "map_Kd"),
                        ]:
                            tex_value = getattr(material, attr_name, None)
                            if tex_value is not None and not isinstance(
                                tex_value, (list, tuple, int, float, str)
                            ):
                                if self._save_texture_image(
                                    tex_value,
                                    alias,
                                    output_folder,
                                    texture_mapping,
                                    str(name),
                                ):
                                    self.logger.info(
                                        f"Extracted texture from material attribute: {attr_name}"
                                    )
                                    texture_counter += 1
                                    found_texture = True

                        # If no texture exists but the material has a solid color, generate a small color texture.
                        if not found_texture:
                            # Prefer baseColorFactor, then baseColor, then diffuse.
                            color = getattr(material, "baseColorFactor", None)
                            if color is None:
                                color = getattr(material, "baseColor", None)
                            if color is None:
                                color = getattr(material, "diffuse", None)
                            self.logger.debug(
                                f"Trying to generate solid-color texture from: {color}"
                            )
                            if (
                                color is not None
                                and hasattr(color, "__len__")
                                and len(color) >= 3
                            ):
                                # Support numpy arrays and list/tuple values.
                                if hasattr(color, "tolist"):
                                    color = color.tolist()
                                r, g, b = color[:3]
                                a = color[3] if len(color) > 3 else 255
                                # Normalize to 0-255.
                                if max(r, g, b, a) <= 1.0:
                                    r, g, b, a = (
                                        int(r * 255),
                                        int(g * 255),
                                        int(b * 255),
                                        int(a * 255),
                                    )
                                else:
                                    r, g, b, a = int(r), int(g), int(b), int(a)
                                # Save as RGBA only when alpha is not opaque.
                                if a < 255:
                                    img = self.Image.new("RGBA", (4, 4), (r, g, b, a))
                                else:
                                    img = self.Image.new("RGB", (4, 4), (r, g, b))
                                texture_filename = f"{name}_auto_color.png"
                                texture_path = output_folder / texture_filename
                                img.save(texture_path)
                                texture_mapping[str(name)] = texture_filename
                                self.logger.info(
                                    f"Generated solid-color texture: {texture_filename} color={(r, g, b, a)} path={texture_path}"
                                )
                                texture_counter += 1

                    if hasattr(visual, "uv") and material is not None:
                        tex_image = getattr(material, "image", None)
                        if tex_image is not None:
                            if self._save_texture_image(
                                tex_image,
                                "texture",
                                output_folder,
                                texture_mapping,
                                str(name),
                            ):
                                self.logger.info(
                                    "Extracted texture from UV material image"
                                )
                                texture_counter += 1

                except Exception as exc:  # pragma: no cover - best effort extraction
                    self.logger.warning(f"Failed to extract texture for {name}: {exc}")
            else:
                self.logger.debug(f"Geometry {name} has no visual information")

        try:
            if hasattr(scene, "metadata") and scene.metadata:
                self._extract_from_metadata(
                    scene.metadata, output_folder, texture_mapping
                )
        except Exception as exc:  # pragma: no cover - best effort extraction
            self.logger.warning(f"Failed to extract textures from metadata: {exc}")

        self.logger.info(f"Extracted {texture_counter} textures")
        return texture_mapping

    def _save_texture_image(
        self,
        image_data: Any,
        texture_name: str,
        output_folder: Path,
        texture_mapping: Dict[str, str],
        mesh_name: str,
    ) -> bool:
        """Save a texture image."""
        try:
            texture_filename = f"{texture_name}.png"
            texture_path = output_folder / texture_filename

            if isinstance(image_data, self.Image.Image):
                image_data.save(texture_path)
            elif isinstance(image_data, self.np.ndarray):
                img = self.Image.fromarray(image_data)
                img.save(texture_path)
            elif hasattr(image_data, "save"):
                image_data.save(texture_path)
            else:
                self.logger.warning(f"Unsupported image type: {type(image_data)}")
                return False

            texture_mapping[mesh_name] = texture_filename
            self.logger.info(f"Saved texture: {texture_filename}")
            return True

        except Exception as exc:
            self.logger.warning(f"Failed to save texture: {exc}")
            return False

    def _extract_from_metadata(
        self,
        metadata: Any,
        output_folder: Path,
        texture_mapping: Dict[str, str],
    ) -> None:
        """Extract texture metadata recursively."""
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if "image" in key.lower() or "texture" in key.lower():
                    self.logger.debug(f"Found texture metadata: {key}")
                    if isinstance(value, dict):
                        self._extract_from_metadata(
                            value, output_folder, texture_mapping
                        )

    def extract_textures_from_glb(
        self, glb_path: Path, output_folder: Path
    ) -> Tuple[Dict[str, str], Optional[Dict[str, Any]]]:
        """Extract embedded texture bytes directly from a GLB file."""
        texture_mapping: Dict[str, str] = {}
        gltf_data: Optional[Dict[str, Any]] = None

        try:
            with open(glb_path, "rb") as file_obj:
                magic = file_obj.read(4)
                if magic != b"glTF":
                    self.logger.warning("Invalid GLB file")
                    return texture_mapping, gltf_data

                version = struct.unpack("<I", file_obj.read(4))[0]
                total_length = struct.unpack("<I", file_obj.read(4))[0]
                self.logger.debug(f"GLB version: {version}, length: {total_length}")

                json_chunk_length = struct.unpack("<I", file_obj.read(4))[0]
                json_chunk_type = file_obj.read(4)
                if json_chunk_type != b"JSON":
                    self.logger.warning("JSON chunk not found")
                    return texture_mapping, gltf_data

                json_data = file_obj.read(json_chunk_length).decode("utf-8")
                gltf_data = json.loads(json_data)

                bin_data = None
                if file_obj.tell() < total_length:
                    bin_chunk_length = struct.unpack("<I", file_obj.read(4))[0]
                    bin_chunk_type = file_obj.read(4)
                    if bin_chunk_type == b"BIN\x00":
                        bin_data = file_obj.read(bin_chunk_length)

                for i, image_info in enumerate(gltf_data.get("images", [])):
                    try:
                        image_name = image_info.get("name", f"texture_{i:03d}")
                        texture_name = self._get_texture_name_from_usage(
                            gltf_data, i, image_name
                        )

                        if "bufferView" in image_info:
                            buffer_view_idx = image_info["bufferView"]
                            buffer_view = gltf_data["bufferViews"][buffer_view_idx]
                            start = buffer_view.get("byteOffset", 0)
                            byte_length = buffer_view["byteLength"]

                            if bin_data and start + byte_length <= len(bin_data):
                                image_data = bin_data[start : start + byte_length]
                                mime_type = image_info.get("mimeType", "image/png")
                                ext = "png" if "png" in mime_type else "jpg"
                                texture_filename = f"{texture_name}.{ext}"
                                texture_path = output_folder / texture_filename
                                with open(texture_path, "wb") as img_file:
                                    img_file.write(image_data)
                                texture_mapping[f"texture_{i}"] = texture_filename
                                self.logger.info(
                                    f"Extracted texture: {texture_filename} ({len(image_data)} bytes)"
                                )

                        elif "uri" in image_info:
                            uri = image_info["uri"]
                            if uri.startswith("data:"):
                                header, data = uri.split(",", 1)
                                image_data = base64.b64decode(data)
                                mime_type = header.split(";")[0].split(":")[1]
                                ext = "png" if "png" in mime_type else "jpg"
                                texture_filename = f"{texture_name}.{ext}"
                                texture_path = output_folder / texture_filename
                                with open(texture_path, "wb") as img_file:
                                    img_file.write(image_data)
                                texture_mapping[f"texture_{i}"] = texture_filename
                                self.logger.info(
                                    f"Extracted texture: {texture_filename} (data URI)"
                                )

                    except (
                        Exception
                    ) as exc:  # pragma: no cover - best effort extraction
                        self.logger.warning(f"Failed to extract texture {i}: {exc}")

            self.logger.info(f"Extracted {len(texture_mapping)} textures from GLB")

        except Exception as exc:
            self.logger.warning(f"Failed to extract textures directly from GLB: {exc}")

        return texture_mapping, gltf_data

    def _get_texture_name_from_usage(
        self, gltf_data: Dict[str, Any], image_index: int, default_name: str
    ) -> str:
        """Derive a texture name from its glTF usage."""
        textures = gltf_data.get("textures", [])

        if "materials" in gltf_data:
            for material in gltf_data["materials"]:
                pbr = material.get("pbrMetallicRoughness", {})
                if "baseColorTexture" in pbr:
                    texture_info = pbr["baseColorTexture"]
                    if texture_info["index"] < len(textures):
                        texture = textures[texture_info["index"]]
                        if texture.get("source") == image_index:
                            return "albedo"

                if "metallicRoughnessTexture" in pbr:
                    texture_info = pbr["metallicRoughnessTexture"]
                    if texture_info["index"] < len(textures):
                        texture = textures[texture_info["index"]]
                        if texture.get("source") == image_index:
                            return "metallic_roughness"

                for key, alias in [
                    ("normalTexture", "normal"),
                    ("occlusionTexture", "occlusion"),
                    ("emissiveTexture", "emissive"),
                ]:
                    texture_info = material.get(key)
                    if texture_info and texture_info["index"] < len(textures):
                        texture = textures[texture_info["index"]]
                        if texture.get("source") == image_index:
                            return alias

        return default_name

    def _build_material_texture_mapping(
        self, gltf_data: Dict[str, Any], texture_mapping: Dict[str, str]
    ) -> Dict[str, Dict[str, str]]:
        """Build a material-to-texture-file mapping."""
        textures = gltf_data.get("textures", [])
        materials = gltf_data.get("materials", [])
        material_texture_map: Dict[str, Dict[str, str]] = {}

        def resolve_texture(texture_index: Optional[int]) -> Optional[str]:
            if texture_index is None or texture_index >= len(textures):
                return None
            source = textures[texture_index].get("source")
            if source is None:
                return None
            return texture_mapping.get(f"texture_{source}")

        for idx, material in enumerate(materials):
            material_name = material.get("name", f"material_{idx}")
            material_texture_map[material_name] = {}

            pbr = material.get("pbrMetallicRoughness", {})
            diffuse = resolve_texture(pbr.get("baseColorTexture", {}).get("index"))
            if diffuse:
                material_texture_map[material_name]["diffuse"] = diffuse

            metallic_roughness = resolve_texture(
                pbr.get("metallicRoughnessTexture", {}).get("index")
            )
            if metallic_roughness:
                material_texture_map[material_name][
                    "metallic_roughness"
                ] = metallic_roughness

            for key, alias in [
                ("normalTexture", "normal"),
                ("occlusionTexture", "occlusion"),
                ("emissiveTexture", "emissive"),
            ]:
                filename = resolve_texture(material.get(key, {}).get("index"))
                if filename:
                    material_texture_map[material_name][alias] = filename

        return material_texture_map

    def create_mtl_file(
        self,
        scene: Any,
        output_folder: Path,
        texture_mapping: Dict[str, str],
        gltf_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Create an MTL material file and preserve base color when textures exist."""
        mtl_path = output_folder / f"{output_folder.name}.mtl"

        material_texture_map: Dict[str, Dict[str, str]] = {}
        gltf_materials = []
        if gltf_data:
            material_texture_map = self._build_material_texture_mapping(
                gltf_data, texture_mapping
            )
            gltf_materials = gltf_data.get("materials", [])

        with open(mtl_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("# Material file generated from GLB\n\n")

            for name, mesh in scene.geometry.items():
                # Prefer the glTF material name, then fall back to a generated name.
                visual = getattr(mesh, "visual", None)
                material = getattr(visual, "material", None)
                mesh_material_name = getattr(material, "name", None)
                material_name = mesh_material_name or f"material_{name}"
                file_obj.write(f"newmtl {material_name}\n")

                # Default material parameters.
                ka = [1.0, 1.0, 1.0]
                ks = [0.5, 0.5, 0.5]
                ns = 207.36
                illum = 2

                kd_written = False
                texture_written = set()
                gltf_idx = None
                if mesh_material_name and gltf_materials:
                    for idx, mat in enumerate(gltf_materials):
                        if mat.get("name") == mesh_material_name:
                            gltf_idx = idx
                            break
                if gltf_idx is None and gltf_materials:
                    gltf_idx = 0
                base_color = None
                # Read material parameters from GLB/glTF when available.
                if (
                    gltf_materials
                    and gltf_idx is not None
                    and gltf_idx < len(gltf_materials)
                ):
                    mat = gltf_materials[gltf_idx]
                    pbr = mat.get("pbrMetallicRoughness", {})
                    base_color = pbr.get("baseColorFactor")
                    # Read Ks/Ns/Ka values when present.
                    if "specularFactor" in pbr:
                        ks = [float(x) for x in pbr["specularFactor"][:3]]
                    if "roughnessFactor" in pbr:
                        ns = float(pbr["roughnessFactor"]) * 1000
                    if "ambientFactor" in mat:
                        ka = [float(x) for x in mat["ambientFactor"][:3]]

                file_obj.write(f"Ka {ka[0]:.6f} {ka[1]:.6f} {ka[2]:.6f}\n")

                # Write all supported texture slots first.
                if mesh_material_name and mesh_material_name in material_texture_map:
                    textures = material_texture_map[mesh_material_name]
                    if "diffuse" in textures:
                        file_obj.write(f"map_Kd {textures['diffuse']}\n")
                        texture_written.add("map_Kd")
                    if "metallic_roughness" in textures:
                        file_obj.write(f"map_Pm {textures['metallic_roughness']}\n")
                        texture_written.add("map_Pm")
                    if "normal" in textures:
                        file_obj.write(f"map_Bump {textures['normal']}\n")
                        texture_written.add("map_Bump")
                    if "emissive" in textures:
                        file_obj.write(f"map_Ke {textures['emissive']}\n")
                        texture_written.add("map_Ke")
                    if "occlusion" in textures:
                        file_obj.write(f"map_Occlusion {textures['occlusion']}\n")
                        texture_written.add("map_Occlusion")

                # If no real texture is mapped, use the first extracted texture as a fallback.
                if "map_Kd" not in texture_written:
                    if not texture_written and texture_mapping:
                        first_texture = next(iter(texture_mapping.values()))
                        file_obj.write(f"map_Kd {first_texture}\n")
                        texture_written.add("map_Kd")

                # Write Kd only when the material has a texture; otherwise keep defaults.
                if texture_written:
                    if base_color and len(base_color) >= 3:
                        r, g, b = base_color[:3]
                        if r > 1.0 or g > 1.0 or b > 1.0:
                            r, g, b = r / 255.0, g / 255.0, b / 255.0
                        file_obj.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
                        kd_written = True
                    elif material is not None:
                        try:
                            color = getattr(material, "diffuse", None)
                            if color is not None and len(color) >= 3:
                                r, g, b = color[:3]
                                if r > 1.0 or g > 1.0 or b > 1.0:
                                    r, g, b = r / 255.0, g / 255.0, b / 255.0
                                file_obj.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
                                kd_written = True
                        except Exception as exc:
                            self.logger.warning(
                                f"Failed to process material color for {name}: {exc}"
                            )
                    if not kd_written:
                        file_obj.write("Kd 0.8 0.8 0.8\n")

                file_obj.write(f"Ks {ks[0]:.6f} {ks[1]:.6f} {ks[2]:.6f}\n")
                file_obj.write(f"Ns {ns:.6f}\n")
                file_obj.write(f"illum {illum}\n")
                file_obj.write("\n")

            if not scene.geometry and texture_mapping:
                file_obj.write("newmtl default\n")
                file_obj.write("Ka 0.2 0.2 0.2\n")
                file_obj.write("Kd 0.8 0.8 0.8\n")
                file_obj.write("Ks 0.1 0.1 0.1\n")
                file_obj.write("Ns 10.0\n")
                file_obj.write("illum 2\n")
                first_texture = next(iter(texture_mapping.values()))
                file_obj.write(f"map_Kd {first_texture}\n\n")

        return mtl_path

    def export_obj_with_materials(
        self, scene: Any, output_folder: Path, mtl_filename: str
    ) -> Path:
        """Export an OBJ file with material references."""
        obj_path = output_folder / f"{output_folder.name}.obj"

        with open(obj_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("# OBJ file generated from GLB\n")
            file_obj.write(f"mtllib {mtl_filename}\n\n")

            vertex_offset = 1
            uv_offset = 1
            normal_offset = 1

            for name, mesh in scene.geometry.items():
                try:
                    file_obj.write(f"g {name}\n")
                    file_obj.write(f"usemtl material_{name}\n")

                    vertices = mesh.vertices
                    for vertex in vertices:
                        file_obj.write(
                            f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n"
                        )

                    has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
                    if has_uv:
                        uv_coords = mesh.visual.uv
                        for uv in uv_coords:
                            file_obj.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
                    else:
                        uv_coords = []

                    has_normals = (
                        hasattr(mesh, "vertex_normals")
                        and mesh.vertex_normals is not None
                    )
                    if has_normals:
                        normals = mesh.vertex_normals
                        for normal in normals:
                            file_obj.write(
                                f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}\n"
                            )
                    else:
                        normals = []

                    for face in mesh.faces:
                        face_line = "f"
                        for vertex_idx in face:
                            obj_vertex_idx = vertex_offset + vertex_idx
                            if has_uv and has_normals:
                                obj_uv_idx = uv_offset + vertex_idx
                                obj_normal_idx = normal_offset + vertex_idx
                                face_line += (
                                    f" {obj_vertex_idx}/{obj_uv_idx}/{obj_normal_idx}"
                                )
                            elif has_uv:
                                obj_uv_idx = uv_offset + vertex_idx
                                face_line += f" {obj_vertex_idx}/{obj_uv_idx}"
                            elif has_normals:
                                obj_normal_idx = normal_offset + vertex_idx
                                face_line += f" {obj_vertex_idx}//{obj_normal_idx}"
                            else:
                                face_line += f" {obj_vertex_idx}"
                        file_obj.write(face_line + "\n")

                    vertex_offset += len(vertices)
                    if has_uv:
                        uv_offset += len(uv_coords)
                    if has_normals:
                        normal_offset += len(normals)

                    file_obj.write("\n")

                except Exception as exc:  # pragma: no cover - best effort export
                    self.logger.warning(f"Failed to process geometry {name}: {exc}")

        return obj_path

    def convert_glb_to_obj(
        self, glb_path: Path, output_folder: Optional[Path] = None
    ) -> bool:
        """Convert one GLB file to OBJ."""
        if output_folder is None:
            output_folder = self.output_dir / glb_path.stem

        output_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Converting: {glb_path.name}")

        try:
            texture_mapping, gltf_data = self.extract_textures_from_glb(
                glb_path, output_folder
            )

            scene = self.trimesh.load(str(glb_path))

            # Merge multi-geometry scenes to avoid many tiny OBJ fragments downstream.
            if self.merge_meshes and isinstance(scene, self.trimesh.Scene):
                merged_mesh = self._merge_scene_to_single_mesh(scene)
                merged_scene = self.trimesh.Scene()
                merged_scene.add_geometry(merged_mesh, node_name=glb_path.stem)
                scene = merged_scene

            if isinstance(scene, self.trimesh.Trimesh):
                mesh = scene
                scene = self.trimesh.Scene()
                scene.add_geometry(mesh, node_name=glb_path.stem)

            scene_textures = self.extract_textures_from_scene(scene, output_folder)
            texture_mapping.update(scene_textures)

            mtl_path = self.create_mtl_file(
                scene, output_folder, texture_mapping, gltf_data
            )
            obj_path = self.export_obj_with_materials(
                scene, output_folder, mtl_path.name
            )

            self.logger.info(f"Conversion complete: {output_folder}")
            self.logger.info(f"OBJ file: {obj_path}")
            self.logger.info(f"MTL file: {mtl_path}")
            self.logger.info(f"Extracted {len(texture_mapping)} textures")
            return True

        except Exception as exc:
            self.logger.error(f"Conversion failed: {exc}")
            return False

    def convert_directory(
        self, input_dir: Path, output_dir: Path, recursive: bool = True
    ) -> Dict[str, bool]:
        """Convert all GLB files in a directory while preserving relative layout."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if recursive:
            glb_files = sorted(input_dir.rglob("*.glb")) + sorted(
                input_dir.rglob("*.GLB")
            )
        else:
            glb_files = sorted(input_dir.glob("*.glb")) + sorted(
                input_dir.glob("*.GLB")
            )

        results: Dict[str, bool] = {}
        if not glb_files:
            self.logger.warning(f"No GLB files found in {input_dir}")
            return results

        self.logger.info(f"Found {len(glb_files)} GLB files")

        for glb_file in glb_files:
            relative_path = glb_file.relative_to(input_dir)
            target_dir = output_dir / relative_path.parent / glb_file.stem
            results[str(glb_file)] = self.convert_glb_to_obj(glb_file, target_dir)

        success_count = sum(1 for success in results.values() if success)
        self.logger.info(
            f"Conversion complete: {success_count}/{len(glb_files)} files succeeded"
        )
        return results

    def convert_batch(self, input_path: str) -> int:
        """Batch convert GLB files from a file or directory path."""
        input_path_obj = Path(input_path)

        if input_path_obj.is_file():
            return 1 if self.convert_glb_to_obj(input_path_obj) else 0

        if input_path_obj.is_dir():
            results = self.convert_directory(
                input_path_obj, self.output_dir, recursive=True
            )
            return sum(1 for success in results.values() if success)

        glb_files = self.find_glb_files(input_path)

        if not glb_files:
            self.logger.warning(f"No GLB files found in {input_path}")
            return 0

        self.logger.info(f"Found {len(glb_files)} GLB files")

        success_count = 0
        for glb_file in glb_files:
            if self.convert_glb_to_obj(glb_file):
                success_count += 1

        self.logger.info(
            f"Conversion complete: {success_count}/{len(glb_files)} files succeeded"
        )
        return success_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert GLB files to OBJ and extract textures."
    )
    parser.add_argument(
        "input", help="Input GLB file or directory containing GLB files."
    )
    parser.add_argument(
        "-o", "--output", default="output", help="Output directory (default: output)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge GLB geometries. This may generate multiple small OBJ files downstream.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input path '{args.input}' does not exist")
        return 1

    converter = GLB2OBJConverter(args.output, merge_meshes=not args.no_merge)
    if args.verbose:
        converter.logger.setLevel("DEBUG")

    converter.convert_batch(args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
