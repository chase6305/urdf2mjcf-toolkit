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

    # By default, compatible GLB geometry is merged while distinct materials retain their groups.
    # Use --no-merge to preserve all geometry grouping.

    # Convert all GLB files in a directory.
    python -m urdf2mjcf.glb_to_obj_converter ./models/ -o ./output/
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

from urdf2mjcf.logging_utils import URDF2MJCFLogger
from urdf2mjcf.mesh_to_obj_converter import MeshToOBJConverter


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


def _read_glb_layout(
    file_obj: BinaryIO,
) -> tuple[dict[str, Any], tuple[int, int] | None]:
    """Validate the GLB container and locate its JSON and optional BIN chunk.

    Check every declared range against the actual file before reading it. The
    BIN payload is left on disk; texture extraction only reads image ranges.
    """
    file_obj.seek(0, os.SEEK_END)
    file_size = file_obj.tell()
    file_obj.seek(0)
    if file_size < 20:
        raise ValueError("Truncated GLB header or JSON chunk")
    magic, version, total_length = struct.unpack("<4sII", file_obj.read(12))
    if magic != b"glTF" or version != 2:
        raise ValueError("Expected a GLB version 2 file")
    if total_length != file_size:
        raise ValueError("GLB declared length does not match its file size")

    metadata = None
    binary_chunk = None
    chunk_index = 0
    while file_obj.tell() < file_size:
        if file_size - file_obj.tell() < 8:
            raise ValueError("Truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack("<I4s", file_obj.read(8))
        start = file_obj.tell()
        if chunk_length % 4 or chunk_length > file_size - start:
            raise ValueError("Invalid GLB chunk length")
        if chunk_index == 0 and chunk_type != b"JSON":
            raise ValueError("The first GLB chunk must contain JSON")
        if chunk_type == b"JSON":
            if chunk_index != 0:
                raise ValueError("GLB must contain exactly one JSON chunk")
            payload = file_obj.read(chunk_length)
            if len(payload) != chunk_length:
                raise ValueError("Truncated GLB JSON payload")
            metadata = json.loads(payload.decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("GLB JSON must contain an object")
        elif chunk_type == b"BIN\x00":
            if chunk_index != 1 or binary_chunk is not None:
                raise ValueError("GLB BIN chunk must occur once, as the second chunk")
            binary_chunk = (start, chunk_length)
        # Unknown chunk types are allowed; skip their payload without reading it.
        file_obj.seek(start + chunk_length)
        chunk_index += 1
    if metadata is None:
        raise ValueError("GLB JSON chunk is missing")
    return metadata, binary_chunk


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

        meshes = list(scene.geometry.values())
        merged = MeshToOBJConverter._as_single_mesh(scene)
        # All callers here have already baked transforms and checked material
        # compatibility. Preserve the original UVs and material instead of
        # relying on a texture atlas generated during trimesh concatenation.
        if meshes and all(
            getattr(mesh.visual, "material", None) is not None for mesh in meshes
        ):
            uv = None
            if all(getattr(mesh.visual, "uv", None) is not None for mesh in meshes):
                uv = self.np.vstack([mesh.visual.uv for mesh in meshes])
            merged.visual = self.trimesh.visual.TextureVisuals(
                uv=uv, material=meshes[0].visual.material.copy()
            )
        return merged

    @staticmethod
    def _scene_texture_filename(mesh_name: str, role: str) -> str:
        digest = hashlib.sha256(f"{mesh_name}\0{role}".encode()).hexdigest()[:16]
        label = (
            role
            if role
            in {"albedo", "diffuse", "baseColor", "map_Kd", "texture", "auto_color"}
            else "image"
        )
        return f"scene_{digest}_{label}.png"

    @staticmethod
    def _obj_material_name(geometry_name: str) -> str:
        # Display names may contain whitespace or OBJ/MTL line separators.
        digest = hashlib.sha256(str(geometry_name).encode()).hexdigest()[:16]
        return f"material_{digest}"

    def find_glb_files(self, input_path: str) -> list[Path]:
        """Find GLB files under an input path."""
        path = Path(input_path)
        glb_files: list[Path] = []

        if path.is_file() and path.suffix.lower() == ".glb":
            glb_files.append(path)
        elif path.is_dir():
            glb_files.extend(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".glb"
            )

        return sorted(set(glb_files))

    def extract_textures_from_scene(
        self, scene: Any, output_folder: Path
    ) -> dict[str, str]:
        """Extract textures from a trimesh scene."""
        texture_mapping: dict[str, str] = {}
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
                            if (
                                tex_value is not None
                                and not isinstance(
                                    tex_value, (list, tuple, int, float, str)
                                )
                                and not (
                                    isinstance(tex_value, self.np.ndarray)
                                    and tex_value.ndim < 2
                                )
                            ) and (
                                self._save_texture_image(
                                    tex_value,
                                    alias,
                                    output_folder,
                                    texture_mapping,
                                    str(name),
                                )
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
                                texture_filename = self._scene_texture_filename(
                                    str(name), "auto_color"
                                )
                                texture_path = output_folder / texture_filename
                                img.save(texture_path)
                                texture_mapping[str(name)] = texture_filename
                                self.logger.info(
                                    f"Generated solid-color texture: {texture_filename} color={(r, g, b, a)} path={texture_path}"
                                )
                                texture_counter += 1

                    if hasattr(visual, "uv") and material is not None:
                        tex_image = getattr(material, "image", None)
                        if (tex_image is not None) and (
                            self._save_texture_image(
                                tex_image,
                                "texture",
                                output_folder,
                                texture_mapping,
                                str(name),
                            )
                        ):
                            self.logger.info("Extracted texture from UV material image")
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
        texture_mapping: dict[str, str],
        mesh_name: str,
    ) -> bool:
        """Save a texture image."""
        try:
            texture_filename = self._scene_texture_filename(mesh_name, texture_name)
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
        texture_mapping: dict[str, str],
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
    ) -> tuple[dict[str, str], dict[str, Any] | None]:
        """Extract embedded images without loading the complete binary buffer."""
        texture_mapping: dict[str, str] = {}
        gltf_data: dict[str, Any] | None = None

        try:
            with open(glb_path, "rb") as file_obj:
                gltf_data, binary_chunk = _read_glb_layout(file_obj)
                buffers = gltf_data.get("buffers", [])
                embedded_buffer = buffers and "uri" not in buffers[0]
                buffer_length = 0
                if embedded_buffer:
                    buffer_length = buffers[0]["byteLength"]
                    if (
                        type(buffer_length) is not int
                        or buffer_length < 0
                        or binary_chunk is None
                        or buffer_length > binary_chunk[1]
                    ):
                        raise ValueError("GLB embedded buffer exceeds its BIN chunk")

                for i, image_info in enumerate(gltf_data.get("images", [])):
                    if "bufferView" in image_info:
                        index = image_info["bufferView"]
                        views = gltf_data.get("bufferViews", [])
                        if type(index) is not int or not 0 <= index < len(views):
                            raise ValueError(f"Invalid bufferView for image {i}")
                        view = views[index]
                        if view["buffer"] != 0 or not embedded_buffer:
                            # External buffers are loaded by trimesh, not the GLB BIN chunk.
                            continue
                        start = view.get("byteOffset", 0)
                        length = view["byteLength"]
                        if (
                            type(start) is not int
                            or type(length) is not int
                            or start < 0
                            or length <= 0
                            or start + length > buffer_length
                        ):
                            raise ValueError(f"Image {i} exceeds the embedded buffer")
                        assert binary_chunk is not None
                        file_obj.seek(binary_chunk[0] + start)
                        image_data = file_obj.read(length)
                        if len(image_data) != length:
                            raise ValueError(f"Truncated image {i} in GLB")
                        mime_type = image_info.get("mimeType", "image/png")
                    elif image_info.get("uri", "").startswith("data:"):
                        header, data = image_info["uri"].split(",", 1)
                        if not header.endswith(";base64"):
                            raise ValueError(f"Unsupported image data URI {i}")
                        image_data = base64.b64decode(data, validate=True)
                        mime_type = header.split(";", 1)[0].split(":", 1)[1]
                    else:
                        continue
                    ext = "png" if "png" in mime_type else "jpg"
                    texture_filename = f"texture_{i:03d}.{ext}"
                    texture_path = output_folder / texture_filename
                    with open(texture_path, "wb") as img_file:
                        img_file.write(image_data)
                    texture_mapping[f"texture_{i}"] = texture_filename
                    self.logger.info(
                        f"Extracted texture: {texture_filename} ({len(image_data)} bytes)"
                    )

            self.logger.info(f"Extracted {len(texture_mapping)} textures from GLB")

        except Exception as exc:
            self.logger.warning(f"Failed to extract textures directly from GLB: {exc}")
            return {}, None

        return texture_mapping, gltf_data

    def _get_texture_name_from_usage(
        self, gltf_data: dict[str, Any], image_index: int, default_name: str
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
        self, gltf_data: dict[str, Any], texture_mapping: dict[str, str]
    ) -> dict[int, dict[str, str]]:
        """Build texture bindings by glTF material index, never display name."""
        textures = gltf_data.get("textures", [])
        materials = gltf_data.get("materials", [])
        material_texture_map: dict[int, dict[str, str]] = {}

        def resolve_texture(texture_index: int | None) -> str | None:
            if texture_index is None or texture_index >= len(textures):
                return None
            source = textures[texture_index].get("source")
            if source is None:
                return None
            return texture_mapping.get(f"texture_{source}")

        for idx, material in enumerate(materials):
            material_texture_map[idx] = {}

            pbr = material.get("pbrMetallicRoughness", {})
            diffuse = resolve_texture(pbr.get("baseColorTexture", {}).get("index"))
            if diffuse:
                material_texture_map[idx]["diffuse"] = diffuse

            metallic_roughness = resolve_texture(
                pbr.get("metallicRoughnessTexture", {}).get("index")
            )
            if metallic_roughness:
                material_texture_map[idx]["metallic_roughness"] = metallic_roughness

            for key, alias in [
                ("normalTexture", "normal"),
                ("occlusionTexture", "occlusion"),
                ("emissiveTexture", "emissive"),
            ]:
                filename = resolve_texture(material.get(key, {}).get("index"))
                if filename:
                    material_texture_map[idx][alias] = filename

        return material_texture_map

    def create_mtl_file(
        self,
        scene: Any,
        output_folder: Path,
        texture_mapping: dict[str, str],
        gltf_data: dict[str, Any] | None = None,
    ) -> Path:
        """Create an MTL material file and preserve base color when textures exist."""
        mtl_path = output_folder / f"{output_folder.name}.mtl"

        material_texture_map: dict[int, dict[str, str]] = {}
        gltf_materials = []
        if gltf_data:
            material_texture_map = self._build_material_texture_mapping(
                gltf_data, texture_mapping
            )
            gltf_materials = gltf_data.get("materials", [])
        material_indices: dict[str, int | None] = {}
        for idx, material in enumerate(gltf_materials):
            name = material.get("name")
            if name:
                # Ambiguous names must use the geometry's actual loaded
                # material, rather than another material's JSON entry.
                material_indices[name] = None if name in material_indices else idx

        with open(mtl_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("# Material file generated from GLB\n\n")

            for name, mesh in scene.geometry.items():
                visual = getattr(mesh, "visual", None)
                material = getattr(visual, "material", None)
                mesh_material_name = getattr(material, "name", None)
                # Match the per-geometry usemtl emitted by the OBJ writer.
                # glTF material names are optional and need not be unique.
                material_name = self._obj_material_name(name)
                file_obj.write(f"newmtl {material_name}\n")

                # Default material parameters.
                ka = [1.0, 1.0, 1.0]
                ks = [0.5, 0.5, 0.5]
                ns = 207.36
                illum = 2

                kd_written = False
                texture_written = set()
                gltf_idx = material_indices.get(mesh_material_name)
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
                if gltf_idx is not None:
                    textures = material_texture_map[gltf_idx]
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

                # A merged scene may have a new texture atlas unrelated to the
                # original glTF material names. Use its own extracted image.
                if ("map_Kd" not in texture_written) and (
                    not texture_written and texture_mapping
                ):
                    fallback_texture = texture_mapping.get(str(name))
                    if fallback_texture and not fallback_texture.endswith(
                        "_auto_color.png"
                    ):
                        file_obj.write(f"map_Kd {fallback_texture}\n")
                        texture_written.add("map_Kd")

                # Solid colors need Kd even without a texture. Atlas materials
                # carry their own (normally white) factor; an arbitrary original
                # glTF material would tint every part with the first color.
                if base_color is None and material is not None:
                    base_color = getattr(material, "baseColorFactor", None)
                    if base_color is None:
                        base_color = getattr(material, "diffuse", None)
                if base_color is not None and len(base_color) >= 3:
                    r, g, b = base_color[:3]
                    if max(r, g, b) > 1.0:
                        r, g, b = r / 255.0, g / 255.0, b / 255.0
                    file_obj.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
                    kd_written = True
                if not kd_written:
                    factor = (
                        "1 1 1"
                        if texture_written
                        or isinstance(
                            material, self.trimesh.visual.material.PBRMaterial
                        )
                        else "0.8 0.8 0.8"
                    )
                    file_obj.write(f"Kd {factor}\n")

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
                    material_name = self._obj_material_name(name)
                    file_obj.write(f"g {material_name}\n")
                    file_obj.write(f"usemtl {material_name}\n")

                    vertices = mesh.vertices
                    for vertex in vertices:
                        file_obj.write(
                            f"v {vertex[0]:.17g} {vertex[1]:.17g} {vertex[2]:.17g}\n"
                        )

                    has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
                    if has_uv:
                        uv_coords = mesh.visual.uv
                        for uv in uv_coords:
                            file_obj.write(f"vt {uv[0]:.17g} {uv[1]:.17g}\n")
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
                                f"vn {normal[0]:.17g} {normal[1]:.17g} {normal[2]:.17g}\n"
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

                except Exception as exc:
                    raise ValueError(
                        f"Failed to export geometry {name}: {exc}"
                    ) from exc

        return obj_path

    def convert_glb_to_obj(
        self, glb_path: Path, output_folder: Path | None = None
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
            if gltf_data is None:
                raise ValueError("Invalid GLB container or embedded image data")

            scene = self.trimesh.load(str(glb_path))

            if isinstance(scene, self.trimesh.Scene):
                baked = self.trimesh.Scene()
                for node in scene.graph.nodes_geometry:
                    transform, geometry_name = scene.graph[node]
                    mesh = scene.geometry[geometry_name].copy()
                    mesh.apply_transform(transform)
                    baked.add_geometry(mesh, geom_name=node, node_name=node)
                scene = baked

            # Merge multi-geometry scenes to avoid many tiny OBJ fragments downstream.
            compatible = True
            if isinstance(scene, self.trimesh.Scene):
                materials = {
                    hash(getattr(mesh.visual, "material", None))
                    for mesh in scene.geometry.values()
                }
                compatible = len(materials) <= 1
            if (
                self.merge_meshes
                and isinstance(scene, self.trimesh.Scene)
                and compatible
            ):
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
    ) -> dict[str, bool]:
        """Convert all GLB files in a directory while preserving relative layout."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
        glb_files = sorted(
            p for p in iterator if p.is_file() and p.suffix.lower() == ".glb"
        )

        results: dict[str, bool] = {}
        if not glb_files:
            self.logger.debug(f"No GLB files found in {input_dir}")
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
        help="Preserve every GLB geometry group. The default merges compatible materials.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input path '{args.input}' does not exist")
        return 1

    converter = GLB2OBJConverter(args.output, merge_meshes=not args.no_merge)
    if args.verbose:
        converter.logger.setLevel("DEBUG")

    files = converter.find_glb_files(args.input)
    if not files:
        converter.logger.error("No GLB files found in %s", args.input)
        return 1
    success_count = converter.convert_batch(args.input)
    return 0 if success_count == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
