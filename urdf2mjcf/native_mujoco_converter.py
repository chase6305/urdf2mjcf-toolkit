"""Convert URDF using MuJoCo's native importer, without an external CLI."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from urdf2mjcf.mesh_identity import (
    prepare_native_mesh_aliases,
    remove_unused_mesh_assets,
)
from urdf2mjcf.physics import restore_urdf_physics
from urdf2mjcf.urdf_to_mjcf_converter import URDFConverterBase


class NativeMujocoConverter(URDFConverterBase):
    """Preserve URDF bodies and visuals and serialize them as MJCF."""

    def __init__(self, urdf_path: Path, output_dir: Path, assets_dir: Path):
        super().__init__(urdf_path, output_dir, assets_dir)
        self.output_xml: Path | None = None

    def convert(self) -> bool:
        self.output_xml = None
        self.last_error = None
        self.last_error_codes = ()
        try:
            import mujoco

            tree = self.prepare_urdf_tree()
            extension = tree.getroot().find("mujoco")
            if extension is None:
                extension = ET.SubElement(tree.getroot(), "mujoco")
            compiler = extension.find("compiler")
            if compiler is None:
                compiler = ET.SubElement(extension, "compiler")
            # URDF defaults would strip paths, omit visuals, and merge the fixed
            # base into the world, preventing later floating-base conversion.
            compiler.set("strippath", "false")
            compiler.set("discardvisual", "false")
            compiler.set("fusestatic", "false")
            compiler.set("usethread", "false")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="native-", dir=self.output_dir
            ) as staging:
                source = Path(staging) / self.urdf_path.name
                aliased, filenames = prepare_native_mesh_aliases(
                    tree, Path(staging) / "mesh-aliases", self.assets_dir
                )
                aliased.write(source, encoding="utf-8", xml_declaration=True)
                spec = mujoco.MjSpec.from_file(str(source))
                for mesh in spec.meshes:
                    mesh.file = filenames[mesh.name]
                result = ET.fromstring(spec.to_xml())
                remove_unused_mesh_assets(result)
                restore_urdf_physics(result, tree.getroot())
                pending = Path(staging) / f"{self.urdf_path.stem}.xml"
                ET.ElementTree(result).write(
                    pending, encoding="utf-8", xml_declaration=True
                )
                candidate = self.output_dir / pending.name
                pending.replace(candidate)
            self.output_xml = candidate
            self.logger.info("Native MuJoCo conversion successful: %s", candidate)
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.logger.error("Native MuJoCo conversion failed: %s", exc)
            return False

    def get_output_xml(self) -> Path | None:
        return self.output_xml
