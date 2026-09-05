#!/usr/bin/env python3
"""
Modular URDF to MuJoCo Conversion Script
Supports both urdf2mjcf and basic conversion methods
"""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path

from urdf2mjcf.asset_paths import MeshPathResolver
from urdf2mjcf.logging_utils import URDF2MJCFLogger
from urdf2mjcf.physics import normalize_urdf_physics, restore_urdf_physics
from urdf2mjcf.urdf_validation import validate_urdf_physics


class URDFConverterBase(ABC):
    """Base class for URDF converters"""

    def __init__(self, urdf_path: Path, output_dir: Path, assets_dir: Path):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.assets_dir = Path(assets_dir).expanduser().resolve()
        self.mesh_search_dir = self.urdf_path.parent
        self.last_error: str | None = None
        self.prepared_tree: ET.ElementTree | None = None
        self.mesh_source_map: dict[Path, Path] | None = None
        self.logger = URDF2MJCFLogger.get_logger(type(self).__name__)

    def prepare_urdf_tree(self) -> ET.ElementTree:
        """Resolve mesh references without modifying the original URDF."""
        resolver = MeshPathResolver(
            self.mesh_search_dir, self.assets_dir, source_map=self.mesh_source_map
        )
        tree = ET.parse(self.urdf_path)
        if tree.getroot().tag != "robot":
            raise ValueError("Expected a <robot> root element in the URDF")
        validate_urdf_physics(tree.getroot())
        normalize_urdf_physics(tree.getroot())
        changed = 0
        for elem in tree.getroot().iter("mesh"):
            for attr in ("filename", "file"):
                if attr in elem.attrib:
                    old = elem.attrib[attr]
                    new = resolver.resolve(old).as_posix()
                    elem.set(attr, new)
                    changed += old != new
        self.logger.debug("Resolved %s mesh references", changed)
        self.prepared_tree = tree
        return tree

    @abstractmethod
    def convert(self) -> bool:
        """Execute conversion"""
        pass

    @abstractmethod
    def get_output_xml(self) -> Path | None:
        """Get output XML file path"""
        pass


class URDF2MJCFConverter(URDFConverterBase):
    """Converter using urdf2mjcf tool"""

    def __init__(self, urdf_path: Path, output_dir: Path, assets_dir: Path):
        super().__init__(urdf_path, output_dir, assets_dir)
        self.output_xml = None

    def convert(self) -> bool:
        """Convert URDF to MJCF using urdf2mjcf"""
        successful = False
        self.last_error = None
        try:
            self.logger.info("Converting URDF to MJCF using urdf2mjcf...")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.assets_dir.mkdir(parents=True, exist_ok=True)

            # Build output file path
            candidate = self.output_dir / f"{self.urdf_path.stem}.xml"
            # Avoid collision if a directory named '<stem>.xml' already exists
            if candidate.exists() and candidate.is_dir():
                candidate = self.output_dir / f"{self.urdf_path.stem}_mjcf.xml"
            self.output_xml = candidate
            previous_outputs = {
                path: (path.stat().st_mtime_ns, path.stat().st_size)
                for path in self.output_dir.glob("*.xml")
                if path.is_file()
            }

            tree = self.prepare_urdf_tree()

            # Unique staging prevents overwriting an input inside assets_dir and
            # removes temporary URDFs on both success and failure.
            with tempfile.TemporaryDirectory(
                prefix="urdf2mjcf-", dir=self.assets_dir
            ) as staging:
                staged_urdf = Path(staging) / self.urdf_path.name
                tree.write(staged_urdf, encoding="utf-8", xml_declaration=True)
                cmd = ["urdf2mjcf", str(staged_urdf), "--output", str(self.output_xml)]
                self.logger.debug("Executing command: %s", cmd)
                result = subprocess.run(
                    cmd,
                    cwd=str(self.assets_dir),
                    capture_output=True,
                    text=True,
                )

            if result.returncode == 0:
                output_changed = False
                if self.output_xml and self.output_xml.exists():
                    signature = (
                        self.output_xml.stat().st_mtime_ns,
                        self.output_xml.stat().st_size,
                    )
                    output_changed = previous_outputs.get(self.output_xml) != signature

                if self.output_xml and output_changed:
                    self._restore_physics(tree)
                    self.logger.info(
                        f"✓ urdf2mjcf conversion successful: {self.output_xml}"
                    )
                    successful = True
                    return True

                # Some urdf2mjcf versions may ignore --output and write to a default XML path.
                xml_candidates = [
                    p
                    for p in self.output_dir.glob("*.xml")
                    if p.is_file()
                    and p.name != self.urdf_path.name
                    and previous_outputs.get(p)
                    != (p.stat().st_mtime_ns, p.stat().st_size)
                ]
                if xml_candidates:
                    xml_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    self.output_xml = xml_candidates[0]
                    self._restore_physics(tree)
                    self.logger.warning(
                        "urdf2mjcf returned success but expected output was missing; "
                        f"using discovered XML instead: {self.output_xml}"
                    )
                    successful = True
                    return True

                self.last_error = "urdf2mjcf returned success but no XML file was generated in output_dir"
                self.logger.error(self.last_error)
                if result.stdout:
                    self.logger.error(f"urdf2mjcf stdout: {result.stdout}")
                if result.stderr:
                    self.logger.error(f"urdf2mjcf stderr: {result.stderr}")
                return False
            else:
                self.last_error = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or f"urdf2mjcf exited with status {result.returncode}"
                )
                self.logger.error("urdf2mjcf conversion failed: %s", self.last_error)
                return False
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Error during conversion: {e}")
            return False
        finally:
            if not successful:
                self.output_xml = None

    def _restore_physics(self, source: ET.ElementTree) -> None:
        converted = ET.parse(self.output_xml)
        restore_urdf_physics(converted.getroot(), source.getroot())
        converted.write(self.output_xml, encoding="utf-8", xml_declaration=True)

    def get_output_xml(self) -> Path | None:
        """Get output XML file path"""
        return self.output_xml
