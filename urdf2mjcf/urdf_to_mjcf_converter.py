#!/usr/bin/env python3
"""
Modular URDF to MuJoCo Conversion Script
Supports both urdf2mjcf and basic conversion methods
"""

import os
import sys
import argparse
import logging
import subprocess

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from abc import ABC, abstractmethod
from urdf2mjcf.logging_utils import URDF2MJCFLogger


class URDFConverterBase(ABC):
    """Base class for URDF converters"""

    def __init__(self, urdf_path: Path, output_dir: Path, assets_dir: Path):
        self.urdf_path = urdf_path
        self.output_dir = output_dir
        self.assets_dir = assets_dir

    @abstractmethod
    def convert(self) -> bool:
        """Execute conversion"""
        pass

    @abstractmethod
    def get_output_xml(self) -> Optional[Path]:
        """Get output XML file path"""
        pass


class URDF2MJCFConverter(URDFConverterBase):
    """Converter using urdf2mjcf tool"""

    def __init__(self, urdf_path: Path, output_dir: Path, assets_dir: Path):
        super().__init__(urdf_path, output_dir, assets_dir)
        self.output_xml = None
        self.logger = URDF2MJCFLogger.get_logger("URDF2MJCFConverter")

    def _move_meshes_to_assets(self):
        """Move files from output_dir into assets_dir."""
        if not self.assets_dir.exists():
            self.assets_dir.mkdir(parents=True, exist_ok=True)
        for file in self.output_dir.iterdir():
            if file.is_file():
                target = self.assets_dir / file.name
                try:
                    file.rename(target)
                    self.logger.info(f"Moved file: {file} -> {target}")
                except Exception as e:
                    self.logger.error(f"Failed to move file {file} to {target}: {e}")

    def convert(self) -> bool:
        """Convert URDF to MJCF using urdf2mjcf"""
        try:
            self.logger.info("Converting URDF to MJCF using urdf2mjcf...")

            # Build output file path
            candidate = self.output_dir / f"{self.urdf_path.stem}.xml"
            # Avoid collision if a directory named '<stem>.xml' already exists
            if candidate.exists() and candidate.is_dir():
                candidate = self.output_dir / f"{self.urdf_path.stem}_mjcf.xml"
            self.output_xml = candidate

            # Ensure URDF mesh paths resolve even if urdf2mjcf compiles a temp URDF in another directory.
            # MuJoCo resolves relative mesh paths against the URDF location. urdf2mjcf internally writes
            # the URDF into a temp dir during compilation, so relative paths like `collision/...` can break.
            # Fix: stage a URDF copy under `assets_dir` and rewrite mesh filenames to absolute paths.

            def _resolve_mesh_ref(mesh_ref: str) -> str:
                mesh_ref = (mesh_ref or "").strip().replace("\\", "/")
                if not mesh_ref:
                    return mesh_ref
                # skip URLs / package refs
                if "://" in mesh_ref:
                    return mesh_ref

                # common normalizations
                rel = mesh_ref
                if rel.startswith("./"):
                    rel = rel[2:]
                if rel.startswith("meshes/"):
                    rel = rel[len("meshes/") :]
                for src, dst in (
                    ("Visual/", "visual/"),
                    ("Collision/", "collision/"),
                ):
                    if rel.startswith(src):
                        rel = dst + rel[len(src) :]

                candidate = (self.assets_dir / rel).resolve()
                if candidate.exists():
                    return candidate.as_posix()

                # fallback: search by basename under assets_dir
                basename = Path(rel).name
                if basename:
                    matches = list(self.assets_dir.rglob(basename))
                    if len(matches) == 1:
                        return matches[0].resolve().as_posix()

                return mesh_ref

            staged_urdf = self.assets_dir / self.urdf_path.name
            staged_ok = False
            try:
                tree = ET.parse(self.urdf_path)
                root = tree.getroot()
                changed = 0
                # URDFs sometimes include XML namespaces; ElementTree's `.findall('.//mesh')`
                # would miss namespaced tags. Use an endswith check instead.
                for elem in root.iter():
                    if not isinstance(elem.tag, str):
                        continue
                    if not elem.tag.endswith("mesh"):
                        continue
                    for attr in ("filename", "file"):
                        if attr not in elem.attrib:
                            continue
                        old = elem.attrib.get(attr, "")
                        new = _resolve_mesh_ref(old)
                        if new and new != old:
                            elem.set(attr, new)
                            changed += 1
                if changed:
                    self.logger.info(
                        f"Staging URDF with absolute mesh paths: updated {changed} mesh references"
                    )
                tree.write(staged_urdf, encoding="utf-8", xml_declaration=True)
                staged_ok = True
            except Exception as exc:
                self.logger.warning(
                    f"Failed to rewrite URDF mesh paths, falling back to raw copy: {exc}"
                )

            if not staged_ok:
                try:
                    staged_urdf.write_text(
                        self.urdf_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )
                except UnicodeDecodeError:
                    staged_urdf.write_bytes(self.urdf_path.read_bytes())

            # Build command without --copy-meshes because assets are staged explicitly.
            cmd = ["urdf2mjcf", str(staged_urdf), "--output", str(self.output_xml)]

            self.logger.info(f"Executing command: {' '.join(cmd)}")

            # Run conversion (cwd helps if tool resolves relative paths against process CWD)
            result = subprocess.run(
                cmd,
                cwd=str(self.assets_dir),
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                if self.output_xml and self.output_xml.exists():
                    self.logger.info(
                        f"✓ urdf2mjcf conversion successful: {self.output_xml}"
                    )
                    return True

                # Some urdf2mjcf versions may ignore --output and write to a default XML path.
                xml_candidates = [
                    p
                    for p in self.output_dir.glob("*.xml")
                    if p.is_file() and p.name != self.urdf_path.name
                ]
                if xml_candidates:
                    xml_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    self.output_xml = xml_candidates[0]
                    self.logger.warning(
                        "urdf2mjcf returned success but expected output was missing; "
                        f"using discovered XML instead: {self.output_xml}"
                    )
                    return True

                self.logger.error(
                    "urdf2mjcf returned success but no XML file was generated in output_dir"
                )
                if result.stdout:
                    self.logger.error(f"urdf2mjcf stdout: {result.stdout}")
                if result.stderr:
                    self.logger.error(f"urdf2mjcf stderr: {result.stderr}")
                return False
            else:
                self.logger.error(f"✗ urdf2mjcf conversion failed: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Error during conversion: {e}")
            return False

    def get_output_xml(self) -> Optional[Path]:
        """Get output XML file path"""
        return self.output_xml
