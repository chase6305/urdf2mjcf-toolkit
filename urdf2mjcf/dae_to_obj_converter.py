#!/usr/bin/env python3
"""
Blender DAE to OBJ Conversion Module
Batch conversion of DAE to OBJ format using Blender, with support for complex scenes and multi-material handling
"""

import os
import sys
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from urdf2mjcf.logging_utils import URDF2MJCFLogger


class DAE2OBJConverter:
    """DAE to OBJ converter using Blender as backend"""

    def __init__(self, blender_executable: str = "blender"):
        """
        Initialize converter

        Args:
            blender_executable: Blender executable path or command
        """
        self.blender_executable = blender_executable
        self.logger = URDF2MJCFLogger.get_logger("DAE2OBJConverter")
        self._blender_checked = False

    def check_blender_available(self) -> bool:
        """Check if Blender is available"""
        if self._blender_checked:
            return True
        try:
            result = subprocess.run(
                [self.blender_executable, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                version_line = result.stdout.split("\n")[0]
                self.logger.info(f"✓ Blender available: {version_line}")
                self._blender_checked = True
                return True
            else:
                raise RuntimeError(f"Blender check failed: {result.stderr}")

        except FileNotFoundError:
            raise RuntimeError(
                f"Blender executable not found: {self.blender_executable}\n"
                "Please ensure Blender is installed and in PATH, or specify via --blender-path"
            )
        except Exception as e:
            raise RuntimeError(f"Error checking Blender: {e}")

    def convert_single_dae(
        self, dae_file: Path, output_dir: Path, preserve_textures: bool = True
    ) -> Tuple[bool, List[Path]]:
        """
        Convert single DAE file

        Args:
            dae_file: DAE file path
            output_dir: Output directory
            preserve_textures: Whether to copy texture files

        Returns:
            Tuple[success status, list of generated OBJ files]
        """
        try:
            if not dae_file.exists():
                self.logger.error(f"DAE file does not exist: {dae_file}")
                return False, []

            self.check_blender_available()

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate Blender script
            blender_script = self._create_single_conversion_script(
                dae_file, output_dir, preserve_textures
            )

            # Execute conversion
            success, output_files = self._run_blender_conversion(
                blender_script, dae_file.name, output_dir
            )

            if success:
                self.logger.info(
                    f"✓ Conversion successful: {dae_file.name} -> {len(output_files)} OBJ files"
                )
            else:
                self.logger.warning(f"✗ Conversion failed: {dae_file.name}")

            return success, output_files

        except Exception as e:
            self.logger.error(f"Error converting single DAE file {dae_file}: {e}")
            return False, []

    def convert_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        recursive: bool = True,
        preserve_textures: bool = True,
    ) -> Dict[str, Tuple[bool, List[Path]]]:
        """
        Batch convert all DAE files in directory

        Args:
            input_dir: Input directory
            output_dir: Output directory
            recursive: Whether to process subdirectories recursively
            preserve_textures: Whether to copy texture files

        Returns:
            Dictionary: {dae file path: (success status, list of generated OBJ files)}
        """
        results = {}

        try:
            # Collect all DAE files
            dae_files = self._collect_dae_files(input_dir, recursive)

            if not dae_files:
                self.logger.warning(f"No DAE files found in directory: {input_dir}")
                return results

            self.logger.info(f"Found {len(dae_files)} DAE files")
            self.check_blender_available()

            # Generate Blender batch conversion script
            blender_script = self._create_batch_conversion_script(
                dae_files, input_dir, output_dir, preserve_textures
            )

            # Execute batch conversion
            self.logger.info("Starting batch DAE to OBJ conversion...")
            success = self._run_blender_batch_conversion(blender_script)

            if success:
                # Collect results
                for dae_file in dae_files:
                    relative_path = dae_file.relative_to(input_dir)
                    output_subdir = output_dir / relative_path.parent / dae_file.stem

                    # Find generated OBJ files
                    obj_files = list(output_subdir.glob("*.obj"))
                    results[str(dae_file)] = (True, obj_files)

                    self.logger.debug(
                        f"  {dae_file.name}: Generated {len(obj_files)} OBJ files"
                    )

            return results

        except Exception as e:
            self.logger.error(f"Error during batch directory conversion: {e}")
            return results

    def _collect_dae_files(self, input_dir: Path, recursive: bool) -> List[Path]:
        """Collect all DAE files"""
        dae_files = []

        if recursive:
            pattern = "**/*.dae"
        else:
            pattern = "*.dae"

        for dae_file in input_dir.glob(pattern):
            dae_files.append(dae_file)

        return sorted(dae_files)

    def _create_single_conversion_script(
        self, dae_file: Path, output_dir: Path, preserve_textures: bool
    ) -> str:
        """Create Blender script for single file conversion"""
        dae_path = str(dae_file).replace("\\", "/")
        output_path = str(output_dir).replace("\\", "/")

        return f'''
import bpy
import os
import glob

def convert_dae_to_obj(dae_path, output_dir):
    """Convert single DAE file to OBJ"""
    print(f"Starting conversion: {{os.path.basename(dae_path)}}")
    
    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Record OBJ files before conversion
    obj_files_before = set(glob.glob(os.path.join(output_dir, '*.obj')))
    
    try:
        # Import DAE
        bpy.ops.wm.collada_import(filepath=dae_path)
        
        # Get imported mesh objects
        imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        
        if not imported_objects:
            print(f"  ⚠ Warning: No mesh objects imported")
            return 0
        
        print(f"  Imported {{len(imported_objects)}} mesh objects")
        
        # Count materials
        materials = set()
        for obj in imported_objects:
            if obj.data.materials:
                for mat in obj.data.materials:
                    if mat:
                        materials.add(mat.name)
        
        print(f"  Found {{len(materials)}} materials")
        
        # Select all mesh objects
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported_objects:
            obj.select_set(True)
        
        bpy.context.view_layer.objects.active = imported_objects[0]
        
        # Export as OBJ
        output_path = os.path.join(output_dir, f"{{os.path.splitext(os.path.basename(dae_path))[0]}}.obj")
        
        # Check Blender version, use appropriate export function
        try:
            # Blender 3.2+
            bpy.ops.wm.obj_export(
                filepath=output_path,
                forward_axis='Y',
                up_axis='Z',
                export_selected_objects=True,
                export_materials=True,
                export_colors=True,
                export_normals=True,
                export_uv=True
            )
        except AttributeError:
            # Blender 3.0-3.1
            bpy.ops.export_scene.obj(
                filepath=output_path,
                axis_forward='Y',
                axis_up='Z',
                use_selection=True,
                use_materials=True,
                use_mesh_modifiers=True,
                use_normals=True,
                use_uvs=True,
                use_blen_objects=True
            )
        
        print(f"  → Exported: {{os.path.basename(output_path)}}")
        
        # Count generated OBJ files
        obj_files_after = set(glob.glob(os.path.join(output_dir, '*.obj')))
        new_obj_files = obj_files_after - obj_files_before
        
        return len(new_obj_files)
        
    except Exception as e:
        print(f"  ✗ Conversion failed: {{e}}")
        import traceback
        traceback.print_exc()
        return 0

# Execute conversion
result = convert_dae_to_obj(r"{dae_path}", r"{output_path}")

if result > 0:
    print(f"✓ Conversion completed, generated {{result}} OBJ files")
else:
    print(f"✗ Conversion failed or no OBJ files generated")
'''

    def _create_batch_conversion_script(
        self,
        dae_files: List[Path],
        input_dir: Path,
        output_dir: Path,
        preserve_textures: bool,
    ) -> str:
        """Create Blender script for batch conversion"""
        dae_paths_str = [str(f).replace("\\", "/") for f in dae_files]
        input_dir_str = str(input_dir).replace("\\", "/")
        output_dir_str = str(output_dir).replace("\\", "/")

        # Create script
        script = f'''
import bpy
import os
import glob
import traceback

def copy_textures(source_dir, dest_dir, dae_file):
    """Copy texture files"""
    try:
        # Get DAE file directory
        dae_dir = os.path.dirname(dae_file)
        
        # Find common texture files
        texture_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tga', '.tiff']
        
        for ext in texture_extensions:
            for texture_file in glob.glob(os.path.join(dae_dir, f'*{{ext}}')):
                texture_name = os.path.basename(texture_file)
                dest_path = os.path.join(dest_dir, texture_name)
                
                if not os.path.exists(dest_path):
                    try:
                        import shutil
                        shutil.copy2(texture_file, dest_path)
                        print(f"    Copied texture: {{texture_name}}")
                    except Exception as e:
                        print(f"    ⚠ Texture copy failed {{texture_name}}: {{e}}")
    
    except Exception as e:
        print(f"    Texture copy process error: {{e}}")

def convert_dae_batch(dae_files, input_dir, output_dir, preserve_textures):
    """Batch convert DAE files"""
    print("=" * 60)
    print("Batch DAE to OBJ Conversion Started")
    print(f"Input directory: {{input_dir}}")
    print(f"Output directory: {{output_dir}}")
    print(f"File count: {{len(dae_files)}}")
    print("=" * 60)
    
    total_converted = 0
    total_objs = 0
    
    for i, dae_file in enumerate(dae_files, 1):
        # Clear scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
        # Calculate output directory
        rel_path = os.path.relpath(dae_file, input_dir)
        component_dir = os.path.join(output_dir, os.path.dirname(rel_path), 
                                    os.path.splitext(os.path.basename(dae_file))[0])
        
        # Ensure output directory exists
        os.makedirs(component_dir, exist_ok=True)
        
        print(f"[{{i}}/{{len(dae_files)}}] Processing: {{rel_path}}")
        
        # Record OBJ files before conversion
        obj_files_before = set(glob.glob(os.path.join(component_dir, '*.obj')))
        
        try:
            # Import DAE file
            print(f"  Importing DAE file...")
            bpy.ops.wm.collada_import(filepath=dae_file)
            
            # Get imported mesh objects
            imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
            
            if not imported_objects:
                print(f"  ⚠ Warning: No mesh objects imported")
                continue
            
            print(f"  Imported {{len(imported_objects)}} mesh objects")
            
            # Count materials
            materials = set()
            for obj in imported_objects:
                if obj.data.materials:
                    for mat in obj.data.materials:
                        if mat:
                            materials.add(mat.name)
            
            print(f"  Found {{len(materials)}} materials")
            
            # Copy texture files (if needed)
            if preserve_textures:
                copy_textures(os.path.dirname(dae_file), component_dir, dae_file)
            
            # Select all mesh objects
            bpy.ops.object.select_all(action='DESELECT')
            for obj in imported_objects:
                obj.select_set(True)
            
            bpy.context.view_layer.objects.active = imported_objects[0]
            
            # Export as OBJ
            output_file = os.path.join(component_dir, 
                                      f"{{os.path.splitext(os.path.basename(dae_file))[0]}}.obj")
            
            # Choose export function based on Blender version
            try:
                # Blender 3.2+
                bpy.ops.wm.obj_export(
                    filepath=output_file,
                    forward_axis='Y',
                    up_axis='Z',
                    export_selected_objects=True,
                    export_materials=True,
                    export_colors=True,
                    export_normals=True,
                    export_uv=True,
                    export_triangulated_mesh=True
                )
            except AttributeError:
                # Blender 3.0-3.1
                bpy.ops.export_scene.obj(
                    filepath=output_file,
                    axis_forward='Y',
                    axis_up='Z',
                    use_selection=True,
                    use_materials=True,
                    use_mesh_modifiers=True,
                    use_normals=True,
                    use_uvs=True,
                    use_blen_objects=True,
                    use_triangles=True
                )
            
            # Count generated OBJ files
            obj_files_after = set(glob.glob(os.path.join(component_dir, '*.obj')))
            new_obj_files = obj_files_after - obj_files_before
            
            if new_obj_files:
                total_converted += 1
                total_objs += len(new_obj_files)
                print(f"  ✓ Conversion successful, generated {{len(new_obj_files)}} OBJ files")
            else:
                print(f"  ✗ No OBJ files generated")
                
        except Exception as e:
            print(f"  ✗ Conversion failed: {{e}}")
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Batch conversion completed!")
    print(f"Successfully converted: {{total_converted}}/{{len(dae_files)}} DAE files")
    print(f"Total generated: {{total_objs}} OBJ files")
    print("=" * 60)
    
    return total_converted > 0

# Execute batch conversion
dae_files = {dae_paths_str}
input_dir = r"{input_dir_str}"
output_dir = r"{output_dir_str}"

success = convert_dae_batch(dae_files, input_dir, output_dir, {preserve_textures})

if success:
    print("Batch conversion completed successfully")
else:
    print("Errors occurred during batch conversion")
'''
        return script

    def _run_blender_conversion(
        self, script: str, dae_name: str, output_dir: Path
    ) -> Tuple[bool, List[Path]]:
        """Run Blender for single file conversion"""
        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                script_path = f.name

            try:
                # Run Blender
                cmd = [
                    self.blender_executable,
                    "--background",
                    "--python",
                    script_path,
                    "--python-exit-code",
                    "1",
                ]

                self.logger.debug(f"Executing Blender command: {' '.join(cmd)}")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )

                # Check output
                if result.returncode != 0:
                    self.logger.warning(
                        f"Blender returned non-zero code: {result.returncode}"
                    )
                    if result.stderr:
                        self.logger.debug(f"Blender error output: {result.stderr}")

                # Check generated OBJ files
                obj_files = list(output_dir.glob(f"{Path(dae_name).stem}*.obj"))
                success = len(obj_files) > 0

                if success:
                    self.logger.debug(
                        f"Conversion output: {result.stdout[-500:] if result.stdout else 'No output'}"
                    )

                return success, obj_files

            finally:
                # Clean up temporary script file
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

        except Exception as e:
            self.logger.error(f"Error running Blender conversion: {e}")
            return False, []

    def _run_blender_batch_conversion(self, script: str) -> bool:
        """Run Blender for batch conversion"""
        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script)
                script_path = f.name

            try:
                # Run Blender
                cmd = [
                    self.blender_executable,
                    "--background",
                    "--python",
                    script_path,
                    "--python-exit-code",
                    "1",
                ]

                self.logger.info("Running Blender for batch conversion...")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )

                # Output Blender log
                if result.stdout:
                    for line in result.stdout.split("\n"):
                        if line.strip() and ":" in line:  # Only show meaningful lines
                            self.logger.debug(f"  {line.strip()}")

                if result.returncode != 0:
                    self.logger.warning(
                        f"Blender batch conversion returned non-zero code: {result.returncode}"
                    )
                    if result.stderr:
                        self.logger.error(f"Blender error: {result.stderr}")

                return result.returncode == 0

            finally:
                # Clean up temporary script file
                try:
                    os.unlink(script_path)
                except OSError:
                    pass

        except Exception as e:
            self.logger.error(f"Error running Blender batch conversion: {e}")
            return False


class TexturePathFixer:
    """Texture path fixer for repairing texture paths in OBJ/MTL files"""

    def __init__(self, assets_dir: Path):
        """
        Initialize texture path fixer

        Args:
            assets_dir: Assets directory
        """
        self.assets_dir = assets_dir
        self.logger = URDF2MJCFLogger.get_logger("TexturePathFixer")

    def fix_mtl_files(self, recursive: bool = True) -> int:
        """
        Fix texture paths in MTL files

        Args:
            recursive: Whether to process recursively

        Returns:
            Number of files fixed
        """
        fixed_count = 0

        try:
            # Find all MTL files
            if recursive:
                mtl_files = list(self.assets_dir.rglob("*.mtl"))
            else:
                mtl_files = list(self.assets_dir.glob("*.mtl"))

            for mtl_file in mtl_files:
                if self._fix_single_mtl(mtl_file):
                    fixed_count += 1
                    self.logger.debug(
                        f"Fixed MTL file: {mtl_file.relative_to(self.assets_dir)}"
                    )

            self.logger.info(f"Fixed {fixed_count} MTL files")

        except Exception as e:
            self.logger.error(f"Error fixing MTL files: {e}")

        return fixed_count

    def _fix_single_mtl(self, mtl_file: Path) -> bool:
        """Fix single MTL file"""
        try:
            # Read file content
            with open(mtl_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = content.split("\n")
            fixed_lines = []

            mtl_dir = mtl_file.parent

            for line in lines:
                if line.strip().startswith("map_Kd"):
                    # Extract texture path
                    parts = line.split()
                    if len(parts) >= 2:
                        texture_path = parts[-1]

                        # Handle different path formats
                        fixed_texture_path = self._resolve_texture_path(
                            texture_path, mtl_dir
                        )

                        if fixed_texture_path:
                            # Update texture path
                            new_line = f"map_Kd {fixed_texture_path}"
                            fixed_lines.append(new_line)
                            self.logger.debug(
                                f"  Fixed texture path: {texture_path} -> {fixed_texture_path}"
                            )
                        else:
                            # If texture file doesn't exist, keep as is but add comment
                            fixed_lines.append(f"# {line}  # Texture file not found")
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)

            # Write back to file
            with open(mtl_file, "w", encoding="utf-8") as f:
                f.write("\n".join(fixed_lines))

            return True

        except Exception as e:
            self.logger.warning(f"Failed to fix MTL file {mtl_file}: {e}")
            return False

    def _resolve_texture_path(self, texture_path: str, mtl_dir: Path) -> Optional[str]:
        """Resolve texture path, return relative path"""
        try:
            # Remove possible file:// prefix
            if texture_path.startswith("file://"):
                texture_path = texture_path[7:]

            # Convert to Path object
            texture_path_obj = Path(texture_path)

            # Check if it's an absolute path
            if texture_path_obj.is_absolute():
                # If absolute path, try to find relative path
                try:
                    rel_path = texture_path_obj.relative_to(self.assets_dir)
                    return str(rel_path)
                except ValueError:
                    # Not in assets directory, copy file
                    return self._copy_texture_file(texture_path_obj, mtl_dir)

            # Already relative path, check if file exists
            possible_paths = [
                mtl_dir / texture_path_obj,  # Relative to MTL file
                mtl_dir / texture_path_obj.name,  # Just filename
                self.assets_dir / texture_path_obj,  # Relative to assets directory
            ]

            for possible_path in possible_paths:
                if possible_path.exists():
                    # Return path relative to MTL file
                    try:
                        rel_path = possible_path.relative_to(mtl_dir)
                        return str(rel_path)
                    except ValueError:
                        # Return path relative to assets directory
                        try:
                            rel_path = possible_path.relative_to(self.assets_dir)
                            return str(rel_path)
                        except ValueError:
                            return texture_path_obj.name

            # File doesn't exist, try to find file with same name in assets directory
            texture_name = texture_path_obj.name
            for possible_texture in self.assets_dir.rglob(texture_name):
                try:
                    rel_path = possible_texture.relative_to(mtl_dir)
                    return str(rel_path)
                except ValueError:
                    try:
                        rel_path = possible_texture.relative_to(self.assets_dir)
                        return str(rel_path)
                    except ValueError:
                        return texture_name

            # Texture file not found
            return None

        except Exception as e:
            self.logger.debug(f"Failed to resolve texture path {texture_path}: {e}")
            return texture_path

    def _copy_texture_file(self, source_path: Path, dest_dir: Path) -> Optional[str]:
        """Copy texture file to destination directory"""
        try:
            if not source_path.exists():
                return None

            dest_path = dest_dir / source_path.name

            # If target file already exists, don't overwrite
            if not dest_path.exists():
                shutil.copy2(source_path, dest_path)
                self.logger.debug(
                    f"Copied texture file: {source_path.name} -> {dest_dir}"
                )

            return source_path.name

        except Exception as e:
            self.logger.warning(f"Failed to copy texture file {source_path}: {e}")
            return None


class DAE2OBJPipeline:
    """
    Complete pipeline for converting DAE files in URDF projects
    Manages the entire conversion process including texture copying and path fixing
    """

    def __init__(self, blender_path: str = "blender"):
        """
        Initialize DAE to OBJ pipeline

        Args:
            blender_path: Path to Blender executable
        """
        self.blender_path = blender_path
        self.logger = URDF2MJCFLogger.get_logger("DAE2OBJPipeline")

        # Initialize components
        self.converter = DAE2OBJConverter(blender_executable=blender_path)
        self.texture_fixer = None  # Will be initialized when assets_dir is known

    def process_urdf_directory(
        self, urdf_dir: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """
        Process all DAE files in a URDF directory

        Args:
            urdf_dir: Directory containing URDF and Visual subdirectory
            output_dir: Output directory for converted files
        Returns:
            Dictionary with conversion statistics
        """
        stats = {
            "dae_files_found": 0,
            "dae_files_converted": 0,
            "obj_files_generated": 0,
            "textures_copied": 0,
            "mtl_files_fixed": 0,
            "success": False,
        }
        try:
            self.logger.info("=" * 60)
            self.logger.info("DAE to OBJ Conversion Pipeline")
            self.logger.info("=" * 60)
            # 1. Find Visual directory
            visual_source_dir = urdf_dir / "Visual"
            visual_output_dir = output_dir / "assets" / "Visual"
            if not visual_source_dir.exists():
                self.logger.warning(
                    f"Visual directory does not exist: {visual_source_dir}"
                )
                return stats
            # 2. Batch convert DAE files. Textures are fixed from MTL files later.
            self.logger.info("Starting batch DAE to OBJ conversion...")
            conversion_results = self.converter.convert_directory(
                input_dir=visual_source_dir,
                output_dir=visual_output_dir,
                recursive=True,
                preserve_textures=False,
            )
            # 3. Count conversion results
            stats["dae_files_found"] = len(conversion_results)
            for dae_file, (success, obj_files) in conversion_results.items():
                if success and obj_files:
                    stats["dae_files_converted"] += 1
                    stats["obj_files_generated"] += len(obj_files)
            # 4. Initialize texture fixer and fix MTL files
            self.logger.info("Fixing texture paths in MTL files...")
            self.texture_fixer = TexturePathFixer(visual_output_dir)
            stats["mtl_files_fixed"] = self.texture_fixer.fix_mtl_files(recursive=True)
            # 5. Update success status
            stats["success"] = stats["dae_files_converted"] > 0
            # 6. Output summary
            self._print_summary(stats)
            return stats
        except Exception as e:
            self.logger.error(f"DAE conversion pipeline error: {e}")
            import traceback

            traceback.print_exc()
            return stats

    def _print_summary(self, stats: Dict[str, Any]) -> None:
        """Print conversion summary"""
        self.logger.info("=" * 60)
        self.logger.info("DAE Conversion Pipeline Summary:")
        self.logger.info(f"  DAE Files Found: {stats['dae_files_found']}")
        self.logger.info(f"  DAE Files Converted: {stats['dae_files_converted']}")
        self.logger.info(f"  OBJ Files Generated: {stats['obj_files_generated']}")
        self.logger.info(f"  Texture Files Copied: {stats['textures_copied']}")
        self.logger.info(f"  MTL Files Fixed: {stats['mtl_files_fixed']}")

        if stats["success"]:
            self.logger.info(f"  Status: ✓ SUCCESS")
        else:
            self.logger.warning(f"  Status: ✗ FAILED (No DAE files converted)")

        self.logger.info("=" * 60)

    def process_single_dae(
        self, dae_file: Path, output_dir: Path, preserve_textures: bool = True
    ) -> Tuple[bool, List[Path]]:
        """
        Process single DAE file with full pipeline

        Args:
            dae_file: DAE file path
            output_dir: Output directory
            preserve_textures: Whether to copy texture files

        Returns:
            Tuple[success status, list of generated OBJ files]
        """
        # Use the converter directly for single file conversion
        return self.converter.convert_single_dae(
            dae_file, output_dir, preserve_textures
        )

    def fix_mtl_files_in_directory(
        self, assets_dir: Path, recursive: bool = True
    ) -> int:
        """
        Fix MTL files in a directory

        Args:
            assets_dir: Assets directory containing MTL files
            recursive: Whether to process recursively

        Returns:
            Number of files fixed
        """
        self.texture_fixer = TexturePathFixer(assets_dir)
        return self.texture_fixer.fix_mtl_files(recursive)


# if __name__ == "__main__":
#     # Command line interface
#     import argparse

#     parser = argparse.ArgumentParser(description='Batch DAE to OBJ conversion tool')
#     parser.add_argument('input_dir', help='Input directory (contains DAE files)')
#     parser.add_argument('output_dir', help='Output directory')
#     parser.add_argument('--blender-path', default='blender', help='Blender executable path')
#     parser.add_argument('--verbose', action='store_true', help='Verbose output')

#     args = parser.parse_args()

#     # Execute conversion using pipeline
#     input_dir = Path(args.input_dir)
#     output_dir = Path(args.output_dir)

#     if not input_dir.exists():
#         print(f"Error: Input directory does not exist: {input_dir}")
#         sys.exit(1)

#     # Create and run pipeline
#     pipeline = DAE2OBJPipeline(blender_path=args.blender_path)
#     stats = pipeline.process_urdf_directory(input_dir, output_dir)

#     if stats['success']:
#         print(f"✓ Pipeline completed successfully!")
#         sys.exit(0)
#     else:
#         print(f"✗ Pipeline failed or no DAE files converted")
#         sys.exit(1)


def main() -> int:
    """Run the DAE-to-OBJ converter from the command line."""
    parser = argparse.ArgumentParser(
        description="Batch convert DAE files to OBJ using Blender."
    )
    parser.add_argument("input_dir", help="Directory containing DAE files.")
    parser.add_argument("output_dir", help="Directory for converted OBJ assets.")
    parser.add_argument(
        "--blender-path",
        default="blender",
        help="Blender executable path or command. Defaults to 'blender'.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only convert DAE files directly under input_dir.",
    )
    parser.add_argument(
        "--no-textures",
        action="store_true",
        help="Do not copy texture files next to generated OBJ files.",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use the URDF project pipeline layout (input_dir/Visual -> output_dir/assets/Visual).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: input directory does not exist: {input_dir}")
        return 1

    if args.pipeline:
        pipeline = DAE2OBJPipeline(blender_path=args.blender_path)
        if args.verbose:
            pipeline.logger.setLevel("DEBUG")
        stats = pipeline.process_urdf_directory(input_dir, output_dir)
        return 0 if stats["success"] else 1

    converter = DAE2OBJConverter(blender_executable=args.blender_path)
    if args.verbose:
        converter.logger.setLevel("DEBUG")
    results = converter.convert_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=not args.no_recursive,
        preserve_textures=not args.no_textures,
    )
    success_count = sum(
        1 for success, obj_files in results.values() if success and obj_files
    )
    return 0 if success_count else 1


if __name__ == "__main__":
    sys.exit(main())
