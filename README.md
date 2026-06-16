# URDF2MJCF Toolkit

URDF2MJCF Toolkit is an automated Python toolchain for converting robot URDF models and their mesh assets into MuJoCo-ready MJCF projects.

## Description

A Python toolkit for converting URDF robot models and mesh assets into MuJoCo-ready MJCF projects.

## Features

- Converts URDF models to MJCF with a single pipeline command.
- Copies and normalizes visual and optional collision assets.
- Converts DAE and GLB meshes to OBJ for MuJoCo-friendly asset loading.
- Generates OBJ-derived MJCF snippets with `obj2mjcf`.
- Repairs problematic inertia values by recalculating them from mesh geometry when conversion fails.
- Post-processes MJCF files to normalize mesh paths, textures, materials, default geom classes, ground plane, and fixed/floating base behavior.
- Includes helper tools for URDF inertia validation and MuJoCo model visualization.

## Requirements

- Python 3.11+
- Blender available on `PATH` for DAE conversion
- `obj2mjcf` and `urdf2mjcf` command-line tools from the Python dependencies
- MuJoCo Python bindings for validation or visualization

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Install Blender on Ubuntu:

```bash
sudo apt install blender
blender --version
```

## Quick Start

Convert a URDF model with a fixed base:

```bash
python -m urdf2mjcf.urdf_to_mujoco_converter /path/to/robot.urdf
```

Specify an output directory and use a floating base:

```bash
python -m urdf2mjcf.urdf_to_mujoco_converter robot.urdf ./output --floating-base
```

Export collision meshes as separate MuJoCo collision geoms:

```bash
python -m urdf2mjcf.urdf_to_mujoco_converter robot.urdf ./output --export-collision
```

Enable debug logging:

```bash
python -m urdf2mjcf.urdf_to_mujoco_converter robot.urdf ./output --verbose
```

## Main CLI

```bash
python -m urdf2mjcf.urdf_to_mujoco_converter <urdf_path> [output_dir] [options]
```

Arguments:

- `<urdf_path>`: input URDF file.
- `[output_dir]`: optional output directory. If omitted, the converter creates a sibling `<input_parent>_mjcf` directory.

Options:

- `--floating-base`: keep or insert a root `<freejoint>` for floating-base models. The default is fixed-base behavior.
- `--export-collision`: copy and process collision assets, then add collision geoms when the model does not already contain collision geometry.
- `--no-inertia-recalc`: disable automatic inertia recalculation when the initial URDF conversion fails.
- `-v, --verbose`: enable debug logging.

## Helper Tools

Convert DAE files to OBJ with Blender:

```bash
python -m urdf2mjcf.dae_to_obj_converter ./meshes/visual ./converted_visual
```

Convert GLB files to OBJ and extract textures:

```bash
python -m urdf2mjcf.glb_to_obj_converter ./meshes/visual -o ./converted_visual
```

Generate MJCF snippets for OBJ asset folders:

```bash
python -m urdf2mjcf.obj_to_mjcf_converter ./output/meshes/visual --recursive
```

Recalculate URDF inertia values in place:

```bash
python -m urdf2mjcf.urdf_inertia_calculator robot.urdf --geometry visual
```

Validate URDF inertia values without modifying the file:

```bash
python kit/urdf_inertia_validator.py robot.urdf
```

Validate a generated MJCF model:

```bash
python kit/visualize_mujoco.py output/robot.xml --validate-only
```

Open the MuJoCo viewer:

```bash
python kit/visualize_mujoco.py output/robot.xml
```

## Conversion Workflow

1. Copy visual assets, and collision assets when requested.
2. Convert DAE and GLB assets under `meshes/visual` to OBJ.
3. Stage the URDF so mesh paths resolve correctly during `urdf2mjcf` conversion.
4. Convert URDF to MJCF, optionally retrying after inertia recalculation.
5. Run `obj2mjcf` over generated OBJ folders.
6. Edit the final MJCF to normalize compiler paths, materials, textures, mesh variants, default geom classes, and base behavior.
7. Save the final MJCF and converted assets in the output directory.

## Output Layout

A typical output directory looks like this:

```text
output_dir/
  robot.xml
  meshes/
    visual/
      ... converted visual OBJ/XML/texture assets ...
    collision/
      ... collision assets when --export-collision is used ...
```

## Troubleshooting

- `Blender executable not found`: install Blender and confirm `blender --version` works, or pass `--blender-path` to the DAE converter.
- `obj2mjcf executable not found`: reinstall dependencies with `python -m pip install -r requirements.txt` and confirm `obj2mjcf` is on `PATH`.
- Missing meshes or textures: check that the original URDF mesh paths point to existing visual/collision asset folders.
- Invalid inertia values: run `python kit/urdf_inertia_validator.py robot.urdf`, then retry conversion with inertia recalculation enabled.
- Model fails to load in MuJoCo: run `python kit/visualize_mujoco.py output/robot.xml --validate-only` for a focused validation pass.

## Development

The repository includes lightweight formatting and lint configuration in `pyproject.toml`.

Recommended local checks:

```bash
python -m compileall urdf2mjcf kit
python -m pip install black ruff
ruff check .
black --check .
```

## Contributing

Contributions are welcome. Please open an issue or pull request with a clear description, reproduction steps for bugs, and example input models when possible.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
