import pytest

from urdf2mjcf.mesh_to_obj_converter import MeshToOBJConverter


ASCII_STL = """solid triangle
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid triangle
"""


def test_stl_is_converted_to_obj(tmp_path):
    source = tmp_path / "triangle.stl"
    source.write_text(ASCII_STL, encoding="ascii")

    output = MeshToOBJConverter().convert_file(source)

    assert output == source.with_suffix(".obj")
    assert output.is_file()
    assert "v 0.00000000 0.00000000 0.00000000" in output.read_text(
        encoding="utf-8"
    )


def test_directory_scan_is_case_insensitive_and_recursive(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "mesh.STL").write_text(ASCII_STL, encoding="ascii")
    (tmp_path / "ignored.txt").touch()

    assert MeshToOBJConverter().find_mesh_files(tmp_path) == [nested / "mesh.STL"]


def test_unsupported_format_is_rejected(tmp_path):
    source = tmp_path / "mesh.fbx"
    source.touch()

    with pytest.raises(ValueError, match="Unsupported mesh format"):
        MeshToOBJConverter().convert_file(source)
