from __future__ import annotations
import numpy as np
import trimesh
from pathlib import Path

def validate(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("文件必须包含三角网格")
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all():
        raise ValueError("顶点为空或包含无效数值")
    if f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or f.min() < 0 or f.max() >= len(v):
        raise ValueError("三角面索引无效")
    span = np.ptp(v, axis=0)
    if span.max() <= 0:
        raise ValueError("模型尺寸为零")
    # The upload/interchange format is GLB, whose positions are float32.
    # Reject a lossy overflow/collapse before exporting an apparently valid mesh.
    with np.errstate(over='ignore',invalid='ignore'):
        encoded=np.asarray(v,dtype=np.float32)
    if not np.isfinite(encoded).all() or np.any((span > 0) & (np.ptp(encoded.astype(np.float64),axis=0) == 0)):
        raise ValueError("模型坐标超出 GLB 精度范围，请先居中并调整模型尺寸")
    return trimesh.Trimesh(v.copy(), f.copy(), process=False)

def load_mesh(path: Path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as x:
            if not np.isrealobj(x["vertices"]):
                raise ValueError("网格顶点必须为实数坐标")
            if not np.issubdtype(x["faces"].dtype,np.integer):
                raise ValueError("三角面索引必须为整数")
            return validate(trimesh.Trimesh(x["vertices"], x["faces"], process=False))
    # External-resource formats GLTF/OBJ intentionally excluded from uploads.
    return validate(trimesh.load(path, process=False, resolver={}, allow_remote=False, skip_materials=True))

def inspect_mesh(mesh):
    mesh = validate(mesh)
    unique, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    areas = mesh.area_faces
    return {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "bounds": mesh.bounds.tolist(), "extents": mesh.extents.tolist(),
            "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
            "boundary_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum()),
            "degenerate_faces": int((areas <= 1e-14).sum()),
            "components": len(trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)))),
            "unchecked": ["自交", "壁厚", "最小零件强度", "切片可打印性"],
            "units": "unknown", "warnings": ["Hi3D 会重新归一化尺寸并可能删除小连通件；不保留原 UV/材质。"]}

def export_mesh(mesh, target: Path, height_mm: float | None = None):
    mesh = validate(mesh)
    if height_mm is not None:
        if not 0 < height_mm <= 100000 or mesh.extents[2] <= 1e-12:
            raise ValueError("目标高度或 Z 轴尺寸无效")
        # glTF/GLB encodes linear distances in meters; STL/PLY use our mm convention.
        coordinate_height=height_mm/1000 if target.suffix.lower()=='.glb' else height_mm
        mesh.apply_scale(coordinate_height / mesh.extents[2])
        mesh=validate(mesh)
    mesh.export(target)
    return {"format": target.suffix, "height_axis": "Z", "height_mm": height_mm,
            "units": ("m" if target.suffix.lower()=='.glb' else "mm") if height_mm else "normalized/unknown", "bounds": mesh.bounds.tolist()}
