#!/usr/bin/env python
# plot_container.py
# ---------------------------------------------------------------------------
# Visualiza malla de tetraedros + contenedor y exporta múltiples formatos.
#
# Uso:
#   python plot_container.py -j 16069283 --pal Blues
#
# Requisitos (requirements.txt):
#   numpy pandas openpyxl pyvista[qt] vtk matplotlib pillow imageio
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib as mpl
import matplotlib.colors as mcolors


# ---------------------------- Rutas comunes --------------------------------
ARTIFACTS      = pathlib.Path("artifacts")
RESULTS_JSON   = ARTIFACTS / "results.json"
MESH_DIR       = ARTIFACTS / "meshes"

HTML_DIR       = ARTIFACTS / "html"
IMG_DIR        = ARTIFACTS / "images"
GLTF_DIR       = ARTIFACTS / "renders" / "gltf"
OBJ_DIR        = ARTIFACTS / "renders" / "obj"
VTM_DIR        = ARTIFACTS / "renders" / "vtm"

for d in (HTML_DIR, IMG_DIR, GLTF_DIR, OBJ_DIR, VTM_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------- Utilidades de carga --------------------------------
def load_record(job_id: str) -> dict:
    if not RESULTS_JSON.exists():
        sys.exit(f"❌ Falta {RESULTS_JSON}")
    with RESULTS_JSON.open(encoding="utf-8") as fh:
        db = json.load(fh)
    if job_id not in db:
        sys.exit(f"❌ El job_id {job_id} no está en {RESULTS_JSON}")
    return db[job_id]


def load_mesh_dataframe(job_id: str) -> pd.DataFrame:
    mesh_path = MESH_DIR / f"{job_id}.csv"
    if not mesh_path.exists():
        sys.exit(f"❌ Falta la malla {mesh_path}")
    df = pd.read_csv(mesh_path)
    req_cols = {"tetra", "vertex", "x", "y", "z"}
    if not req_cols.issubset(df.columns):
        sys.exit(f"❌ {mesh_path} no contiene {req_cols}")
    return df


def compute_center(df: pd.DataFrame) -> tuple[float, float, float]:
    return tuple(df[["x", "y", "z"]].mean().values)


# ------------------------ Construcción escena ------------------------------
def build_scene(record: dict,
                df: pd.DataFrame,
                palette: str = "tab20") -> tuple[pv.Plotter, List[pv.DataSet]]:
    """
    Devuelve (plotter, exports) donde exports es una lista de PolyData para
    las exportaciones 3D.
    """
    pv.global_theme.smooth_shading = True
    plotter = pv.Plotter()
    plotter.set_background("white")

    # -------- Contenedor ----------------------------------------------------
    container_type = (record.get("container_type") or "").lower()
    radius_value   = record.get("radius") or record.get("side")
    cube_side      = record.get("side")
    height_value   = record.get("height")
    center         = compute_center(df)

    exports: List[pv.DataSet] = []

    def _add_container(geom: pv.DataSet):
        plotter.add_mesh(geom, color="lightgray", opacity=0.15, show_edges=False)
        plotter.add_mesh(geom, style="wireframe", color="black",
                         opacity=0.25, line_width=1)
        exports.append(geom.extract_surface())

    if "sphere" in container_type and radius_value:
        _add_container(
            pv.Sphere(radius=radius_value, center=center,
                      theta_resolution=40, phi_resolution=40)
        )
    elif "cube" in container_type and cube_side:
        _add_container(
            pv.Cube(center=center,
                    x_length=cube_side, y_length=cube_side, z_length=cube_side)
        )
    elif "cylinder" in container_type and radius_value and height_value:
        _add_container(
            pv.Cylinder(center=center, direction=(0, 0, 1),
                        radius=radius_value, height=height_value,
                        resolution=60, capping=True)
        )
    else:
        warnings.warn("Contenedor no reconocido o faltan datos; sólo se mostrará la malla.")

    # -------- Colores por tetraedro ----------------------------------------
    unique_tets = sorted(df["tetra"].unique())
    num_tets    = len(unique_tets)

    try:
        cmap = mpl.colormaps.get_cmap(palette)
    except ValueError:
        warnings.warn(f"Paleta '{palette}' no encontrada; se usa 'tab20'.")
        cmap = mpl.colormaps.get_cmap("tab20")

    # Resample para obtener n colores
    if hasattr(cmap, "resampled"):
        cmap = cmap.resampled(num_tets)
        colors_array = cmap(range(num_tets))
    else:  # versión < 3.5
        colors_array = cmap(np.linspace(0, 1, num_tets))

    color_by_tet = {
        tet_id: mcolors.to_hex(colors_array[i])
        for i, tet_id in enumerate(unique_tets)
    }

    # -------- Tetraedros ----------------------------------------------------
    for tet_id, group in df.groupby("tetra"):
        verts = group.sort_values("vertex")[["x", "y", "z"]].values[:4]
        cells = np.hstack([[4, 0, 1, 2, 3]]).astype(np.int64)
        tet   = pv.UnstructuredGrid(cells, np.array([10]), verts)  # VTK_TETRA

        plotter.add_mesh(
            tet,
            color=color_by_tet[tet_id],
            opacity=1.0,
            smooth_shading=True,
            show_edges=True
        )
        exports.append(tet.extract_surface())

    return plotter, exports


# --------------------------- Exportaciones ---------------------------------
def export_assets(job_id: str,
                  plotter: pv.Plotter,
                  surfaces: List[pv.DataSet]) -> None:
    """
    Genera archivos en artifacts/:
      html/, images/, renders/gltf/, renders/obj/, renders/vtk/
    """
    html_path = HTML_DIR / f"{job_id}.html"
    img_path  = IMG_DIR  / f"{job_id}.png"
    glb_path  = GLTF_DIR / f"{job_id}.glb"
    obj_path  = OBJ_DIR  / f"{job_id}.obj"
    vtm_path  = VTM_DIR  / f"{job_id}.vtm"

    # 1) HTML interactivo
    plotter.export_html(str(html_path))

    # 2) GLB (glTF 2.0)
    plotter.export_gltf(str(glb_path))

    # 3) OBJ (fusiona superficies)
    if surfaces:
        surf_comb = surfaces[0].copy()
        for s in surfaces[1:]:
            surf_comb = surf_comb.merge(s)
        surf_comb.save(str(obj_path))

    # 4) VTM multiblock
    pv.MultiBlock(surfaces).save(str(vtm_path))

    # 5) Mostrar escena y capturar PNG
    plotter.show(
        title=f"Job {job_id}",
        window_size=[1400, 1000],
        screenshot=str(img_path)     # captura al cerrar la ventana
    )

    # --- Log
    print(f"✓ HTML  → {html_path}")
    print(f"✓ PNG   → {img_path}")
    print(f"✓ GLB   → {glb_path}")
    print(f"✓ OBJ   → {obj_path}")
    print(f"✓ VTM   → {vtm_path}")


# ------------------------------ CLI ----------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renderiza malla y contenedor, exportando múltiples formatos."
    )
    parser.add_argument("-j", "--job_id", required=True,
                        help="Identificador del job en artifacts/results.json")
    parser.add_argument("-p", "--pal", default="tab20",
                        help="Nombre de la paleta Matplotlib (p. ej. Blues, pink, hsv...)")
    args = parser.parse_args()

    job_id  = str(args.job_id)
    palette = args.pal

    record = load_record(job_id)
    df     = load_mesh_dataframe(job_id)

    plotter, surfaces = build_scene(record, df, palette=palette)
    export_assets(job_id, plotter, surfaces)


if __name__ == "__main__":
    main()
