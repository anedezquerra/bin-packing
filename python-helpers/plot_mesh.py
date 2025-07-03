# plot_container.py
# ------------------------------------------------------------------
# Renderiza tetraedros + contenedor (esfera, cubo o cilindro) y
# exporta HTML interactivo, PNG, GLB (glTF 2.0), OBJ y VTM.
#
# Uso:
#   python plot_container.py -j 16069283
#
# Requiere: numpy, pandas, openpyxl, pyvista[qt], vtk, imageio, pillow
# ------------------------------------------------------------------
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import warnings
from itertools import cycle

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl  

import numpy as np
import pandas as pd
import pyvista as pv


# ---------------------------- Rutas ----------------------------------------
ARTIFACTS     = pathlib.Path("artifacts")
RESULTS_JSON  = ARTIFACTS / "results.json"
MESH_DIR      = ARTIFACTS / "meshes"

HTML_DIR      = ARTIFACTS / "html"
IMG_DIR       = ARTIFACTS / "images"
RENDER_BASE   = ARTIFACTS / "renders"
GLTF_DIR      = RENDER_BASE / "gltf"
OBJ_DIR       = RENDER_BASE / "obj"
VTM_DIR       = RENDER_BASE / "vtm"

for d in (HTML_DIR, IMG_DIR, GLTF_DIR, OBJ_DIR, VTM_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------- Carga ----------------------------------------
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
    req = {"tetra", "vertex", "x", "y", "z"}
    if not req.issubset(df.columns):
        sys.exit(f"❌ {mesh_path} no contiene {req}")
    return df


def compute_center(df: pd.DataFrame) -> tuple[float, float, float]:
    return tuple(df[["x", "y", "z"]].mean().values)


def build_scene(record: dict,
                df: pd.DataFrame) -> tuple[pv.Plotter, list[pv.DataSet]]:
    """Devuelve (plotter, exports).  'exports' es una lista de PolyData."""
    pv.global_theme.smooth_shading = True
    plotter = pv.Plotter()
    plotter.set_background("white")

    container_type = (record.get("container_type") or "").lower()
    radius_value   = record.get("radius") or record.get("side")
    cube_side      = record.get("side")
    height_value   = record.get("height")
    center         = compute_center(df)

    exports: list[pv.DataSet] = []      # ← list que acabaremos devolviendo

    # --------------- Contenedor --------------------------------------------
    if "sphere" in container_type and radius_value:
        geom = pv.Sphere(radius=radius_value, center=center,
                         theta_resolution=40, phi_resolution=40)
        _add_container(plotter, geom)
        exports.append(geom.extract_surface())

    elif "cube" in container_type and cube_side:
        geom = pv.Cube(center=center,
                       x_length=cube_side, y_length=cube_side, z_length=cube_side)
        _add_container(plotter, geom)
        exports.append(geom.extract_surface())

    elif "cylinder" in container_type and radius_value and height_value:
        geom = pv.Cylinder(center=center, direction=(0, 0, 1),
                           radius=radius_value, height=height_value,
                           resolution=60, capping=True)
        _add_container(plotter, geom)
        exports.append(geom.extract_surface())

    # --------------- Paleta de colores -------------------------------------
    unique_tets = sorted(df["tetra"].unique())
    num_tets    = len(unique_tets)

    if num_tets <= 20:
        cmap = mpl.colormaps.get_cmap("tab20").resampled(num_tets)
    else:
        cmap = mpl.colormaps.get_cmap("hsv").resampled(num_tets)

    color_by_tet = {
        tet_id: mpl.colors.to_hex(cmap(i)) for i, tet_id in enumerate(unique_tets)
    }

    # --------------- Tetraedros --------------------------------------------
    for tet_id, group in df.groupby("tetra"):
        verts = group.sort_values("vertex")[["x", "y", "z"]].values[:4]
        cells = np.hstack([[4, 0, 1, 2, 3]]).astype(np.int64)   # VTK_TETRA
        tet   = pv.UnstructuredGrid(cells, np.array([10]), verts)

        plotter.add_mesh(
            tet,
            color=color_by_tet[tet_id],
            opacity=1.0,
            smooth_shading=True,
            show_edges=True
        )
        exports.append(tet.extract_surface())

    # ---------------- Fin ---------------------------------------------------
    return plotter, exports     # ← añade a la lista 'exports'


def _add_container(plotter: pv.Plotter, geom: pv.DataSet) -> None:
    plotter.add_mesh(geom, color="lightgray", opacity=0.15, show_edges=False)
    plotter.add_mesh(geom, style="wireframe", color="black",
                     opacity=0.25, line_width=1)


# ---------------------------------------------------------------------------
# Exporta HTML, PNG, GLB, OBJ y VTM (versión estable: PNG vía show)
# ---------------------------------------------------------------------------
def export_assets(job_id: str,
                  plotter: pv.Plotter,
                  surfaces: list[pv.DataSet]) -> None:
    """
    Genera:
      • artifacts/html/<job_id>.html
      • artifacts/images/<job_id>.png
      • artifacts/renders/gltf/<job_id>.glb
      • artifacts/renders/obj/<job_id>.obj
      • artifacts/renders/vtk/<job_id>.vtm
    """
    html_path = HTML_DIR / f"{job_id}.html"
    img_path  = IMG_DIR  / f"{job_id}.png"
    glb_path  = GLTF_DIR / f"{job_id}.glb"
    obj_path  = OBJ_DIR  / f"{job_id}.obj"
    vtm_path  = VTM_DIR  / f"{job_id}.vtm"

    # 1) HTML interactivo
    plotter.export_html(str(html_path))

    # 2) GLB
    plotter.export_gltf(str(glb_path))

    # 3) OBJ (fusiona superficies)
    if surfaces:
        surf_comb = surfaces[0].copy()
        for s in surfaces[1:]:
            surf_comb = surf_comb.merge(s)
        surf_comb.save(str(obj_path))

    # 4) VTM multiblock
    pv.MultiBlock(surfaces).save(str(vtm_path))

    # 5) Mostrar escena y capturar PNG en la misma llamada
    plotter.show(
        title=f"Job {job_id}",
        window_size=[1400, 1000],
        screenshot=str(img_path),   # ← PyVista guarda la captura al cerrar
        auto_close=True             # se cierra al cerrar la ventana
    )

    # --- Consola
    print(f"✓ HTML  → {html_path}")
    print(f"✓ PNG   → {img_path}")
    print(f"✓ GLB   → {glb_path}")
    print(f"✓ OBJ   → {obj_path}")
    print(f"✓ VTM   → {vtm_path}")



# ----------------------------- Main CLI ------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renderiza malla y contenedor, "
                    "exportando HTML/PNG/GLB/OBJ/VTM."
    )
    parser.add_argument("-j", "--job_id", required=True,
                        help="ID presente en artifacts/results.json")
    args = parser.parse_args()
    job_id = str(args.job_id)

    record = load_record(job_id)
    df     = load_mesh_dataframe(job_id)

    plotter, surfaces = build_scene(record, df)
    export_assets(job_id, plotter, surfaces)


if __name__ == "__main__":
    main()
