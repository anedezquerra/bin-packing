#!/usr/bin/env python
# o‑manager.py
# ---------------------------------------------------------------------------
# Unified CLI for the tetrahedra‑packing project
#
# Sub‑commands
#   • neos‑fetch     → download NEOS results, parse & update results.json (+ meshes)
#   • data           → maintenance utilities on results.json & artifacts/*
#   • json2xlsx      → convert results.json → Excel (optionally adds packing ratio)
#   • eda            → exploratory graphics from an Excel results file
#   • plot           → render a single job (HTML, PNG, GLB, OBJ, VTM)
#
# Run “python o‑manager.py <sub‑command> -h” for details on each one.
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse
import logging
import os
import sys
import json
import math
import re
import shutil
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

# Third‑party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors
import pyvista as pv
import xmlrpc.client
from tqdm import tqdm

# ------------- CONSTANTS ---------------------------------------------------
ARTIFACTS = Path("artifacts")
RESULTS_JSON = ARTIFACTS / "results.json"
MESH_DIR = ARTIFACTS / "meshes"
HTML_DIR = ARTIFACTS / "html"
IMG_DIR = ARTIFACTS / "images"
GLTF_DIR = ARTIFACTS / "renders" / "gltf"
OBJ_DIR = ARTIFACTS / "renders" / "obj"
VTM_DIR = ARTIFACTS / "renders" / "vtm"

ARTIFACT_DIRS = [
    "artifacts/html",
    "artifacts/images",
    "artifacts/meshes",
    "artifacts/renders/gltf",
    "artifacts/renders/obj",
    "artifacts/renders/vtm",
]

for d in (
    ARTIFACTS,
    MESH_DIR,
    HTML_DIR,
    IMG_DIR,
    GLTF_DIR,
    OBJ_DIR,
    VTM_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

SOLVERS = ("BARON", "Knitro", "IPOPT")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ======================================================================
#  SECTION 1 ▸ DATA‑MANAGEMENT HELPERS  (from data-manager.py)
# ======================================================================


# ======================================================================
#  HELPER ▸ parse flexible --delete input
# ======================================================================
def _parse_delete_input(value: str) -> List[str]:
    """Return a list of job‑id strings from CLI input."""
    import ast
    path_str = value.lstrip("@")                    # support optional "@file"
    path = Path(path_str)
    # ---- treat as file -------------------------------------------------
    if path.exists() and path.is_file():
        ids = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        return ids
    # ---- treat as Python/JSON list -------------------------------------
    if value.startswith("[") and value.endswith("]"):
        try:
            seq = ast.literal_eval(value)
            if isinstance(seq, (list, tuple)):
                return [str(x).strip() for x in seq]
        except Exception:
            pass
    # ---- treat as CSV --------------------------------------------------
    if "," in value:
        return [x.strip() for x in value.split(",") if x.strip()]
    # ---- single id -----------------------------------------------------
    return [value.strip()]



def _dm_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _dm_save(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def _dm_remove_job_files(base_dir: Path, job_id: str) -> None:
    for rel_dir in ARTIFACT_DIRS:
        full_dir = base_dir / rel_dir
        if not full_dir.exists():
            continue
        for f in full_dir.iterdir():
            if f.name.startswith(job_id):
                try:
                    f.unlink()
                except OSError as e:
                    logging.warning("Could not remove %s: %s", f, e)


def _dm_prune_null(data: Dict[str, Any], base_dir: Path) -> List[str]:
    to_delete = [
        jid
        for jid, rec in data.items()
        if rec.get("container_type") is None
        and rec.get("structural_conservation_type") is None
        and rec.get("solve_result_num") is None
    ]
    for jid in to_delete:
        data.pop(jid)
        _dm_remove_job_files(base_dir, jid)
    return to_delete


def _dm_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    combs = defaultdict(int)
    for rec in data.values():
        key = (rec.get("container_type"), rec.get("structural_conservation_type"))
        combs[key] += 1
    return {
        "total_entries": len(data),
        "type_combinations": OrderedDict(
            sorted(
                combs.items(),
                key=lambda kv: (str(kv[0][0] or ""), str(kv[0][1] or "")),
            )
        ),
    }


def _dm_validate_neos(data: Dict[str, Any], base_dir: Path) -> int:
    neos_dir = base_dir / "neos_results"
    if not neos_dir.exists():
        logging.warning("neos_results directory not found at %s", neos_dir)
        return 0
    json_jobs = set(data.keys())
    removed = 0
    for f in neos_dir.iterdir():
        job_id = f.stem
        if job_id in json_jobs:
            continue
        _dm_remove_job_files(base_dir, job_id)
        removed += 1
    return removed


# ======================================================================
#  SECTION 2 ▸ NEOS PARSER & INCREMENT  (from parser.py)
# ======================================================================
_REGEXES = {
    "container_type": r"Container type:\s*([^\n\r]+)",
    "structural_conservation_type": r"Structural conservation type:\s*([^\n\r]+)",
    "solve_result_num": r"solve_result_num\s*=\s*([^\s]+)",
    "solve_result": r"solve_result\s*=\s*([^\s]+)",
    "card_figures": r"card\(figures\)\s*=\s*([^\s]+)",
    "radius": r"radius\s*=\s*([^\s]+)",
    "side": r"side\s*=\s*([^\s]+)",
    "height": r"height\s*=\s*([^\s]+)",
    "softness": r"softness\s*=\s*([^\s]+)",
    "ampl_time": r"_ampl_time\s*=\s*([^\s]+)",
    "total_solve_time": r"_total_solve_time\s*=\s*([^\s]+)",
    "ampl_elapsed_time": r"_ampl_elapsed_time\s*=\s*([^\s]+)",
    "ampl_user_time": r"_ampl_user_time\s*=\s*([^\s]+)",
    "total_time_elapsed": r"Total time elapsed:\s*\$?([0-9.+\-Ee]+\.?)",
}


def _parse_scalar_fields(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"solver": None}
    for s in SOLVERS:
        if re.search(rf"\b{s}\b", text, re.I):
            out["solver"] = s
            break
    for key, pat in _REGEXES.items():
        m = re.search(pat, text, re.I)
        if not m:
            out[key] = None
        else:
            val = m.group(1).strip()
            if key in {
                "solve_result",
                "container_type",
                "structural_conservation_type",
            }:
                out[key] = val
            else:
                val = val.rstrip(".")
                try:
                    num = float(val)
                    out[key] = int(num) if num.is_integer() else num
                except ValueError:
                    out[key] = None
    return out


def _parse_tetra_volume_sum(text: str) -> Optional[float]:
    m = re.search(
        r"tetrahedron_volume\s*\[\*\]\s*:=\s*(.*?)\s*;", text, re.S | re.I
    )
    if not m:
        return None
    toks = re.findall(
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", m.group(1)
    )
    return sum(float(toks[i]) for i in range(1, len(toks), 2))


def _parse_coords_df(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    dim_maps = {1: [], 2: [], 3: []}
    dim = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if re.match(r"\s*coords\s*\[\*,\*,1\]", ln):
            dim = 1
            i += 2
            continue
        if dim and re.match(r"\s*\[\*,\*,2\]", ln):
            dim = 2
            i += 2
            continue
        if dim and re.match(r"\s*\[\*,\*,3\]", ln):
            dim = 3
            i += 2
            continue
        if dim and ln.strip().startswith(";"):
            dim = None
        elif dim and re.match(r"\s*\d", ln):
            parts = ln.strip().split()
            dim_maps[dim].append((int(parts[0]), list(map(float, parts[1:]))))
        i += 1

    if not all(dim_maps[d] for d in (1, 2, 3)):
        return pd.DataFrame()

    x_dict = {r: v for r, v in dim_maps[1]}
    y_dict = {r: v for r, v in dim_maps[2]}
    z_dict = {r: v for r, v in dim_maps[3]}

    rows = []
    for tet_id, xs in x_dict.items():
        ys, zs = y_dict[tet_id], z_dict[tet_id]
        for vidx, (x, y, z) in enumerate(zip(xs, ys, zs), 1):
            rows.append({"tetra": tet_id, "vertex": vidx, "x": x, "y": y, "z": z})
    return pd.DataFrame(rows)


def parse_and_increment(text: str, job_id: str, password: str) -> None:
    job_id = str(job_id)
    record: Dict[str, Any] = {
        "job_id": job_id,
        "password": password,
        "processing_datetime": datetime.now().isoformat(),
        **_parse_scalar_fields(text),
        "tetrahedron_volume_sum": _parse_tetra_volume_sum(text),
    }

    mesh_df = _parse_coords_df(text)
    if not mesh_df.empty:
        mesh_path = MESH_DIR / f"{job_id}.csv"
        mesh_df.to_csv(mesh_path, index=False)
        record["mesh_file"] = str(mesh_path)
    else:
        record["mesh_file"] = None

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_JSON.exists():
        try:
            db = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logging.warning("results.json corrupt → re‑created.")
            db = {}
    else:
        db = {}

    if job_id in db:
        logging.info("Job %s already in results.json → skipped.", job_id)
        return
    db[job_id] = record
    RESULTS_JSON.write_text(json.dumps(db, indent=2), encoding="utf-8")
    logging.info("Added job %s to %s", job_id, RESULTS_JSON)


# ======================================================================
#  SECTION 3 ▸ JSON → XLSX (merges json2xlsx.py + toxlsx.py)
# ======================================================================
def json_to_xlsx(
    input_path: Path,
    output_path: Path,
    item_vol: Optional[float] = None,
) -> None:
    with input_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    records = list(data.values()) if isinstance(data, dict) else data
    df = pd.DataFrame.from_records(records)

    base_cols = [
        "job_id",
        "solver",
        "container_type",
        "structural_conservation_type",
        "solve_result",
        "card_figures",
        "radius",
        "side",
        "height",
        "softness",
        "ampl_time",
        "total_solve_time",
        "ampl_elapsed_time",
        "ampl_user_time",
    ]

    df = df[base_cols]

    if item_vol is not None:
        df["container_volume"] = np.nan
        cyl = df["container_type"] == "cylinder"
        sph = df["container_type"] == "sphere"
        cub = df["container_type"] == "cube"

        df.loc[cyl, "container_volume"] = (
            np.pi * df.loc[cyl, "radius"] ** 2 * df.loc[cyl, "height"]
        )
        df.loc[sph, "container_volume"] = (4 / 3) * np.pi * df.loc[sph, "radius"] ** 3
        df.loc[cub, "container_volume"] = df.loc[cub, "side"] ** 3

        df["items_volume"] = df["card_figures"] * item_vol
        df["packing_ratio"] = np.where(
            df["container_volume"] > 0,
            df["items_volume"] / df["container_volume"],
            np.nan,
        )
        df["valid_result"] = df["packing_ratio"] <= 1
        base_cols.extend(
            ["container_volume", "items_volume", "packing_ratio", "valid_result"]
        )

    df = df.rename(
        columns={
            "container_type": "container",
            "structural_conservation_type": "conservation",
            "solve_result": "result",
            "card_figures": "items",
        }
    )

    df = (
        df[df["result"] == "solved"]
        .sort_values(["container", "conservation", "softness", "items"])
        .drop_duplicates(subset=["container", "conservation", "softness", "items"])
        .fillna(np.nan)
    )

    df.to_excel(output_path, index=False)
    logging.info("✔️ Excel saved → %s", output_path.resolve())


# ======================================================================
#  SECTION 4 ▸ EDA PLOTS  (eda-analysis.py merged)
# ======================================================================
def run_eda(xlsx_path: Path, output_folder: Path) -> None:
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (20, 10)
    df = pd.read_excel(xlsx_path)
    df = df[df["result"] == "solved"].copy()
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1. Softness vs packing ratio
    sns.lineplot(data=df, x="softness", y="packing_ratio", hue="items", marker="o")
    plt.title("Softness vs Packing ratio (cube container)")
    plt.tight_layout()
    plt.savefig(output_folder / "softness_vs_packing_ratio.png")
    plt.close()

    # 2. Solve time by container (boxplot, log‑scale)
    sns.boxplot(data=df, x="container", y="total_solve_time", hue="conservation")
    plt.yscale("log")
    plt.title("Solve‑time distribution by container")
    plt.tight_layout()
    plt.savefig(output_folder / "solve_time_by_container.png")
    plt.close()

    # 3. Heatmap softness × items   (average solve time)
    piv = df.pivot_table(
        values="total_solve_time", index="softness", columns="items", aggfunc="mean"
    )
    sns.heatmap(piv, cmap="YlGnBu", annot=True, fmt=".1f")
    plt.title("Avg solve‑time (s) vs softness × items")
    plt.tight_layout()
    plt.savefig(output_folder / "heatmap_solve_time.png")
    plt.close()

    # 4. One heatmap per (container, conservation)
    combos = df[["container", "conservation"]].drop_duplicates()
    for _, row in combos.iterrows():
        cont, cons = row
        sub = df[(df["container"] == cont) & (df["conservation"] == cons)]
        if sub.empty:
            continue
        piv = sub.pivot_table(
            values="total_solve_time", index="softness", columns="items", aggfunc="mean"
        )
        if piv.empty or piv.isna().all().all():
            continue
        sns.heatmap(piv, cmap="YlGnBu", annot=True, fmt=".1f")
        safe_cont = cont.replace(" ", "_").lower()
        safe_cons = cons.replace(" ", "_").lower()
        plt.title(f"Avg solve‑time\n{cont} | {cons}")
        plt.tight_layout()
        plt.savefig(output_folder / f"heatmap_{safe_cont}_{safe_cons}.png")
        plt.close()

    logging.info("EDA figures saved in %s", output_folder.resolve())


# ======================================================================
#  SECTION 5 ▸ 3D VISUALISATION  (plot_mesh_pal.py merged)
# ======================================================================
def _compute_center(df: pd.DataFrame) -> Tuple[float, float, float]:
    return tuple(df[["x", "y", "z"]].mean().values)


def _build_scene(
    record: Dict[str, Any], df: pd.DataFrame, palette: str = "tab20"
) -> Tuple[pv.Plotter, List[pv.DataSet]]:
    pv.global_theme.smooth_shading = True
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")
    exports: List[pv.DataSet] = []

    # ---------------- container geometry ---------------------------------
    container_type = (record.get("container_type") or "").lower()
    radius = record.get("radius") or record.get("side")
    cube_side = record.get("side")
    height = record.get("height")
    center = _compute_center(df)

    def _add_container(geom: pv.DataSet):
        plotter.add_mesh(geom, color="lightgray", opacity=0.15)
        plotter.add_mesh(geom, style="wireframe", color="black", opacity=0.25)
        exports.append(geom.extract_surface())

    if "sphere" in container_type and radius:
        _add_container(
            pv.Sphere(radius=radius, center=center, theta_resolution=40, phi_resolution=40)
        )
    elif "cube" in container_type and cube_side:
        _add_container(
            pv.Cube(
                center=center,
                x_length=cube_side,
                y_length=cube_side,
                z_length=cube_side,
            )
        )
    elif "cylinder" in container_type and radius and height:
        _add_container(
            pv.Cylinder(
                center=center,
                direction=(0, 0, 1),
                radius=radius,
                height=height,
                resolution=60,
                capping=True,
            )
        )

    # ---------------- palette --------------------------------------------
    unique_tets = sorted(df["tetra"].unique())
    n = len(unique_tets)
    if palette in mpl.colormaps:
        cmap = mpl.colormaps[palette].resampled(n) if hasattr(
            mpl.colormaps[palette], "resampled"
        ) else mpl.colormaps[palette]
        colors = cmap(np.linspace(0, 1, n))
    else:
        logging.warning("Palette %s not found → using tab20", palette)
        cmap = mpl.colormaps["tab20"]
        colors = cmap(np.linspace(0, 1, n))
    color_map = {tid: mcolors.to_hex(colors[i]) for i, tid in enumerate(unique_tets)}

    # ---------------- tetrahedra meshes ----------------------------------
    for tet_id, grp in df.groupby("tetra"):
        verts = grp.sort_values("vertex")[["x", "y", "z"]].values[:4]
        cells = np.hstack([[4, 0, 1, 2, 3]]).astype(np.int64)
        tet = pv.UnstructuredGrid(cells, np.array([10]), verts)  # VTK_TETRA
        plotter.add_mesh(tet, color=color_map[tet_id], show_edges=True)
        exports.append(tet.extract_surface())

    return plotter, exports


def _export_assets(
    job_id: str, plotter: pv.Plotter, surfaces: List[pv.DataSet]
) -> None:
    html_path = HTML_DIR / f"{job_id}.html"
    img_path = IMG_DIR / f"{job_id}.png"
    glb_path = GLTF_DIR / f"{job_id}.glb"
    obj_path = OBJ_DIR / f"{job_id}.obj"
    vtm_path = VTM_DIR / f"{job_id}.vtm"

    plotter.export_html(str(html_path))
    plotter.export_gltf(str(glb_path))

    if surfaces:
        merged = surfaces[0].copy()
        for s in surfaces[1:]:
            merged = merged.merge(s)
        merged.save(str(obj_path))
    pv.MultiBlock(surfaces).save(str(vtm_path))

    plotter.off_screen = True
    plotter.screenshot(str(img_path), window_size=[1400, 1000])

    logging.info("Assets exported → html/png/gltf/obj/vtm for job %s", job_id)


# ======================================================================
#  SECTION 6 ▸ CLI GLUE
# ======================================================================
def cmd_neos_fetch(args: argparse.Namespace) -> None:
    neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333")
    job_pairs = pd.read_excel(args.xlsx, usecols=[args.job_col, args.pass_col]).itertuples(
        index=False, name=None
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for job_id, password in tqdm(job_pairs, desc="Fetching NEOS jobs"):
        status = neos.getJobStatus(int(job_id), password)
        if status != "Done":
            continue
        res_bytes = neos.getFinalResults(int(job_id), password)
        result_text = res_bytes.data.decode("utf-8")
        (outdir / f"job_{job_id}.txt").write_text(result_text, encoding="utf-8")
        parse_and_increment(result_text, str(job_id), password)


def cmd_data_manage(args: argparse.Namespace) -> None:
    data = _dm_load(Path(args.input))
    base = Path(args.base_dir)
    changed = False

    if args.prune:
        removed = _dm_prune_null(data, base)
        logging.info("Deleted %d null entries", len(removed))
        changed = True
    if args.delete:
        delete_ids = _parse_delete_input(args.delete)
        for jid in delete_ids:
            if jid in data:
                del data[jid]
                _dm_remove_job_files(base, jid)
                logging.info("Deleted job %s", jid)
                changed = True
            else:
                logging.warning("Job %s not found in database", jid)
    if args.val:
        n = _dm_validate_neos(data, base)
        logging.info("Removed %d orphaned NEOS artifacts", n)
    if changed:
        _dm_save(Path(args.input), data)

    if args.stats:
        st = _dm_stats(data)
        print("Total entries:", st["total_entries"])
        print("Combination counts (container, conservation):")
        for (c, s), cnt in st["type_combinations"].items():
            print(f"  ({c}, {s}): {cnt}")
    if args.item:
        print(json.dumps(data.get(args.item, {}), indent=2))
    if args.list is not None:
        items = list(data.items())
        res = dict(items[args.list :] if args.list < 0 else items[: args.list])
        print(json.dumps(res, indent=2))


def cmd_json2xlsx(args: argparse.Namespace) -> None:
    json_to_xlsx(Path(args.input), Path(args.output), args.item_vol)


def cmd_eda(args: argparse.Namespace) -> None:
    run_eda(Path(args.xlsx), Path(args.outdir))


def cmd_plot(args: argparse.Namespace) -> None:
    db = _dm_load(RESULTS_JSON)
    if args.job_id not in db:
        logging.error("Job %s not found in %s", args.job_id, RESULTS_JSON)
        sys.exit(1)
    record = db[args.job_id]
    mesh_path = MESH_DIR / f"{args.job_id}.csv"
    if not mesh_path.exists():
        logging.error("Mesh file %s missing", mesh_path)
        sys.exit(1)
    df = pd.read_csv(mesh_path)
    plotter, surfaces = _build_scene(record, df, palette=args.palette)
    _export_assets(args.job_id, plotter, surfaces)
    if args.show:
        plotter_show, _ = _build_scene(record, df, palette=args.palette)
        plotter_show.show(title=f"Job {args.job_id}", window_size=[1400, 1000])


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="o‑manager.py",
        description="All‑in‑one manager for tetrahedra packing experiments.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # neos‑fetch
    s = sub.add_parser("neos-fetch", help="Download & parse NEOS job results")
    s.add_argument("-x", "--xlsx", required=True, type=Path, help="Excel with Job IDs")
    s.add_argument("--job-col", default=0, type=int, help="Zero‑based column of job id")
    s.add_argument(
        "--pass-col", default=1, type=int, help="Zero‑based column of password"
    )
    s.add_argument("-o", "--outdir", default="neos_results", help="Folder for raw txt")
    s.set_defaults(func=cmd_neos_fetch)

    # data
    s = sub.add_parser("data", help="Manage results.json & artifacts")
    s.add_argument("--input", default=RESULTS_JSON, help="Path to results.json")
    s.add_argument("--base-dir", default=".", help="Project base dir")
    s.add_argument("--prune", action="store_true", help="Prune null entries")
    s.add_argument("--stats", action="store_true", help="Show stats")
    s.add_argument("--item", help="Show single job record")
    s.add_argument(
        "--delete",
        metavar="ID|LIST|FILE",
        help=(
            "Remove one job, several (comma‑list or [list]), or all IDs "
            "contained in a text file (use @file or plain path)."
        ),
    )
    s.add_argument("--list", type=int, help="List first (N) or last (-N) jobs")
    s.add_argument("--val", action="store_true", help="Validate orphaned NEOS files")
    s.set_defaults(func=cmd_data_manage)

    # json2xlsx
    s = sub.add_parser("json2xlsx", help="Export results.json → Excel")
    s.add_argument("-i", "--input", default=RESULTS_JSON, help="results.json path")
    s.add_argument("-o", "--output", required=True, type=Path, help="Output .xlsx")
    s.add_argument(
        "-v",
        "--item-vol",
        type=float,
        help="Item volume (adds container/items/packing columns)",
    )
    s.set_defaults(func=cmd_json2xlsx)

    # eda
    s = sub.add_parser("eda", help="Generate EDA figures from Excel")
    s.add_argument("-x", "--xlsx", required=True, type=Path)
    s.add_argument("-o", "--outdir", default="artifacts/figures", type=Path)
    s.set_defaults(func=cmd_eda)

    # plot
    s = sub.add_parser("plot", help="Render 3D mesh & container for one job")
    s.add_argument("-j", "--job-id", required=True, help="Job identifier")
    s.add_argument("-p", "--palette", default="tab20", help="Matplotlib colormap name")
    s.add_argument("--show", action="store_true", help="Open interactive viewer")
    s.set_defaults(func=cmd_plot)

    return p


def main() -> None:
    cli = build_cli()
    args = cli.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
