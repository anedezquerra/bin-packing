# ---------------------------------------------------------------------------
#  UTILIDAD: Parseo de resultados AMPL/NEOS  ➜  JSON incremental + malla CSV
# ---------------------------------------------------------------------------
#  - Estructura JSON:  artifacts/results.json  (dict indexado por job_id)
#  - Mallas:           artifacts/meshes/<job_id>.csv
#  - Re-ejecuciones:   si el job_id ya existe en el JSON **no** se sobre-escribe
#
#  Requisitos:  pandas  (pip install pandas)
# ---------------------------------------------------------------------------


# -*- coding:utf-8 -*-

from __future__ import annotations
import xmlrpc.client
import json, os, re, pathlib, logging
from typing import Any, Dict, List, Tuple

import datetime
from dateutil import parser as date_parser

import pandas as pd


ARTIFACTS_DIR      = pathlib.Path("artifacts")
RESULTS_JSON_PATH  = ARTIFACTS_DIR / "results.json"
MESH_FOLDER        = ARTIFACTS_DIR / "meshes"
MESH_FOLDER.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)


# ----------  API principal  -------------------------------------------------
def parse_and_increment(text: str, job_id: int | str, password: str) -> None:
    """
    Extrae los campos requeridos, actualiza results.json (keyed por job_id)
    y genera artifacts/meshes/<job_id>.csv si existen coords.

    La función es idempotente: si el job_id ya está registrado, no lo re-graba.
    """
    job_id = str(job_id)

    # ------------------------------------------------------------------ JSON
    record: Dict[str, Any] = {
        "job_id": job_id,
        "password": password,    
        "processing_datetime": datetime.datetime.now().isoformat(),
        # se almacena tal cual (dato no sensible)
        **_parse_scalar_fields(text),    # solver, tiempos, radio, etc.
        "tetrahedron_volume_sum": _parse_tetra_volume_sum(text),
    }

    # ------------------------------------------------------------------ Mesh
    mesh_df = _parse_coords_to_dataframe(text)
    if not mesh_df.empty:
        mesh_path = MESH_FOLDER / f"{job_id}.csv"
        mesh_df.to_csv(mesh_path, index=False)
        record["mesh_file"] = str(mesh_path)
    else:
        record["mesh_file"] = None

    # ----------------------------------------------------------- Persistencia
    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_JSON_PATH.exists():
        try:
            with RESULTS_JSON_PATH.open(encoding="utf-8") as fh:
                db = json.load(fh)
        except json.JSONDecodeError:
            logging.warning("results.json estaba corrupto; se reinicia.")
            db = {}
    else:
        db = {}

    if job_id in db:
        logging.info("Job %s ya estaba en results.json → se omite.", job_id)
        return

    db[job_id] = record
    with RESULTS_JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=2)
    logging.info("Añadido job %s a %s", job_id, RESULTS_JSON_PATH)


# ----------  Parsing de campos simples  ------------------------------------
_REGEXES: dict[str, str] = {
    "container_type":               r"Container type:\s*([^\n\r]+)",
    "structural_conservation_type": r"Structural conservation type:\s*([^\n\r]+)",
    "solve_result_num":             r"solve_result_num\s*=\s*([^\s]+)",
    "solve_result":                 r"solve_result\s*=\s*([^\s]+)",
    "card_figures":                 r"card\(figures\)\s*=\s*([^\s]+)",
    "radius":                       r"radius\s*=\s*([^\s]+)",
    "side":                         r"side\s*=\s*([^\s]+)",
    "height":                       r"height\s*=\s*([^\s]+)",
    "softness":                     r"softness\s*=\s*([^\s]+)",
    "ampl_time":                    r"_ampl_time\s*=\s*([^\s]+)",
    "total_solve_time":             r"_total_solve_time\s*=\s*([^\s]+)",
    "ampl_elapsed_time":            r"_ampl_elapsed_time\s*=\s*([^\s]+)",
    "ampl_user_time":               r"_ampl_user_time\s*=\s*([^\s]+)",
    "total_time_elapsed":           r"Total time elapsed:\s*\$?([0-9.+\-Ee]+\.?)",
}

_SOLVERS = ("BARON", "Knitro", "IPOPT")


def _parse_scalar_fields(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"solver": None}

    # --- Solvers: primer match
    for s in _SOLVERS:
        if re.search(rf"\b{s}\b", text, re.I):
            out["solver"] = s
            break

    # --- Resto de campos
    for key, pattern in _REGEXES.items():
        m = re.search(pattern, text, re.I)
        if not m:
            out[key] = None
            continue

        val = m.group(1).strip()
        if key in {"solve_result",
                   "container_type",
                   "structural_conservation_type"}:
            out[key] = val
        else:  # numéricos
            val = val.rstrip(".")  # quita punto sobrante si acaba en "."
            try:
                num = float(val)
                out[key] = int(num) if num.is_integer() else num
            except ValueError:
                out[key] = None
    return out


# ----------  Tetrahedron_volume --------------------------------------------
def _parse_tetra_volume_sum(text: str) -> float | None:
    """
    Suma todos los valores (excluyendo los índices) del bloque:
      tetrahedron_volume [*] :=
          1 0.117851   2 0.117851 ...
      ;
    """
    m = re.search(
        r"tetrahedron_volume\s*\[\*\]\s*:=\s*(.*?)\s*;",
        text, re.S | re.I
    )
    if not m:
        return None

    tokens = re.findall(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", m.group(1))
    # pares: índice, valor —> nos quedamos con los valores (1,3,5…)
    total = sum(float(tokens[i]) for i in range(1, len(tokens), 2))
    return total


# ----------  Coords ➜ DataFrame  -------------------------------------------
def _parse_coords_to_dataframe(text: str) -> pd.DataFrame:
    """
    Extrae coords[*,*,1-3] → DataFrame con columnas (tetra, vertex, x, y, z).
    Ya **NO** se añade el primer vértice como último.
    """
    lines = text.splitlines()
    dim_maps = {1: [], 2: [], 3: []}
    dim = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if re.match(r"\s*coords\s*\[\*,\*,1\]", ln):
            dim = 1; i += 2; continue
        if dim and re.match(r"\s*\[\*,\*,2\]", ln):
            dim = 2; i += 2; continue
        if dim and re.match(r"\s*\[\*,\*,3\]", ln):
            dim = 3; i += 2; continue
        if dim and ln.strip().startswith(";"):
            dim = None
        elif dim and re.match(r"\s*\d", ln):
            parts = ln.strip().split()
            dim_maps[dim].append((int(parts[0]), list(map(float, parts[1:]))))
        i += 1

    if not (dim_maps[1] and dim_maps[2] and dim_maps[3]):
        return pd.DataFrame()

    x_dict = {r: v for r, v in dim_maps[1]}
    y_dict = {r: v for r, v in dim_maps[2]}
    z_dict = {r: v for r, v in dim_maps[3]}

    rows = []
    for tetra_id, x_vals in x_dict.items():
        y_vals, z_vals = y_dict[tetra_id], z_dict[tetra_id]
        for v_idx, (x, y, z) in enumerate(zip(x_vals, y_vals, z_vals), start=1):
            rows.append(
                {"tetra": tetra_id, "vertex": v_idx, "x": x, "y": y, "z": z}
            )
    return pd.DataFrame(rows)


def read_xlsx_columns(file_path: str, columns: List[str]) -> List[Tuple]:
    """
    Reads an Excel file and returns a list of tuples containing the values
    of the specified columns.

    Parameters:
    - file_path (str): Path to the Excel file.
    - columns (List[str]): List of column names to extract.

    Returns:
    - List[Tuple]: Each tuple contains values from the specified columns.
    """
    try:
        df = pd.read_excel(file_path, usecols=columns)
        return list(df.itertuples(index=False, name=None))
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def generate_summary_excel():
    """Generates a summary Excel file from the results.json database"""
    if not RESULTS_JSON_PATH.exists():
        logging.error("No results.json found. Cannot generate summary.")
        return

    with open(RESULTS_JSON_PATH, 'r', encoding='utf-8') as f:
        db = json.load(f)

    # Prepare dataframe with enhanced columns
    records = []
    for job_id, data in db.items():
        record = data.copy()
        
        # Handle container type safely
        container_type = record.get('container_type')
        if container_type is not None:
            container_type = container_type.lower()
        else:
            container_type = "unknown"
        
        # Calculate container volume
        container_volume = None
        if container_type == 'cylinder':
            radius = record.get('radius', 0)
            height = record.get('height', 0)
            if radius and height:
                container_volume = 3.1415926535 * (radius ** 2) * height
        elif container_type == 'prism':
            side = record.get('side', 0)
            height = record.get('height', 0)
            if side and height:
                container_volume = (side ** 2) * height
        
        record['container_volume'] = container_volume
        
        # Calculate packing percentage
        tetra_sum = record.get('tetrahedron_volume_sum')
        if container_volume and tetra_sum:
            record['packing_percentage'] = (tetra_sum / container_volume) * 100
        else:
            record['packing_percentage'] = None
        
        # Add job ID as column
        record['job_id'] = job_id
        records.append(record)

    if not records:
        logging.warning("No records found for summary Excel")
        return

    df = pd.DataFrame(records)
    
    # Generate filename with current datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
    excel_filename = f"preprocessing_{now}.xlsx"
    
    # Select and order columns
    columns = [
        'job_id', 'processing_datetime', 'solver', 'container_type',
        'container_volume', 'tetrahedron_volume_sum', 'packing_percentage',
        'radius', 'side', 'height', 'softness', 'solve_result',
        'ampl_elapsed_time', 'total_solve_time', 'password'
    ]
    
    # Ensure only existing columns are included
    available_cols = [col for col in columns if col in df.columns]
    missing_cols = set(columns) - set(available_cols)
    if missing_cols:
        logging.warning("Missing columns in data: %s", ", ".join(missing_cols))
    
    # Write to Excel
    df[available_cols].to_excel(excel_filename, index=False)
    logging.info("Summary Excel generated: %s", excel_filename)


def main():
    neos = xmlrpc.client.ServerProxy("https://neos-server.org:3333")
    job_list = read_xlsx_columns('all-done.xlsx', ['Job number', 'password'])
    
    # Create folder for raw results
    raw_output_folder = "neos_results"
    os.makedirs(raw_output_folder, exist_ok=True)
    
    # Load already processed jobs
    processed_jobs = set()
    if RESULTS_JSON_PATH.exists():
        try:
            with open(RESULTS_JSON_PATH, 'r', encoding='utf-8') as f:
                db = json.load(f)
                processed_jobs = set(db.keys())
        except json.JSONDecodeError:
            logging.warning("Corrupt results.json - will be overwritten")

    # Process jobs
    for job_number, job_password in job_list:
        job_id = str(job_number)
        
        if job_id in processed_jobs:
            logging.info("Skipping job %s (already processed)", job_id)
            continue

        try:
            logging.info("Processing job %s...", job_id)
            
            # 1. Check NEOS status
            status = neos.getJobStatus(job_number, job_password)
            if status != "Done":
                logging.warning("Job %s not completed (status: %s)", job_id, status)
                continue

            # 2. Get results
            result_bytes = neos.getFinalResults(job_number, job_password)
            result_text = result_bytes.data.decode("utf-8")
            
            # 3. Save raw results
            raw_path = os.path.join(raw_output_folder, f"job_{job_id}.txt")
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(result_text)
            
            # 4. Parse and update artifacts (with datetime)
            parse_and_increment(result_text, job_id, job_password)
            
        except xmlrpc.client.Fault as e:
            logging.error("NEOS error for job %s: %s", job_id, e.faultString)
        except Exception as e:
            logging.exception("Unexpected error for job %s", job_id)
    
    # Generate summary Excel after all processing
    generate_summary_excel()



if __name__ == '__main__':
    main()
