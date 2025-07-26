#-*- coding:utf-8 -*-

# Built-in libraries
import argparse
import json
from pathlib import Path
from typing import Union
import logging
import math

# Third-party libraries
import numpy as np
import pandas as pd


def json_to_xlsx(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    item_vol: float
) -> None:
    
    input_path, output_path = Path(input_path), Path(output_path)

    with input_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    records = list(data.values()) if isinstance(data, dict) else data
    df = pd.DataFrame.from_records(records)
    
    # Calculate container volume based on container type
    df['container_volume'] = np.nan
    
    # Cylinder: V = πr²h
    cylinder_mask = df['container_type'] == 'cylinder'
    df.loc[cylinder_mask, 'container_volume'] = (
        math.pi * 
        df.loc[cylinder_mask, 'radius']**2 * 
        df.loc[cylinder_mask, 'height']
    )
    
    # Sphere: V = (4/3)πr³
    sphere_mask = df['container_type'] == 'sphere'
    df.loc[sphere_mask, 'container_volume'] = (
        (4/3) * 
        math.pi * 
        df.loc[sphere_mask, 'radius']**3
    )
    
    # Cube: V = side³
    cube_mask = df['container_type'] == 'cube'
    df.loc[cube_mask, 'container_volume'] = (
        df.loc[cube_mask, 'side']**3
    )
    
    # Calculate items volume
    df['items_volume'] = df['card_figures'] * item_vol
    
    # Calculate packing ratio with zero division handling
    df['packing_ratio'] = np.where(
        df['container_volume'] > 0,
        df['items_volume'] / df['container_volume'],
        np.nan
    )
    
    
    df['valid_result'] = np.where(
        df['packing_ratio'] > 1,
        False,
        True
    )    
    
    # Select and rename columns
    df = df[
        [
            'job_id',    
            'solver',    
            'container_type',    
            'structural_conservation_type',    
            'solve_result',    
            'card_figures',    
            'radius',    
            'side',    
            'height',    
            'softness',    
            'ampl_time',
            'total_solve_time',    
            'ampl_elapsed_time',    
            'ampl_user_time',
            'container_volume',
            'items_volume',
            'packing_ratio',
            'valid_result'
        ]
    ]
    
    # Filter solved cases
    df = df[df['solve_result'] == 'solved'].copy()
    
    # Rename columns
    df = df.rename(
        columns={
           'container_type': 'container',
           'structural_conservation_type': 'conservation',
           'solve_result': 'result',
           'card_figures': 'items' 
        }
    )
    
    # Sort data
    df = df.sort_values(
        by=[
            'container',
            'conservation',
            'softness',
            'items'
        ],
        ascending=[
            True,
            True,
            False,
            False
        ]
    )
    
    # Remove duplicates
    df = df.drop_duplicates(
        subset=[
            'container',
            'conservation',
            'softness',
            'items'             
        ]
    )
    
    # Handle missing values
    df = df.fillna('N/A')
    
    # Save to Excel
    df.to_excel(output_path, index=False)
    logging.info(f"✔️ Excel generado: {output_path.resolve()}")

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="JSON → XLSX (una fila por registro)")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Archivo JSON")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Archivo XLSX")
    parser.add_argument("-v", "--item-vol", type=float, required=True, help="Volumen de un item individual")
    args = parser.parse_args()
    
    try:
        json_to_xlsx(args.input, args.output, args.item_vol)
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    
# python toxlsx.py --input artifacts/results.json --output artifacts/results.xlsx -v 0.11785113020