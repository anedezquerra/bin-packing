#-*- coding:utf-8 -*-

# Built-in libraries
import argparse
import json
from pathlib import Path
from typing import Union
import logging

# Third-party libraries
import numpy as np
import pandas as pd


def json_to_xlsx(
    input_path: Union[str, Path],
    output_path: Union[str, Path]
) -> None:
    
    input_path, output_path = Path(input_path), Path(output_path)

    with input_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    records = list(data.values()) if isinstance(data, dict) else data
    df = pd.DataFrame.from_records(records)
    
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
            'ampl_user_time'
        ]
    ]
    
    df = df[df['solve_result'] == 'solved'].copy()
    
    df = df.rename(
        columns={
           'container_type': 'container',
           'structural_conservation_type': 'conservation',
           'solve_result': 'result',
           'card_figures': 'items' 
        }
    )
    
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
    
    df = df.drop_duplicates(
        subset=[
            'container',
            'conservation',
            'softness',
            'items'             
        ]
    )
    
    df = df.fillna(np.nan)
    
    df.to_excel(output_path, index=False)
    print(f"✔️ Excel generado: {output_path.resolve()}")

def main() -> None:
    parser = argparse.ArgumentParser(description="JSON → XLSX (una fila por registro)")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Archivo JSON")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Archivo XLSX")
    args = parser.parse_args()
    json_to_xlsx(args.input, args.output)

if __name__ == "__main__":
    main()


#