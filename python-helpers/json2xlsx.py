#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
from typing import Union

def json_to_xlsx(input_path: Union[str, Path],
                 output_path: Union[str, Path]) -> None:
    input_path, output_path = Path(input_path), Path(output_path)

    with input_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    records = list(data.values()) if isinstance(data, dict) else data
    df = pd.DataFrame.from_records(records)
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
