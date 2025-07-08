# -*- coding:utf-8 -*-


# Built-in libraries
from pathlib import Path

# Third-party libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Configuración global
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (20, 10)

def load_data(xlsx_path: str) -> pd.DataFrame:
    """Carga los datos desde el archivo XLSX."""
    xls = pd.ExcelFile(xlsx_path)
    return xls.parse(xls.sheet_names[0])


def plot_heatmaps_by_combination(df: pd.DataFrame, output_folder: str):
    """Genera un heatmap para cada combinación de contenedor y conservación."""
    df = df.copy()
    unique_combinations = df[["container_type", "structural_conservation_type"]].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        cont_type = row["container_type"]
        cons_type = row["structural_conservation_type"]

        subset = df[(df["container_type"] == cont_type) &
                    (df["structural_conservation_type"] == cons_type)]

        if subset.empty:
            print(f"⏭️ Combinación vacía: {cont_type} + {cons_type}")
            continue

        pivot = subset.pivot_table(
            values="total_solve_time",
            index="softness",
            columns="card_figures",
            aggfunc="mean"
        )

        if pivot.empty or pivot.isna().all().all():
            print(f"⏭️ Datos insuficientes para: {cont_type} + {cons_type}")
            continue

        plt.figure(figsize=(20, 10))
        sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".1f")
        plt.title(f"Tiempo promedio de resolución\nContenedor: {cont_type} | Conservación: {cons_type}")
        plt.xlabel("Número de tetraedros")
        plt.ylabel("Suavidad")
        plt.tight_layout()

        safe_cont = cont_type.replace(" ", "_").lower()
        safe_cons = cons_type.replace(" ", "_").lower()
        filename = f"heatmap_{safe_cont}_{safe_cons}.png"
        plt.savefig(f"{output_folder}/{filename}")
        plt.close()


def plot_softness_vs_packing(df: pd.DataFrame, output_folder: str):
    """Gráfico: Suavidad vs. Densidad aparente de empaquetado."""
    df = df.copy()
    df["packing_ratio"] = df["tetrahedron_volume_sum"] / df["side"]**3  # Solo si el contenedor es cubo
    df = df[df["container_type"] == "cube"]

    sns.lineplot(data=df, x="softness", y="packing_ratio", hue="card_figures", marker="o")
    plt.title("Relación entre suavidad y tasa de empaquetado (contenedor cúbico)")
    plt.xlabel("Suavidad")
    plt.ylabel("Tasa de empaquetado")
    plt.legend(title="Tetraedros")
    plt.tight_layout()
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_folder}/softness_vs_packing_ratio.png")
    plt.close()


def plot_solve_time_by_container(df: pd.DataFrame, output_folder: str):
    """Boxplot: Tiempo total de resolución por tipo de contenedor."""
    sns.boxplot(data=df, x="container_type", y="total_solve_time", hue="structural_conservation_type")
    plt.yscale("log")
    plt.title("Distribución del tiempo de resolución por contenedor")
    plt.xlabel("Contenedor")
    plt.ylabel("Tiempo total de resolución (s)")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/solve_time_by_container.png")
    plt.close()


def plot_heatmap_softness_vs_figures(df: pd.DataFrame, output_folder: str):
    """Heatmap: Tiempo medio de resolución según suavidad y número de tetraedros."""
    pivot = df.pivot_table(values="total_solve_time", index="softness", columns="card_figures", aggfunc="mean")
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".1f")
    plt.title("Tiempo de resolución promedio (s) según suavidad y número de piezas")
    plt.xlabel("Número de tetraedros")
    plt.ylabel("Suavidad")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/heatmap_solve_time.png")
    plt.close()


# def plot_error_vs_softness(df: pd.DataFrame, output_folder: str):
#     """Gráfico hipotético si se incluye una columna de error de conservación."""
#     if "volume_error" not in df.columns:
#         print("⚠️ La columna 'volume_error' no está presente.")
#         return

#     sns.lineplot(data=df, x="softness", y="volume_error", hue="card_figures", marker="o")
#     plt.title("Error de conservación de volumen vs. Suavidad")
#     plt.xlabel("Suavidad")
#     plt.ylabel("Error relativo")
#     plt.tight_layout()
#     plt.savefig(f"{output_folder}/volume_error_vs_softness.png")
#     plt.close()


def main():
    # Ruta local
    xlsx_path = "artifacts/results.xlsx"
    output_folder = "figures"

    df = load_data(xlsx_path)
    
    df = df[df['solve_result_num'] == 0].copy()

    plot_softness_vs_packing(df, output_folder)
    plot_solve_time_by_container(df, output_folder)
    plot_heatmap_softness_vs_figures(df, output_folder)
    #plot_error_vs_softness(df, output_folder)
    plot_heatmaps_by_combination(df, output_folder)


if __name__ == "__main__":
    main()
