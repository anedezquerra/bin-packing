# =========================================
# 0) Carga y utilidades
# =========================================
import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

import warnings

warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams['xtick.labelsize'] = 16  # Increase x-tick label size
plt.rcParams['ytick.labelsize'] = 16  # Increase y-tick label size
plt.rcParams['axes.labelsize'] = 18   # Increase x-axis and y-axis label size
plt.rcParams['axes.titlesize'] = 18  # Increase title font size, if you have a title
plt.rcParams['axes.titleweight'] = 'bold' 

plt.rcParams['legend.fontsize'] = 16       # Tamaño de la fuente de la leyenda
plt.rcParams['legend.loc'] = 'upper center' # Posición de la leyenda
plt.rcParams['legend.frameon'] = True      # Habilitar borde en la leyenda
plt.rcParams['legend.framealpha'] = 0.9    # Transparencia del fondo de la leyenda
plt.rcParams['legend.facecolor'] = 'white' # Color de fondo de la leyenda
plt.rcParams['legend.edgecolor'] = 'black' # Color del borde de la leyenda

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Set a default available font
plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Set a default available font


pd.options.display.float_format = '{:.2f}'.format


DATA_PATH = "data/results.xlsx"  # ajusta si es necesario
OUT_DIR = "images"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    # Filtramos resultados válidos para métricas (puedes cambiar según tu criterio)
    dval = df[df["valid_result"] == True].copy()
    # Limpiezas ligeras
    for col in ["packing_ratio", "items", "softness", "total_solve_time"]:
        dval = dval[~dval[col].isna()]
    # Asegurar categorías ordenadas si quieres
    dval["container"] = dval["container"].astype("category")
    dval["conservation"] = dval["conservation"].astype("category")
    return df, dval

def ci95(series: pd.Series):
    """IC 95% para la media (t de Student)."""
    x = series.dropna().astype(float)
    n = x.shape[0]
    if n < 2:
        return (np.nan, np.nan)
    m = x.mean()
    se = x.std(ddof=1) / math.sqrt(n)
    tval = stats.t.ppf(0.975, df=n-1)
    return (m - tval*se, m + tval*se)


df_raw, dfv = load_data()

# =========================================
# 1) Cobertura y calidad experimental
# =========================================
# Tabla de cobertura por container × conservation
cov = (df_raw
       .groupby(["container","conservation"], dropna=False)
       .agg(n=("job_id","count"),
            valid=("valid_result", lambda s: s.fillna(False).sum()))
       .reset_index())
cov["valid_pct"] = 100*cov["valid"]/cov["n"]

print("\nCobertura por contenedor × regla:\n", cov)

# Histograma de items (una figura)
plt.figure()
df_raw["items"].dropna().plot(kind="hist", bins=30)
plt.title("Distribución de items (tamaño del problema)")
plt.xlabel("items")
plt.ylabel("frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_items.png"), dpi=200)

# Histograma de softness (una figura)
plt.figure()
df_raw["softness"].dropna().plot(kind="hist", bins=30)
plt.title("Distribución de softness")
plt.xlabel("softness")
plt.ylabel("frecuencia")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hist_softness.png"), dpi=200)

# =========================================
# 2) % empaquetado por contenedor y regla
#    (barras con media + IC95%) + boxplots
# =========================================
g = (dfv
     .groupby(["container","conservation"])
     .agg(n=("packing_ratio","count"),
          mean=("packing_ratio","mean"))
     .reset_index())

# Calcular IC95% por grupo
ci_lo, ci_hi = [], []
for (c, r), sub in dfv.groupby(["container","conservation"]):
    lo, hi = ci95(sub["packing_ratio"])
    ci_lo.append(lo); ci_hi.append(hi)
g["ci_lo"], g["ci_hi"] = ci_lo, ci_hi
g["err_lower"] = g["mean"] - g["ci_lo"]
g["err_upper"] = g["ci_hi"] - g["mean"]

# Barras con errores (una figura)
labels = [f"{c}\n{r}" for c,r in zip(g["container"], g["conservation"])]
plt.figure()
plt.bar(range(len(g)), g["mean"], yerr=[g["err_lower"], g["err_upper"]])
plt.xticks(range(len(g)), labels, rotation=90)
plt.ylabel("packing_ratio (media ± IC95%)")
plt.title("% empaquetado por contenedor × regla")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "bar_packing_by_group.png"), dpi=200)

# Boxplots por grupo (una figura)
data = []
labels = []
for (c, r), sub in dfv.groupby(["container","conservation"]):
    data.append(sub["packing_ratio"].values)
    labels.append(f"{c}\n{r}")

plt.figure()
plt.boxplot(data, tick_labels=labels, showmeans=True)
plt.xticks(range(len(g)), labels, rotation=90)
plt.ylabel("packing_ratio")
plt.title("Distribución de packing_ratio por contenedor × regla")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "box_packing_by_group.png"), dpi=200)


# =========================================
# 3) Tendencias vs items y vs softness
#    (dispersión + mediana rodante)
# =========================================
def rolling_xy(x, y, window=15):
    """Curva suavizada por mediana rodante en el eje X (ordenando por X)."""
    d = pd.DataFrame({"x": x, "y": y}).sort_values("x")
    if len(d) < window: 
        return d["x"].values, d["y"].values
    med = d["y"].rolling(window, center=True).median()
    return d["x"].values, med.values

# packing_ratio vs items, una figura por contenedor (tres gráficas = tres archivos)
for c in dfv["container"].cat.categories:
    sub = dfv[dfv["container"]==c]
    plt.figure()
    plt.scatter(sub["items"], sub["packing_ratio"], alpha=0.6)
    rx, ry = rolling_xy(sub["items"], sub["packing_ratio"], window=21)
    plt.plot(rx, ry)
    plt.xlabel("items")
    plt.ylabel("packing_ratio")
    plt.title(f"packing_ratio vs items — {c}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"trend_items_{c}.png"), dpi=200)

# packing_ratio vs softness (una figura por contenedor)
for c in dfv["container"].cat.categories:
    sub = dfv[dfv["container"]==c]
    plt.figure()
    plt.scatter(sub["softness"], sub["packing_ratio"], alpha=0.6)
    rx, ry = rolling_xy(sub["softness"], sub["packing_ratio"], window=21)
    plt.plot(rx, ry)
    plt.xlabel("softness")
    plt.ylabel("packing_ratio")
    plt.title(f"packing_ratio vs softness — {c}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"trend_softness_{c}.png"), dpi=200)


# =========================================
# 4) Escalamiento de tiempo de cómputo
#    (log–log + ajuste ley de potencias)
# =========================================
def power_fit(x, y):
    """Ajusta y = a * x^b en log–log. Retorna (a, b)."""
    x = np.asarray(x); y = np.asarray(y)
    m = (x > 0) & (y > 0)
    X = np.log(x[m]); Y = np.log(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    b, loga = np.linalg.lstsq(A, Y, rcond=None)[0]  # Y = b*X + log(a)
    return float(np.exp(loga)), float(b)

for c in dfv["container"].cat.categories:
    sub = dfv[(dfv["container"]==c) & (dfv["total_solve_time"]>0) & (dfv["items"]>0)]
    a, b = power_fit(sub["items"], sub["total_solve_time"])
    plt.figure()
    # Dispersión en log–log
    plt.loglog(sub["items"], sub["total_solve_time"], linestyle="", marker="o", alpha=0.6)
    # Curva ajustada
    xs = np.linspace(sub["items"].min(), sub["items"].max(), 200)
    ys = a * (xs ** b)
    plt.loglog(xs, ys)
    plt.xlabel("items (log)")
    plt.ylabel("total_solve_time (log)")
    plt.title(f"Escalamiento tiempo vs items — {c} (t ≈ {a:.3g}·items^{b:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"time_scaling_{c}.png"), dpi=200)


# =========================================
# 5) ANOVA de dos vías + supuestos
# =========================================
# Modelo lineal: packing_ratio ~ container * conservation
mod = smf.ols("packing_ratio ~ C(container) * C(conservation)", data=dfv).fit()
anova_tbl = sm.stats.anova_lm(mod, typ=2)
print("\nANOVA (dos vías):\n", anova_tbl)

# Supuestos
resid = mod.resid
# Shapiro-Wilk en muestra (para evitar sobre-sensibilidad con n grande)
sample = resid.sample(n=min(500, len(resid)), random_state=0)
W, p_shapiro = stats.shapiro(sample)
print(f"\nShapiro–Wilk residuales: W={W:.3f}, p={p_shapiro:.3g}")

# Levene (homogeneidad de varianzas) por contenedor
lev_stat, lev_p = stats.levene(*[dfv[dfv["container"]==c]["packing_ratio"].values 
                                 for c in dfv["container"].cat.categories])
print(f"Levene por contenedor: stat={lev_stat:.2f}, p={lev_p:.3g}")

# QQ-plot residuales (una figura)
plt.figure()
sm.ProbPlot(resid).qqplot(line="45")
plt.title("QQ-plot residuales (modelo lineal)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "qqplot_residuales.png"), dpi=200)

# =========================================
# 6) Alternativa no paramétrica:
#    Kruskal–Wallis + post hoc (Mann–Whitney con Holm)
# =========================================
# Definimos grupos como container × conservation
dfv["grp"] = dfv["container"].astype(str) + " | " + dfv["conservation"].astype(str)
groups = {k: v["packing_ratio"].values for k, v in dfv.groupby("grp")}

# Kruskal global
kw_stat, kw_p = stats.kruskal(*groups.values())
print(f"\nKruskal–Wallis global: H={kw_stat:.2f}, p={kw_p:.3g}")

# Post hoc pareado (Holm)
def mannwhitney_holm(groups_dict):
    keys = list(groups_dict.keys())
    comps = list(itertools.combinations(keys, 2))
    results = []
    for a, b in comps:
        u_stat, p = stats.mannwhitneyu(groups_dict[a], groups_dict[b], alternative="two-sided")
        n1, n2 = len(groups_dict[a]), len(groups_dict[b])
        # Tamaño de efecto (correlación biserial por rangos aproximada)
        U = u_stat
        r_rb = 1 - (2*U)/(n1*n2)
        results.append({"A": a, "B": b, "p_raw": p, "r_rank_biserial": r_rb})
    # Corrección Holm
    m = len(results)
    results_sorted = sorted(results, key=lambda d: d["p_raw"])
    for i, d in enumerate(results_sorted):
        d["p_holm"] = min(1.0, d["p_raw"] * (m - i))
    # Volver al orden original de comparación
    return pd.DataFrame(results_sorted)

posthoc = mannwhitney_holm(groups)
print("\nPost hoc (Mann–Whitney + Holm) — top 15 por p_holm:\n",
      posthoc.sort_values("p_holm").head(15))
# Guarda tabla completa
posthoc.to_csv(os.path.join(OUT_DIR, "posthoc_mannwhitney_holm.csv"), index=False)

# =========================================
# 7) Modelo explicativo opcional (lineal)
# =========================================
mod2 = smf.ols("packing_ratio ~ C(container) * C(conservation) + items + softness", data=dfv).fit()
print("\nModelo explicativo (OLS):\n", mod2.summary())

# (Opcional) Logístico para 'valid_result' (usa df_raw)
try:
    dglm = df_raw.copy()
    dglm["is_valid"] = dglm["valid_result"].astype(int)
    dglm = dglm.dropna(subset=["items","softness"])
    glm = smf.glm("is_valid ~ C(container) * C(conservation) + items + softness",
                  data=dglm, family=sm.families.Binomial()).fit()
    print("\nGLM Binomial (valid_result):\n", glm.summary())
except Exception as e:
    print("GLM opcional no ejecutado:", e)


# =========================================
# 8) Exportables para LaTeX
# =========================================
# Resumen por contenedor × regla
summary = (dfv.groupby(["container","conservation"])
           .agg(n=("packing_ratio","count"),
                mean=("packing_ratio","mean"),
                median=("packing_ratio","median"),
                sd=("packing_ratio","std"))
           .reset_index())
summary["mean"] = summary["mean"].round(3)
summary["median"] = summary["median"].round(3)
summary["sd"] = summary["sd"].round(3)

# CSV y LaTeX
summary.to_csv(os.path.join(OUT_DIR, "summary_packing.csv"), index=False)
with open(os.path.join(OUT_DIR, "summary_packing.tex"), "w", encoding="utf-8") as f:
    f.write(summary.to_latex(index=False, caption="Resumen de packing por contenedor y regla",
                             label="tab:summary_packing"))
print("\nTablas exportadas en 'figs/': summary_packing.csv y summary_packing.tex")




