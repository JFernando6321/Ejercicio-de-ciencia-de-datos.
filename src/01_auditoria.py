"""
Fase 0b — Auditoría profunda: hojas extra, llaves, deriva y leakage.
"""
import pandas as pd, numpy as np, pathlib
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 100)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OUT = ROOT / "outputs" / "tablas"

# ---------------------------------------------------------------- 1. HOJAS EXTRA
print("=" * 90); print("1. HOJAS ADICIONALES DEL ARCHIVO MODELO"); print("=" * 90)
for sh in ["Hoja1 (2)", "Hoja3"]:
    df = pd.read_excel(DATA / "Base de Datos Modelo.xlsx", sheet_name=sh)
    print(f"\n--- {sh}: shape={df.shape} ---")
    print("columnas:", list(df.columns))
    print(df.head(12).to_string())

hist = pd.read_excel(DATA / "Base de Datos Modelo.xlsx", sheet_name="Hoja1")
pred = pd.read_excel(DATA / "Base de Datos Predicción.xlsx")

# ---------------------------------------------------------------- 2. LLAVES
print("\n" + "=" * 90); print("2. ESTRUCTURA DE LLAVES"); print("=" * 90)
print(f"hist: filas={len(hist)}  clientes únicos={hist['CODIGO CLIENTE'].nunique()}  prestamos únicos={hist['CODIGO PRESTAMO'].nunique()}")
print(f"pred: filas={len(pred)}  clientes únicos={pred['CODIGO CLIENTE'].nunique()}  prestamos únicos={pred['CODIGO PRESTAMO'].nunique()}")
print("\n¿CODIGO PRESTAMO == CODIGO CLIENTE * 10?")
print("  hist:", (hist["CODIGO PRESTAMO"] == hist["CODIGO CLIENTE"] * 10).mean())
print("  pred:", (pred["CODIGO PRESTAMO"] == pred["CODIGO CLIENTE"] * 10).mean())
print("\n¿pred['CODIGO PRESTAMO']/10 coincide con clientes del histórico?")
pred_cli_implicito = pred["CODIGO PRESTAMO"] / 10
print("  ¿es entero?:", (pred_cli_implicito % 1 == 0).mean())
inter = set(pred_cli_implicito.astype(int)) & set(hist["CODIGO CLIENTE"])
print(f"  intersección con CODIGO CLIENTE histórico: {len(inter)} de {pred['CODIGO PRESTAMO'].nunique()}")
inter2 = set(pred["CODIGO PRESTAMO"]) & set(hist["CODIGO PRESTAMO"])
print(f"  intersección CODIGO PRESTAMO pred vs hist: {len(inter2)}")
inter3 = set(pred["CODIGO CLIENTE"]) & set(hist["CODIGO CLIENTE"])
print(f"  intersección CODIGO CLIENTE pred vs hist: {len(inter3)}")

# ---------------------------------------------------------------- 3. OBJETIVO
print("\n" + "=" * 90); print("3. VARIABLE OBJETIVO"); print("=" * 90)
print(hist["ESTADO"].value_counts(dropna=False))
print(f"\nTasa de fuga (Cliente Retirado) = {(hist['ESTADO']=='Cliente Retirado').mean():.4f}")

# ---------------------------------------------------------------- 4. CALIDAD CATEGÓRICAS
print("\n" + "=" * 90); print("4. CALIDAD DE CATEGÓRICAS (espacios / mayúsculas)"); print("=" * 90)
for c in ["TIPO DE CREDITO", "SEXO", "ETAPA"]:
    for nom, df in [("hist", hist), ("pred", pred)]:
        raw = df[c].astype(str)
        clean = raw.str.strip().str.upper()
        print(f"{nom:5s} {c:18s} niveles_raw={raw.nunique():3d} niveles_limpios={clean.nunique():3d} -> {sorted(clean.unique())}")

# ---------------------------------------------------------------- 5. DERIVA hist vs pred
print("\n" + "=" * 90); print("5. DERIVA (histórico vs predicción)"); print("=" * 90)

def psi(a, b, bins=10):
    """PSI con cortes por cuantiles del histórico."""
    a = pd.Series(a).dropna(); b = pd.Series(b).dropna()
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3: return np.nan
    qs[0], qs[-1] = -np.inf, np.inf
    pa = pd.cut(a, qs).value_counts(normalize=True, sort=False)
    pb = pd.cut(b, qs).value_counts(normalize=True, sort=False)
    pa, pb = pa.replace(0, 1e-6), pb.replace(0, 1e-6)
    return float(((pa - pb) * np.log(pa / pb)).sum())

num = ["TASA_NOMINAL", "CAPITAL_CONCEDIDO", "SALDO_CAPITAL", "CREDITOS ANTERIORES",
       "REGION", "AGENCIA", "PRODUCTO", "SUBPRODUCTO"]
rows = []
for c in num:
    rows.append({
        "variable": c,
        "media_hist": hist[c].mean(), "media_pred": pred[c].mean(),
        "p50_hist": hist[c].median(), "p50_pred": pred[c].median(),
        "min_hist": hist[c].min(), "min_pred": pred[c].min(),
        "max_hist": hist[c].max(), "max_pred": pred[c].max(),
        "PSI": psi(hist[c], pred[c]),
    })
drift = pd.DataFrame(rows)
print(drift.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
drift.to_csv(OUT / "01_deriva_psi.csv", index=False)

print("\n--- Casos extremos en PREDICCIÓN ---")
print("TASA_NOMINAL == 0 :", (pred["TASA_NOMINAL"] == 0).mean())
print("CAPITAL_CONCEDIDO <= 1 :", (pred["CAPITAL_CONCEDIDO"] <= 1).mean())
print("SALDO_CAPITAL == 0 :", (pred["SALDO_CAPITAL"] == 0).mean())
print("\n--- Casos extremos en HISTÓRICO ---")
print("TASA_NOMINAL == 0 :", (hist["TASA_NOMINAL"] == 0).mean())
print("SALDO_CAPITAL == 0 :", (hist["SALDO_CAPITAL"] == 0).mean())
print("\nSALDO_CAPITAL == 0 por ESTADO:")
print(hist.groupby("ESTADO")["SALDO_CAPITAL"].apply(lambda s: (s == 0).mean()))

print("\n--- Distribución de categóricas: hist vs pred (proporciones) ---")
for c in ["TIPO DE CREDITO", "SEXO", "ETAPA"]:
    a = hist[c].astype(str).str.strip().str.upper().value_counts(normalize=True)
    b = pred[c].astype(str).str.strip().str.upper().value_counts(normalize=True)
    print(f"\n· {c}")
    print(pd.DataFrame({"hist": a, "pred": b}).fillna(0).to_string(float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------- 6. AUC UNIVARIADO (leakage)
print("\n" + "=" * 90); print("6. AUC UNIVARIADO — DETECTOR DE LEAKAGE"); print("=" * 90)
from sklearn.metrics import roc_auc_score
y = (hist["ESTADO"] == "Cliente Retirado").astype(int)
res = []
for c in num:
    s = hist[c].fillna(hist[c].median())
    a = roc_auc_score(y, s)
    res.append({"variable": c, "AUC_univariado": max(a, 1 - a), "direccion": "+" if a > .5 else "-"})
for c in ["TIPO DE CREDITO", "SEXO", "ETAPA"]:
    tasa = hist.groupby(hist[c].astype(str).str.strip().str.upper())["ESTADO"].apply(lambda s: (s == "Cliente Retirado").mean())
    s = hist[c].astype(str).str.strip().str.upper().map(tasa)
    a = roc_auc_score(y, s)
    res.append({"variable": c, "AUC_univariado": max(a, 1 - a), "direccion": "cat"})
res = pd.DataFrame(res).sort_values("AUC_univariado", ascending=False)
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
res.to_csv(OUT / "01_auc_univariado.csv", index=False)

print("\n--- Tasa de fuga por nivel categórico ---")
for c in ["TIPO DE CREDITO", "SEXO", "ETAPA"]:
    g = hist.assign(k=hist[c].astype(str).str.strip().str.upper()).groupby("k").agg(
        n=("ESTADO", "size"), tasa_fuga=("ESTADO", lambda s: (s == "Cliente Retirado").mean()))
    print(f"\n· {c}\n{g.to_string(float_format=lambda x: f'{x:.4f}')}")

# ---------------------------------------------------------------- 7. DUPLICADOS
print("\n" + "=" * 90); print("7. DUPLICADOS"); print("=" * 90)
feat = [c for c in hist.columns if c not in ["CODIGO CLIENTE", "CODIGO PRESTAMO", "ESTADO"]]
print("duplicados exactos (todas las columnas):", hist.duplicated().sum())
print("duplicados por vector de features (sin llaves ni objetivo):", hist.duplicated(subset=feat).sum())
d = hist.groupby(feat, dropna=False)["ESTADO"].nunique()
print("grupos de features idénticos con AMBOS estados (ruido irreducible):", (d > 1).sum())
