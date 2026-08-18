"""
Fase 0 — Reconocimiento de datos.
Objetivo: entender estructura, llaves, objetivo, esquemas y compatibilidad
histórico vs predicción ANTES de cualquier modelado.
Salida: outputs/tablas/00_*.csv y un resumen por consola.
"""
import pandas as pd, numpy as np, json, pathlib

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "tablas"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("HOJAS DISPONIBLES")
print("=" * 90)
for f in ["Base de Datos Modelo.xlsx", "Base de Datos Predicción.xlsx"]:
    xl = pd.ExcelFile(DATA / f)
    print(f"{f}: {xl.sheet_names}")

hist = pd.read_excel(DATA / "Base de Datos Modelo.xlsx")
pred = pd.read_excel(DATA / "Base de Datos Predicción.xlsx")

print("\n" + "=" * 90)
print(f"HISTÓRICO: {hist.shape}   |   PREDICCIÓN: {pred.shape}")
print("=" * 90)

print("\n--- COLUMNAS HISTÓRICO ---")
for i, c in enumerate(hist.columns):
    print(f"{i:3d} {c!r:45s} {str(hist[c].dtype):12s} nulos={hist[c].isna().mean():6.2%} nunique={hist[c].nunique()}")

print("\n--- COLUMNAS PREDICCIÓN ---")
for i, c in enumerate(pred.columns):
    print(f"{i:3d} {c!r:45s} {str(pred[c].dtype):12s} nulos={pred[c].isna().mean():6.2%} nunique={pred[c].nunique()}")

print("\n--- COMPATIBILIDAD DE ESQUEMAS ---")
solo_hist = [c for c in hist.columns if c not in pred.columns]
solo_pred = [c for c in pred.columns if c not in hist.columns]
print(f"Solo en histórico (candidatas a objetivo / inutilizables): {solo_hist}")
print(f"Solo en predicción: {solo_pred}")

print("\n--- MUESTRA HISTÓRICO ---")
print(hist.head(8).to_string())
print("\n--- MUESTRA PREDICCIÓN ---")
print(pred.head(8).to_string())

print("\n--- DESCRIPTIVOS NUMÉRICOS (histórico) ---")
print(hist.describe(include=[np.number]).T.to_string())

print("\n--- CATEGÓRICAS (histórico, top 12 niveles) ---")
for c in hist.select_dtypes(include=["object", "category"]).columns:
    vc = hist[c].value_counts(dropna=False)
    print(f"\n· {c}  (nunique={hist[c].nunique()})")
    print(vc.head(12).to_string())

# Guardar diccionario de datos preliminar
dic = pd.DataFrame({
    "columna": hist.columns,
    "dtype": [str(hist[c].dtype) for c in hist.columns],
    "pct_nulos": [hist[c].isna().mean() for c in hist.columns],
    "nunique": [hist[c].nunique() for c in hist.columns],
    "en_prediccion": [c in pred.columns for c in hist.columns],
    "ejemplo": [hist[c].dropna().iloc[0] if hist[c].notna().any() else None for c in hist.columns],
})
dic.to_csv(OUT / "00_diccionario_preliminar.csv", index=False)
print(f"\n[OK] Diccionario preliminar -> {OUT/'00_diccionario_preliminar.csv'}")
