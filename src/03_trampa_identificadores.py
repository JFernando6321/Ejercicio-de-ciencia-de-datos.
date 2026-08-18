"""
Fase 0d — Confirmación forense de las dos trampas de fuga de información:
 (T1) los identificadores están ordenados por la variable objetivo
 (T2) CREDITOS ANTERIORES == 1 determina el objetivo en el histórico
"""
import pandas as pd, numpy as np, pathlib
from sklearn.metrics import roc_auc_score
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 100)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
hist = pd.read_excel(DATA / "Base de Datos Modelo.xlsx", sheet_name="Hoja1")
pred = pd.read_excel(DATA / "Base de Datos Predicción.xlsx")
hist["y"] = (hist["ESTADO"] == "Cliente Retirado").astype(int)

print("=" * 90); print("T1. IDENTIFICADORES ORDENADOS POR EL OBJETIVO"); print("=" * 90)
g = hist.groupby("ESTADO")["CODIGO CLIENTE"].agg(["count", "min", "max"])
print(g.to_string())
corte = hist.loc[hist.y == 1, "CODIGO CLIENTE"].max()
print(f"\nMáximo CODIGO CLIENTE entre RETIRADOS  = {corte}")
print(f"Mínimo CODIGO CLIENTE entre RENOVADOS  = {hist.loc[hist.y==0,'CODIGO CLIENTE'].min()}")
print(f"¿Los rangos son disjuntos? -> {hist.loc[hist.y==1,'CODIGO CLIENTE'].max() < hist.loc[hist.y==0,'CODIGO CLIENTE'].min()}")
print(f"\nRegla trivial 'CODIGO CLIENTE <= {corte} => RETIRADO' acierta: "
      f"{((hist['CODIGO CLIENTE'] <= corte).astype(int) == hist.y).mean():.6f}")
print("\n=> El identificador NO es un dato del cliente: es el resultado de haber\n"
      "   ordenado el archivo por ESTADO antes de numerarlo. Usarlo da AUC=1.0 en\n"
      "   entrenamiento y ruido puro en la base de predicción (renumerada 1..10,000).")

print("\n--- Verificación en la base de predicción ---")
print(f"CODIGO CLIENTE pred: 1..10000 correlativo -> ¿ordenado por algo real? "
      f"corr con SALDO_CAPITAL = {pred['CODIGO CLIENTE'].corr(pred['SALDO_CAPITAL']):.4f}")
print(f"corr(CODIGO PRESTAMO, SALDO_CAPITAL) en pred = {pred['CODIGO PRESTAMO'].corr(pred['SALDO_CAPITAL']):.4f}")
print(f"corr(CODIGO PRESTAMO, SALDO_CAPITAL) en hist = {hist['CODIGO PRESTAMO'].corr(hist['SALDO_CAPITAL']):.4f}")

print("\n" + "=" * 90); print("T2. CREDITOS ANTERIORES == 1"); print("=" * 90)
t = hist.groupby(hist["CREDITOS ANTERIORES"] == 1).agg(
    n=("y", "size"), tasa_fuga=("y", "mean"))
t.index = ["CREDITOS ANTERIORES > 1", "CREDITOS ANTERIORES == 1"]
print(t.to_string(float_format=lambda x: f"{x:.6f}"))
print(f"\nEn PREDICCIÓN: % con CREDITOS ANTERIORES == 1 -> {(pred['CREDITOS ANTERIORES']==1).mean():.4%} "
      f"(n={(pred['CREDITOS ANTERIORES']==1).sum()})")
print(f"En HISTÓRICO : % con CREDITOS ANTERIORES == 1 -> {(hist['CREDITOS ANTERIORES']==1).mean():.4%} "
      f"(n={(hist['CREDITOS ANTERIORES']==1).sum()})")

print("\nDistribución CREDITOS ANTERIORES (proporciones):")
print(pd.DataFrame({
    "hist": hist["CREDITOS ANTERIORES"].clip(upper=10).value_counts(normalize=True).sort_index(),
    "pred": pred["CREDITOS ANTERIORES"].clip(upper=10).value_counts(normalize=True).sort_index()
}).fillna(0).to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 90); print("T3. AUC UNIVARIADO SOBRE EL SUBCONJUNTO SIN LA REGLA TRIVIAL"); print("=" * 90)
sub = hist[hist["CREDITOS ANTERIORES"] > 1].copy()
print(f"n = {len(sub)}   tasa de fuga = {sub.y.mean():.4f}")
sub["ratio_saldo_cap"] = sub["SALDO_CAPITAL"] / sub["CAPITAL_CONCEDIDO"]
for c in ["SALDO_CAPITAL", "CAPITAL_CONCEDIDO", "ratio_saldo_cap", "CREDITOS ANTERIORES",
          "TASA_NOMINAL", "REGION", "AGENCIA", "PRODUCTO", "SUBPRODUCTO"]:
    s = sub[c].fillna(sub[c].median()); a = roc_auc_score(sub.y, s)
    print(f"{c:22s} AUC = {max(a,1-a):.4f}")
for c in ["ETAPA", "TIPO DE CREDITO", "SEXO"]:
    k = sub[c].astype(str).str.strip().str.upper()
    a = roc_auc_score(sub.y, k.map(sub.groupby(k)["y"].mean()))
    print(f"{c:22s} AUC = {max(a,1-a):.4f}  (codificado por tasa, optimista)")

print("\nETAPA dentro del subconjunto (CREDITOS ANTERIORES > 1):")
print(sub.groupby("ETAPA").agg(n=("y","size"), tasa=("y","mean")).to_string(float_format=lambda x: f"{x:.4f}"))
print("\nCruce ETAPA x (CREDITOS ANTERIORES==1) en histórico:")
print(pd.crosstab(hist["ETAPA"], hist["CREDITOS ANTERIORES"] == 1, normalize="columns").to_string(float_format=lambda x: f"{x:.4f}"))
print("\nETAPA en predicción vs histórico:")
print(pd.DataFrame({"hist": hist["ETAPA"].value_counts(normalize=True),
                    "pred": pred["ETAPA"].value_counts(normalize=True)}).to_string(float_format=lambda x: f"{x:.4f}"))
