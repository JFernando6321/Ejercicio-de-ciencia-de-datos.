"""
Fase 0c — Cierre de auditoría: hojas duplicadas, trampa de identificadores,
deriva real de montos y estructura de ETAPA.
"""
import pandas as pd, numpy as np, pathlib
from sklearn.metrics import roc_auc_score
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 100)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OUT = ROOT / "outputs" / "tablas"

hist = pd.read_excel(DATA / "Base de Datos Modelo.xlsx", sheet_name="Hoja1")
h2   = pd.read_excel(DATA / "Base de Datos Modelo.xlsx", sheet_name="Hoja1 (2)")
h3   = pd.read_excel(DATA / "Base de Datos Modelo.xlsx", sheet_name="Hoja3")
pred = pd.read_excel(DATA / "Base de Datos Predicción.xlsx")
y = (hist["ESTADO"] == "Cliente Retirado").astype(int)

print("=" * 90); print("A. ¿Hoja1 (2) aporta algo?"); print("=" * 90)
print("¿Hoja1 == Hoja1 (2)? ->", hist.equals(h2))
print("Hoja3['Aleatorio'] no nulos:", h3["Aleatorio"].notna().sum(), "/", len(h3))
print("¿Hoja3 respeta el mismo orden de clientes? ->",
      (h3["CODIGO CLIENTE"].values == hist["CODIGO CLIENTE"].values).all())

print("\n" + "=" * 90); print("B. TRAMPA DE IDENTIFICADORES"); print("=" * 90)
for c in ["CODIGO CLIENTE", "CODIGO PRESTAMO"]:
    a = roc_auc_score(y, hist[c])
    print(f"{c:18s} AUC univariado en histórico = {max(a,1-a):.4f}")
print(f"\nRango CODIGO CLIENTE  hist=[{hist['CODIGO CLIENTE'].min()}, {hist['CODIGO CLIENTE'].max()}]"
      f"  pred=[{pred['CODIGO CLIENTE'].min()}, {pred['CODIGO CLIENTE'].max()}]")
print(f"Rango CODIGO PRESTAMO hist=[{hist['CODIGO PRESTAMO'].min()}, {hist['CODIGO PRESTAMO'].max()}]"
      f"  pred=[{pred['CODIGO PRESTAMO'].min()}, {pred['CODIGO PRESTAMO'].max()}]")
print("¿pred CODIGO CLIENTE es exactamente 1..10000? ->",
      (np.sort(pred["CODIGO CLIENTE"].values) == np.arange(1, 10001)).all())

print("\n" + "=" * 90); print("C. DERIVA REAL DE MONTOS (bins fijos, no cuantiles)"); print("=" * 90)
bins = [0, 1, 1200, 2000, 3000, 5000, 6000, 10000, 15000, 25000, 50000, 100000, np.inf]
for c in ["CAPITAL_CONCEDIDO", "SALDO_CAPITAL"]:
    a = pd.cut(hist[c], bins, right=True).value_counts(normalize=True, sort=False)
    b = pd.cut(pred[c], bins, right=True).value_counts(normalize=True, sort=False)
    t = pd.DataFrame({"hist": a, "pred": b}).fillna(0)
    t["dif"] = t["pred"] - t["hist"]
    aa, bb = t["hist"].replace(0, 1e-6), t["pred"].replace(0, 1e-6)
    print(f"\n· {c}  PSI(bins fijos) = {((bb-aa)*np.log(bb/aa)).sum():.4f}")
    print(t.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n--- Valores más frecuentes de CAPITAL_CONCEDIDO ---")
print(pd.DataFrame({
    "hist": hist["CAPITAL_CONCEDIDO"].value_counts(normalize=True).head(15),
    "pred": pred["CAPITAL_CONCEDIDO"].value_counts(normalize=True).head(15)}).fillna(0)
    .to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 90); print("D. ETAPA: ¿escalón de deterioro (IFRS9) o mora?"); print("=" * 90)
g = hist.assign(r=hist["SALDO_CAPITAL"] / hist["CAPITAL_CONCEDIDO"]).groupby("ETAPA").agg(
    n=("ESTADO", "size"),
    tasa_fuga=("ESTADO", lambda s: (s == "Cliente Retirado").mean()),
    saldo_medio=("SALDO_CAPITAL", "mean"),
    capital_medio=("CAPITAL_CONCEDIDO", "mean"),
    ratio_saldo_cap=("r", "mean"),
    creditos_ant=("CREDITOS ANTERIORES", "mean"),
    tasa_nominal=("TASA_NOMINAL", "mean"))
print(g.to_string(float_format=lambda x: f"{x:,.3f}"))

print("\n" + "=" * 90); print("E. RATIO SALDO/CAPITAL — candidata a variable clave"); print("=" * 90)
r = (hist["SALDO_CAPITAL"] / hist["CAPITAL_CONCEDIDO"]).replace([np.inf, -np.inf], np.nan)
print("descriptivos:", r.describe().to_dict())
print(f"AUC univariado ratio = {max(roc_auc_score(y, r.fillna(r.median())), 1-roc_auc_score(y, r.fillna(r.median()))):.4f}")
print("\nTasa de fuga por decil del ratio:")
print(hist.assign(r=r, dec=pd.qcut(r, 10, duplicates="drop", labels=False))
        .groupby("dec").agg(n=("ESTADO","size"), r_med=("r","median"),
                            tasa=("ESTADO", lambda s:(s=="Cliente Retirado").mean()))
        .to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 90); print("F. CREDITOS ANTERIORES — forma de la relación"); print("=" * 90)
print(hist.assign(k=hist["CREDITOS ANTERIORES"].clip(upper=12)).groupby("k").agg(
    n=("ESTADO","size"), tasa=("ESTADO", lambda s:(s=="Cliente Retirado").mean()))
    .to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 90); print("G. CARDINALIDAD Y COBERTURA DE AGENCIA / PRODUCTO"); print("=" * 90)
for c in ["REGION", "AGENCIA", "PRODUCTO", "SUBPRODUCTO"]:
    sh, sp = set(hist[c].unique()), set(pred[c].unique())
    cob = pred[c].isin(sh).mean()
    print(f"{c:12s} niveles_hist={len(sh):4d} niveles_pred={len(sp):4d} "
          f"nuevos_en_pred={len(sp-sh):3d}  cobertura_filas_pred={cob:.4%}")
