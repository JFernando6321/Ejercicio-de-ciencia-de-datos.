"""
Fase 0e — ¿SALDO_CAPITAL está medido en el mismo momento en ambas bases?
Es la pregunta que decide si el ratio saldo/capital es un predictor legítimo
o una segunda tautología.
"""
import pandas as pd, numpy as np, pathlib
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 100)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
hist = pd.read_excel(DATA / "Base de Datos Modelo.xlsx", sheet_name="Hoja1")
pred = pd.read_excel(DATA / "Base de Datos Predicción.xlsx")
hist["y"] = (hist["ESTADO"] == "Cliente Retirado").astype(int)
hist["ratio"] = hist["SALDO_CAPITAL"] / hist["CAPITAL_CONCEDIDO"]
pred["ratio"] = pred["SALDO_CAPITAL"] / pred["CAPITAL_CONCEDIDO"]

print("=" * 90); print("1. DISTRIBUCIÓN DEL RATIO SALDO/CAPITAL"); print("=" * 90)
qs = [0, .05, .1, .25, .5, .75, .9, .95, 1]
tab = pd.DataFrame({
    "hist_TODOS":    hist["ratio"].quantile(qs).values,
    "hist_RENOVADO": hist.loc[hist.y == 0, "ratio"].quantile(qs).values,
    "hist_RETIRADO": hist.loc[hist.y == 1, "ratio"].quantile(qs).values,
    "pred_ACTIVOS":  pred["ratio"].quantile(qs).values,
}, index=[f"p{int(q*100)}" for q in qs])
print(tab.to_string(float_format=lambda x: f"{x:.4f}"))

print("\nMedias:  hist_todos=%.4f  renovado=%.4f  retirado=%.4f  pred=%.4f" % (
    hist["ratio"].mean(), hist.loc[hist.y == 0, "ratio"].mean(),
    hist.loc[hist.y == 1, "ratio"].mean(), pred["ratio"].mean()))

print("\n>> Si el perfil de los ACTIVOS se parece al de los RENOVADOS y no a la mezcla,")
print("   entonces el saldo del histórico está medido DESPUÉS del desenlace (tautología).")
print("   Si se parece a la mezcla completa, la medición es comparable y el ratio es legítimo.\n")

print("Distancia (Kolmogorov-Smirnov) del ratio de pred contra cada grupo:")
from scipy.stats import ks_2samp
for nom, s in [("hist_TODOS", hist["ratio"]),
               ("hist_RENOVADO", hist.loc[hist.y == 0, "ratio"]),
               ("hist_RETIRADO", hist.loc[hist.y == 1, "ratio"])]:
    print(f"  pred vs {nom:14s} KS = {ks_2samp(pred['ratio'], s).statistic:.4f}")

print("\n" + "=" * 90); print("2. ¿EL RATIO SE CONCENTRA EN VALORES DE 'CRÉDITO RECIÉN DESEMBOLSADO'?"); print("=" * 90)
bins = [-0.001, 0.001, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 1.05, np.inf]
t = pd.DataFrame({
    "hist_RENOVADO": pd.cut(hist.loc[hist.y == 0, "ratio"], bins).value_counts(normalize=True, sort=False),
    "hist_RETIRADO": pd.cut(hist.loc[hist.y == 1, "ratio"], bins).value_counts(normalize=True, sort=False),
    "pred_ACTIVOS":  pd.cut(pred["ratio"], bins).value_counts(normalize=True, sort=False)})
print(t.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 90); print("3. MISMO CHEQUEO PARA SALDO_CAPITAL ABSOLUTO"); print("=" * 90)
tab2 = pd.DataFrame({
    "hist_RENOVADO": hist.loc[hist.y == 0, "SALDO_CAPITAL"].quantile(qs).values,
    "hist_RETIRADO": hist.loc[hist.y == 1, "SALDO_CAPITAL"].quantile(qs).values,
    "pred_ACTIVOS":  pred["SALDO_CAPITAL"].quantile(qs).values,
}, index=[f"p{int(q*100)}" for q in qs])
print(tab2.to_string(float_format=lambda x: f"{x:,.2f}"))

print("\n" + "=" * 90); print("4. RESTRINGIDO A CREDITOS ANTERIORES > 1 (población no degenerada)"); print("=" * 90)
h2 = hist[hist["CREDITOS ANTERIORES"] > 1]; p2 = pred[pred["CREDITOS ANTERIORES"] > 1]
tab3 = pd.DataFrame({
    "hist_RENOVADO": h2.loc[h2.y == 0, "ratio"].quantile(qs).values,
    "hist_RETIRADO": h2.loc[h2.y == 1, "ratio"].quantile(qs).values,
    "hist_TODOS":    h2["ratio"].quantile(qs).values,
    "pred_ACTIVOS":  p2["ratio"].quantile(qs).values,
}, index=[f"p{int(q*100)}" for q in qs])
print(tab3.to_string(float_format=lambda x: f"{x:.4f}"))
for nom, s in [("hist_TODOS", h2["ratio"]), ("hist_RENOVADO", h2.loc[h2.y==0,"ratio"]),
               ("hist_RETIRADO", h2.loc[h2.y==1,"ratio"])]:
    print(f"  KS pred vs {nom:14s} = {ks_2samp(p2['ratio'], s).statistic:.4f}")

print("\n" + "=" * 90); print("5. TASA DE FUGA POR CRUCE ratio x CREDITOS ANTERIORES (k>=2)"); print("=" * 90)
h2 = h2.copy()
h2["ratio_b"] = pd.cut(h2["ratio"], [-0.001, .1, .2, .4, .6, np.inf],
                       labels=["<0.10", "0.10-0.20", "0.20-0.40", "0.40-0.60", ">0.60"])
h2["k_b"] = pd.cut(h2["CREDITOS ANTERIORES"], [1, 2, 3, 5, 8, np.inf],
                   labels=["2", "3", "4-5", "6-8", "9+"])
print("Tasa de fuga:")
print(pd.crosstab(h2["ratio_b"], h2["k_b"], values=h2.y, aggfunc="mean").to_string(float_format=lambda x: f"{x:.3f}"))
print("\nn:")
print(pd.crosstab(h2["ratio_b"], h2["k_b"]).to_string())
