"""
Fase 5 — Probabilidad de fuga de los 10,000 activos (1d) y selección
de campaña por valor esperado (1e).

Tratamiento del segmento de primer crédito
------------------------------------------
En el histórico, CREDITOS ANTERIORES == 1 implica y == 1 en 12,791 de 12,791
casos, porque "Cliente Renovado" exige por definición un segundo crédito. Es
la variable objetivo escrita de otra forma, no un predictor. Por lo tanto:
  · el ORDENAMIENTO dentro del segmento se toma del modelo entrenado en k>=2;
  · el NIVEL de probabilidad se estima extrapolando la curva empírica de fuga
    frente al número de créditos (ajustada en k = 2..5, donde la extrapolación
    a k = 1 es creíble) y se aplica como corrección de prior en escala logit,
    que es monótona y por tanto no altera el ordenamiento.
Se acompaña de análisis de sensibilidad.
"""
import pandas as pd, numpy as np, joblib, json, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import logit, expit

import features as F, estilo
estilo.aplicar()

ROOT = F.ROOT
OUT_T = ROOT / "outputs" / "tablas"; OUT_F = ROOT / "outputs" / "figuras"
SEED = F.SEED

art = joblib.load(ROOT / "outputs" / "modelos" / "modelo_fuga.joblib")
lgb_full, iso, CATS_REF, COLS = art["lgbm"], art["isotonica"], art["cats"], art["cols"]

hist, pred = F.cargar()
hist_f, tasa_prod = F.construir_features(hist)
pred_f, _ = F.construir_features(pred, tasa_media_producto=art["tasa_prod"])

def a_cat(df):
    d = df[COLS].copy()
    for c in F.CATEGORICAS:
        d[c] = pd.Categorical(d[c].astype(str), categories=CATS_REF[c])
    return d

# ============================================================ 1. PUNTUACIÓN BASE
p_raw = lgb_full.predict_proba(a_cat(pred_f))[:, 1]
p_cal = iso.predict(p_raw) if iso is not None else p_raw
p_cal = np.clip(p_cal, 1e-4, 1 - 1e-4)
pred_f["p_modelo"] = p_cal

es_k1 = (pred_f["CREDITOS_ANTERIORES"] == 1).values
print(f"Base de predicción: {len(pred_f):,} clientes | primer crédito: {es_k1.sum():,} "
      f"({es_k1.mean():.2%}) | con historial (k>=2): {(~es_k1).sum():,}")

# ============================================================ 2. NIVEL DEL SEGMENTO k=1
curva = (hist_f.groupby("CREDITOS_ANTERIORES")["y"].agg(["size", "mean"])
         .rename(columns={"size": "n", "mean": "tasa_fuga"}))
print("\n=== CURVA EMPÍRICA DE FUGA POR NÚMERO DE CRÉDITOS ===")
print(curva.head(10).to_string(float_format=lambda x: f"{x:.4f}"))

aj = curva.loc[2:5]
X_aj = np.log(aj.index.values.astype(float)); Y_aj = logit(aj["tasa_fuga"].values)
b, a = np.polyfit(X_aj, Y_aj, 1)
tasa_k1_est = float(expit(a + b * np.log(1.0)))
print(f"\nAjuste logit(tasa) = {a:.4f} {b:+.4f}·log(k)  sobre k=2..5")
print(f">> Tasa de fuga estimada para el primer crédito: {tasa_k1_est:.4f}")
print(f"   (observada en el histórico para k=2: {curva.loc[2,'tasa_fuga']:.4f}; "
      f"el segmento de primer crédito es el de mayor riesgo de la cartera)")

def corregir_prior(p, objetivo):
    """Desplazamiento en logit que lleva la media de p al valor objetivo.
    Es monótono: preserva por completo el ordenamiento de riesgo."""
    f = lambda c: expit(logit(p) + c).mean() - objetivo
    return expit(logit(p) + brentq(f, -15, 15))

p_final = p_cal.copy()
p_final[es_k1] = corregir_prior(p_cal[es_k1], tasa_k1_est)
pred_f["prob_fuga"] = p_final

print(f"\nProbabilidad media  k=1: {p_final[es_k1].mean():.4f} | k>=2: {p_final[~es_k1].mean():.4f} "
      f"| global: {p_final.mean():.4f}")
print(f"Fugas esperadas en la base de predicción: {p_final.sum():,.0f} de {len(p_final):,} clientes")

# --- Sensibilidad del supuesto del primer crédito -------------------------
sens_k1 = []
for t in [0.30, tasa_k1_est, 0.60, 0.85]:
    pp = p_final.copy(); pp[es_k1] = corregir_prior(p_cal[es_k1], t)
    sens_k1.append({"tasa_supuesta_k1": t, "prob_media_global": pp.mean(),
                    "fugas_esperadas": pp.sum(),
                    "clientes_k1_en_top1000": int(es_k1[np.argsort(-pp)[:1000]].sum()),
                    "clientes_k1_en_top2000": int(es_k1[np.argsort(-pp)[:2000]].sum())})
sens_k1 = pd.DataFrame(sens_k1)
print("\n=== SENSIBILIDAD AL SUPUESTO DEL PRIMER CRÉDITO ===")
print(sens_k1.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
sens_k1.to_csv(OUT_T / "07_sensibilidad_primer_credito.csv", index=False)

# ============================================================ 3. DECILES Y SEGMENTOS
pred_f["decil_riesgo"] = pd.qcut(pred_f["prob_fuga"].rank(method="first", ascending=False),
                                 10, labels=range(1, 11)).astype(int)
pred_f["segmento_riesgo"] = pd.cut(pred_f["prob_fuga"], [-0.01, 0.20, 0.50, 1.01],
                                   labels=["Bajo", "Medio", "Alto"]).astype(str)

dec = pred_f.groupby("decil_riesgo").agg(
    clientes=("prob_fuga", "size"), prob_media=("prob_fuga", "mean"),
    prob_min=("prob_fuga", "min"), prob_max=("prob_fuga", "max"),
    fugas_esperadas=("prob_fuga", "sum"),
    saldo_en_riesgo=("SALDO_CAPITAL", "sum"),
    pct_primer_credito=("CREDITOS_ANTERIORES", lambda s: (s == 1).mean()))
dec["captura_acum_%"] = 100 * dec["fugas_esperadas"].cumsum() / dec["fugas_esperadas"].sum()
dec["lift"] = dec["prob_media"] / pred_f["prob_fuga"].mean()
print("\n=== DECILES DE RIESGO — BASE DE PREDICCIÓN ===")
print(dec.to_string(float_format=lambda x: f"{x:,.4f}"))
dec.to_csv(OUT_T / "07_deciles_prediccion.csv")

seg = pred_f.groupby("segmento_riesgo").agg(
    clientes=("prob_fuga", "size"), prob_media=("prob_fuga", "mean"),
    fugas_esperadas=("prob_fuga", "sum"), saldo=("SALDO_CAPITAL", "sum"))
seg["%_cartera"] = 100 * seg["saldo"] / seg["saldo"].sum()
print("\n=== SEGMENTOS DE RIESGO ===")
print(seg.to_string(float_format=lambda x: f"{x:,.2f}"))
seg.to_csv(OUT_T / "07_segmentos_riesgo.csv")

# ============================================================ 4. CAMPAÑA POR VALOR ESPERADO
MARGEN_NETO = 0.30      # margen neto sobre el ingreso financiero del ciclo
COSTO_CONTACTO = 25.0   # COP por cliente contactado (llamada + tiempo de asesor)

pred_f["valor_cliente"] = (pred_f["CAPITAL_CONCEDIDO"] * pred_f["TASA_NOMINAL"] / 100 * MARGEN_NETO)
print(f"\nValor estimado del cliente (margen del próximo ciclo): "
      f"mediana ${pred_f['valor_cliente'].median():,.0f}, media ${pred_f['valor_cliente'].mean():,.0f}")

filas = []
for efec in [0.10, 0.20, 0.30]:
    ben = pred_f["prob_fuga"] * efec * pred_f["valor_cliente"] - COSTO_CONTACTO
    sel = ben > 0
    filas.append({"efectividad": efec, "clientes_seleccionados": int(sel.sum()),
                  "%_de_la_base": sel.mean(),
                  "fugas_esperadas_cubiertas": pred_f.loc[sel, "prob_fuga"].sum(),
                  "%_fugas_cubiertas": pred_f.loc[sel, "prob_fuga"].sum() / pred_f["prob_fuga"].sum(),
                  "retenciones_esperadas": (pred_f.loc[sel, "prob_fuga"] * efec).sum(),
                  "costo_total": sel.sum() * COSTO_CONTACTO,
                  "beneficio_esperado": ben[sel].sum(),
                  "ROI": ben[sel].sum() / (sel.sum() * COSTO_CONTACTO)})
sens = pd.DataFrame(filas)
print("\n=== SELECCIÓN POR VALOR ESPERADO — SENSIBILIDAD A LA EFECTIVIDAD ===")
print(sens.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
sens.to_csv(OUT_T / "07_sensibilidad_campana.csv", index=False)

EFEC_BASE = 0.20
pred_f["beneficio_esperado"] = (pred_f["prob_fuga"] * EFEC_BASE * pred_f["valor_cliente"]
                                - COSTO_CONTACTO)
pred_f["recomendar_campana"] = np.where(pred_f["beneficio_esperado"] > 0, "Sí", "No")
n_sel = (pred_f["recomendar_campana"] == "Sí").sum()
print(f"\n>> Escenario base (efectividad {EFEC_BASE:.0%}): se recomienda contactar a "
      f"{n_sel:,} clientes ({n_sel/len(pred_f):.1%} de la base)")

# --- Versión con restricción de capacidad ---------------------------------
cap = []
for K in [500, 1000, 1500, 2000, 3000]:
    idx = pred_f["beneficio_esperado"].nlargest(K).index
    cap.append({"capacidad": K,
                "prob_media": pred_f.loc[idx, "prob_fuga"].mean(),
                "fugas_esperadas": pred_f.loc[idx, "prob_fuga"].sum(),
                "%_fugas_totales": pred_f.loc[idx, "prob_fuga"].sum() / pred_f["prob_fuga"].sum(),
                "retenciones_esperadas": (pred_f.loc[idx, "prob_fuga"] * EFEC_BASE).sum(),
                "saldo_protegido": pred_f.loc[idx, "SALDO_CAPITAL"].sum(),
                "costo": K * COSTO_CONTACTO,
                "beneficio_esperado": pred_f.loc[idx, "beneficio_esperado"].sum()})
cap = pd.DataFrame(cap)
print("\n=== CAMPAÑA CON RESTRICCIÓN DE CAPACIDAD (ordenado por valor esperado) ===")
print(cap.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
cap.to_csv(OUT_T / "07_capacidad_campana.csv", index=False)

pred_f["prioridad_campana"] = pred_f["beneficio_esperado"].rank(ascending=False, method="first").astype(int)

# ============================================================ 5. FIGURAS
# Distribución de probabilidades
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.hist(pred_f.loc[~es_k1, "prob_fuga"], bins=40, color=estilo.PRIMARIO, alpha=.85,
        label=f"Con historial (k≥2, n={(~es_k1).sum():,})")
ax.hist(pred_f.loc[es_k1, "prob_fuga"], bins=40, color=estilo.ACENTO, alpha=.75,
        label=f"Primer crédito (n={es_k1.sum():,})")
ax.set_xlabel("Probabilidad de fuga"); ax.set_ylabel("Clientes")
ax.set_title("Distribución de la probabilidad de fuga — 10,000 clientes activos")
ax.legend()
plt.tight_layout(); plt.savefig(OUT_F / "07_distribucion_prob.png"); plt.close()

# Curva de ganancia acumulada
orden = np.argsort(-pred_f["prob_fuga"].values)
cum = np.cumsum(pred_f["prob_fuga"].values[orden]) / pred_f["prob_fuga"].sum()
xx = np.arange(1, len(cum) + 1) / len(cum)
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.plot(xx * 100, cum * 100, color=estilo.PRIMARIO, lw=2, label="Modelo")
ax.plot([0, 100], [0, 100], color=estilo.GRIS, ls="--", lw=1, label="Sin modelo (azar)")
for f in [0.1, 0.2, 0.3]:
    v = cum[int(f * len(cum)) - 1] * 100
    ax.plot([f * 100, f * 100], [0, v], color=estilo.ACENTO, ls=":", lw=1)
    ax.annotate(f"{v:.0f}%", (f * 100, v), textcoords="offset points", xytext=(4, -10),
                color=estilo.ACENTO, fontsize=8.5, fontweight="bold")
ax.set_xlabel("% de clientes contactados (ordenados por riesgo)")
ax.set_ylabel("% de fugas capturadas")
ax.set_title("Curva de ganancia acumulada")
ax.legend(loc="lower right")
plt.tight_layout(); plt.savefig(OUT_F / "07_curva_ganancia.png"); plt.close()

# Deciles: saldo en riesgo
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.bar(dec.index, dec["saldo_en_riesgo"] / 1e6, color=estilo.PRIMARIO)
ax2 = ax.twinx(); ax2.plot(dec.index, dec["prob_media"] * 100, color=estilo.ACENTO, marker="o", lw=1.8)
ax2.set_ylabel("Probabilidad media de fuga (%)", color=estilo.ACENTO); ax2.grid(False)
ax.set_xlabel("Decil de riesgo (1 = mayor riesgo)"); ax.set_ylabel("Saldo en riesgo (millones Q)")
ax.set_title("Saldo expuesto y riesgo por decil"); ax.set_xticks(range(1, 11))
plt.tight_layout(); plt.savefig(OUT_F / "07_deciles_saldo.png"); plt.close()

# ============================================================ 6. GUARDAR
cols_out = ["CODIGO CLIENTE", "CODIGO PRESTAMO", "prob_fuga", "decil_riesgo",
            "segmento_riesgo", "valor_cliente", "beneficio_esperado",
            "prioridad_campana", "recomendar_campana"]
pred_f[cols_out].to_csv(ROOT / "outputs" / "modelos" / "predicciones.csv", index=False)

json.dump({"tasa_k1_estimada": tasa_k1_est,
           "prob_media_global": float(p_final.mean()),
           "fugas_esperadas": float(p_final.sum()),
           "n_campana_base": int(n_sel),
           "margen_neto_supuesto": MARGEN_NETO, "costo_contacto": COSTO_CONTACTO,
           "efectividad_base": EFEC_BASE,
           "captura_top10": float(cum[999]), "captura_top20": float(cum[1999]),
           "captura_top30": float(cum[2999]),
           "saldo_total_base": float(pred_f["SALDO_CAPITAL"].sum()),
           "saldo_riesgo_alto": float(pred_f.loc[pred_f.segmento_riesgo == "Alto", "SALDO_CAPITAL"].sum())},
          open(OUT_T / "07_resumen_prediccion.json", "w"), indent=2)
print("\n" + json.dumps(json.load(open(OUT_T / "07_resumen_prediccion.json")), indent=2))
