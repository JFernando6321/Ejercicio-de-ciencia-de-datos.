"""
Fase 9 — Agregados para el tablero (pregunta 3).
Genera un único JSON con todo lo que consume el dashboard. No se exporta ningún
dato individual identificable más allá del código de cliente ya anonimizado.
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
import features as F

ROOT = F.ROOT
OUT_T = ROOT / "outputs" / "tablas"
ENT = ROOT / "entregables"

ent = pd.read_excel(ENT / "Genesis_Churn_Prediccion_Julio_Vicente.xlsx", sheet_name="Predicción 10,000")
perfil = pd.read_csv(OUT_T / "08_perfil_clusters.csv")
res_act = pd.read_csv(OUT_T / "08_resumen_clusters_activos.csv")
shap_imp = pd.read_csv(OUT_T / "06_importancia_shap.csv")
dec_hold = pd.read_csv(OUT_T / "05_deciles_holdout.csv")
dec_pred = pd.read_csv(OUT_T / "07_deciles_prediccion.csv")
escalera = pd.read_csv(OUT_T / "05_escalera_modelos.csv")
sens = pd.read_csv(OUT_T / "07_sensibilidad_campana.csv")
cap = pd.read_csv(OUT_T / "07_capacidad_campana.csv")
reglas = pd.read_csv(OUT_T / "09_reglas_seleccionadas.csv")
prod = pd.read_csv(OUT_T / "09_soporte_productos.csv")
resumen_m = json.load(open(OUT_T / "05_resumen_modelo.json"))
resumen_p = json.load(open(OUT_T / "07_resumen_prediccion.json"))
resumen_b = json.load(open(OUT_T / "09_resumen_bakery.json"))

NOM = dict(zip(res_act["cluster"], ent.drop_duplicates("cluster").set_index("cluster")["cluster_nombre"]))

# --- Curva de ganancia -----------------------------------------------------
p = ent["Probabilidad Fuga"].values
orden = np.argsort(-p)
cum = np.cumsum(p[orden]) / p.sum()
ganancia = [{"x": round(100 * (i + 1) / len(cum), 1), "y": round(100 * cum[i], 2)}
            for i in range(0, len(cum), 100)]

# --- Histograma de probabilidad -------------------------------------------
h, bordes = np.histogram(p, bins=20, range=(0, 1))
histo = [{"bin": f"{bordes[i]:.2f}", "centro": round((bordes[i] + bordes[i+1]) / 2, 3),
          "n": int(h[i])} for i in range(len(h))]

# --- Por agencia y región --------------------------------------------------
def agrupar(col, top_n=None):
    g = ent.groupby(col).agg(clientes=("CODIGO CLIENTE", "size"),
                             prob_media=("Probabilidad Fuga", "mean"),
                             fugas=("Probabilidad Fuga", "sum"),
                             saldo=("SALDO_CAPITAL", "sum"),
                             campana=("recomendar_campana", lambda s: (s == "Sí").sum())).reset_index()
    g = g.sort_values("fugas", ascending=False)
    if top_n:
        g = g.head(top_n)
    g[col] = g[col].astype(str)
    for c in ["prob_media"]:
        g[c] = g[c].round(4)
    for c in ["fugas", "saldo"]:
        g[c] = g[c].round(0)
    return g.to_dict("records")

datos = {
    "kpi": {
        "clientes": int(len(ent)),
        "prob_media": round(float(p.mean()), 4),
        "fugas_esperadas": int(round(p.sum())),
        "riesgo_alto": int((ent.segmento_riesgo == "Alto").sum()),
        "saldo_total": float(ent.SALDO_CAPITAL.sum()),
        "saldo_riesgo_alto": float(ent.loc[ent.segmento_riesgo == "Alto", "SALDO_CAPITAL"].sum()),
        "campana": int((ent.recomendar_campana == "Sí").sum()),
        "auc_roc": round(resumen_m["holdout_auc_roc"], 4),
        "auc_pr": round(resumen_m["holdout_auc_pr"], 4),
        "ks": round(resumen_m["holdout_ks"], 4),
        "brier": round(resumen_m["holdout_brier_isotonica"], 4),
        "captura20": round(100 * resumen_m["captura_top20"], 1),
        "lift1": round(resumen_m["lift_decil_1"], 2),
        "tasa_hist": 0.3807,
        "tasa_k1": round(resumen_p["tasa_k1_estimada"], 4),
    },
    "ganancia": ganancia,
    "histograma": histo,
    "deciles_pred": dec_pred.round(4).to_dict("records"),
    "deciles_holdout": dec_hold.round(4).to_dict("records"),
    "escalera": escalera.round(4).to_dict("records"),
    "shap": shap_imp.head(12)[["nombre_negocio", "peso_relativo_%", "direccion"]].round(2).to_dict("records"),
    "clusters": [{
        "id": int(r.cluster), "nombre": NOM.get(int(r.cluster), f"Segmento {r.cluster}"),
        "n_hist": int(perfil.loc[perfil.cluster == r.cluster, "n"].iloc[0]),
        "pct_clientes": round(float(perfil.loc[perfil.cluster == r.cluster, "%_clientes"].iloc[0]), 2),
        "pct_cartera": round(float(perfil.loc[perfil.cluster == r.cluster, "%_cartera"].iloc[0]), 2),
        "tasa_hist": round(float(r.tasa_fuga_historica), 4),
        "prob_activos": round(float(r.prob_media_fuga), 4),
        "n_activos": int(r.clientes_activos),
        "saldo": float(r.saldo),
        "campana": int(r.recomendados_campana),
        "capital_medio": float(perfil.loc[perfil.cluster == r.cluster, "capital_medio"].iloc[0]),
        "saldo_medio": float(perfil.loc[perfil.cluster == r.cluster, "saldo_medio"].iloc[0]),
        "ratio_ciclo": round(float(perfil.loc[perfil.cluster == r.cluster, "ratio_ciclo"].iloc[0]), 3),
        "creditos": float(perfil.loc[perfil.cluster == r.cluster, "creditos"].iloc[0]),
        "tasa_nominal": float(perfil.loc[perfil.cluster == r.cluster, "tasa_nominal"].iloc[0]),
        "pct_M2M3": round(float(perfil.loc[perfil.cluster == r.cluster, "pct_M2_M3"].iloc[0]), 4),
        "pct_mujeres": round(float(perfil.loc[perfil.cluster == r.cluster, "pct_mujeres"].iloc[0]), 4),
        "accion": ent.loc[ent.cluster == r.cluster, "accion_recomendada"].iloc[0],
    } for r in res_act.itertuples()],
    "cruce": pd.crosstab(ent["cluster"], ent["segmento_riesgo"]).reset_index().to_dict("records"),
    "cruce_saldo": pd.crosstab(ent["cluster"], ent["segmento_riesgo"],
                               values=ent["SALDO_CAPITAL"], aggfunc="sum").fillna(0).round(0).reset_index().to_dict("records"),
    "region": agrupar("REGION"),
    "agencia": agrupar("AGENCIA", 20),
    "producto": agrupar("PRODUCTO", 15),
    "etapa": agrupar("ETAPA"),
    "sensibilidad": sens.round(4).to_dict("records"),
    "capacidad": cap.round(2).to_dict("records"),
    "bakery": {
        "resumen": resumen_b,
        "reglas": reglas.head(14)[["productos_del_combo", "n_productos", "n_tickets",
                                    "support", "confidence", "lift", "leverage"]].round(4).to_dict("records"),
        "productos": prod.head(15).round(4).to_dict("records"),
    },
    "clientes_top": ent.head(200)[["CODIGO CLIENTE", "AGENCIA", "prob_fuga_pct", "decil_riesgo",
                                    "segmento_riesgo", "cluster_nombre", "SALDO_CAPITAL",
                                    "beneficio_esperado"]].to_dict("records"),
    # Vectores completos de los 10,000 para que el simulador del tablero calcule
    # sobre la base real y no sobre una aproximación.
    "vec": {
        "p": [round(float(v), 4) for v in ent["Probabilidad Fuga"]],
        "valor": [int(round(v)) for v in ent["valor_cliente"]],
        "saldo": [int(round(v)) for v in ent["SALDO_CAPITAL"]],
        "cluster": [int(v) for v in ent["cluster"]],
    },
    "costo_contacto": 25.0,
}

ruta = ROOT / "outputs" / "datos_dashboard.json"
json.dump(datos, open(ruta, "w"), ensure_ascii=False, allow_nan=False)
print(f"[OK] {ruta}  ({ruta.stat().st_size/1024:.0f} KB)")
print(json.dumps(datos["kpi"], indent=2, ensure_ascii=False))
