"""
Fase 10 — Genera los fragmentos .tex de las tablas del informe.
El informe las incluye con \\input{}, de modo que si el modelo cambia,
el documento se actualiza sin edición manual.
"""
import pandas as pd, numpy as np, json, pathlib, warnings, re
warnings.filterwarnings("ignore")
import features as F

ROOT = F.ROOT
OUT_T = ROOT / "outputs" / "tablas"
TEX = ROOT / "outputs" / "tex"; TEX.mkdir(parents=True, exist_ok=True)

def esc(s):
    """Escapa texto crudo procedente de los datos. Se aplica UNA sola vez, en el
    punto donde el texto entra; las celdas ya formateadas pasan sin tocar."""
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("<", r"$<$"), (">", r"$>$")]:
        s = s.replace(a, b)
    return s

def tabla(df, cols, encabezados, aligns, nombre):
    """Escribe un tabular booktabs listo para \\input.
    Las celdas se toman tal cual: ya vienen formateadas como LaTeX válido."""
    L = [r"\begin{tabular}{" + aligns + "}", r"\toprule",
         " & ".join(rf"\textbf{{{h}}}" for h in encabezados) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        L.append(" & ".join(str(r[c]) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TEX / f"{nombre}.tex").write_text("\n".join(L), encoding="utf8")
    print(f"  [tex] {nombre}.tex")

pct = lambda v, d=1: f"{100*float(v):.{d}f}\\%"
num = lambda v, d=0: f"{float(v):,.{d}f}".replace(",", "\\,")
q   = lambda v: "\\$" + f"{float(v):,.0f}".replace(",", "\\,")   # pesos colombianos

# ---------------------------------------------------------------- 1. Escalera
e = pd.read_csv(OUT_T / "05_escalera_modelos.csv")
e["mod"] = e["modelo"].str.replace(r"^\d+b?\.\s*", "", regex=True).map(esc)
e["a"] = e["AUC_PR"].map(lambda v: f"{v:.3f}")
e["b"] = e["AUC_ROC"].map(lambda v: f"{v:.3f}")
e["c"] = e["KS"].map(lambda v: f"{v:.3f}")
e["d"] = e["lift_decil_1"].map(lambda v: f"{v:.2f}")
e["f"] = e["captura_top20pct"].map(lambda v: pct(v, 1))
tabla(e, ["mod", "a", "b", "c", "d", "f"],
      ["Modelo", "AUC-PR", "AUC-ROC", "KS", "Lift decil 1", "Captura al 20\\%"],
      "@{}lrrrrr@{}", "escalera")

# ---------------------------------------------------------------- 2. Calibración
d = pd.read_csv(OUT_T / "05_deciles_holdout.csv")
d["a"] = d["n"].map(lambda v: num(v))
d["b"] = d["prob_media_predicha"].map(lambda v: pct(v))
d["c"] = d["tasa_real_observada"].map(lambda v: pct(v))
d["e"] = d["lift"].map(lambda v: f"{v:.2f}")
d["f"] = d["captura_acum_%"].map(lambda v: f"{v:.1f}\\%")
tabla(d, ["decil", "a", "b", "c", "e", "f"],
      ["Decil", "Clientes", "Probabilidad predicha", "Tasa real observada", "Lift", "Captura acum."],
      "@{}lrrrrr@{}", "calibracion")

# ---------------------------------------------------------------- 3. SHAP
s = pd.read_csv(OUT_T / "06_importancia_shap.csv").head(10)
s["a"] = s["peso_relativo_%"].map(lambda v: f"{v:.1f}\\%")
s["b"] = s["direccion"].map({"reduce la fuga": "Reduce la fuga",
                             "aumenta la fuga": "Aumenta la fuga",
                             "categórica": "Categórica",
                             "no monótona": "No monótona"}).fillna("--")
s["nombre_negocio"] = s["nombre_negocio"].map(esc)
tabla(s, ["nombre_negocio", "a", "b"],
      ["Variable", "Peso relativo", "Sentido del efecto"],
      "@{}lrl@{}", "shap")

# ---------------------------------------------------------------- 4. Deciles predicción
p = pd.read_csv(OUT_T / "07_deciles_prediccion.csv")
p["a"] = p["clientes"].map(lambda v: num(v))
p["b"] = p["prob_media"].map(lambda v: pct(v))
p["c"] = p["fugas_esperadas"].map(lambda v: num(v))
p["e"] = p["saldo_en_riesgo"].map(q)
p["f"] = p["captura_acum_%"].map(lambda v: f"{v:.1f}\\%")
tabla(p, ["decil_riesgo", "a", "b", "c", "e", "f"],
      ["Decil", "Clientes", "Prob. media", "Fugas esperadas", "Saldo", "Captura acum."],
      "@{}lrrrrr@{}", "deciles_pred")

# ---------------------------------------------------------------- 5. Sensibilidad campaña
c = pd.read_csv(OUT_T / "07_sensibilidad_campana.csv")
c["a"] = c["efectividad"].map(lambda v: pct(v, 0))
c["b"] = c["clientes_seleccionados"].map(lambda v: num(v))
c["c"] = c["%_fugas_cubiertas"].map(lambda v: pct(v, 0))
c["e"] = c["retenciones_esperadas"].map(lambda v: num(v))
c["f"] = c["costo_total"].map(q)
c["g"] = c["beneficio_esperado"].map(q)
c["h"] = c["ROI"].map(lambda v: f"{v:.1f}$\\times$")
tabla(c, ["a", "b", "c", "e", "f", "g", "h"],
      ["Efectividad", "Clientes", "\\% de fugas", "Retenciones", "Costo", "Beneficio", "Retorno"],
      "@{}lrrrrrr@{}", "sensibilidad")

# ---------------------------------------------------------------- 6. Capacidad
k = pd.read_csv(OUT_T / "07_capacidad_campana.csv")
k["a"] = k["capacidad"].map(lambda v: num(v))
k["b"] = k["prob_media"].map(lambda v: pct(v, 0))
k["c"] = k["fugas_esperadas"].map(lambda v: num(v))
k["e"] = k["%_fugas_totales"].map(lambda v: pct(v, 0))
k["f"] = k["retenciones_esperadas"].map(lambda v: num(v))
k["g"] = k["beneficio_esperado"].map(q)
tabla(k, ["a", "b", "c", "e", "f", "g"],
      ["Capacidad", "Prob. media", "Fugas en lista", "\\% del total", "Retenciones", "Beneficio"],
      "@{}lrrrrr@{}", "capacidad")

# ---------------------------------------------------------------- 7. Clusters
NOM = F.NOMBRES_CLUSTER
pf = pd.read_csv(OUT_T / "08_perfil_clusters.csv")
ra = pd.read_csv(OUT_T / "08_resumen_clusters_activos.csv")
cl = pf.merge(ra[["cluster", "clientes_activos", "prob_media_fuga", "recomendados_campana"]], on="cluster")
cl["nom"] = cl["cluster"].map(NOM).map(esc)
cl = cl.sort_values("tasa_fuga_historica", ascending=False)
cl["a"] = cl["n"].map(lambda v: num(v))
cl["b"] = cl["%_clientes"].map(lambda v: f"{v:.1f}\\%")
cl["c"] = cl["%_cartera"].map(lambda v: f"{v:.1f}\\%")
cl["e"] = cl["tasa_fuga_historica"].map(lambda v: pct(v, 1))
cl["f"] = cl["prob_media_fuga"].map(lambda v: pct(v, 1))
cl["g"] = cl["clientes_activos"].map(lambda v: num(v))
tabla(cl, ["nom", "a", "b", "c", "e", "f", "g"],
      ["Segmento", "Hist.", "\\% clientes", "\\% cartera", "Fuga obs.", "Prob. activos", "Activos"],
      "@{}p{4.6cm}rrrrrr@{}", "clusters")

cl["h"] = cl["capital_medio"].map(q)
cl["i"] = cl["saldo_medio"].map(q)
cl["j"] = cl["ratio_ciclo"].map(lambda v: f"{100*(1-v):.0f}\\%")
cl["k"] = cl["creditos"].map(lambda v: f"{v:.0f}")
cl["l"] = cl["pct_M2_M3"].map(lambda v: pct(v, 1))
cl["m"] = cl["pct_mujeres"].map(lambda v: pct(v, 0))
tabla(cl, ["nom", "h", "i", "j", "k", "l", "m"],
      ["Segmento", "Capital", "Saldo", "\\% pagado", "Créditos", "En mora", "Mujeres"],
      "@{}p{4.6cm}rrrrrr@{}", "clusters_perfil")

# ---------------------------------------------------------------- 8. Cruce
cn = pd.read_csv(OUT_T / "08_cruce_cluster_riesgo_n.csv")
cs = pd.read_csv(OUT_T / "08_cruce_cluster_riesgo_saldo.csv")
cn["nom"] = cn["cluster"].map(NOM).map(esc)
for c_ in ["Alto", "Medio", "Bajo"]:
    cn[c_] = cn[c_].map(lambda v: num(v))
cn["sal"] = cs["Alto"].map(q)
tabla(cn, ["nom", "Alto", "Medio", "Bajo", "sal"],
      ["Segmento", "Riesgo alto", "Riesgo medio", "Riesgo bajo", "Saldo en riesgo alto"],
      "@{}p{4.6cm}rrrr@{}", "cruce")

# ---------------------------------------------------------------- 9. Reglas
r = pd.read_csv(OUT_T / "09_reglas_seleccionadas.csv").head(14)
r["combo"] = r["productos_del_combo"].map(lambda s: esc(s).replace(" + ", " $+$ "))
r["a"] = r["n_tickets"].map(lambda v: num(v))
r["b"] = r["support"].map(lambda v: pct(v, 1))
r["c"] = r["confidence"].map(lambda v: pct(v, 0))
r["e"] = r["lift"].map(lambda v: f"{v:.1f}$\\times$")
tabla(r, ["combo", "a", "b", "c", "e"],
      ["Canasta", "Tickets", "Soporte", "Confianza", "Lift"],
      "@{}p{7.6cm}rrrr@{}", "reglas")

# ---------------------------------------------------------------- 10. Auditoría de variables
aud = pd.DataFrame([
    ("CODIGO CLIENTE", "Descartada", "AUC univariado = 1.000. El archivo fue ordenado por ESTADO antes de numerarse: retirados 1--34\\,354, renovados 34\\,355--88\\,828. En la base de predicción se renumeró 1--10\\,000."),
    ("CODIGO PRESTAMO", "Descartada", "Es exactamente diez veces el código de cliente en el histórico. Hereda la misma fuga."),
    ("CREDITOS ANTERIORES", "Tratada aparte", "El valor 1 implica el objetivo por definición (12\\,791 de 12\\,791 retirados): renovar exige un segundo crédito. Se modela como segmento separado."),
    ("SALDO\\_CAPITAL", "Conservada", "Comparable entre bases: la distribución de los activos se sitúa entre la de renovados y retirados (KS = 0.144 contra el histórico completo)."),
    ("CAPITAL\\_CONCEDIDO", "Conservada con cautela", "Deriva severa entre bases (PSI = 2.74; rejillas de valores casi disjuntas). Se prioriza el ratio saldo/capital, invariante a escala."),
    ("ETAPA", "Conservada", "Escalón de deterioro observable al momento de decidir. M3 concentra 98.2\\% de fuga en el histórico."),
    ("TIPO DE CREDITO", "Conservada y limpiada", "12 niveles aparentes se reducen a 8 tras normalizar espacios y mayúsculas."),
    ("SEXO", "Conservada y limpiada", "Un registro con valor vacío se recodifica como `ND'. AUC univariado 0.502: sin poder predictivo."),
], columns=["v", "d", "j"])
tabla(aud, ["v", "d", "j"], ["Variable", "Decisión", "Justificación"],
      "@{}p{3.3cm}p{2.6cm}p{8.3cm}@{}", "auditoria")

# ---------------------------------------------------------------- 11. Cifras clave
m = json.load(open(OUT_T / "05_resumen_modelo.json"))
pr = json.load(open(OUT_T / "07_resumen_prediccion.json"))
bk = json.load(open(OUT_T / "09_resumen_bakery.json"))
macros = {
    "AUCROC": f"{m['holdout_auc_roc']:.3f}", "AUCPR": f"{m['holdout_auc_pr']:.3f}",
    "KSmodelo": f"{m['holdout_ks']:.3f}", "BrierSC": f"{m['holdout_brier_sin_calibrar']:.4f}",
    "BrierCal": f"{m['holdout_brier_isotonica']:.4f}",
    "CapturaDiez": f"{100*m['captura_top10']:.0f}", "CapturaVeinte": f"{100*m['captura_top20']:.0f}",
    "CapturaTreinta": f"{100*m['captura_top30']:.0f}", "LiftUno": f"{m['lift_decil_1']:.2f}",
    "AUCSinSaldo": f"{m['auc_roc_sin_variables_saldo']:.3f}",
    "AUCROCcv": f"{float(e.loc[e.modelo.str.startswith('3. '),'AUC_ROC'].iloc[0]):.3f}",
    "NModelable": f"{m['n_modelable']:,}".replace(",", "\\,"),
    "PrevalenciaKdos": f"{100*m['prevalencia_k2']:.1f}",
    "TasaKuno": f"{100*pr['tasa_k1_estimada']:.1f}",
    "ProbMedia": f"{100*pr['prob_media_global']:.1f}",
    "FugasEsperadas": f"{pr['fugas_esperadas']:,.0f}".replace(",", "\\,"),
    "NCampana": f"{pr['n_campana_base']:,}".replace(",", "\\,"),
    "SaldoTotal": f"{pr['saldo_total_base']/1e6:.1f}",
    "SaldoAlto": f"{pr['saldo_riesgo_alto']/1e6:.1f}",
    "NTickets": f"{bk['n_tickets']:,}".replace(",", "\\,"),
    "NReglas": f"{bk['n_reglas_lift_gt_1']:,}".replace(",", "\\,"),
    "NCanastas": str(bk["n_reglas_seleccionadas"]),
    "LiftMax": f"{bk['lift_maximo']:.1f}",
    "ProdTop": bk["producto_top"], "SoporteTop": f"{100*bk['soporte_top']:.1f}",
    "ItemsTicket": f"{bk['items_por_ticket_media']:.2f}",
}
(TEX / "cifras.tex").write_text(
    "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()), encoding="utf8")
print("  [tex] cifras.tex")
print("\n[OK] Fragmentos LaTeX generados en", TEX)
