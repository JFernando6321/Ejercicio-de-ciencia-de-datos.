"""
Fase 8 — Construcción de los archivos de entrega (XLSX).
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings("ignore")
import features as F

ROOT = F.ROOT
OUT_T = ROOT / "outputs" / "tablas"; OUT_M = ROOT / "outputs" / "modelos"
ENT = ROOT / "entregables"; ENT.mkdir(exist_ok=True)
APELLIDO = "Vicente"

NOMBRES_CLUSTER = F.NOMBRES_CLUSTER
ACCION_CLUSTER = F.ACCION_CLUSTER

# ============================================================ 1. BASE ORIGINAL + PREDICCIONES
pred_orig = pd.read_excel(ROOT / "data" / "Base de Datos Predicción.xlsx")
pred_orig = pred_orig.drop(columns=[c for c in pred_orig.columns
                                    if c.strip() in ("Probabilidad Fuga", "Predicción")])

pr = pd.read_csv(OUT_M / "predicciones.csv")
cl = pd.read_csv(OUT_M / "clusters_pred.csv")

ent = (pred_orig
       .merge(pr, on=["CODIGO CLIENTE", "CODIGO PRESTAMO"], how="left")
       .merge(cl, on="CODIGO CLIENTE", how="left"))

assert len(ent) == 10000 and ent["prob_fuga"].notna().all(), "Fallo en el cruce de predicciones"

ent["Probabilidad Fuga"] = ent["prob_fuga"].round(4)
ent["prob_fuga_pct"] = (ent["prob_fuga"] * 100).round(1)
ent["Predicción"] = np.where(ent["recomendar_campana"] == "Sí", "Se retira", "Se mantiene")
ent["cluster_nombre"] = ent["cluster"].map(NOMBRES_CLUSTER)
ent["accion_recomendada"] = ent["cluster"].map(ACCION_CLUSTER)
ent["valor_cliente"] = ent["valor_cliente"].round(2)
ent["beneficio_esperado"] = ent["beneficio_esperado"].round(2)

COLS_ENT = ["CODIGO CLIENTE", "CODIGO PRESTAMO", "REGION", "AGENCIA", "PRODUCTO",
            "SUBPRODUCTO", "TIPO DE CREDITO", "TASA_NOMINAL", "SEXO",
            "CAPITAL_CONCEDIDO", "SALDO_CAPITAL", "ETAPA", "CREDITOS ANTERIORES",
            "Probabilidad Fuga", "prob_fuga_pct", "decil_riesgo", "segmento_riesgo",
            "cluster", "cluster_nombre", "valor_cliente", "beneficio_esperado",
            "prioridad_campana", "recomendar_campana", "accion_recomendada", "Predicción"]
ent = ent[COLS_ENT].sort_values("prioridad_campana")

DICC = pd.DataFrame([
    ("Probabilidad Fuga", "Probabilidad calibrada de que el cliente no renueve (0 a 1, 4 decimales)."),
    ("prob_fuga_pct", "La misma probabilidad expresada en porcentaje con 1 decimal."),
    ("decil_riesgo", "1 = 10% de mayor riesgo ... 10 = 10% de menor riesgo."),
    ("segmento_riesgo", "Alto (p>0.50), Medio (0.20-0.50), Bajo (p<=0.20)."),
    ("cluster", "Segmento asignado por el modelo de clustering (k=6)."),
    ("cluster_nombre", "Nombre comercial del segmento."),
    ("valor_cliente", "Margen financiero neto estimado del próximo ciclo de crédito (Q)."),
    ("beneficio_esperado", "prob_fuga x efectividad(20%) x valor_cliente - costo de contacto($25 COP)."),
    ("prioridad_campana", "Orden de contacto: 1 = contactar primero. Útil si hay límite de capacidad."),
    ("recomendar_campana", "Sí = el beneficio esperado de contactarlo es positivo."),
    ("accion_recomendada", "Acción comercial sugerida según el segmento."),
    ("Predicción", "'Se retira' para los clientes recomendados a campaña; 'Se mantiene' en caso contrario."),
], columns=["Columna", "Definición"])

f1 = ENT / f"Genesis_Churn_Prediccion_Julio_{APELLIDO}.xlsx"
with pd.ExcelWriter(f1, engine="xlsxwriter") as w:
    ent.to_excel(w, sheet_name="Predicción 10,000", index=False)
    DICC.to_excel(w, sheet_name="Diccionario", index=False)
    pd.read_csv(OUT_T / "07_deciles_prediccion.csv").to_excel(w, sheet_name="Deciles", index=False)
    pd.read_csv(OUT_T / "07_sensibilidad_campana.csv").to_excel(w, sheet_name="Sensibilidad campaña", index=False)
    pd.read_csv(OUT_T / "07_capacidad_campana.csv").to_excel(w, sheet_name="Capacidad", index=False)
    for sh, ancho in [("Predicción 10,000", 16), ("Diccionario", 30)]:
        w.sheets[sh].set_column(0, 30, ancho)
print(f"[OK] {f1.name}  ({len(ent):,} filas x {len(ent.columns)} columnas)")

# ============================================================ 2. SEGMENTACIÓN
perfil = pd.read_csv(OUT_T / "08_perfil_clusters.csv")
perfil["nombre"] = perfil["cluster"].map(NOMBRES_CLUSTER)
perfil["accion_recomendada"] = perfil["cluster"].map(ACCION_CLUSTER)
res_act = pd.read_csv(OUT_T / "08_resumen_clusters_activos.csv")
res_act["nombre"] = res_act["cluster"].map(NOMBRES_CLUSTER)

f2 = ENT / f"Genesis_Segmentacion_Julio_{APELLIDO}.xlsx"
with pd.ExcelWriter(f2, engine="xlsxwriter") as w:
    perfil.to_excel(w, sheet_name="Perfil de segmentos", index=False)
    res_act.to_excel(w, sheet_name="Segmentos vs activos", index=False)
    pd.read_csv(OUT_T / "08_indice_clusters.csv").to_excel(w, sheet_name="Índice base 100", index=False)
    pd.read_csv(OUT_T / "08_cruce_cluster_riesgo_n.csv").to_excel(w, sheet_name="Cruce segmento-riesgo (n)", index=False)
    pd.read_csv(OUT_T / "08_cruce_cluster_riesgo_saldo.csv").to_excel(w, sheet_name="Cruce segmento-riesgo (Q)", index=False)
    pd.read_csv(OUT_T / "08_eleccion_k.csv").to_excel(w, sheet_name="Elección de k", index=False)
    pd.read_csv(OUT_T / "08_estabilidad_ari.csv").to_excel(w, sheet_name="Estabilidad ARI", index=False)
    ent[["CODIGO CLIENTE", "cluster", "cluster_nombre", "Probabilidad Fuga",
         "segmento_riesgo"]].to_excel(w, sheet_name="Asignación por cliente", index=False)
print(f"[OK] {f2.name}")

# ============================================================ 3. REGLAS DE ASOCIACIÓN
f3 = ENT / f"Genesis_Reglas_Asociacion_Julio_{APELLIDO}.xlsx"
with pd.ExcelWriter(f3, engine="xlsxwriter") as w:
    pd.read_csv(OUT_T / "09_reglas_seleccionadas.csv").to_excel(w, sheet_name="Canastas seleccionadas", index=False)
    pd.read_csv(OUT_T / "09_combos.csv").to_excel(w, sheet_name="Combos propuestos", index=False)
    pd.read_csv(OUT_T / "09_reglas_todas.csv").to_excel(w, sheet_name="Todas las reglas", index=False)
    pd.read_csv(OUT_T / "09_soporte_productos.csv").to_excel(w, sheet_name="Penetración productos", index=False)
print(f"[OK] {f3.name}")

# ============================================================ 4. AUDITORÍA
f4 = ENT / f"Genesis_Auditoria_Datos_Julio_{APELLIDO}.xlsx"
with pd.ExcelWriter(f4, engine="xlsxwriter") as w:
    pd.read_csv(OUT_T / "00_diccionario_preliminar.csv").to_excel(w, sheet_name="Diccionario original", index=False)
    pd.read_csv(OUT_T / "01_auc_univariado.csv").to_excel(w, sheet_name="AUC univariado (leakage)", index=False)
    pd.read_csv(OUT_T / "01_deriva_psi.csv").to_excel(w, sheet_name="Deriva PSI", index=False)
    pd.read_csv(OUT_T / "05_escalera_modelos.csv").to_excel(w, sheet_name="Escalera de modelos", index=False)
    pd.read_csv(OUT_T / "05_deciles_holdout.csv").to_excel(w, sheet_name="Calibración holdout", index=False)
    pd.read_csv(OUT_T / "06_importancia_shap.csv").to_excel(w, sheet_name="Importancia SHAP", index=False)
    pd.read_csv(OUT_T / "06_estabilidad_ranking.csv").to_excel(w, sheet_name="Estabilidad importancia", index=False)
print(f"[OK] {f4.name}")

# ============================================================ 5. RESUMEN
print("\n=== RESUMEN DE LA ENTREGA ===")
print(f"Clientes con recomendación de campaña : {(ent.recomendar_campana=='Sí').sum():,}")
print(f"Probabilidad media de fuga            : {ent['Probabilidad Fuga'].mean():.4f}")
print(f"Fugas esperadas                       : {ent['Probabilidad Fuga'].sum():,.0f}")
print(f"Saldo total de la base                : Q{ent.SALDO_CAPITAL.sum():,.0f}")
print("\nDistribución por segmento de riesgo:")
print(ent.groupby("segmento_riesgo").agg(clientes=("CODIGO CLIENTE", "size"),
                                          prob_media=("Probabilidad Fuga", "mean"),
                                          saldo=("SALDO_CAPITAL", "sum")).to_string(float_format=lambda x: f"{x:,.3f}"))
print("\nDistribución por segmento (cluster):")
print(ent.groupby(["cluster", "cluster_nombre"]).agg(
    clientes=("CODIGO CLIENTE", "size"), prob_media=("Probabilidad Fuga", "mean"),
    campana=("recomendar_campana", lambda s: (s == "Sí").sum()),
    saldo=("SALDO_CAPITAL", "sum")).to_string(float_format=lambda x: f"{x:,.3f}"))
print("\nMuestra de las 10 primeras filas de la entrega:")
print(ent.head(10)[["CODIGO CLIENTE", "Probabilidad Fuga", "prob_fuga_pct", "decil_riesgo",
                    "segmento_riesgo", "cluster_nombre", "recomendar_campana"]].to_string(index=False))
