"""
Módulo compartido: carga, limpieza e ingeniería de variables.

Decisiones de diseño (justificadas en la auditoría, src/00-04):
  · Se excluyen CODIGO CLIENTE y CODIGO PRESTAMO: AUC univariado = 1.0000 porque
    el archivo histórico fue ordenado por ESTADO antes de numerarse
    (retirados 1..34,354 / renovados 34,355..88,828). En la base de predicción
    los códigos fueron renumerados 1..10,000, por lo que no contienen señal.
  · CREDITOS ANTERIORES == 1 es tautológico en el histórico (renovar implica >= 2
    créditos): 12,791 de 12,791 casos son retirados. Se modela aparte.
  · CAPITAL_CONCEDIDO presenta rejillas de valores casi disjuntas entre bases
    (PSI = 2.74). Se prefieren transformaciones de escala relativa (ratio, log).
"""
import pandas as pd, numpy as np, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42

# ---- Variables descartadas por fuga de información -------------------------
EXCLUIDAS_LEAKAGE = ["CODIGO CLIENTE", "CODIGO PRESTAMO"]

CATEGORICAS = ["REGION", "AGENCIA", "PRODUCTO", "SUBPRODUCTO",
               "TIPO_CREDITO", "SEXO", "ETAPA"]
# Nota: se excluye deliberadamente `amortizado` (= 1 - ratio_saldo_capital) por ser
# una transformación afín exacta del ratio: no aporta información y reparte de forma
# artificial la importancia SHAP entre dos columnas que son la misma señal.
NUMERICAS = ["TASA_NOMINAL", "CAPITAL_CONCEDIDO", "SALDO_CAPITAL",
             "CREDITOS_ANTERIORES", "ratio_saldo_capital", "log_capital",
             "log_saldo", "capital_por_credito", "saldo_por_credito",
             "tasa_rel_producto", "ratio_x_creditos",
             "etapa_num", "es_saldo_cero", "es_ratio_alto"]


# ---- Nombres comerciales de los segmentos (asignados tras perfilar, ver 08) ----
NOMBRES_CLUSTER = {
    0: "Microcrédito pequeño por vencer",
    1: "Crédito vigente en ciclo temprano",
    2: "Crédito mediano próximo a cancelar",
    3: "Primer crédito en deterioro",
    4: "Recurrente de monto bajo",
    5: "Cliente consolidado de alto monto",
}
ACCION_CLUSTER = {
    0: "Oferta de renovación anticipada con monto escalonado al alcanzar el 80% de amortización.",
    1: "Sin acción: monitoreo pasivo. Contactarlos gastaría presupuesto sin efecto.",
    2: "Renovación proactiva con incremento de monto; es cartera de valor en punto de decisión.",
    3: "Gestión de cobranza y reestructuración, no campaña comercial. La retención aquí es recuperación.",
    4: "Campaña de fidelización con beneficio por antigüedad y mejora de tasa.",
    5: "Gestión personalizada por ejecutivo de cuenta: concentran la mitad de la cartera.",
}


def _limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza categóricas de texto (espacios, mayúsculas) y renombra."""
    df = df.copy()
    df["TIPO_CREDITO"] = df["TIPO DE CREDITO"].astype(str).str.strip().str.upper()
    df["SEXO"] = df["SEXO"].astype(str).str.strip().str.upper().replace({"": "ND", "NAN": "ND"})
    df["ETAPA"] = df["ETAPA"].astype(str).str.strip().str.upper()
    df["CREDITOS_ANTERIORES"] = df["CREDITOS ANTERIORES"].astype(int)
    return df


def construir_features(df: pd.DataFrame, tasa_media_producto: pd.Series | None = None):
    """Ingeniería de variables. `tasa_media_producto` se ajusta SOLO con train."""
    df = _limpiar(df)

    cap = df["CAPITAL_CONCEDIDO"].replace(0, np.nan)
    df["ratio_saldo_capital"] = (df["SALDO_CAPITAL"] / cap).clip(0, 1.5).fillna(0)
    df["amortizado"] = 1 - df["ratio_saldo_capital"]          # % del crédito ya pagado
    df["log_capital"] = np.log1p(df["CAPITAL_CONCEDIDO"])
    df["log_saldo"] = np.log1p(df["SALDO_CAPITAL"])
    df["capital_por_credito"] = df["CAPITAL_CONCEDIDO"] / df["CREDITOS_ANTERIORES"]
    df["saldo_por_credito"] = df["SALDO_CAPITAL"] / df["CREDITOS_ANTERIORES"]
    df["ratio_x_creditos"] = df["ratio_saldo_capital"] * df["CREDITOS_ANTERIORES"]
    df["etapa_num"] = df["ETAPA"].map({"M1": 1, "M2": 2, "M3": 3}).fillna(1).astype(int)
    df["es_saldo_cero"] = (df["SALDO_CAPITAL"] <= 1).astype(int)
    df["es_ratio_alto"] = (df["ratio_saldo_capital"] > 0.85).astype(int)

    # Tasa relativa al producto: mide si al cliente le cobran por encima de su par
    if tasa_media_producto is None:
        tasa_media_producto = df.groupby("PRODUCTO")["TASA_NOMINAL"].mean()
    df["tasa_rel_producto"] = df["TASA_NOMINAL"] - df["PRODUCTO"].map(tasa_media_producto).fillna(
        df["TASA_NOMINAL"].mean())

    for c in CATEGORICAS:
        df[c] = df[c].astype("category")
    return df, tasa_media_producto


def cargar():
    hist = pd.read_excel(DATA / "Base de Datos Modelo.xlsx", sheet_name="Hoja1")
    pred = pd.read_excel(DATA / "Base de Datos Predicción.xlsx")
    hist["y"] = (hist["ESTADO"] == "Cliente Retirado").astype(int)
    return hist, pred


def alinear_categorias(train: pd.DataFrame, otros: list[pd.DataFrame]):
    """Fuerza las mismas categorías en todas las bases (evita fallos de LightGBM)."""
    for c in CATEGORICAS:
        cats = train[c].cat.categories
        for d in otros:
            d[c] = pd.Categorical(d[c].astype(str), categories=[str(x) for x in cats])
        train[c] = pd.Categorical(train[c].astype(str), categories=[str(x) for x in cats])
    return train, otros
