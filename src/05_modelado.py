"""
Fase 3 — Escalera de modelos, validación y calibración (preguntas 1a, 1b).

Diseño de validación
--------------------
La base NO tiene fechas ni estructura de panel (una fila = un cliente,
78,829 clientes únicos), por lo que una partición fuera de tiempo es
IMPOSIBLE. Se declara explícitamente y se sustituye por:
  · Partición por grupos en tres bloques: entrenamiento 60% / calibración 20%
    / prueba (holdout) 20%. El bloque de calibración es independiente del de
    entrenamiento y el holdout no se toca hasta el reporte final.
  · Validación cruzada 5-fold AGRUPADA por vector de atributos (hash), porque
    7,803 filas tienen atributos idénticos a otra fila; con K-fold simple esas
    filas caerían en train y test a la vez e inflarían las métricas.

Población de entrenamiento
--------------------------
Sólo CREDITOS ANTERIORES >= 2 (n = 66,038). El nivel 1 es tautológico
(12,791/12,791 retirados) y se trata en 06_prediccion.py.
"""
import pandas as pd, numpy as np, pathlib, json, warnings, hashlib
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve)
from sklearn.calibration import calibration_curve
from scipy.stats import rankdata
import lightgbm as lgb
import joblib

import features as F

ROOT = F.ROOT
OUT_T = ROOT / "outputs" / "tablas"; OUT_T.mkdir(parents=True, exist_ok=True)
OUT_M = ROOT / "outputs" / "modelos"; OUT_M.mkdir(parents=True, exist_ok=True)
SEED = F.SEED

# ============================================================ 1. DATOS
hist, pred = F.cargar()
hist_f, tasa_prod = F.construir_features(hist)

modelable = hist_f[hist_f["CREDITOS_ANTERIORES"] >= 2].copy()
print(f"Población modelable (CREDITOS ANTERIORES >= 2): n={len(modelable):,}  "
      f"tasa de fuga={modelable.y.mean():.4f}")

COLS = F.CATEGORICAS + F.NUMERICAS
X = modelable[COLS].copy()
for c in F.CATEGORICAS:                       # a texto: control total de categorías
    X[c] = X[c].astype(str)
y = modelable["y"].values

grupo = (X.astype(str).agg("|".join, axis=1)
         .map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:12], 16)).values)
print(f"Grupos únicos de atributos: {len(np.unique(grupo)):,} sobre {len(X):,} filas")

g_unicos = np.unique(grupo)
g_rest, g_te = train_test_split(g_unicos, test_size=0.20, random_state=SEED)
g_tr, g_cal = train_test_split(g_rest, test_size=0.25, random_state=SEED)   # 0.25*0.8 = 0.20

m_tr, m_cal, m_te = (np.isin(grupo, g) for g in (g_tr, g_cal, g_te))
X_tr, y_tr, grp_tr = X[m_tr], y[m_tr], grupo[m_tr]
X_cal, y_cal = X[m_cal], y[m_cal]
X_te, y_te = X[m_te], y[m_te]
print(f"Train={len(X_tr):,} (fuga {y_tr.mean():.4f}) | "
      f"Calib={len(X_cal):,} ({y_cal.mean():.4f}) | Holdout={len(X_te):,} ({y_te.mean():.4f})")

CATS_REF = {c: sorted(X[c].unique()) for c in F.CATEGORICAS}

def a_categorico(df):
    d = df.copy()
    for c in F.CATEGORICAS:
        d[c] = pd.Categorical(d[c], categories=CATS_REF[c])
    return d

# ============================================================ 2. MÉTRICAS
def lift_decil(y_true, p, k=1):
    n = int(len(p) * 0.1 * k); idx = np.argsort(-p)[:n]
    return y_true[idx].mean() / y_true.mean()

def captura(y_true, p, frac=0.2):
    n = int(len(p) * frac); idx = np.argsort(-p)[:n]
    return y_true[idx].sum() / y_true.sum()

def ks(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p); return float(np.max(tpr - fpr))

def evaluar(nombre, y_true, p):
    return {"modelo": nombre,
            "AUC_PR": average_precision_score(y_true, p),
            "AUC_ROC": roc_auc_score(y_true, p),
            "KS": ks(y_true, p),
            "Brier": brier_score_loss(y_true, p),
            "lift_decil_1": lift_decil(y_true, p),
            "captura_top20pct": captura(y_true, p, 0.2)}

# ============================================================ 3. ESCALERA DE MODELOS
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X_tr, y_tr, groups=grp_tr))

def preproc_lineal():
    return ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                               sparse_output=False))]), F.CATEGORICAS),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), F.NUMERICAS)])

LGBM_PARAMS = dict(objective="binary", n_estimators=700, learning_rate=0.05,
                   num_leaves=48, min_child_samples=60, subsample=0.85,
                   subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.5,
                   reg_lambda=2.0, random_state=SEED, n_jobs=-1, verbose=-1)

resultados, oof_store = [], {}

# ---- Nivel 0: regla de negocio -------------------------------------------
regla = ((X_tr["ratio_saldo_capital"] < 0.15) | (X_tr["etapa_num"] >= 2)).astype(float).values
resultados.append({**evaluar("0. Regla de negocio (ratio<0.15 ó ETAPA>=M2)", y_tr, regla),
                   "conjunto": "train"})

def cv_modelo(nombre, constructor, es_lgbm=False):
    oof = np.zeros(len(X_tr))
    for tr_i, va_i in folds:
        m = constructor()
        if es_lgbm:
            m.fit(a_categorico(X_tr.iloc[tr_i]), y_tr[tr_i], categorical_feature=F.CATEGORICAS)
            oof[va_i] = m.predict_proba(a_categorico(X_tr.iloc[va_i]))[:, 1]
        else:
            m.fit(X_tr.iloc[tr_i], y_tr[tr_i])
            oof[va_i] = m.predict_proba(X_tr.iloc[va_i])[:, 1]
    oof_store[nombre] = oof
    r = {**evaluar(nombre, y_tr, oof), "conjunto": "CV 5-fold agrupada (OOF)"}
    resultados.append(r)
    print(f"  {nombre:46s} AUC-PR={r['AUC_PR']:.4f}  AUC-ROC={r['AUC_ROC']:.4f}  KS={r['KS']:.4f}")

print("\n--- Validación cruzada agrupada (out-of-fold) ---")
cv_modelo("1. Regresión logística",
          lambda: Pipeline([("pp", preproc_lineal()),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                       random_state=SEED))]))
cv_modelo("2. Random Forest",
          lambda: Pipeline([("pp", preproc_lineal()),
                            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=15,
                                                           class_weight="balanced_subsample",
                                                           n_jobs=-1, random_state=SEED))]))
cv_modelo("3. LightGBM", lambda: lgb.LGBMClassifier(**LGBM_PARAMS), es_lgbm=True)

oof_ens = np.mean([rankdata(oof_store[k]) / len(X_tr) for k in oof_store], axis=0)
r = {**evaluar("4. Ensamble (promedio de rangos)", y_tr, oof_ens), "conjunto": "CV 5-fold agrupada (OOF)"}
resultados.append(r)
print(f"  {r['modelo']:46s} AUC-PR={r['AUC_PR']:.4f}  AUC-ROC={r['AUC_ROC']:.4f}  KS={r['KS']:.4f}")

# ---- Nivel 3b: prueba de robustez, SIN variables derivadas del saldo ------
SIN_SALDO = [c for c in COLS if c not in
             ["SALDO_CAPITAL", "log_saldo", "ratio_saldo_capital", "amortizado",
              "saldo_por_credito", "ratio_x_creditos", "es_saldo_cero", "es_ratio_alto"]]
oof_ss = np.zeros(len(X_tr))
for tr_i, va_i in folds:
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(a_categorico(X_tr.iloc[tr_i])[SIN_SALDO], y_tr[tr_i], categorical_feature=F.CATEGORICAS)
    oof_ss[va_i] = m.predict_proba(a_categorico(X_tr.iloc[va_i])[SIN_SALDO])[:, 1]
r = {**evaluar("3b. LightGBM SIN variables de saldo (robustez)", y_tr, oof_ss),
     "conjunto": "CV 5-fold agrupada (OOF)"}
resultados.append(r)
print(f"  {r['modelo']:46s} AUC-PR={r['AUC_PR']:.4f}  AUC-ROC={r['AUC_ROC']:.4f}  KS={r['KS']:.4f}")

tab = pd.DataFrame(resultados)[["modelo", "conjunto", "AUC_PR", "AUC_ROC", "KS",
                                "Brier", "lift_decil_1", "captura_top20pct"]]
print("\n=== ESCALERA DE MODELOS ===")
print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
tab.to_csv(OUT_T / "05_escalera_modelos.csv", index=False)

# ============================================================ 4. CALIBRACIÓN EXPLÍCITA
print("\n--- Calibración isotónica: ajustada en CALIB, evaluada en HOLDOUT ---")
base = lgb.LGBMClassifier(**LGBM_PARAMS)
base.fit(a_categorico(X_tr), y_tr, categorical_feature=F.CATEGORICAS)

p_cal = base.predict_proba(a_categorico(X_cal))[:, 1]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_cal, y_cal)

p_te_raw = base.predict_proba(a_categorico(X_te))[:, 1]
p_te_iso = iso.predict(p_te_raw)

# Platt como comparación
from sklearn.linear_model import LogisticRegression as LR
platt = LR(max_iter=1000).fit(p_cal.reshape(-1, 1), y_cal)
p_te_platt = platt.predict_proba(p_te_raw.reshape(-1, 1))[:, 1]

final = pd.DataFrame([
    {**evaluar("LightGBM sin calibrar", y_te, p_te_raw), "conjunto": "HOLDOUT 20%"},
    {**evaluar("LightGBM + isotónica", y_te, p_te_iso), "conjunto": "HOLDOUT 20%"},
    {**evaluar("LightGBM + Platt", y_te, p_te_platt), "conjunto": "HOLDOUT 20%"}])
print(final.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
final.to_csv(OUT_T / "05_holdout_calibracion.csv", index=False)

usar_iso = brier_score_loss(y_te, p_te_iso) < brier_score_loss(y_te, p_te_raw)
p_final = p_te_iso if usar_iso else p_te_raw
print(f"\n>> Calibrador seleccionado: {'isotónica' if usar_iso else 'ninguno (el modelo ya está calibrado)'}")

# ---- Tabla decil: predicho vs observado ----------------------------------
d = pd.DataFrame({"y": y_te, "p": p_final})
d["decil"] = pd.qcut(d["p"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
dec = d.groupby("decil").agg(n=("y", "size"), prob_media_predicha=("p", "mean"),
                             tasa_real_observada=("y", "mean"), fugas=("y", "sum"))
dec["lift"] = dec["tasa_real_observada"] / d["y"].mean()
dec["captura_acum_%"] = 100 * dec["fugas"].cumsum() / dec["fugas"].sum()
print("\n=== CALIBRACIÓN POR DECIL (holdout) ===")
print(dec.to_string(float_format=lambda x: f"{x:.4f}"))
dec.to_csv(OUT_T / "05_deciles_holdout.csv")

rel = []
for nom, p in [("sin calibrar", p_te_raw), ("isotónica", p_te_iso)]:
    ft, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    rel.append(pd.DataFrame({"modelo": nom, "prob_predicha": mp, "frac_positivos": ft}))
pd.concat(rel).to_csv(OUT_T / "05_curva_calibracion.csv", index=False)

# ============================================================ 5. MODELO FINAL
# Reentrenar con el 100% de la población modelable y recalibrar con OOF agrupado.
print("\n--- Modelo final: reentrenamiento con el 100% + calibración por OOF ---")
folds_full = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups=grupo))
oof_full = np.zeros(len(X))
for tr_i, va_i in folds_full:
    m = lgb.LGBMClassifier(**LGBM_PARAMS)
    m.fit(a_categorico(X.iloc[tr_i]), y[tr_i], categorical_feature=F.CATEGORICAS)
    oof_full[va_i] = m.predict_proba(a_categorico(X.iloc[va_i]))[:, 1]
print(f"OOF completo: AUC-PR={average_precision_score(y, oof_full):.4f}  "
      f"AUC-ROC={roc_auc_score(y, oof_full):.4f}  Brier={brier_score_loss(y, oof_full):.4f}")

iso_full = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof_full, y)
lgb_full = lgb.LGBMClassifier(**LGBM_PARAMS)
lgb_full.fit(a_categorico(X), y, categorical_feature=F.CATEGORICAS)

joblib.dump({"lgbm": lgb_full, "isotonica": iso_full if usar_iso else None,
             "cats": CATS_REF, "cols": COLS, "tasa_prod": tasa_prod,
             "prevalencia_k2": float(y.mean()), "seed": SEED,
             "params": LGBM_PARAMS},
            OUT_M / "modelo_fuga.joblib")
print(f"[OK] Modelo guardado -> {OUT_M/'modelo_fuga.joblib'}")

resumen = {"n_modelable": int(len(X)), "prevalencia_k2": float(y.mean()),
           "calibrador": "isotonica" if usar_iso else "ninguno",
           "holdout_auc_pr": float(average_precision_score(y_te, p_final)),
           "holdout_auc_roc": float(roc_auc_score(y_te, p_final)),
           "holdout_ks": ks(y_te, p_final),
           "holdout_brier_sin_calibrar": float(brier_score_loss(y_te, p_te_raw)),
           "holdout_brier_isotonica": float(brier_score_loss(y_te, p_te_iso)),
           "holdout_brier_platt": float(brier_score_loss(y_te, p_te_platt)),
           "captura_top10": float(captura(y_te, p_final, .1)),
           "captura_top20": float(captura(y_te, p_final, .2)),
           "captura_top30": float(captura(y_te, p_final, .3)),
           "lift_decil_1": float(lift_decil(y_te, p_final)),
           "auc_roc_sin_variables_saldo": float(roc_auc_score(y_tr, oof_ss))}
json.dump(resumen, open(OUT_T / "05_resumen_modelo.json", "w"), indent=2)
np.save(OUT_M / "oof_full.npy", oof_full)
print(json.dumps(resumen, indent=2))
