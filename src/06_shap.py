"""
Fase 4 — Variables de mayor impacto (pregunta 1c).
SHAP sobre el modelo final + contraste con la regresión logística + estabilidad.
"""
import pandas as pd, numpy as np, joblib, warnings, hashlib, json
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import shap, lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import features as F, estilo
estilo.aplicar()

ROOT = F.ROOT
OUT_T = ROOT / "outputs" / "tablas"; OUT_F = ROOT / "outputs" / "figuras"
OUT_F.mkdir(parents=True, exist_ok=True)
SEED = F.SEED

art = joblib.load(ROOT / "outputs" / "modelos" / "modelo_fuga.joblib")
lgb_full, CATS_REF, COLS = art["lgbm"], art["cats"], art["cols"]

hist, _ = F.cargar()
hist_f, _ = F.construir_features(hist)
mod = hist_f[hist_f["CREDITOS_ANTERIORES"] >= 2].copy()
X = mod[COLS].copy()
for c in F.CATEGORICAS:
    X[c] = X[c].astype(str)
y = mod["y"].values

def a_cat(df):
    d = df.copy()
    for c in F.CATEGORICAS:
        d[c] = pd.Categorical(d[c], categories=CATS_REF[c])
    return d

NOMBRES = {
    "ratio_saldo_capital": "Saldo pendiente / capital concedido",
    "amortizado": "Porcentaje del crédito ya amortizado",
    "SALDO_CAPITAL": "Saldo de capital vigente (Q)",
    "log_saldo": "Saldo vigente (escala logarítmica)",
    "CREDITOS_ANTERIORES": "Número de créditos del cliente",
    "ETAPA": "Etapa de deterioro (M1/M2/M3)",
    "etapa_num": "Etapa de deterioro (numérica)",
    "CAPITAL_CONCEDIDO": "Capital concedido (Q)",
    "log_capital": "Capital concedido (escala logarítmica)",
    "TASA_NOMINAL": "Tasa nominal",
    "tasa_rel_producto": "Tasa vs. promedio de su producto",
    "TIPO_CREDITO": "Tipo de crédito",
    "PRODUCTO": "Producto", "SUBPRODUCTO": "Subproducto",
    "AGENCIA": "Agencia", "REGION": "Región", "SEXO": "Sexo",
    "capital_por_credito": "Capital por crédito acumulado",
    "saldo_por_credito": "Saldo por crédito acumulado",
    "ratio_x_creditos": "Interacción saldo/capital x nº créditos",
    "es_saldo_cero": "Indicador de saldo cero",
    "es_ratio_alto": "Indicador de crédito recién desembolsado",
}

# ============================================================ 1. SHAP
print("Calculando SHAP (muestra de 8,000 clientes)...")
rng = np.random.RandomState(SEED)
idx = rng.choice(len(X), size=min(8000, len(X)), replace=False)
Xs = a_cat(X.iloc[idx])
expl = shap.TreeExplainer(lgb_full)
sv = expl.shap_values(Xs)
if isinstance(sv, list):
    sv = sv[1]
print("shap_values:", np.shape(sv))

imp = pd.DataFrame({
    "variable": COLS,
    "shap_medio_abs": np.abs(sv).mean(axis=0),
}).sort_values("shap_medio_abs", ascending=False).reset_index(drop=True)
imp["nombre_negocio"] = imp["variable"].map(NOMBRES).fillna(imp["variable"])
imp["peso_relativo_%"] = 100 * imp["shap_medio_abs"] / imp["shap_medio_abs"].sum()
imp["acumulado_%"] = imp["peso_relativo_%"].cumsum()
print("\n=== IMPORTANCIA SHAP ===")
print(imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
imp.to_csv(OUT_T / "06_importancia_shap.csv", index=False)

# --- Dirección del efecto: correlación entre valor y su SHAP ---------------
dirs = []
for i, c in enumerate(COLS):
    if c in F.CATEGORICAS:
        dirs.append("categórica")
    else:
        v = Xs[c].astype(float).values
        r = np.corrcoef(v, sv[:, i])[0, 1] if np.std(v) > 0 else 0
        dirs.append("aumenta la fuga" if r > 0.05 else
                    ("reduce la fuga" if r < -0.05 else "no monótona"))
imp2 = imp.merge(pd.DataFrame({"variable": COLS, "direccion": dirs}), on="variable")
imp2.to_csv(OUT_T / "06_importancia_shap.csv", index=False)

# --- Figura beeswarm -------------------------------------------------------
Xs_plot = Xs.copy()
for c in F.CATEGORICAS:
    Xs_plot[c] = Xs_plot[c].cat.codes
Xs_plot.columns = [NOMBRES.get(c, c) for c in Xs_plot.columns]
plt.figure(figsize=(8.5, 6))
shap.summary_plot(sv, Xs_plot, max_display=14, show=False, plot_size=None)
plt.title("Impacto de cada variable en la probabilidad de fuga (SHAP)", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig(OUT_F / "06_shap_beeswarm.png"); plt.close()

# --- Barra de importancia --------------------------------------------------
top = imp2.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(top["nombre_negocio"], top["peso_relativo_%"], color=estilo.PRIMARIO)
ax.set_xlabel("Peso relativo en la decisión del modelo (%)")
ax.set_title("Variables de mayor impacto en la fuga de clientes")
for i, (v, n) in enumerate(zip(top["peso_relativo_%"], top["nombre_negocio"])):
    ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8, color=estilo.GRIS)
plt.tight_layout(); plt.savefig(OUT_F / "06_shap_barras.png"); plt.close()

# --- Dependence plots de las 3 principales --------------------------------
for c in imp2["variable"].head(4):
    if c in F.CATEGORICAS:
        continue
    i = COLS.index(c)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    v = Xs[c].astype(float).values
    lo, hi = np.percentile(v, [0.5, 99.5])
    m = (v >= lo) & (v <= hi)
    ax.scatter(v[m], sv[m, i], s=4, alpha=0.25, color=estilo.PRIMARIO)
    ax.axhline(0, color=estilo.ACENTO, lw=0.8, ls="--")
    ax.set_xlabel(NOMBRES.get(c, c)); ax.set_ylabel("Efecto SHAP (log-odds)")
    ax.set_title(f"Efecto de: {NOMBRES.get(c,c)}", fontsize=10)
    plt.tight_layout(); plt.savefig(OUT_F / f"06_dep_{c}.png"); plt.close()

# ============================================================ 2. CONTRASTE CON LOGÍSTICA
print("\n=== CONTRASTE CON LA REGRESIÓN LOGÍSTICA ===")
pp = ColumnTransformer([
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                           sparse_output=False))]), F.CATEGORICAS),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), F.NUMERICAS)])
pipe = Pipeline([("pp", pp), ("clf", LogisticRegression(max_iter=2000,
                                                        class_weight="balanced", random_state=SEED))])
pipe.fit(X, y)
nombres_oh = pipe.named_steps["pp"].get_feature_names_out()
coef = pd.DataFrame({"variable": nombres_oh, "coef": pipe.named_steps["clf"].coef_[0]})
coef["abs"] = coef["coef"].abs()
coef["odds_ratio"] = np.exp(coef["coef"])
coef = coef.sort_values("abs", ascending=False)
print(coef.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
coef.to_csv(OUT_T / "06_coeficientes_logistica.csv", index=False)

top_shap = set(imp2["variable"].head(6))
top_log = set(c.split("__")[1].split("_")[0] for c in coef["variable"].head(10))
print(f"\nTop-6 SHAP: {sorted(top_shap)}")
print(f"Variables del top-10 logístico: {sorted(top_log)}")

# ============================================================ 3. ESTABILIDAD DEL RANKING
print("\n=== ESTABILIDAD DEL RANKING DE IMPORTANCIA (5 folds) ===")
grupo = (X.astype(str).agg("|".join, axis=1)
         .map(lambda s: int(hashlib.md5(s.encode()).hexdigest()[:12], 16)).values)
rank_folds = []
for k, (tr_i, _) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups=grupo)):
    m = lgb.LGBMClassifier(**art["params"])
    m.fit(a_cat(X.iloc[tr_i]), y[tr_i], categorical_feature=F.CATEGORICAS)
    e = shap.TreeExplainer(m)
    s = e.shap_values(a_cat(X.iloc[tr_i]).sample(3000, random_state=SEED))
    if isinstance(s, list):
        s = s[1]
    r = pd.Series(np.abs(s).mean(axis=0), index=COLS).rank(ascending=False)
    rank_folds.append(r.rename(f"fold_{k+1}"))
est = pd.concat(rank_folds, axis=1)
est["rango_medio"] = est.mean(axis=1)
est["desv_est"] = est.iloc[:, :5].std(axis=1)
est = est.sort_values("rango_medio")
est.index = [NOMBRES.get(i, i) for i in est.index]
print(est.head(12).to_string(float_format=lambda x: f"{x:.2f}"))
est.to_csv(OUT_T / "06_estabilidad_ranking.csv")
print(f"\n[OK] Figuras -> {OUT_F}")
