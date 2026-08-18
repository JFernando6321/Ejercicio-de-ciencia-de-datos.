"""
Fase 6 — Segmentación de clientes (pregunta 1f).

Diseño
------
· Marco RFM adaptado a cartera de crédito, con variables elegidas por intención
  (no se mete todo el catálogo):
    Monto      -> log(capital concedido)
    Frecuencia -> nº de créditos del cliente
    Ciclo      -> saldo pendiente / capital concedido (posición en la vida del crédito)
    Precio     -> tasa nominal
    Deterioro  -> etapa M1/M2/M3
· Sólo variables numéricas: k-means con one-hot sobre categóricas de alta
  cardinalidad (138 agencias) produce distancias sin sentido. Las categóricas
  se usan para PERFILAR los clusters ya formados.
· RobustScaler por la presencia de outliers de monto.
· k se elige con codo + silueta + Davies-Bouldin, y se valida con estabilidad
  por bootstrap medida con el índice de Rand ajustado (ARI).
· El modelo se AJUSTA en el histórico (78,829 clientes) y se APLICA a los
  10,000 activos, lo que permite cruzar cada segmento con su tasa de fuga
  histórica observada y con la probabilidad media de los activos.
"""
import pandas as pd, numpy as np, joblib, json, warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)
from sklearn.decomposition import PCA

import features as F, estilo
estilo.aplicar()

ROOT = F.ROOT
OUT_T = ROOT / "outputs" / "tablas"; OUT_F = ROOT / "outputs" / "figuras"
SEED = F.SEED
rng = np.random.RandomState(SEED)

VARS_CLUSTER = ["log_capital", "log_creditos", "ratio_saldo_capital",
                "TASA_NOMINAL", "etapa_num"]

hist, pred = F.cargar()
hist_f, tasa_prod = F.construir_features(hist)
pred_f, _ = F.construir_features(pred, tasa_media_producto=tasa_prod)

# Winsorización en los percentiles 1 y 99 del histórico. Sin esto, k-means gasta
# segmentos enteros en un puñado de casos extremos (hasta 106 créditos), lo que
# produce clusters de 50-80 clientes, inservibles para el área comercial, y
# derrumba la estabilidad medida por ARI.
for d in (hist_f, pred_f):
    d["log_creditos"] = np.log1p(d["CREDITOS_ANTERIORES"])
LIM = {v: (hist_f[v].quantile(0.01), hist_f[v].quantile(0.99)) for v in VARS_CLUSTER}
print("Límites de winsorización (p1, p99):")
for v, (lo, hi) in LIM.items():
    print(f"  {v:22s} [{lo:.4f}, {hi:.4f}]")

def winsor(df):
    return pd.DataFrame({v: df[v].clip(*LIM[v]) for v in VARS_CLUSTER}, index=df.index)

sc = RobustScaler().fit(winsor(hist_f))
Z_hist = sc.transform(winsor(hist_f))
Z_pred = sc.transform(winsor(pred_f))

# ============================================================ 1. ELECCIÓN DE k
print("=== ELECCIÓN DEL NÚMERO DE SEGMENTOS ===")
sub = Z_hist[rng.choice(len(Z_hist), 20000, replace=False)]
diag = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(sub)
    diag.append({"k": k, "inercia": km.inertia_,
                 "silueta": silhouette_score(sub, km.labels_, sample_size=10000, random_state=SEED),
                 "davies_bouldin": davies_bouldin_score(sub, km.labels_),
                 "calinski_harabasz": calinski_harabasz_score(sub, km.labels_)})
diag = pd.DataFrame(diag)
print(diag.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
diag.to_csv(OUT_T / "08_eleccion_k.csv", index=False)

# ============================================================ 2. ESTABILIDAD (BOOTSTRAP + ARI)
print("\n=== ESTABILIDAD POR BOOTSTRAP (ARI entre particiones) ===")
est = []
for k in range(3, 8):
    base = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(sub)
    aris = []
    for b in range(10):
        idx = rng.choice(len(sub), len(sub), replace=True)
        kb = KMeans(n_clusters=k, n_init=10, random_state=SEED + b + 1).fit(sub[idx])
        aris.append(adjusted_rand_score(base.predict(sub[idx]), kb.labels_))
    est.append({"k": k, "ARI_medio": np.mean(aris), "ARI_min": np.min(aris), "ARI_desv": np.std(aris)})
    print(f"  k={k}  ARI medio={np.mean(aris):.4f}  min={np.min(aris):.4f}  desv={np.std(aris):.4f}")
est = pd.DataFrame(est)
est.to_csv(OUT_T / "08_estabilidad_ari.csv", index=False)

# k = 6 domina a las alternativas en los tres criterios simultáneamente:
# mejor silueta (0.262 vs 0.244 de k=5), menor Davies-Bouldin (1.187 vs 1.292) y
# una estabilidad por bootstrap muy superior (ARI 0.971 con desviación 0.012,
# frente a 0.794 con desviación 0.226 de k=5). Además sigue siendo un número de
# segmentos manejable para un área comercial.
K = 6
print(f"\n>> k seleccionado = {K}: mejor silueta, menor Davies-Bouldin y ARI de "
      f"estabilidad muy superior; sigue siendo operable para el área comercial.")

# ============================================================ 3. MODELO FINAL
km = KMeans(n_clusters=K, n_init=25, random_state=SEED).fit(Z_hist)
hist_f["cluster"] = km.labels_
pred_f["cluster"] = km.predict(Z_pred)

print(f"\nSilueta (muestra 20k) del modelo final: "
      f"{silhouette_score(Z_hist[:20000], km.labels_[:20000]):.4f}")

# ============================================================ 4. PERFILADO
perfil = hist_f.groupby("cluster").agg(
    n=("y", "size"),
    tasa_fuga_historica=("y", "mean"),
    capital_medio=("CAPITAL_CONCEDIDO", "median"),
    saldo_medio=("SALDO_CAPITAL", "median"),
    ratio_ciclo=("ratio_saldo_capital", "median"),
    creditos=("CREDITOS_ANTERIORES", "median"),
    tasa_nominal=("TASA_NOMINAL", "median"),
    etapa_media=("etapa_num", "mean"),
    pct_M2_M3=("etapa_num", lambda s: (s >= 2).mean()),
    pct_primer_credito=("CREDITOS_ANTERIORES", lambda s: (s == 1).mean()),
    pct_mujeres=("SEXO", lambda s: (s == "F").mean()),
    saldo_total=("SALDO_CAPITAL", "sum"))
perfil["%_clientes"] = 100 * perfil["n"] / perfil["n"].sum()
perfil["%_cartera"] = 100 * perfil["saldo_total"] / perfil["saldo_total"].sum()

act = pred_f.groupby("cluster").agg(n_activos=("CODIGO CLIENTE", "size"),
                                    saldo_activos=("SALDO_CAPITAL", "sum"))
perfil = perfil.join(act)

print("\n=== PERFIL DE LOS SEGMENTOS (histórico) ===")
print(perfil.to_string(float_format=lambda x: f"{x:,.3f}"))

# Índice vs promedio global (base 100)
idx_vars = ["capital_medio", "saldo_medio", "ratio_ciclo", "creditos", "tasa_nominal", "etapa_media"]
glob = {"capital_medio": hist_f["CAPITAL_CONCEDIDO"].median(),
        "saldo_medio": hist_f["SALDO_CAPITAL"].median(),
        "ratio_ciclo": hist_f["ratio_saldo_capital"].median(),
        "creditos": hist_f["CREDITOS_ANTERIORES"].median(),
        "tasa_nominal": hist_f["TASA_NOMINAL"].median(),
        "etapa_media": hist_f["etapa_num"].mean()}
indice = pd.DataFrame({v: 100 * perfil[v] / glob[v] for v in idx_vars})
print("\n=== ÍNDICE VS PROMEDIO GLOBAL (base 100) ===")
print(indice.to_string(float_format=lambda x: f"{x:,.0f}"))
indice.to_csv(OUT_T / "08_indice_clusters.csv")

# ---- Nombres de negocio (asignados según el perfil observado) -------------
orden_riesgo = perfil["tasa_fuga_historica"].sort_values(ascending=False)
print("\nOrden por tasa de fuga histórica:", orden_riesgo.round(3).to_dict())

# ============================================================ 5. CRUCE CLUSTER x RIESGO
predic = pd.read_csv(ROOT / "outputs" / "modelos" / "predicciones.csv")
pred_f = pred_f.merge(predic[["CODIGO CLIENTE", "prob_fuga", "decil_riesgo",
                              "segmento_riesgo", "recomendar_campana"]],
                      on="CODIGO CLIENTE", how="left")

cruce_n = pd.crosstab(pred_f["cluster"], pred_f["segmento_riesgo"])
cruce_saldo = pd.crosstab(pred_f["cluster"], pred_f["segmento_riesgo"],
                          values=pred_f["SALDO_CAPITAL"], aggfunc="sum").fillna(0)
print("\n=== CRUCE SEGMENTO x RIESGO — Nº DE CLIENTES ACTIVOS ===")
print(cruce_n.to_string())
print("\n=== CRUCE SEGMENTO x RIESGO — SALDO (Q) ===")
print(cruce_saldo.to_string(float_format=lambda x: f"{x:,.0f}"))
cruce_n.to_csv(OUT_T / "08_cruce_cluster_riesgo_n.csv")
cruce_saldo.to_csv(OUT_T / "08_cruce_cluster_riesgo_saldo.csv")

resumen_act = pred_f.groupby("cluster").agg(
    clientes_activos=("prob_fuga", "size"),
    prob_media_fuga=("prob_fuga", "mean"),
    fugas_esperadas=("prob_fuga", "sum"),
    saldo=("SALDO_CAPITAL", "sum"),
    recomendados_campana=("recomendar_campana", lambda s: (s == "Sí").sum()))
resumen_act = resumen_act.join(perfil[["tasa_fuga_historica", "%_clientes"]])
print("\n=== SEGMENTOS: HISTÓRICO vs ACTIVOS (el cruce que amarra el ejercicio) ===")
print(resumen_act.to_string(float_format=lambda x: f"{x:,.4f}"))
resumen_act.to_csv(OUT_T / "08_resumen_clusters_activos.csv")
perfil.to_csv(OUT_T / "08_perfil_clusters.csv")

# ============================================================ 6. FIGURAS
pca = PCA(n_components=2, random_state=SEED).fit(Z_hist)
P = pca.transform(Z_hist[rng.choice(len(Z_hist), 12000, replace=False)])
lab = km.labels_[rng.choice(len(Z_hist), 12000, replace=False)]
fig, ax = plt.subplots(figsize=(5.8, 4.6))
for c in range(K):
    m = lab == c
    ax.scatter(P[m, 0], P[m, 1], s=5, alpha=.35, color=estilo.PALETA[c], label=f"Segmento {c}")
ax.set_xlabel(f"Componente 1 ({pca.explained_variance_ratio_[0]:.0%} de la varianza)")
ax.set_ylabel(f"Componente 2 ({pca.explained_variance_ratio_[1]:.0%})")
ax.set_title("Separación de los segmentos (2 componentes principales)")
ax.legend(markerscale=3, fontsize=8)
plt.tight_layout(); plt.savefig(OUT_F / "08_pca_clusters.png"); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
axes[0].plot(diag["k"], diag["inercia"], marker="o", color=estilo.PRIMARIO)
axes[0].axvline(K, color=estilo.ACENTO, ls="--"); axes[0].set_title("Codo (inercia)")
axes[1].plot(diag["k"], diag["silueta"], marker="o", color=estilo.PRIMARIO)
axes[1].axvline(K, color=estilo.ACENTO, ls="--"); axes[1].set_title("Coeficiente de silueta")
axes[2].plot(diag["k"], diag["davies_bouldin"], marker="o", color=estilo.PRIMARIO)
axes[2].axvline(K, color=estilo.ACENTO, ls="--"); axes[2].set_title("Davies-Bouldin (menor mejor)")
for a in axes: a.set_xlabel("número de segmentos (k)")
plt.tight_layout(); plt.savefig(OUT_F / "08_eleccion_k.png"); plt.close()

import textwrap
ord_r = perfil["tasa_fuga_historica"].sort_values(ascending=False).index
etiquetas = [textwrap.fill(F.NOMBRES_CLUSTER.get(c, f"Segmento {c}"), 15) for c in ord_r]
fig, ax = plt.subplots(figsize=(8.6, 4.4))
x = np.arange(K); w = 0.38
ax.bar(x - w/2, perfil.loc[ord_r, "tasa_fuga_historica"] * 100, w, color=estilo.PRIMARIO,
       label="Tasa de fuga histórica observada")
ax.bar(x + w/2, resumen_act.loc[ord_r, "prob_media_fuga"] * 100, w, color=estilo.ACENTO,
       label="Probabilidad media de los activos")
for i, c in enumerate(ord_r):
    ax.text(i - w/2, perfil.loc[c, "tasa_fuga_historica"] * 100 + 1.5,
            f"{100*perfil.loc[c,'tasa_fuga_historica']:.0f}", ha="center", fontsize=8,
            color=estilo.PRIMARIO, fontweight="bold")
    ax.text(i + w/2, resumen_act.loc[c, "prob_media_fuga"] * 100 + 1.5,
            f"{100*resumen_act.loc[c,'prob_media_fuga']:.0f}", ha="center", fontsize=8,
            color=estilo.ACENTO, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(etiquetas, fontsize=7.6, linespacing=1.25)
ax.set_ylabel("% de fuga"); ax.set_ylim(0, 108)
ax.set_title("Riesgo de fuga por segmento: histórico vs. cartera activa")
ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT_F / "08_riesgo_por_cluster.png"); plt.close()

pred_f[["CODIGO CLIENTE", "cluster"]].to_csv(ROOT / "outputs" / "modelos" / "clusters_pred.csv", index=False)
hist_f[["CODIGO CLIENTE", "cluster"]].to_csv(ROOT / "outputs" / "modelos" / "clusters_hist.csv", index=False)
joblib.dump({"scaler": sc, "kmeans": km, "vars": VARS_CLUSTER, "k": K, "limites_winsor": LIM},
            ROOT / "outputs" / "modelos" / "modelo_clusters.joblib")
print(f"\n[OK] Segmentación guardada. k={K}")
