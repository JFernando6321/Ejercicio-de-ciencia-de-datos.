"""
Fase 7 — Reglas de asociación, base Bakery (preguntas 2a, 2b, 2c).

La base viene ya en formato matriz transaccional binaria: 9,981 tickets x 50
productos, separador ';'. No incluye fecha, hora, cantidad ni precio, por lo
que no es posible el análisis por franja horaria ni la valoración económica
de las reglas; se documenta como limitación.
"""
import pandas as pd, numpy as np, json, warnings, re, itertools
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import fpgrowth, association_rules

import features as F, estilo
estilo.aplicar()

ROOT = F.ROOT
OUT_T = ROOT / "outputs" / "tablas"; OUT_F = ROOT / "outputs" / "figuras"

df = pd.read_csv(ROOT / "data" / "Bakery.csv", sep=";")
df = df.set_index("id")
df.columns = [re.sub(r"\s+", " ", c.replace("'", "").strip()) for c in df.columns]
df = df.astype(bool)

print(f"Tickets: {len(df):,} | Productos: {df.shape[1]}")
n_items = df.sum(axis=1)
print(f"\nArtículos por ticket: media={n_items.mean():.2f} mediana={n_items.median():.0f} "
      f"min={n_items.min()} max={n_items.max()}")
print(n_items.value_counts().sort_index().to_string())

vacios = (n_items == 0).sum()
print(f"\nTickets vacíos (se eliminan): {vacios}")
df = df[n_items > 0]

top = df.mean().sort_values(ascending=False)
print(f"\n--- Top 12 productos por penetración ---")
print((top.head(12) * 100).round(2).to_string())
print(f"--- Cola: productos con soporte < 1% ---")
print((top[top < 0.01] * 100).round(3).to_string())
top.rename("soporte").to_frame().to_csv(OUT_T / "09_soporte_productos.csv")

# ============================================================ 1. FP-GROWTH
SOP_MIN = 0.01          # ~100 tickets: umbral bajo, adecuado a tickets de 1-5 artículos
frec = fpgrowth(df, min_support=SOP_MIN, use_colnames=True, max_len=5)
frec["n_items"] = frec["itemsets"].apply(len)
print(f"\nItemsets frecuentes (soporte >= {SOP_MIN}): {len(frec):,}")
print(frec.groupby("n_items").size().rename("cantidad").to_string())

reglas = association_rules(frec, metric="lift", min_threshold=1.0)
for c in ["antecedents", "consequents"]:
    reglas[c + "_txt"] = reglas[c].apply(lambda s: " + ".join(sorted(s)))
reglas["n_tickets"] = (reglas["support"] * len(df)).round().astype(int)
print(f"Reglas con lift > 1: {len(reglas):,}")

# ============================================================ 2. CRITERIO DE SELECCIÓN (2b)
# Una regla se acepta sólo si supera SIMULTÁNEAMENTE:
#   (i)   lift > 1.20      -> asociación real, no coincidencia por popularidad
#   (ii)  n_tickets >= 100 -> respaldo muestral suficiente
#   (iii) confianza > soporte del consecuente x 1.2 -> mejora sobre la línea base
#   (iv)  leverage > 0     -> co-ocurrencia por encima de la independencia
LIFT_MIN, N_MIN = 1.20, 100
sel = reglas[(reglas["lift"] > LIFT_MIN) &
             (reglas["n_tickets"] >= N_MIN) &
             (reglas["confidence"] > reglas["consequent support"] * 1.2) &
             (reglas["leverage"] > 0)].copy()
print(f"\nReglas que superan los cuatro filtros: {len(sel):,}")

# --- (v) Colapso de redundancia -------------------------------------------
# En esta base el problema NO es el lift inflado por un producto ubicuo, sino la
# REDUNDANCIA: las 322 reglas son permutaciones de un puñado de canastas reales.
# {Lemon Cookie, Raspberry Cookie, Lemon Lemonade, Raspberry Lemonade} genera por
# sí sola decenas de reglas A->C con métricas casi idénticas.
# Criterio: dos reglas con el mismo itemset (antecedente U consecuente) describen
# el MISMO hallazgo comercial. Se conserva una sola representante por itemset —
# la de mayor lift — y se descartan los itemsets contenidos en otro ya retenido
# cuyo lift no sea superior (subsunción).
sel["itemset"] = sel.apply(lambda r: frozenset(r["antecedents"] | r["consequents"]), axis=1)
sel = sel.sort_values(["lift", "confidence"], ascending=False)
repres = sel.drop_duplicates(subset="itemset").reset_index(drop=True)
print(f"Reglas -> canastas distintas: {len(sel)} reglas describen {len(repres)} itemsets únicos")

descartar = set()
for i, r in repres.iterrows():
    for j, s in repres.iterrows():
        if i == j or j in descartar or i in descartar:
            continue
        if s["itemset"] < r["itemset"] and s["lift"] <= r["lift"]:
            descartar.add(j)          # s está contenida en r y no mejora el lift
sel_f = repres.drop(index=list(descartar)).reset_index(drop=True)
print(f"Itemsets subsumidos eliminados: {len(descartar)}  ->  hallazgos finales: {len(sel_f)}")
sel_f["productos_del_combo"] = sel_f["itemset"].apply(lambda s: " + ".join(sorted(s)))
sel_f["n_productos"] = sel_f["itemset"].apply(len)

cols = ["productos_del_combo", "n_productos", "antecedents_txt", "consequents_txt",
        "n_tickets", "support", "confidence", "lift", "leverage", "conviction"]
sel_f = sel_f.sort_values("lift", ascending=False)
print("\n=== CANASTAS IDENTIFICADAS (una regla representante por hallazgo) ===")
print(sel_f[["productos_del_combo", "n_tickets", "support", "confidence", "lift", "leverage"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
sel_f[cols].to_csv(OUT_T / "09_reglas_seleccionadas.csv", index=False)
reglas.sort_values("lift", ascending=False)[
    ["antecedents_txt", "consequents_txt", "n_tickets", "support", "confidence",
     "lift", "leverage", "conviction"]].to_csv(OUT_T / "09_reglas_todas.csv", index=False)

# ============================================================ 3. JUSTIFICACIÓN DEL CRITERIO (2b)
print("\n=== POR QUÉ ESTE CRITERIO Y NO 'LAS DE MAYOR CONFIANZA' ===")
print(f"Producto más frecuente: {top.index[0]} con soporte {top.iloc[0]:.4f}.")
print("No existe un producto ubicuo (tipo pan o café en toda compra), por lo que")
print("la patología clásica 'confianza alta con lift ≈ 1' casi no aparece aquí:")
trampa = reglas[(reglas["confidence"] > 0.20) & (reglas["lift"] < 1.20)]
print(f"  reglas con confianza > 20% y lift < 1.20: {len(trampa)} de {len(reglas)}")
print("\nLa patología dominante en esta base es la REDUNDANCIA:")
print(f"  · {len(reglas):,} reglas con lift > 1")
print(f"  · describen sólo {len(repres)} itemsets distintos")
print(f"  · que se reducen a {len(sel_f)} hallazgos comerciales independientes")
print("Presentar las 322 reglas como '322 hallazgos' sería un error de lectura:")
print("son permutaciones internas de unas pocas canastas reales.")

ejemplo = sel.sort_values("lift", ascending=False).head(12)
print("\nEjemplo: reglas distintas que describen LA MISMA canasta")
mayor = sel_f.iloc[0]["itemset"]
mismo = sel[sel["itemset"].apply(lambda s: s == mayor)]
print(mismo[["antecedents_txt", "consequents_txt", "confidence", "lift"]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================ 4. FIGURAS
fig, ax = plt.subplots(figsize=(6.6, 4.2))
t12 = top.head(15).iloc[::-1]
ax.barh(t12.index, t12.values * 100, color=estilo.PRIMARIO)
ax.set_xlabel("% de tickets que contienen el producto")
ax.set_title("Productos más vendidos (penetración en tickets)")
plt.tight_layout(); plt.savefig(OUT_F / "09_top_productos.png"); plt.close()

fig, ax = plt.subplots(figsize=(6.2, 4.4))
sc_ = ax.scatter(reglas["support"] * 100, reglas["confidence"] * 100,
                 c=reglas["lift"], s=14, cmap="viridis", alpha=.6)
ax.scatter(sel_f["support"] * 100, sel_f["confidence"] * 100, s=30,
           facecolors="none", edgecolors=estilo.ACENTO, linewidths=1.1,
           label=f"Seleccionadas (n={len(sel_f)})")
plt.colorbar(sc_, ax=ax, label="Lift")
ax.axhline(0, color="none")
ax.set_xlabel("Soporte (%)"); ax.set_ylabel("Confianza (%)")
ax.set_title("Universo de reglas y subconjunto seleccionado")
ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT_F / "09_dispersion_reglas.png"); plt.close()

# Grafo de co-ocurrencia. Cada canasta seleccionada se dibuja como un clique
# entre sus productos: así se ve la estructura real (grupos de 4-5 productos que
# se compran juntos) y no sólo los pares, que serían una fracción del hallazgo.
G = nx.Graph()
for _, r in sel_f.iterrows():
    prods = sorted(r["itemset"])
    for a, b in itertools.combinations(prods, 2):
        if G.has_edge(a, b):
            G[a][b]["weight"] = max(G[a][b]["weight"], r["lift"])
        else:
            G.add_edge(a, b, weight=r["lift"])
comp = list(nx.connected_components(G))
color_comp = {}
for i, cset in enumerate(sorted(comp, key=len, reverse=True)):
    for n in cset:
        color_comp[n] = estilo.PALETA[i % len(estilo.PALETA)]

# Disposición: un componente por celda de una rejilla, con los productos de cada
# canasta en círculo. Un spring_layout global colapsa los cliques unos sobre otros
# y deja las etiquetas ilegibles.
comps = sorted(comp, key=len, reverse=True)
NCOL = 4
pos = {}
for i, cset in enumerate(comps):
    fila, columna = divmod(i, NCOL)
    cx, cy = columna * 3.0, -fila * 2.6
    sub = G.subgraph(cset)
    if len(cset) == 2:
        p = {n: np.array([-0.5, 0.0]) if k == 0 else np.array([0.5, 0.0])
             for k, n in enumerate(sorted(cset))}
    else:
        p = nx.circular_layout(sub, scale=0.62 + 0.06 * len(cset))
    for n, xy in p.items():
        pos[n] = np.array([cx + xy[0], cy + xy[1]])

fig, ax = plt.subplots(figsize=(11.2, 2.9 * ((len(comps) + NCOL - 1) // NCOL)))
pesos = [G[u][v]["weight"] for u, v in G.edges()]
mx_w = max(pesos)
nx.draw_networkx_edges(G, pos, width=[0.7 + 2.8 * w / mx_w for w in pesos],
                       alpha=.45, edge_color=estilo.GRIS, ax=ax)
tam = [260 + 2600 * float(top.get(n, 0.02)) for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=tam,
                       node_color=[color_comp[n] for n in G.nodes()],
                       alpha=.92, edgecolors="white", linewidths=1.4, ax=ax)
for n, (x, y) in pos.items():
    ax.text(x, y - 0.30, n, ha="center", va="top", fontsize=6.4,
            color=estilo.GRIS, fontweight="600", zorder=5)
ax.set_title(f"Canastas de compra: {len(comps)} grupos de productos que se venden juntos\n"
             f"cada grupo es una canasta · el grosor del enlace es el lift · "
             f"el tamaño del círculo es la frecuencia del producto", fontsize=10.5)
ax.axis("off"); ax.margins(.06, .12)
plt.tight_layout(); plt.savefig(OUT_F / "09_red_productos.png"); plt.close()
print(f"\nGrafo: {G.number_of_nodes()} productos, {G.number_of_edges()} enlaces, "
      f"{len(comp)} grupos conexos de tamaños {sorted((len(c) for c in comp), reverse=True)}")

# ============================================================ 5. COMBOS RECOMENDADOS (2c)
combos = sel_f.head(10)[["productos_del_combo", "n_productos", "n_tickets",
                         "support", "confidence", "lift"]].copy()
print("\n=== COMBOS PROPUESTOS (ordenados por lift) ===")
print(combos.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
combos.to_csv(OUT_T / "09_combos.csv", index=False)

json.dump({"n_tickets": int(len(df)), "n_productos": int(df.shape[1]),
           "items_por_ticket_media": float(n_items[n_items > 0].mean()),
           "soporte_minimo": SOP_MIN, "n_itemsets": int(len(frec)),
           "n_reglas_lift_gt_1": int(len(reglas)),
           "n_reglas_seleccionadas": int(len(sel_f)),
           "lift_maximo": float(sel_f["lift"].max()),
           "producto_top": str(top.index[0]), "soporte_top": float(top.iloc[0])},
          open(OUT_T / "09_resumen_bakery.json", "w"), indent=2)
print("\n" + json.dumps(json.load(open(OUT_T / "09_resumen_bakery.json")), indent=2))
