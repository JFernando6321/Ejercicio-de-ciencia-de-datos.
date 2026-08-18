# Registro de decisiones metodológicas

Cada decisión con su evidencia y su defensa en una frase. Ordenadas por impacto
sobre el resultado.

---

## 1. Descartar `CODIGO CLIENTE` y `CODIGO PRESTAMO`

**Evidencia** (`src/03_trampa_identificadores.py`)

| | mín | máx | n |
|---|---|---|---|
| Cliente Retirado | 1 | 34,354 | 30,014 |
| Cliente Renovado | 34,355 | 88,828 | 48,815 |

Rangos **disjuntos**. La regla `CODIGO CLIENTE <= 34354 ⇒ Retirado` acierta en el
**100.0000%** de los 78,829 registros. AUC univariado = 1.0000.
`CODIGO PRESTAMO = CODIGO CLIENTE × 10` en todas las filas, así que hereda la fuga.

En la base de predicción, `CODIGO CLIENTE` es exactamente 1…10,000 (verificado) y
la intersección de `CODIGO PRESTAMO` entre bases es **0**.

**Defensa en 30 segundos:** el archivo se ordenó por estado antes de numerarse.
El código no dice nada del cliente: dice en qué parte del archivo quedó. Un modelo
que lo use reporta AUC 1.00 en entrenamiento y produce ruido sobre los 10,000.

---

## 2. Modelar aparte el segmento de primer crédito

**Evidencia** (`src/03_trampa_identificadores.py`)

- `CREDITOS ANTERIORES == 1` → 12,791 de 12,791 son Retirados (tasa 1.000000).
- `CREDITOS ANTERIORES >= 2` → 66,038 registros, tasa 0.2608.
- En la base de predicción: 1,873 clientes (18.7%) tienen valor 1.

**Por qué es tautología, no señal:** «Cliente Renovado» significa que tomó otro
crédito, luego tiene ≥ 2 por definición. El valor 1 es equivalente lógico de «no
ha renovado». Es la variable objetivo escrita de otra forma.

**Tratamiento adoptado** (`src/07_prediccion.py`)

1. Modelo principal entrenado sólo con `k >= 2`.
2. Ordenamiento dentro del segmento `k = 1`: heredado del modelo.
3. Nivel: ajuste logit-lineal de la tasa de fuga contra `log(k)` sobre `k = 2..5`,
   extrapolado a `k = 1` → **49.6%**. Se aplica como desplazamiento en logit, que
   es monótono y por tanto no altera el orden.
4. Sensibilidad reportada para 30%, 49.6%, 60% y 85%.

**Por qué `k = 2..5` y no todo el rango:** la curva cae muy rápido a partir de
`k = 6` (0.088, 0.056, 0.043…). Extrapolar desde el tramo contiguo al punto que se
quiere estimar es más creíble que ajustar toda la curva.

---

## 3. Sin validación fuera de tiempo — y decirlo

**Evidencia:** ninguna de las dos bases contiene columna de fecha
(`src/00_reconocimiento.py`). Una fila por cliente: 78,829 códigos únicos en
78,829 filas. No es panel.

**Sustituto adoptado:** tres bloques disjuntos (60/20/20) más validación cruzada
de 5 particiones, **agrupando por vector de atributos**.

**Por qué la agrupación:** 7,803 filas tienen atributos idénticos a otra fila. Con
K-fold aleatorio simple, filas idénticas caerían en entrenamiento y prueba a la
vez, inflando las métricas.

**Techo del problema:** 587 grupos de atributos idénticos presentan ambos
desenlaces. Es ruido irreducible con las variables disponibles.

---

## 4. LightGBM para ordenar + isotónica para calibrar

**Evidencia** (`src/05_modelado.py`, holdout 20%)

| Variante | AUC-PR | AUC-ROC | Brier |
|---|---|---|---|
| Sin calibrar | 0.845 | 0.941 | 0.0877 |
| **+ isotónica** | 0.835 | 0.941 | **0.0869** |
| + Platt | 0.845 | 0.941 | 0.0917 |

**Honestidad:** la ganancia de calibrar fue **marginal** — el LightGBM ya salía
prácticamente calibrado. Se conserva la isotónica porque el inciso 1.d pide un
porcentaje y la garantía no tiene costo relevante. La pérdida de 0.010 en AUC-PR
se declara.

**Descartado:** `CalibratedClassifierCV(cv=5)` degradaba el Brier de 0.082 a 0.146
y el AUC-ROC de 0.947 a 0.877. Se sustituyó por calibración explícita sobre un
bloque independiente.

**Sin SMOTE ni submuestreo:** con 38% de prevalencia el desbalance es moderado, y
ambas técnicas destruyen la calibración, que es justo lo que se necesita.

---

## 5. Eliminar `amortizado` del conjunto de variables

`amortizado = 1 - ratio_saldo_capital`: transformación afín exacta. Con ambas, SHAP
repartía la importancia entre dos columnas que son la misma señal (18.1% + 13.7%).
Al eliminarla, `ratio_saldo_capital` queda correctamente en 31.0%.

**Defensa:** una variable que es función lineal exacta de otra no aporta
información y distorsiona la lectura de importancias.

---

## 6. Prueba de robustez sin variables de saldo

**Riesgo evaluado:** ¿el saldo del histórico está medido después del desenlace? Si
un renovado ya tiene su crédito nuevo, su saldo sería alto por construcción.

**Verificación** (`src/04_comparabilidad.py`) — distancia de Kolmogorov-Smirnov del
ratio saldo/capital de los activos contra cada grupo del histórico:

| Comparación | KS |
|---|---|
| activos vs histórico completo | **0.144** |
| activos vs sólo renovados | 0.306 |
| activos vs sólo retirados | 0.391 |

Los activos se parecen a la **mezcla**, no a los renovados. Si la medición fuera
posterior al desenlace, se parecerían a los renovados. La variable es comparable.

**Cota adicional:** un LightGBM sin ninguna variable de saldo alcanza AUC-ROC 0.850
frente a 0.944 del modelo completo. El resultado no depende de una sola familia.

---

## 7. Priorizar por ratio y no por monto absoluto

**Evidencia** (`src/02_auditoria_fina.py`): PSI de `CAPITAL_CONCEDIDO` = **2.74**
con cortes fijos. Las rejillas de valores son casi disjuntas:

| Monto | histórico | predicción |
|---|---|---|
| $3,000 | 12.6% | 0.0% |
| $5,000 | 7.6% | 42.6% |
| $15,000 | 6.5% | 0.0% |
| $40,000 | 0.0% | 4.5% |

**Decisión:** apoyar el modelo en `ratio_saldo_capital`, invariante a la escala del
monto. Resultó ser la variable más predictiva (AUC univariado 0.867 en la población
modelable, 31.0% del peso SHAP).

---

## 8. k = 6 en la segmentación, elegido por evidencia

**Evidencia** (`src/08_clustering.py`)

| k | Silueta | Davies-Bouldin | ARI medio | Desv. ARI |
|---|---|---|---|---|
| 4 | 0.231 | 1.534 | 0.810 | 0.257 |
| 5 | 0.244 | 1.292 | 0.794 | 0.226 |
| **6** | **0.262** | **1.187** | **0.971** | **0.012** |
| 7 | 0.256 | 1.168 | 0.895 | 0.147 |

k = 6 domina en los tres criterios a la vez. Se cambió la elección inicial (k = 5)
al ver los datos.

**Winsorización previa (p1, p99):** sin ella, k-means dedicaba segmentos enteros a
casos extremos (hasta 106 créditos), produciendo grupos de 54 y 83 clientes,
inservibles comercialmente, y el ARI de k=5 caía a 0.646.

**Sólo variables numéricas:** k-means con one-hot sobre 138 agencias produce
distancias sin sentido. Las categóricas se usan para perfilar, no para agrupar.

---

## 9. Umbral de campaña por valor esperado, no 0.5

```
Beneficio(i) = P(fuga_i) × efectividad × valor_cliente_i − costo_contacto
Contactar si Beneficio(i) > 0
```

**Supuestos declarados y parametrizables:**
- `valor_cliente` = capital × tasa nominal / 100 × 0.30 (margen neto). Mediana $832.
- `costo_contacto` = $25.
- `efectividad` = desconocida → sensibilidad con 10%, 20%, 30%.

**Por qué no priorizar por saldo en riesgo:** los clientes de mayor probabilidad de
fuga son los de **menor** saldo, porque saldo bajo significa crédito por terminar.
El decil 10 (menor riesgo) concentra $22.1 M de saldo; el decil 1 apenas $0.9 M.
Priorizar por saldo llevaría exactamente a los clientes equivocados. Lo que se
pierde al no retener no es el saldo actual sino el margen del ciclo siguiente.

---

## 10. Depurar reglas de asociación por itemset, no por métrica

**Evidencia** (`src/09_bakery.py`): 352 reglas con lift > 1 describen sólo **74
conjuntos de productos distintos**, que se reducen a **14 hallazgos independientes**.

- La canasta {Lemon Cookie, Raspberry Cookie, Lemon Lemonade, Raspberry Lemonade,
  Green Tea} genera **180 reglas** por sí sola.
- La canasta de los cuatro productos de manzana genera **50**.

**La patología esperada no aplica aquí:** el producto más vendido (Coffee Eclair)
está en apenas el 10.9% de los tickets. No hay artículo ubicuo, y ninguna regla
tiene confianza > 20% con lift < 1.20. La patología real es la **redundancia**.

**Criterio final:** lift > 1.20 **y** ≥ 100 tickets **y** confianza > 1.2 × soporte
del consecuente **y** leverage > 0; después, una regla representante por itemset
(la de mayor lift) y eliminación de itemsets subsumidos.

---

## 11. Tablero web autónomo en lugar de Power BI

El enunciado admite «Power BI o cualquier otro software que facilite la
visualización». Se eligió un tablero web autónomo por dos razones:

1. **Confidencialidad:** son datos de clientes de una entidad financiera. Subirlos
   a un servicio público de alojamiento de tableros sería inadecuado. Este tablero
   no envía datos a ningún servidor.
2. **Portabilidad:** se abre en cualquier navegador, sin licencia ni instalación.

Los XLSX entregados quedan listos para cargarse en Power BI si se prefiere.

---

## Limitaciones que se declaran en el informe

1. El modelo predice **quién se va**, no **a quién convence la campaña**. Requiere
   uplift con grupo de control. Recomendación: 10% de control en la primera campaña.
2. Sin validación fuera de tiempo (no hay fechas).
3. Faltan las variables más predictivas del negocio: días de mora, puntualidad,
   fechas de desembolso, montos de créditos anteriores. Sin ellas no hay variables
   de tendencia.
4. Deriva real entre bases (PSI 2.74 en capital). Mitigada, no eliminada.
5. La tasa del primer crédito (49.6%) es una extrapolación, no una observación.
6. Bakery no trae fecha, hora, cantidad ni precio: no hay análisis por franja
   horaria ni priorización por impacto económico.
