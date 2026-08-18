# Prueba técnica — Especialista en Ciencia de Datos
 
Predicción de fuga de clientes, segmentación y reglas de asociación.

---

## Resultados en una línea

- **AUC-ROC 0.940**, AUC-PR 0.835, KS 0.737 sobre el 20% de datos reservados.
- Contactando al **20%** de la cartera se captura el **62%** de las fugas.
- Probabilidad **calibrada**: decil 1 predice 93.3%, se observa 92.2%.
- **4,957** clientes recomendados a campaña; retorno esperado 3.6×.
- **6 segmentos** estables (ARI 0.971) con tasas de fuga de 1.3% a 97.7%.
- **14 canastas** de compra depuradas de 352 reglas.

---

## Reproducir en tres pasos

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src && for s in 0*.py 1*.py; do python $s; done
latexmk -pdf informe.tex          # desde la raíz
```

Semilla fija (42) en particiones, modelos y remuestreos: la ejecución es
determinista.

---

## Estructura

```
data/                      Bases originales 
src/
  00_reconocimiento.py       Estructura, esquemas, compatibilidad entre bases
  01_auditoria.py            AUC univariado, deriva PSI, calidad, duplicados
  02_auditoria_fina.py       Hojas extra, deriva de montos, forma de relaciones
  03_trampa_identificadores.py  Confirmación forense de las dos fugas de información
  04_comparabilidad.py       ¿El saldo es comparable entre bases? (test KS)
  features.py                Limpieza e ingeniería de variables compartida
  estilo.py                  Estilo gráfico único
  05_modelado.py             Escalera de modelos, validación, calibración
  06_shap.py                 Importancia SHAP, contraste con logística, estabilidad
  07_prediccion.py           Probabilidades de los 10,000, deciles, valor esperado
  08_clustering.py           Segmentación, elección de k, estabilidad, perfilado
  09_bakery.py               Reglas de asociación y depuración
  10_entregables.py          Construcción de los XLSX
  11_datos_dashboard.py      Agregados para el tablero
  12_construir_dashboard.py  Tablero autónomo
  13_tablas_latex.py         Fragmentos .tex del informe
outputs/
  tablas/   figuras/   modelos/   tex/
dashboard/
  plantilla.html             Plantilla del tablero
  tablero_cartera_riesgo.html  Tablero autónomo (abrir en cualquier navegador)
entregables/                 Los cinco archivos que se envían
informe.tex                  Informe (compilar con latexmk)
DECISIONES.md                Registro de decisiones metodológicas con su evidencia
```

---

## Entregables

| Archivo | Contenido | Responde |
|---|---|---|
| `Prueba_Tecnica_CD_Julio_Vicente.pdf` | Informe PDF   | todo |
| `Genesis_Churn_Prediccion_Julio_Vicente.xlsx` | 10,000 clientes con probabilidad, decil, segmento y recomendación | 1.d, 1.e |
| `Genesis_Segmentacion_Julio_Vicente.xlsx` | Perfiles de segmento y asignación por cliente | 1.f |
| `Genesis_Reglas_Asociacion_Julio_Vicente.xlsx` | Canastas y reglas completas | 2 |
| `Genesis_Auditoria_Datos_Julio_Vicente.xlsx` | Evidencia de la auditoría de datos | — |
| `Tablero_Cartera_Riesgo_Julio_Vicente.html` | Tablero autónomo de 5 páginas, incluye simulador de campaña | 3 |

---

## Los dos hallazgos que condicionan todo

**1. Los identificadores contienen la respuesta.** En la base histórica los
retirados ocupan los códigos 1–34,354 y los renovados 34,355–88,828: rangos
disjuntos, AUC univariado 1.000. El archivo fue ordenado por estado antes de
numerarse. En la base de predicción los códigos fueron renumerados 1–10,000, por
lo que un modelo apoyado en ellos obtiene métricas perfectas en entrenamiento y
ruido puro al aplicarse.

**2. El primer crédito es la variable objetivo escrita de otra forma.** Los 12,791
clientes históricos con un solo crédito son retirados sin excepción, porque
«renovado» exige por definición un segundo crédito. Los 1,873 activos en esa
situación no pueden recibir probabilidad 100%.

 
---

 
