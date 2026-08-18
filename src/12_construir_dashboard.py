"""Fase 9b — Inyecta el JSON de datos en la plantilla y produce el tablero autónomo.

El resultado es un único .html sin dependencias externas: se abre con doble clic
en cualquier navegador y no requiere servidor ni conexión.
"""
import json, pathlib
import features as F

ROOT = F.ROOT
plantilla = (ROOT / "dashboard" / "plantilla.html").read_text(encoding="utf8")
datos = json.load(open(ROOT / "outputs" / "datos_dashboard.json"))
# </script> no puede aparecer literal dentro del bloque JSON embebido
crudo = json.dumps(datos, ensure_ascii=False).replace("</", "<\\/")
salida = plantilla.replace("__DATOS__", crudo)
destino = ROOT / "dashboard" / "tablero_cartera_riesgo.html"
destino.write_text(salida, encoding="utf8")
print(f"[OK] {destino}  ({destino.stat().st_size/1024:.0f} KB)")

copia = ROOT / "entregables" / "Tablero_Cartera_Riesgo_Julio_Vicente.html"
copia.write_text(salida, encoding="utf8")
print(f"[OK] {copia}")
