"""Estilo gráfico único para todas las figuras del informe."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRIMARIO = "#1F4E5F"
ACENTO   = "#C8102E"
GRIS     = "#4A4A4A"
CLARO    = "#EEF3F5"
PALETA   = ["#1F4E5F", "#C8102E", "#5B8C9E", "#E8A33D", "#7A9A7E", "#8C6A9E"]

def aplicar():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 11, "axes.labelsize": 9.5,
        "axes.titleweight": "bold", "axes.edgecolor": "#BBBBBB",
        "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.facecolor": "white",
        "axes.prop_cycle": plt.cycler(color=PALETA),
    })
