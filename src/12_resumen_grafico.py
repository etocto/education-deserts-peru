"""Paso 12 – Resumen gráfico (graphical abstract) para Education Sciences (MDPI).

Requisitos de MDPI: imagen original, distinta de las figuras del artículo, mínimo 560 × 1100 px (alto × ancho),
PNG/JPEG/TIFF, texto legible. Se genera a 2200 × 1120 px.
Usa las salidas de los pasos 4, 4c, 4e, 4b y 10 (se leen de salidas/tablas y datos/processed).

Salida: manuscrito_mdpi/resumen_grafico.png (no se incluye en main.tex; se sube aparte al enviar).
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import FAMILIAS_EN, RAIZ, ruta

TINTA, TINTA2, NARANJA, GRIS = "#1a202c", "#4a5568", "#dd6b20", "#cbd5e0"


def main():
    lim = gpd.read_file(ruta("limites_distrital"))[["UBIGEO", "geometry"]].rename(columns={"UBIGEO": "ubigeo"})
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0)]
    res = pd.read_csv(ruta("tablas") / "tabla_desiertos_resumen.csv")
    res = res[res.escenario == "post"].set_index("familia").pct

    fig = plt.figure(figsize=(11, 5.6), dpi=200)
    fig.patch.set_facecolor("white")
    fig.text(0.02, 0.94, "University deserts in Peru after the licensing reform", fontsize=17, weight="bold", color=TINTA)
    fig.text(0.02, 0.895, "537,451 final-year secondary students · 3587 licensed undergraduate programs · 60 km and 90 min thresholds",
             fontsize=10, color=TINTA2)

    # 1) mapa
    ax = fig.add_axes([0.0, 0.03, 0.30, 0.84])
    m = lim.merge(acc[acc.familia == "Total"][["ubigeo", "pct_quinto_en_desierto"]], on="ubigeo", how="left")
    m.plot(ax=ax, column="pct_quinto_en_desierto", cmap="Oranges", vmin=0, vmax=1, linewidth=0,
           missing_kwds={"color": "#edf2f7"})
    ax.set_axis_off()
    cax = fig.add_axes([0.035, 0.10, 0.012, 0.22])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="Oranges", norm=plt.Normalize(0, 100)), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=TINTA2, length=0)
    cb.set_ticks([0, 50, 100], labels=["0%", "50%", "100%"])
    fig.text(0.02, 0.035, "Students with no program of any\nfield within 60 km (by district)", fontsize=8.5, color=TINTA2)

    # 2) barras por familia
    ax = fig.add_axes([0.45, 0.10, 0.22, 0.72])
    s = res.drop("Total").sort_values()
    etiquetas = [FAMILIAS_EN[f].replace(", mathematics and ICT", " & ICT").replace(", journalism and law", " & law")
                 .replace(", manufacturing and construction", "").replace(" and administration", "")
                 .replace(" and environment", "").replace(" and welfare", "").replace(" and humanities", "") for f in s.index]
    ax.barh(range(len(s)), s.values, color=[NARANJA if round(v) >= 20 else GRIS for v in s.values], height=0.7)
    for y, v in enumerate(s.values):
        ax.text(v + 0.8, y, f"{v:.0f}%", va="center", fontsize=9, color=TINTA)
    ax.set_yticks(range(len(s)), etiquetas, fontsize=9, color=TINTA)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color(GRIS)
    ax.set_title(f"Deserts are field deserts\n{res['Total']:.1f}% lack any program; by field:", fontsize=11,
                 loc="left", color=TINTA, x=-0.55)

    # 3) cifras clave
    lic = pd.read_csv(ruta("tablas") / "efecto_licenciamiento_familias.csv").set_index("familia")
    ins = pd.read_csv(ruta("tablas") / "institutos_en_desiertos.csv").set_index("familia")
    asis = pd.read_csv(ruta("tablas") / "asistencia_modelos.csv")
    coef = asis[(asis.dependiente == "asiste") & (asis.acceso == "desierto")].coef.iloc[0] * 100
    oster = pd.read_csv(ruta("tablas") / "oster_asistencia.csv")          # paso 10b: estimación corregida (Oster)
    coef_oster = oster[oster.acceso == "desierto"].beta_corregido_delta1.iloc[0] * 100
    brecha_min, brecha_max = sorted([abs(coef_oster), abs(coef)])        # rango 5-7 pp, como en el resumen y la Tabla 9
    cifras = [
        ("18,074", "students in hidden deserts:\nwithin 60 km but >90 min away (poorest group)"),
        (f"{brecha_min:.0f}\u2013{brecha_max:.0f} pp", "lower higher-education attendance\nin full deserts (youth 17-24)"),
        (f"-{lic.loc['Total', 'caida_pct_oferta_60km']:.1f}%", "nearby supply after 45 license\ndenials; poorer districts pushed into deserts"),
        (f"{ins.loc['Total', 'pct_con_instituto']:.0f}%", "of students with no university within 60 km\nhave a (mostly public) institute nearby"),
    ]
    for k, (num, txt) in enumerate(cifras):
        y = 0.78 - k * 0.19
        fig.text(0.705, y, num, fontsize=20, weight="bold", color=NARANJA)
        fig.text(0.705, y - 0.075, txt, fontsize=9, color=TINTA2, va="bottom")
    p = RAIZ / "manuscrito_mdpi" / "resumen_grafico.png"
    fig.savefig(p, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  -> {p.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
