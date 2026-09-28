"""Paso 9 – Mapas y figuras para el manuscrito.

  mapa_desierto_total.png         % de alumnos de 5.º en desierto (todas las carreras), por distrito
  mapa_desierto_familias.png      panel 3 × 3 por familia de carreras
  fig_desiertos_por_familia.png   barras: % nacional de alumnos de 5.º en desierto por familia
  fig_distancia_pobreza.png       distancia media al programa más cercano vs. pobreza distrital

Las figuras se copian a manuscrito/figuras para Overleaf.
"""
from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from utils import FAMILIAS, FAMILIAS_EN, ruta

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#b9b8b3"


def main():
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0)]
    geo = gpd.read_file(ruta("limites_distrital"))
    geo["ubigeo"] = geo.UBIGEO.astype(str).str.zfill(6)
    figs = []

    tot = geo.merge(acc[acc.familia == "Total"], on="ubigeo", how="left")
    ax = tot.plot(column="pct_quinto_en_desierto", cmap="Oranges", legend=True, figsize=(6, 8),
                  missing_kwds={"color": "#eeeeee"}, linewidth=0)
    ax.set_axis_off()
    ax.set_title("Final-year students in university deserts (all fields)", fontsize=9)
    p = ruta("mapas") / "mapa_desierto_total.png"
    plt.savefig(p, dpi=600, bbox_inches="tight"); plt.close(); figs.append(p)

    fig, axs = plt.subplots(3, 3, figsize=(10, 13))
    for ax, f in zip(axs.flat, FAMILIAS):
        g = geo.merge(acc[acc.familia == f], on="ubigeo", how="left")
        g.plot(column="pct_quinto_en_desierto", cmap="Oranges", vmin=0, vmax=1, ax=ax, linewidth=0,
               missing_kwds={"color": "#eeeeee"})
        ax.set_title(FAMILIAS_EN[f], fontsize=8); ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap="Oranges", norm=plt.Normalize(0, 100))
    fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.025, pad=0.02,
                 label="% of final-year students without a program of the field within 60 km")
    p = ruta("mapas") / "mapa_desierto_familias.png"
    plt.savefig(p, dpi=600, bbox_inches="tight"); plt.close(); figs.append(p)

    t = pd.read_csv(ruta("tablas") / "tabla_desiertos_resumen.csv")
    t = t[(t.escenario == "post") & (t.familia != "Total")].sort_values("pct")
    plt.figure(figsize=(7, 4))
    plt.barh(t.familia.map(FAMILIAS_EN), t.pct, color=AZUL)
    for y, v in enumerate(t.pct):
        plt.text(v + 0.5, y, f"{v:.1f} %", va="center", fontsize=8)
    plt.xlabel("% of final-year students without a program within 60 km")
    plt.gca().spines[["top", "right"]].set_visible(False)
    p = ruta("figuras") / "fig_desiertos_por_familia.png"
    plt.tight_layout(); plt.savefig(p, dpi=600); plt.close(); figs.append(p)

    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    d = base.merge(acc[acc.familia == "Total"][["ubigeo", "dist_min_km"]], on="ubigeo")
    d = d[d.quinto > 0]
    plt.figure(figsize=(6, 4.5))
    plt.scatter(d.pobreza_punto_medio, d.dist_min_km, s=d.quinto / 40, alpha=.35, color=AZUL, edgecolor="none")
    plt.xlabel("Monetary poverty 2018 (%, midpoint of interval)")
    plt.ylabel("Distance to nearest program (km)")
    plt.gca().spines[["top", "right"]].set_visible(False)
    p = ruta("figuras") / "fig_distancia_pobreza.png"
    plt.tight_layout(); plt.savefig(p, dpi=600); plt.close(); figs.append(p)

    for f in figs:
        shutil.copy(f, ruta("manuscrito_figuras") / f.name)
        print("  ->", f.name)


if __name__ == "__main__":
    main()
