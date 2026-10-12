"""Paso 23 – Figuras regeneradas para la versión revisada (revisión, obs. 17).

  fig1_mapa_desiertos.(png|pdf)        % de alumnos de 5.º en desierto (cualquier carrera, 60 km), por distrito, con
                                       límites departamentales, barra de color en %, escala gráfica y rampa secuencial
                                       de un solo tono (legible en escala de grises)
  fig2_licenciamiento.(png|pdf)        % de alumnos en desierto por familia antes y después de las denegatorias,
                                       con valores rotulados y márgenes suficientes
  figS_mapas_familias.(png|pdf)        panel 3 × 3 por familia, misma escala de 0 a 100 %, con límites departamentales

Salida: salidas/figuras/ y manuscrito_hep/figuras/ (carpeta del manuscrito para Higher Education Policy)
"""
from __future__ import annotations

import shutil

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from utils import FAMILIAS, FAMILIAS_EN, RAIZ, ruta

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#8a8a85",
                     "axes.labelcolor": "#333333", "xtick.color": "#555555", "ytick.color": "#555555"})
RAMPA = LinearSegmentedColormap.from_list("azul", ["#eaf2fc", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
AZUL, NARANJA, TINTA = "#2a78d6", "#eb6834", "#333333"
DESTINO = RAIZ / "manuscrito_hep/figuras"
UTM = 32718


def escala(ax, km=200):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x, y = x0 + 0.06 * (x1 - x0), y0 + 0.05 * (y1 - y0)
    ax.plot([x, x + km * 1000], [y, y], color=TINTA, lw=2, solid_capstyle="butt")
    ax.text(x + km * 500, y + 0.012 * (y1 - y0), f"{km} km", ha="center", va="bottom", fontsize=7, color=TINTA)


def guardar(fig, nombre):
    DESTINO.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        p = ruta("figuras") / f"{nombre}.{ext}"
        fig.savefig(p, dpi=600 if ext == "png" else None, bbox_inches="tight")
        shutil.copy(p, DESTINO / p.name)
    plt.close(fig)
    print("  ->", nombre)


def main():
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0)]
    geo = gpd.read_file(ruta("limites_distrital")).to_crs(UTM)
    geo["ubigeo"] = geo.UBIGEO.astype(str).str.zfill(6)
    # límites departamentales: disolución de los distritos (cierra las astillas internas del archivo departamental)
    dep = geo.assign(dpto=geo.ubigeo.str[:2]).dissolve(by="dpto")
    dep["geometry"] = dep.buffer(200).buffer(-200)

    # figura 1
    g = geo.merge(acc[acc.familia == "Total"], on="ubigeo", how="left")
    g["pct"] = 100 * g.pct_quinto_en_desierto
    fig, ax = plt.subplots(figsize=(5.2, 7))
    g.plot(column="pct", cmap=RAMPA, vmin=0, vmax=100, ax=ax, linewidth=0.05, edgecolor="face", missing_kwds={"color": "#d9d9d5", "edgecolor": "face"})
    dep.boundary.plot(ax=ax, color="#4d4d4a", linewidth=0.4)
    ax.set_axis_off(); escala(ax)
    sm = plt.cm.ScalarMappable(cmap=RAMPA, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02)
    cb.set_label("Final-year students with no licensed undergraduate programme within 60 km (%)", fontsize=7)
    cb.outline.set_visible(False)
    ax.text(0.98, 0.02, "Grey: districts without final-year students.\nLines: department boundaries.", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6, color="#555555")
    guardar(fig, "fig1_mapa_desiertos")

    # figura 2 (cualquier carrera arriba; tres ofertas: 2024, 2024 + relicenciadas, reconstruida)
    f = pd.read_csv(ruta("tablas") / "efecto_licenciamiento_familias.csv").set_index("familia")
    r = pd.read_csv(ruta("tablas") / "escenario_relicenciadas.csv").set_index("familia")
    f["pct_desierto_rel"] = r.post_relicenciadas
    orden = list(f.drop(index="Total").sort_values("pct_desierto_post").index) + ["Total"]
    f = f.loc[orden]
    y = np.arange(len(f)).astype(float)
    y[-1] += 0.6                                  # separa "cualquier carrera"
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    series = [("pct_desierto_post", AZUL, "Licensed universities only (2024)", -0.26, TINTA),
              ("pct_desierto_rel", "#8fb8ea", "Adding the 2018 supply of the six universities relicensed in 2024--2026", 0.0, "#555555"),
              ("pct_desierto_pre", NARANJA, "Adding the 2018 supply of all denied universities", 0.26, "#666666")]
    for col, color, etiqueta, dy, ct in series:
        ax.barh(y + dy, f[col], 0.25, color=color, label=etiqueta.replace("--", "\u2013"))
        for yi, v in zip(y, f[col]):
            ax.text(v + 0.5, yi + dy, f"{v + 1e-9:.1f}", va="center", fontsize=6, color=ct)
    ax.set_yticks(y, [FAMILIAS_EN[i] if i != "Total" else "Any field" for i in f.index])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.axhline(y[-2] + 0.8, color="#bdbdb8", lw=0.6)
    ax.set_xlabel("Final-year students with no programme of the field within 60 km (%)")
    ax.set_xlim(0, f[["pct_desierto_post", "pct_desierto_pre"]].to_numpy().max() * 1.12)
    ax.xaxis.grid(True, color="#e6e6e3", lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1)
    guardar(fig, "fig2_licenciamiento")

    # figura suplementaria
    fig, axs = plt.subplots(3, 3, figsize=(10, 12.5), gridspec_kw={"wspace": 0.0, "hspace": 0.06})
    for ax, fam in zip(axs.flat, FAMILIAS):
        gg = geo.merge(acc[acc.familia == fam], on="ubigeo", how="left")
        gg["pct"] = 100 * gg.pct_quinto_en_desierto
        gg.plot(column="pct", cmap=RAMPA, vmin=0, vmax=100, ax=ax, linewidth=0.05, edgecolor="face", missing_kwds={"color": "#d9d9d5", "edgecolor": "face"})
        dep.boundary.plot(ax=ax, color="#4d4d4a", linewidth=0.25)
        ax.set_title(FAMILIAS_EN[fam], fontsize=8, color=TINTA); ax.set_axis_off()
    cb = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.02, pad=0.02)
    cb.set_label("Final-year students with no programme of the field within 60 km (%)", fontsize=8)
    cb.outline.set_visible(False)
    guardar(fig, "figS_mapas_familias")


if __name__ == "__main__":
    main()
