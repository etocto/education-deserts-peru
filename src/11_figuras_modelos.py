"""Paso 11 – Figuras de los modelos para el manuscrito: mapa de la tipología (paso 7) y mapa de la
localización óptima (paso 8). Guarda en salidas/figuras y en la carpeta de figuras del manuscrito (config.yaml).

  fig_tipologia_mapa.png      distritos coloreados por tipo (nombres asignados según el perfil de cada grupo)
  fig_localizacion_mapa.png   un panel por familia: % de 5.º en desierto por distrito y los 20 distritos elegidos
                              (círculos: solución con 5; las soluciones no son anidadas)
  salidas/tablas/tipologia_nombres.csv  grupo -> nombre en inglés, con su perfil
"""
from __future__ import annotations

import shutil

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from utils import FAMILIAS, FAMILIAS_EN, guardar, ruta

DPI = 600


def nombre_tipo(p: pd.Series) -> str:
    """Nombre del tipo según la proporción de alumnos en desierto por familia (perfil medio del grupo)."""
    f = p[FAMILIAS]
    if f.mean() < 0.10:
        return "Oases"
    if f.mean() > 0.90:
        return "Full deserts"
    otras = f.drop(["Artes y humanidades", "Servicios"])
    if f["Artes y humanidades"] > 0.9 and otras.mean() < 0.3 and f["Servicios"] > 0.9:
        return "No arts or services"
    if f["Artes y humanidades"] > 0.9 and otras.mean() < 0.3:
        return "No arts"
    if f["Salud y bienestar"] < 0.1 and f["Educación"] > 0.9:
        return "Narrow offer (poorest)"
    return "Partial deserts"


COLORES = {"Oases": "#2b6cb0", "No arts": "#90cdf4", "No arts or services": "#faf089",
           "Partial deserts": "#f6ad55", "Narrow offer (poorest)": "#c53030", "Full deserts": "#63171b"}


def guardar_fig(nombre):
    p = ruta("figuras") / nombre
    plt.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close()
    shutil.copy(p, ruta("manuscrito_figuras") / nombre)
    print(f"  -> {nombre}")


def main():
    lim = gpd.read_file(ruta("limites_distrital"))[["UBIGEO", "geometry"]].rename(columns={"UBIGEO": "ubigeo"})
    dep = gpd.read_file(ruta("limites_departamental"))

    # ---------------- tipología
    t = pd.read_csv(ruta("processed") / "tipologia_distritos.csv", dtype={"ubigeo": str})
    perf = t.groupby("grupo")[FAMILIAS].mean()
    nombres = perf.apply(nombre_tipo, axis=1)
    resumen = t.groupby("grupo").agg(distritos=("ubigeo", "size"), quinto=("quinto", "sum"))
    resumen["pobreza"] = t.groupby("grupo").apply(lambda d: np.average(d.pobreza_punto_medio.fillna(d.pobreza_punto_medio.mean()), weights=d.quinto + 1e-9))
    resumen["rural"] = t.groupby("grupo").apply(lambda d: 100 * np.average(d.pct_rural_quinto, weights=d.quinto + 1e-9))
    resumen = resumen.join(100 * t.groupby("grupo").apply(lambda d: pd.Series(
        {f: np.average(d[f], weights=d.quinto + 1e-9) for f in FAMILIAS})))
    resumen.insert(0, "tipo", nombres)
    guardar(resumen.round(1).reset_index(), "tipologia_nombres.csv", "tablas")
    print(resumen.round(1).to_string())

    g = lim.merge(t[["ubigeo", "grupo"]], on="ubigeo", how="left")
    g["tipo"] = g.grupo.map(nombres)
    fig, ax = plt.subplots(figsize=(6, 8))
    g[g.tipo.isna()].plot(ax=ax, color="#e2e8f0", linewidth=0)
    for tipo, col in COLORES.items():
        g[g.tipo == tipo].plot(ax=ax, color=col, linewidth=0)
    dep.boundary.plot(ax=ax, color="white", linewidth=0.4)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in COLORES.items() if k in set(nombres)] +
              [Patch(color="#e2e8f0", label="No final-year students")], loc="lower left", fontsize=8, frameon=False)
    ax.set_axis_off()
    guardar_fig("fig_tipologia_mapa.png")

    # ---------------- localización óptima
    loc = pd.read_csv(ruta("tablas") / "localizacion_optima.csv", dtype={"ubigeo": str})
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0)]
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    fams = [f for f in ["Salud y bienestar", "Ingeniería, industria y construcción", "Administración y negocios"] if f in set(loc.familia)]
    fig, axs = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 6.2))
    for ax, fam in zip(np.atleast_1d(axs), fams):
        a = lim.merge(acc[acc.familia == fam][["ubigeo", "pct_quinto_en_desierto"]], on="ubigeo", how="left")
        a.plot(ax=ax, column="pct_quinto_en_desierto", cmap="Greys", vmin=0, vmax=1.4, linewidth=0,
               missing_kwds={"color": "white"})
        dep.boundary.plot(ax=ax, color="#a0aec0", linewidth=0.3)
        sel = loc[(loc.familia == fam) & (loc.p == loc.p.max())].merge(base[["ubigeo", "lat_media", "lon_media"]], on="ubigeo")
        s5 = loc[(loc.familia == fam) & (loc.p == loc.p.min())].merge(base[["ubigeo", "lat_media", "lon_media"]], on="ubigeo")
        ax.scatter(sel.lon_media, sel.lat_media, s=14, color="#e53e3e", edgecolor="white", linewidth=0.4, zorder=3,
                   label=f"Solution with {loc.p.max()} sites")
        ax.scatter(s5.lon_media, s5.lat_media, s=60, facecolor="none", edgecolor="#1a202c", linewidth=1.0, zorder=4,
                   label=f"Solution with {loc.p.min()} sites")
        ax.set_title(FAMILIAS_EN[fam], fontsize=9)
        ax.set_axis_off()
    np.atleast_1d(axs)[0].legend(loc="lower left", fontsize=7, frameon=False)
    guardar_fig("fig_localizacion_mapa.png")


if __name__ == "__main__":
    main()
