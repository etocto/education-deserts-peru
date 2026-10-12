"""Paso 4c – ¿Creó desiertos el licenciamiento? Comparación del escenario pre (con las universidades denegadas,
según carnés 2018) y post (solo licenciadas, 2024), por familia de carreras.

Salidas (salidas/tablas):
  efecto_licenciamiento_familias.csv   alumnos que pasan a desierto y caída de la oferta cercana, por familia
  efecto_licenciamiento_distritos.csv  distritos que pasan a desierto (con pobreza y ruralidad)
  efecto_licenciamiento_departamentos.csv  caída de programas a 60 km por departamento
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FAMILIAS, FAMILIAS_EN, guardar, ruta


def main():
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    acc = acc[acc.factor_sinuosidad == 1.0]
    w = acc.pivot_table(index=["COD_MOD", "ANEXO", "ubigeo", "quinto", "familia"], columns="escenario",
                        values=["desierto", "n_prog_60km"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    w["nuevo_desierto"] = ((w.desierto_pre == 0) & (w.desierto_post == 1)).astype(int)
    w["perdida_prog"] = w.n_prog_60km_pre - w.n_prog_60km_post
    fam = w.groupby("familia").apply(lambda d: pd.Series({
        "quinto_desierto_pre": (d.quinto * d.desierto_pre).sum(),
        "quinto_desierto_post": (d.quinto * d.desierto_post).sum(),
        "quinto_nuevo_desierto": (d.quinto * d.nuevo_desierto).sum(),
        "prog_60km_pre_medio": np.average(d.n_prog_60km_pre, weights=d.quinto + 1e-9),
        "prog_60km_post_medio": np.average(d.n_prog_60km_post, weights=d.quinto + 1e-9),
        "quinto_con_perdida": (d.quinto * (d.perdida_prog > 0)).sum()})).reset_index()
    tot = w.quinto[w.familia == "Total"].sum()
    fam["pct_desierto_pre"] = 100 * fam.quinto_desierto_pre / tot
    fam["pct_desierto_post"] = 100 * fam.quinto_desierto_post / tot
    fam["caida_pct_oferta_60km"] = 100 * (1 - fam.prog_60km_post_medio / fam.prog_60km_pre_medio)
    fam["pct_quinto_con_perdida"] = 100 * fam.quinto_con_perdida / tot
    guardar(fam.round(4), "efecto_licenciamiento_familias.csv", "tablas")
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = fam[fam.familia != "Total"].sort_values("pct_desierto_post")
    y = np.arange(len(g))
    plt.figure(figsize=(7, 4.2))
    plt.barh(y - 0.2, g.pct_desierto_pre, 0.4, color="#b9b8b3", label="Before licensing (incl. denied universities, 2018)")
    plt.barh(y + 0.2, g.pct_desierto_post, 0.4, color="#eb6834", label="After licensing (licensed universities, 2024)")
    plt.yticks(y, g.familia.map(FAMILIAS_EN), fontsize=8)
    plt.xlabel("% of final-year students without a program within 60 km")
    plt.legend(frameon=False, fontsize=7, loc="lower right")
    plt.gca().spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = ruta("figuras") / "fig_efecto_licenciamiento.png"
    plt.savefig(p, dpi=600); plt.close()
    shutil.copy(p, ruta("manuscrito_figuras") / p.name)
    print(fam[["familia", "pct_desierto_pre", "pct_desierto_post", "quinto_nuevo_desierto", "caida_pct_oferta_60km",
               "pct_quinto_con_perdida"]].round(1).to_string(index=False))

    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    t = w[w.familia == "Total"].groupby("ubigeo").agg(quinto=("quinto", "sum"), nuevo=("nuevo_desierto", "max"),
                                                      prog_pre=("n_prog_60km_pre", "mean"), prog_post=("n_prog_60km_post", "mean"))
    t = t.join(base.set_index("ubigeo")[["departamento", "provincia", "distrito", "pobreza_punto_medio", "pct_rural_quinto"]])
    nuevos = t[t.nuevo == 1].sort_values("quinto", ascending=False)
    guardar(nuevos.reset_index(), "efecto_licenciamiento_distritos.csv", "tablas")
    print(f"\ndistritos con colegios que quedan sin ningún programa a 60 km tras el licenciamiento: {len(nuevos)}")
    print(f"pobreza media: nuevos desiertos {nuevos.pobreza_punto_medio.mean():.1f} % vs resto {t[t.nuevo == 0].pobreza_punto_medio.mean():.1f} %")
    t["dpto"] = t.index.str[:2]
    dep = t.groupby("departamento").apply(lambda d: pd.Series({
        "prog_60km_pre": np.average(d.prog_pre, weights=d.quinto + 1e-9),
        "prog_60km_post": np.average(d.prog_post, weights=d.quinto + 1e-9)})).assign(
        caida_pct=lambda x: 100 * (1 - x.prog_60km_post / x.prog_60km_pre)).sort_values("caida_pct", ascending=False)
    guardar(dep.round(4).reset_index(), "efecto_licenciamiento_departamentos.csv", "tablas")
    print(dep.round(1).head(10).to_string())


if __name__ == "__main__":
    main()
