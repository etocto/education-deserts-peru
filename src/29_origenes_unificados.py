"""Paso 29 – Orígenes de los desiertos en categorías mutuamente excluyentes y sin mezclar criterios (N3, N14).

Cada alumno de 5.º que cumple al menos un criterio de desierto (sin programa a 60 km en línea recta O a más de
90 min por la red; oferta licenciada 2024) se asigna a una de cuatro categorías:

  structural     a más de 60 km y de 90 min con la oferta 2024, y a más de 60 km también con la oferta reconstruida
  licensing      a más de 60 km y de 90 min con la oferta 2024, pero a 60 km o menos con la reconstruida
  topographic    a 60 km o menos en línea recta, pero a más de 90 min
  distance_only  a más de 60 km en línea recta, pero a 90 min o menos (no es desierto por tiempo)

A diferencia del paso 13, los 11,003 alumnos "a más de 60 km pero a menos de 90 min" forman su propia fila y no se
cuentan como estructurales o de licenciamiento. Se reporta además cuántos de ellos serían de licenciamiento por
distancia. Para cada categoría y familia: alumnos, % del total de 5.º, perfil (pobreza, % rural censal,
% indígena; promedios distritales ponderados: describen los distritos donde viven los alumnos, no a los alumnos) y
% indígena esperado según la composición de sus departamentos.

Requiere los pasos 4 y 4b. Salida: salidas/tablas/origenes_unificados.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FAMILIAS, guardar, ruta

VARS = {"pobreza_punto_medio": "poverty", "pct_rural_censal": "rural", "pct_autoid_indigena": "indigenous"}
CATS = ["structural", "licensing", "topographic", "distance_only"]


def main():
    k = ["COD_MOD", "ANEXO", "ubigeo", "familia"]
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt, usecols=[*k, "quinto", "escenario", "factor_sinuosidad", "desierto"])
    acc = acc[acc.factor_sinuosidad == 1.0].pivot_table(index=k + ["quinto"], columns="escenario", values="desierto").reset_index()
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt, usecols=[*k, "desierto_t"])
    d = acc.merge(tie, on=k)
    D, Dp, T = d.post == 1, d.pre == 1, d.desierto_t == 1
    d["cat"] = np.select([D & Dp & T, D & ~Dp & T, ~D & T, D & ~T], CATS, default="none")
    d["lic_distancia"] = D & ~Dp
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})[["ubigeo", "departamento", "pobreza_punto_medio"]]
    cen = pd.read_csv(ruta("censo2017_agregado"), dtype={"ubigeo": str})[["ubigeo", "pct_rural_censal", "pct_autoid_indigena"]]
    d = d.merge(base, on="ubigeo", how="left").merge(cen, on="ubigeo", how="left")
    for v in VARS:
        d[v] = d[v].fillna(d[v].mean())
        if v != "pobreza_punto_medio":
            d[v] = 100 * d[v]
    todos = d[d.familia == "Total"]
    total = todos.quinto.sum()
    dep_media = todos.groupby("departamento").apply(lambda g: pd.Series({v: np.average(g[v], weights=g.quinto + 1e-9) for v in VARS}))
    filas = []
    for fam in ["Total", *FAMILIAS]:
        x = d[d.familia == fam]
        for cat in [*CATS, "union"]:
            g = x[x.cat != "none"] if cat == "union" else x[x.cat == cat]
            w = g.quinto + 1e-9
            wd = g.groupby("departamento").quinto.sum()
            fila = {"familia": fam, "origen": cat, "alumnos": g.quinto.sum(), "pct_quinto": 100 * g.quinto.sum() / total,
                    "departamento_principal": wd.idxmax() if len(wd) else "",
                    "lic_distancia_en_categoria": g.quinto[g.lic_distancia].sum()}
            for v, nom in VARS.items():
                fila[nom] = np.average(g[v], weights=w) if len(g) else np.nan
                fila[f"{nom}_esperado_dep"] = np.average(dep_media.loc[wd.index, v], weights=wd + 1e-9) if len(wd) else np.nan
            filas.append(fila)
        fila = {"familia": fam, "origen": "fuera_de_desierto"}
        g = x[x.cat == "none"]
        for v, nom in VARS.items():
            fila[nom] = np.average(g[v], weights=g.quinto + 1e-9)
        fila["alumnos"] = g.quinto.sum()
        filas.append(fila)
    t = pd.DataFrame(filas)
    t["pct_del_desierto"] = 100 * t.alumnos / t.familia.map(t[t.origen == "union"].set_index("familia").alumnos)
    guardar(t.round(4), "origenes_unificados.csv", "tablas")
    print(t[t.familia == "Total"].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
