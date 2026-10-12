"""Paso 18 – Perfil de los desiertos asociados al licenciamiento, por familia, frente a su departamento (revisión, obs. 5).

Para cada familia y cada origen se compara el perfil de los alumnos en desierto (pobreza, % rural censal,
% autoidentificación indígena, % lengua originaria; promedios distritales ponderados por alumnos de 5.º) con el de
TODOS los alumnos de 5.º de los mismos departamentos, ponderando cada departamento por los alumnos en desierto que
aporta (perfil esperado si el desierto se repartiera dentro de cada departamento sin relación con esas variables).
Diferencia = perfil del desierto - perfil esperado. Se reporta también el departamento con más alumnos en el grupo.

Requiere los pasos 4 y 4b. Salida: salidas/tablas/perfil_desiertos_vs_departamento.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FAMILIAS, guardar, ruta

VARS = {"pobreza_punto_medio": "poverty", "pct_rural_censal": "rural", "pct_autoid_indigena": "indigenous",
        "pct_lengua_originaria": "indigenous_tongue"}


def main():
    k = ["COD_MOD", "ANEXO", "ubigeo", "familia"]
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt, usecols=[*k, "quinto", "escenario", "factor_sinuosidad", "desierto"])
    acc = acc[acc.factor_sinuosidad == 1.0].pivot_table(index=k + ["quinto"], columns="escenario", values="desierto").reset_index()
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt, usecols=[*k, "desierto_t"])
    d = acc.merge(tie, on=k)
    d["cat"] = np.select([(d.post == 1) & (d.pre == 0), (d.post == 1) & (d.pre == 1), (d.post == 0) & (d.desierto_t == 1)],
                         ["licensing", "structural", "topographic"], default="none")
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})[["ubigeo", "departamento", "pobreza_punto_medio"]]
    cen = pd.read_csv(ruta("censo2017_agregado"), dtype={"ubigeo": str})[["ubigeo", "pct_rural_censal", "pct_autoid_indigena", "pct_lengua_originaria"]]
    d = d.merge(base, on="ubigeo", how="left").merge(cen, on="ubigeo", how="left")
    for v in VARS:
        d[v] = d[v].fillna(d[v].mean())
        if v != "pobreza_punto_medio":
            d[v] = 100 * d[v]
    todos = d[d.familia == "Total"]
    dep_media = todos.groupby("departamento").apply(lambda g: pd.Series({v: np.average(g[v], weights=g.quinto + 1e-9) for v in VARS}))
    filas = []
    for (fam, cat), g in d[d.cat != "none"].groupby(["familia", "cat"]):
        w_dep = g.groupby("departamento").quinto.sum()
        fila = {"familia": fam, "origen": cat, "alumnos": g.quinto.sum(), "departamento_principal": w_dep.idxmax(),
                "pct_en_departamento_principal": 100 * w_dep.max() / w_dep.sum()}
        for v, nom in VARS.items():
            obs = np.average(g[v], weights=g.quinto + 1e-9)
            esp = np.average(dep_media.loc[w_dep.index, v], weights=w_dep + 1e-9)
            fila |= {f"{nom}": obs, f"{nom}_esperado_dep": esp, f"{nom}_diferencia": obs - esp}
        filas.append(fila)
    t = pd.DataFrame(filas)
    t["familia"] = pd.Categorical(t.familia, ["Total", *FAMILIAS], ordered=True)
    t = t.sort_values(["origen", "familia"])
    guardar(t.round(4), "perfil_desiertos_vs_departamento.csv", "tablas")
    cols = ["familia", "alumnos", "departamento_principal", "pct_en_departamento_principal", "poverty", "poverty_esperado_dep",
            "indigenous", "indigenous_esperado_dep", "indigenous_diferencia"]
    print(t[t.origen == "licensing"][cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
