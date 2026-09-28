"""Paso 8 – Implicancias de política: ¿dónde abrir programas para reducir los desiertos? (máxima cobertura).

Para una familia de carreras f y un número p de nuevos locales, se eligen p distritos candidatos que maximizan
los alumnos de 5.º grado actualmente en desierto que quedarían a menos de radio_cobertura_km
(Maximal Covering Location Problem, Church y ReVelle, 1974). Se resuelve con programación entera (PuLP/CBC).

Candidatos: distritos con al menos 300 alumnos de 5.º grado (ajustable). Demanda: colegios en desierto para f.

Salidas: salidas/tablas/localizacion_<familia>.csv con los distritos elegidos y la cobertura lograda.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pulp

from utils import CFG, guardar, haversine_matriz, ruta

L = CFG["localizacion"]


def resolver(fam: str, p: int, min_quinto_candidato: int = 300):
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.familia == fam) & (acc.factor_sinuosidad == 1.0) & (acc.desierto == 1)]
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    dem = acc.merge(col[["COD_MOD", "ANEXO", "lat", "lon"]], on=["COD_MOD", "ANEXO"])
    dist = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    cand = dist[(dist.quinto >= min_quinto_candidato) & dist.lat_media.notna()].reset_index(drop=True)
    cubre = haversine_matriz(dem.lat, dem.lon, cand.lat_media, cand.lon_media) <= L["radio_cobertura_km"]
    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)
    # PuLP 4 retiró LpVariable.dicts; add_variable_dicts existe desde PuLP 3.3
    variables = getattr(prob, "add_variable_dicts", None) or pulp.LpVariable.dicts
    xj = variables("abrir", range(len(cand)), cat="Binary")
    yi = variables("cubierto", range(len(dem)), cat="Binary")
    prob += pulp.lpSum(dem.quinto.iloc[i] * yi[i] for i in range(len(dem)))
    for i in range(len(dem)):
        js = np.flatnonzero(cubre[i])
        prob += yi[i] <= pulp.lpSum(xj[j] for j in js)
    prob += pulp.lpSum(xj.values()) == p
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=600))
    elegidos = [j for j in range(len(cand)) if xj[j].value() > 0.5]
    cubiertos = sum(dem.quinto.iloc[i] for i in range(len(dem)) if yi[i].value() > 0.5)
    out = cand.loc[elegidos, ["ubigeo", "departamento", "provincia", "distrito", "quinto", "pobreza_punto_medio"]].assign(
        familia=fam, p=p, quinto_en_desierto=dem.quinto.sum(), quinto_cubierto=cubiertos,
        pct_cubierto=100 * cubiertos / dem.quinto.sum())
    return out


def main(familias):
    res = [resolver(f, p) for f in familias for p in L["p_nuevos"]]
    t = pd.concat(res, ignore_index=True)
    guardar(t, "localizacion_optima.csv", "tablas")
    print(t.groupby(["familia", "p"]).pct_cubierto.first().round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or ["Salud y bienestar", "Ingeniería, industria y construcción", "Administración y negocios"])
