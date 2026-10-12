"""Paso 27 – Desiertos topográficos y colegios no conectados a la red de OSM (seguimiento, N4).

Los colegios sin conexión a la red vial/fluvial reciben tiempo infinito y quedan "a más de 90 min". Si tienen un
programa a 60 km en línea recta, caen en el origen topográfico: parte de ese grupo puede deberse a la completitud
de OpenStreetMap y no a la geografía. Este paso reporta, por familia:
  - alumnos topográficos en colegios no conectados (número y %), y su departamento principal;
  - el perfil del grupo topográfico (pobreza, % rural censal, % indígena, % lengua originaria; promedios distritales
    ponderados por alumnos de 5.º) con y sin los colegios no conectados, junto al de los otros orígenes;
  - topográficos por departamento con y sin no conectados (Loreto y Pasco, señalados en el texto).

Requiere los pasos 4 y 4b. Salidas (salidas/tablas): topografico_conectividad.csv, topografico_departamentos.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FAMILIAS, guardar, ruta

VARS = {"pobreza_punto_medio": "poverty", "pct_rural_censal": "rural", "pct_autoid_indigena": "indigenous",
        "pct_lengua_originaria": "indigenous_tongue"}


def cargar():
    k = ["COD_MOD", "ANEXO", "ubigeo", "familia"]
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt, usecols=[*k, "quinto", "escenario", "factor_sinuosidad", "desierto"])
    acc = acc[acc.factor_sinuosidad == 1.0].pivot_table(index=k + ["quinto"], columns="escenario", values="desierto").reset_index()
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt, usecols=[*k, "t_min", "desierto_t"])
    d = acc.merge(tie, on=k)
    d["no_conectado"] = ~np.isfinite(d.t_min)
    d["cat"] = np.select([(d.post == 1) & (d.pre == 0), (d.post == 1) & (d.pre == 1), (d.post == 0) & (d.desierto_t == 1)],
                         ["licensing", "structural", "topographic"], default="none")
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})[["ubigeo", "departamento", "pobreza_punto_medio"]]
    cen = pd.read_csv(ruta("censo2017_agregado"), dtype={"ubigeo": str})[["ubigeo", *[v for v in VARS if v != "pobreza_punto_medio"]]]
    d = d.merge(base, on="ubigeo", how="left").merge(cen, on="ubigeo", how="left")
    for v in VARS:
        d[v] = d[v].fillna(d[v].mean())
        if v != "pobreza_punto_medio":
            d[v] = 100 * d[v]
    return d


def perfil(g: pd.DataFrame) -> dict:
    w = g.quinto + 1e-9
    return {"alumnos": int(g.quinto.sum()), "colegios": len(g),
            **{nom: np.average(g[v], weights=w) if len(g) else np.nan for v, nom in VARS.items()}}


def main():
    d = cargar()
    filas = []
    for fam in ["Total", *FAMILIAS]:
        x = d[d.familia == fam]
        for cat in ["structural", "topographic", "licensing"]:
            g = x[x.cat == cat]
            filas.append({"familia": fam, "origen": cat, "version": "todos", **perfil(g)})
            if cat == "topographic":
                nc = g[g.no_conectado]
                filas.append({"familia": fam, "origen": cat, "version": "sin_no_conectados", **perfil(g[~g.no_conectado])})
                filas.append({"familia": fam, "origen": cat, "version": "solo_no_conectados", **perfil(nc),
                              "departamento_principal": nc.groupby("departamento").quinto.sum().idxmax() if len(nc) else ""})
    t = pd.DataFrame(filas)
    tot = t[(t.origen == "topographic") & (t.version == "todos")].set_index("familia").alumnos
    nc = t[(t.origen == "topographic") & (t.version == "solo_no_conectados")].set_index("familia").alumnos
    t = t.merge((100 * nc / tot.replace(0, np.nan)).rename("pct_topografico_no_conectado").reset_index(), on="familia", how="left")
    guardar(t.round(4), "topografico_conectividad.csv", "tablas")

    x = d[(d.familia == "Total") & (d.cat == "topographic")]
    dep = x.groupby("departamento").apply(lambda g: pd.Series({"alumnos": g.quinto.sum(),
                                                               "alumnos_no_conectados": g.quinto[g.no_conectado].sum(),
                                                               "colegios_no_conectados": int(g.no_conectado.sum())}))
    dep["alumnos_sin_no_conectados"] = dep.alumnos - dep.alumnos_no_conectados
    dep = dep.sort_values("alumnos", ascending=False)
    guardar(dep.reset_index(), "topografico_departamentos.csv", "tablas")

    print("Cualquier carrera:")
    print(t[t.familia == "Total"].drop(columns="familia").round(1).to_string(index=False))
    print("\n% de alumnos topográficos en colegios no conectados, por familia:")
    print(t[(t.version == "todos") & (t.origen == "topographic")][["familia", "alumnos", "pct_topografico_no_conectado"]].round(1).to_string(index=False))
    print("\nTopográficos por departamento (cualquier carrera):")
    print(dep.head(10).to_string())


if __name__ == "__main__":
    main()
