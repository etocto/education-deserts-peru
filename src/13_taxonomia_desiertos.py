"""Paso 13 – Taxonomía de los desiertos según su origen (responde a los revisores 1 y 5).

Cada alumno de 5.º grado que está en desierto (por distancia a 60 km O por tiempo a 90 min; oferta licenciada 2024)
se asigna a UNA de tres categorías, según la causa que puede identificarse con los datos:

  policy       "creado por la política": no estaba en desierto con la oferta de 2018 (incluidas las universidades
               denegadas) y sí lo está con la oferta licenciada -> barrera maleable, producida por el licenciamiento.
  structural   "estructural / remoto": sin programa a 60 km ni antes ni después del licenciamiento -> barrera
               persistente, ligada a la baja densidad de población y a la historia de la oferta.
  topographic  "topográfico / oculto": tiene un programa a menos de 60 km en línea recta, pero a más de 90 min por
               la red vial y fluvial -> barrera física (relieve, ríos, estado de las vías).

Para cada categoría: alumnos, % del total de 5.º, y perfil (pobreza, % rural, % indígena, % lengua originaria),
por familia de carreras y, para "cualquier carrera", por departamento.

Requiere: datos/processed/accesibilidad_colegios.csv (paso 4) y datos/processed/tiempos_colegios.csv (paso 4b).
Salidas: salidas/tablas/taxonomia_desiertos.csv, salidas/tablas/taxonomia_desiertos_departamentos.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FAMILIAS, guardar, ruta

CATS = ["policy", "structural", "topographic"]


def main():
    k = ["COD_MOD", "ANEXO", "ubigeo", "familia"]
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str},
                      usecols=[*k, "quinto", "escenario", "factor_sinuosidad", "desierto"])
    acc = acc[acc.factor_sinuosidad == 1.0].pivot_table(index=k + ["quinto"], columns="escenario", values="desierto").reset_index()
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str},
                      usecols=[*k, "desierto_t"])
    d = acc.merge(tie, on=k, how="inner")
    d["cat"] = np.select([(d.post == 1) & (d.pre == 0), (d.post == 1) & (d.pre == 1), (d.post == 0) & (d.desierto_t == 1)],
                         CATS, default="none")

    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})[["ubigeo", "departamento", "pobreza_punto_medio"]]
    cen = pd.read_csv(ruta("censo2017_agregado"), dtype={"ubigeo": str})[["ubigeo", "pct_rural_censal", "pct_autoid_indigena", "pct_lengua_originaria"]]
    d = d.merge(base, on="ubigeo", how="left").merge(cen, on="ubigeo", how="left")
    total_quinto = d[d.familia == "Total"].quinto.sum()

    def perfil(x):
        w = x.quinto
        prom = lambda v: np.average(x[v].fillna(x[v].mean()), weights=w) if w.sum() > 0 else np.nan  # noqa: E731
        return pd.Series({"quinto": w.sum(), "pct_quinto": 100 * w.sum() / total_quinto,
                          "pobreza": prom("pobreza_punto_medio"), "rural": 100 * prom("pct_rural_censal"),
                          "indigena": 100 * prom("pct_autoid_indigena"), "lengua_originaria": 100 * prom("pct_lengua_originaria")})

    t = d.groupby(["familia", "cat"]).apply(perfil).reset_index()
    t["familia"] = pd.Categorical(t.familia, ["Total", *FAMILIAS], ordered=True)
    t = t.sort_values(["familia", "cat"])
    guardar(t.round(4), "taxonomia_desiertos.csv", "tablas")
    print(t[t.cat != "none"].round(1).to_string(index=False))

    dep = d[(d.familia == "Total")].pivot_table(index="departamento", columns="cat", values="quinto", aggfunc="sum", fill_value=0)
    dep = 100 * dep.div(dep.sum(axis=1), axis=0)
    dep = dep[[c for c in CATS if c in dep]].sort_values("structural", ascending=False)
    guardar(dep.round(4).reset_index(), "taxonomia_desiertos_departamentos.csv", "tablas")
    print("\n% de alumnos de 5.º por categoría (cualquier carrera), por departamento:")
    print(dep.round(1).head(12).to_string())


if __name__ == "__main__":
    main()
