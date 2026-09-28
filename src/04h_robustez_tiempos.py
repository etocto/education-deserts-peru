"""Paso 4h – Métricas de robustez de los desiertos por tiempo de viaje (responde al revisor 2).

1. Coeficiente de variación (CV = desviación estándar / media) del % de alumnos en desierto a 90 min entre los
   seis escenarios de velocidad del paso 4d (base, vías ×0,75 y ×1,25, río 10,6 y 18,0 km/h, sin ríos), por familia.
   Solo necesita salidas/tablas/sensibilidad_tiempos.csv.
2. Estabilidad de la clasificación: % de alumnos cuyo estado (desierto / no desierto a 90 min) es el MISMO en los
   seis escenarios, y % "siempre desierto" y "nunca desierto". Necesita datos/processed/sensibilidad_colegios.csv.gz,
   que genera el paso 4d (volver a correrlo si no existe: python src/04d_sensibilidad_tiempos.py).
   También se reporta la estabilidad solo frente a los supuestos fluviales (base, río 10,6, río 18,0, sin ríos).

Salida: salidas/tablas/robustez_tiempos.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FAMILIAS, guardar, ruta

ESC_VELOCIDAD = ["base", "vias_x0.75", "vias_x1.25", "rio_10.6", "rio_18.0", "sin_rios"]
ESC_RIO = ["base", "rio_10.6", "rio_18.0", "sin_rios"]


def main():
    s = pd.read_csv(ruta("tablas") / "sensibilidad_tiempos.csv")
    s = s[(s.umbral_min == 90) & s.escenario.isin(ESC_VELOCIDAD)]
    w = s.pivot(index="familia", columns="escenario", values="pct_desierto")[ESC_VELOCIDAD]
    t = pd.DataFrame({"pct_base": w["base"], "pct_min": w.min(axis=1), "pct_max": w.max(axis=1),
                      "cv_pct": 100 * w.std(axis=1, ddof=0) / w.mean(axis=1)})
    t["rango_rank"] = 0
    rk = w.drop(index="Total", errors="ignore").rank(ascending=False)
    t.loc[rk.index, "rango_rank"] = rk.max(axis=1) - rk.min(axis=1)   # 0 = mismo puesto en todos los escenarios

    f = ruta("processed") / "sensibilidad_colegios.csv.gz"
    if f.exists():
        c = pd.read_csv(f, dtype={"COD_MOD": str, "ANEXO": str})
        for nombre, esc in [("", ESC_VELOCIDAD), ("_rio", ESC_RIO)]:
            m = c[c.escenario.isin(esc)].pivot_table(index=["COD_MOD", "ANEXO", "familia"], columns="escenario",
                                                     values="desierto_90").reset_index()
            m = m.merge(c[["COD_MOD", "ANEXO", "quinto"]].drop_duplicates(["COD_MOD", "ANEXO"]), on=["COD_MOD", "ANEXO"])
            sumas = m[esc].sum(axis=1)
            m["siempre"], m["nunca"] = (sumas == len(esc)), (sumas == 0)
            g = m.groupby("familia").apply(lambda d: pd.Series({
                f"pct_estable{nombre}": 100 * d.quinto[d.siempre | d.nunca].sum() / d.quinto.sum(),
                f"pct_siempre_desierto{nombre}": 100 * d.quinto[d.siempre].sum() / d.quinto.sum()}))
            t = t.join(g)
    else:
        print("Aviso: falta datos/processed/sensibilidad_colegios.csv.gz -> corre el paso 4d para la estabilidad por colegio.")
    t = t.loc[[x for x in ["Total", *FAMILIAS] if x in t.index]]
    guardar(t.round(2).reset_index(), "robustez_tiempos.csv", "tablas")
    print(t.round(2).to_string())


if __name__ == "__main__":
    main()
