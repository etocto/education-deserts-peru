"""Paso 19 – Estadísticos descriptivos de las variables distritales y datos faltantes (revisión, obs. 12; STROBE 12c y 14).

1. Media, desviación estándar, mínimo, máximo y n de todas las variables de los modelos de conteo (unidad: distrito con
   alumnos de 5.º grado). Variable dependiente: número de programas a menos de 60 km = promedio, ponderado por alumnos
   de 5.º, de los conteos de los colegios del distrito, redondeado a entero (regla de agregación del paso 5).
2. Datos faltantes: colegios sin coordenadas válidas, distritos sin pobreza o sin covariables censales, programas sin
   coordenadas, alumnos de 5.º no asignados.

Salidas: salidas/tablas/descriptivos_distritales.csv, salidas/tablas/datos_faltantes.csv
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from utils import guardar, ruta

ETIQUETAS = {
    "y_Total": "Programmes within 60 km, all fields (count)",
    "quinto": "Final-year students (count)",
    "log_quinto": "Log final-year students",
    "pobreza": "Monetary poverty 2018, interval midpoint (%)",
    "pobreza_ancho_ic": "Width of the poverty confidence interval (pp)",
    "rural": "Final-year students in rural schools (%)",
    "rural_censal": "Rural population, 2017 Census (%)",
    "pct_autoid_indigena": "Indigenous self-identification, ages 15-29 (%)",
    "pct_lengua_originaria": "Indigenous mother tongue, ages 15-29 (%)",
    "pct_afro": "Afro-Peruvian self-identification, ages 15-29 (%)",
    "pct_sin_superior": "Adults 25+ without higher education (%)",
    "dist_min_km": "Distance to nearest programme, all fields (km)",
    "pct_desierto": "Final-year students in deserts, all fields (%)",
}


def main():
    m5 = importlib.import_module("05_modelos_conteo")
    d = m5.datos_distritales()
    for c in ["pct_autoid_indigena", "pct_lengua_originaria", "pct_afro", "pct_sin_superior"]:
        d[c] = 100 * d[c]
    d["rural_censal"] = 100 * d.pct_rural_censal
    d["pobreza_ancho_ic"] = d.pobreza_ic_sup - d.pobreza_ic_inf
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0) & (acc.familia == "Total")]
    d = d.merge(acc[["ubigeo", "dist_min_km", "pct_quinto_en_desierto"]], on="ubigeo", how="left")
    d["pct_desierto"] = 100 * d.pct_quinto_en_desierto
    d["y_Total"] = d.y_Total.round()
    t = pd.DataFrame({ETIQUETAS[c]: d[c].describe()[["mean", "std", "min", "max", "count"]] for c in ETIQUETAS}).T
    t.columns = ["Mean", "SD", "Min", "Max", "n"]
    guardar(t.round(2).reset_index(names="Variable"), "descriptivos_distritales.csv", "tablas")
    print(t.round(2).to_string())

    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"ubigeo": str, "ubigeo_2025": str})
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    prog = pd.read_csv(ruta("processed") / "oferta_programas_pregrado.csv")
    con5 = base[base.quinto > 0]
    f = [
        ("Secondary schools without valid coordinates (placed at district centroid)", int((col.coord_valida == 0).sum()),
         int(col.loc[col.coord_valida == 0, "quinto"].fillna(0).sum())),
        ("Schools in districts created after 2017 (reassigned by location)", int((col.ubigeo_2025 != col.ubigeo).sum()),
         int(col.loc[col.ubigeo_2025 != col.ubigeo, "quinto"].fillna(0).sum())),
        ("Districts with final-year students and no poverty estimate", int(con5.pobreza_punto_medio.isna().sum()),
         int(con5.loc[con5.pobreza_punto_medio.isna(), "quinto"].sum())),
        ("Districts with final-year students and no census covariates", int(d[["pct_autoid_indigena", "pct_sin_superior"]].isna().any(axis=1).sum()),
         int(d.loc[d[["pct_autoid_indigena", "pct_sin_superior"]].isna().any(axis=1), "quinto"].sum())),
        ("Undergraduate programmes without campus coordinates", int(prog.lat.isna().sum()), np.nan),
        ("Districts in the 2017 geography without final-year students", int((base.quinto == 0).sum()), 0),
    ]
    fal = pd.DataFrame(f, columns=["Item", "Units", "Final-year students"])
    guardar(fal, "datos_faltantes.csv", "tablas")
    print("\n", fal.to_string(index=False))


if __name__ == "__main__":
    main()
