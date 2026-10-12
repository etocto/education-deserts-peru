"""Paso 15 – Sensibilidad al umbral de distancia y tabla cruzada distancia × tiempo (revisión, obs. 2).

1. % de alumnos de 5.º grado en desierto por familia para radios de 30, 45, 60, 75 y 90 km (línea recta, oferta 2024).
2. Orígenes (estructural, topográfico, asociado al licenciamiento) recalculados con cada radio
   (el criterio topográfico usa siempre 90 min por la red).
3. Tabla cruzada 2 × 2 por familia: desierto a 60 km × desierto a 90 min
   (ambos; solo distancia = "lejos pero a menos de 90 min"; solo tiempo = "oculto"; ninguno).
4. Calibración: radio (km) cuya proporción de alumnos en desierto iguala la de 90 min, por familia.

Requiere: datos/processed/accesibilidad_colegios.csv (paso 4) y datos/processed/tiempos_colegios.csv (paso 4b).
Salidas (salidas/tablas): sensibilidad_radio.csv, sensibilidad_radio_origenes.csv, cruce_distancia_tiempo.csv,
                          calibracion_radio_tiempo.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import FAMILIAS, guardar, ruta

RADIOS = [30, 45, 60, 75, 90]
K = ["COD_MOD", "ANEXO", "ubigeo", "familia"]


def cargar() -> pd.DataFrame:
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str},
                      usecols=[*K, "quinto", "escenario", "factor_sinuosidad", "dist_min_km"])
    acc = acc[acc.factor_sinuosidad == 1.0].pivot_table(index=K + ["quinto"], columns="escenario",
                                                        values="dist_min_km").reset_index()
    acc = acc.rename(columns={"post": "d_post", "pre": "d_pre"})
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str},
                      usecols=[*K, "t_min", "desierto_t"])
    return acc.merge(tie, on=K, how="inner")


def main():
    d = cargar()
    tot = d[d.familia == "Total"].quinto.sum()
    fam_orden = ["Total", *FAMILIAS]

    # 1 y 2. radios
    filas, orig = [], []
    for r in RADIOS:
        des = (d.d_post > r).astype(int)
        for fam, g in d.assign(des=des).groupby("familia"):
            filas.append({"familia": fam, "radio_km": r, "pct_desierto": 100 * (g.quinto * g.des).sum() / tot})
        cat = np.select([(d.d_post > r) & (d.d_pre <= r), (d.d_post > r) & (d.d_pre > r), (d.d_post <= r) & (d.desierto_t == 1)],
                        ["licensing", "structural", "topographic"], default="none")
        o = d.assign(cat=cat).query("cat != 'none'").groupby(["familia", "cat"]).quinto.sum().unstack(fill_value=0)
        o["radio_km"] = r
        orig.append(o.reset_index())
    t = pd.DataFrame(filas).pivot(index="familia", columns="radio_km", values="pct_desierto").reindex(fam_orden)
    guardar(t.round(4).reset_index(), "sensibilidad_radio.csv", "tablas")
    print("% de 5.º en desierto por radio (línea recta):\n", t.round(1).to_string())
    o = pd.concat(orig, ignore_index=True)
    guardar(o, "sensibilidad_radio_origenes.csv", "tablas")
    print("\norígenes (alumnos), cualquier carrera:\n", o[o.familia == "Total"].to_string(index=False))

    # 3. tabla cruzada 60 km × 90 min
    d["dist60"] = d.d_post > 60
    d["tiempo90"] = d.desierto_t == 1
    d["celda"] = np.select([d.dist60 & d.tiempo90, d.dist60 & ~d.tiempo90, ~d.dist60 & d.tiempo90],
                           ["both", "distance_only", "time_only"], default="neither")
    c = d.groupby(["familia", "celda"]).quinto.sum().unstack(fill_value=0).reindex(fam_orden)
    c = c.reindex(columns=["both", "distance_only", "time_only", "neither"], fill_value=0)
    c["pct_distance"] = 100 * (c.both + c.distance_only) / tot
    c["pct_time"] = 100 * (c.both + c.time_only) / tot
    c["pct_disagree"] = 100 * (c.distance_only + c.time_only) / tot
    guardar(c.round(4).reset_index(), "cruce_distancia_tiempo.csv", "tablas")
    print("\ntabla cruzada 60 km × 90 min (alumnos):\n", c.round(1).to_string())

    # 4. calibración del radio equivalente a 90 min
    cal = []
    for fam, g in d.groupby("familia"):
        objetivo = (g.quinto * g.tiempo90).sum() / tot
        grilla = np.arange(20, 151)
        pct = np.array([(g.quinto * (g.d_post > r)).sum() / tot for r in grilla])
        cal.append({"familia": fam, "pct_90min": 100 * objetivo, "radio_equivalente_km": int(grilla[np.argmin(abs(pct - objetivo))])})
    cal = pd.DataFrame(cal).set_index("familia").reindex(fam_orden).reset_index()
    guardar(cal.round(4), "calibracion_radio_tiempo.csv", "tablas")
    print("\nradio equivalente a 90 min:\n", cal.round(1).to_string(index=False))
    # mediana del tiempo modelado de los alumnos cuyo programa más cercano está a 55-65 km
    g = d[(d.familia == "Total") & d.d_post.between(55, 65) & np.isfinite(d.t_min)]
    print(f"\ntiempo mediano (min) para colegios con el programa más cercano a 55-65 km: "
          f"{np.average(g.t_min, weights=g.quinto + 1e-9):.0f} (media ponderada), {g.t_min.median():.0f} (mediana), n={len(g)}")


if __name__ == "__main__":
    main()
