"""Paso 22 – Muestra para verificación manual de distancias, tiempos y clasificación (revisión, obs. 13).

Selecciona 20 colegios (semilla fija; 5 en desierto estructural, 5 topográfico, 5 asociado al licenciamiento y 5 sin
desierto, cualquier carrera) y para cada uno anota: coordenadas, local universitario más cercano en línea recta
(universidad, coordenadas), distancia, tiempo modelado, clasificación y origen. Los autores comprueban a mano la
distancia (p. ej., con la herramienta de medición de un visor de mapas), la plausibilidad del tiempo y la categoría,
y registran el resultado en las columnas verificado_* del archivo.

Salida: datos/manual/muestra_verificacion_manual.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from utils import CFG, RAIZ, haversine_matriz, ruta


def main(n_por_grupo: int = 5):
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    k = ["COD_MOD", "ANEXO", "ubigeo", "familia"]
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt, usecols=[*k, "quinto", "escenario", "factor_sinuosidad", "desierto", "dist_min_km"])
    acc = acc[(acc.factor_sinuosidad == 1.0) & (acc.familia == "Total")]
    w = acc.pivot_table(index=k + ["quinto"], columns="escenario", values=["desierto", "dist_min_km"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt, usecols=[*k, "t_min", "desierto_t"])
    w = w.merge(tie[tie.familia == "Total"], on=k)
    w["origen"] = np.select([(w.desierto_post == 1) & (w.desierto_pre == 0), (w.desierto_post == 1) & (w.desierto_pre == 1),
                             (w.desierto_post == 0) & (w.desierto_t == 1)], ["licensing", "structural", "topographic"], default="none")
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype=dt)[["COD_MOD", "ANEXO", "CEN_EDU", "D_DPTO", "D_DIST", "lat", "lon", "coord_valida"]]
    w = w.merge(col, on=["COD_MOD", "ANEXO"])
    w = w[(w.coord_valida == 1) & (w.quinto > 0)]
    m = (w.sample(frac=1, random_state=CFG["modelos"]["semilla"]).groupby("origen").head(n_por_grupo)
           .sort_values(["origen", "D_DPTO"]).reset_index(drop=True))
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"]).reset_index(drop=True)
    d = haversine_matriz(m.lat, m.lon, loc.lat, loc.lon)
    j = d.argmin(axis=1)
    out = pd.DataFrame({"COD_MOD": m.COD_MOD, "ANEXO": m.ANEXO, "colegio": m.CEN_EDU, "departamento": m.D_DPTO, "distrito": m.D_DIST,
                        "lat": m.lat, "lon": m.lon, "local_mas_cercano": loc.universidad.values[j] + " (" + loc.distrito.values[j] + ")",
                        "lat_local": loc.lat.values[j], "lon_local": loc.lon.values[j],
                        "distancia_km": m.dist_min_km_post.round(1), "distancia_recalculada_km": d.min(axis=1).round(1),
                        "tiempo_modelado_min": m.t_min.round(0), "desierto_60km": m.desierto_post.astype(int),
                        "desierto_90min": m.desierto_t.astype(int), "origen": m.origen,
                        "verificado_distancia": "", "verificado_tiempo_plausible": "", "verificado_categoria": "", "nota": ""})
    assert np.allclose(out.distancia_km, out.distancia_recalculada_km, atol=0.1), "la distancia guardada no coincide con la recalculada"
    p = RAIZ / "datos/manual/muestra_verificacion_manual.csv"
    if p.exists() and "--forzar" not in sys.argv:
        print(f"{p.relative_to(RAIZ)} ya existe y está verificada a mano; no se sobrescribe (usar --forzar)")
        return
    out.to_csv(p, index=False, encoding="utf-8")
    print(f"muestra de verificación: {p.relative_to(RAIZ)} ({len(out)} colegios)")
    print(out[["colegio", "departamento", "distancia_km", "tiempo_modelado_min", "origen"]].to_string(index=False))


if __name__ == "__main__":
    main()
