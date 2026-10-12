"""Paso 26 – Tiempo de viaje a la oferta reconstruida de 2018 y descomposición de orígenes con una sola definición (N3).

El artículo separaba los orígenes mezclando definiciones (estructural y licenciamiento por distancia; topográfico
por tiempo). Las sedes de 2018 de las universidades denegadas tienen coordenadas, así que el tiempo puede calcularse
en la misma red de OSM que el paso 4b. Este paso:

  1. Reconstruye la red de 4b (mismas velocidades, ríos y embarcaderos) y calcula, para cada colegio y familia, el
     tiempo mínimo a tres ofertas: post (licenciadas 2024), rel (post + seis relicenciadas, oferta 2018) y
     pre (post + todas las denegadas, oferta 2018). Solo el tiempo mínimo (Dijkstra multi-fuente): no recalcula el
     2SFCA, por eso es mucho más rápido que 4b (unos minutos tras leer el .pbf).
  2. Descomposición por TIEMPO (umbral de 90 min), coherente en una sola definición:
        licensing_t    > 90 min con post y <= 90 min con pre
        topographic_t  > 90 min con post y con pre, pero con un programa a <= 60 km en línea recta (post)
        structural_t   > 90 min con post y con pre, y sin programa a 60 km
  3. Descomposición por DISTANCIA (la del artículo) con fila propia para los alumnos a más de 60 km pero a
     <= 90 min ("distance_only"), en lugar de contarlos como estructurales.
  4. Ambas descomposiciones con la oferta post y con la oferta rel (cota de las relicenciadas).

Requiere: datos/raw/osm/peru-latest.osm.pbf (lo descarga 4b) y los pasos 1, 1b, 2 y 4.
Ejecutar aparte (lectura del .pbf: 10–30 min; usa src/red_liviana.py, unos 3-4 GB de RAM):  python src/26_tiempos_oferta_2018.py
Salidas: datos/processed/tiempos_colegios_escenarios.csv
         salidas/tablas/origenes_por_tiempo.csv, origenes_por_distancia_completa.csv, origenes_comparacion.csv
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd

from utils import CFG, FAMILIAS, RAIZ, guardar, haversine_matriz, ruta

RELICENCIADAS = ["053", "048", "050", "078", "090", "117"]


def cargar_04b():
    spec = importlib.util.spec_from_file_location("m04b", RAIZ / "src" / "04b_tiempos_viaje.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ofertas():
    post = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    den = pd.read_csv(ruta("processed") / "oferta_locales_denegadas_2018.csv", dtype={"codigo_entidad": str}).dropna(subset=["lat", "lon"])
    rel = den[den.codigo_entidad.isin(RELICENCIADAS)]
    return {"post": post, "rel": pd.concat([post, rel], ignore_index=True), "pre": pd.concat([post, den], ignore_index=True)}


def calcular_tiempos():
    m = cargar_04b()
    if not m.PBF.exists():
        m.descargar_pbf()
    from red_liviana import construir_red   # misma red que 4b, con menos memoria
    g, lat_red, lon_red = construir_red(m.PBF)
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    col["quinto"] = col.quinto.fillna(0)
    nc, acc_c, _ = m.anclar(lat_red, lon_red, col.lat, col.lon)
    filas = []
    for esc, loc in ofertas().items():
        nl, acc_l, _ = m.anclar(lat_red, lon_red, loc.lat, loc.lon)
        for fam in ["Total", *FAMILIAS]:
            sel = ((loc.n_programas if fam == "Total" else loc[fam]).fillna(0) > 0).to_numpy()
            if not sel.any():
                continue
            t = m.tiempos_desde_fuentes(g, nl[sel], acc_l[sel])[nc] + acc_c
            filas.append(pd.DataFrame({"COD_MOD": col.COD_MOD.values, "ANEXO": col.ANEXO.values, "familia": fam,
                                       "escenario": esc, "t_min": t}))
            print(f"  {esc:5s} {fam}: listo")
    res = pd.concat(filas, ignore_index=True).pivot_table(index=["COD_MOD", "ANEXO", "familia"], columns="escenario",
                                                          values="t_min").reset_index()
    res.columns = ["COD_MOD", "ANEXO", "familia", *[f"t_{c}" for c in res.columns[3:]]]
    guardar(res, "tiempos_colegios_escenarios.csv")
    return res


def descomponer(tie: pd.DataFrame):
    radio, umbral = CFG["accesibilidad"]["radio_desierto_km"], CFG["tiempos"]["umbral_desierto_min"]
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype=dt)
    col["quinto"] = col.quinto.fillna(0)
    of = ofertas()
    filas_t, filas_d = [], []
    for fam in ["Total", *FAMILIAS]:
        # desiertos por distancia en las tres ofertas
        D = {}
        for esc, loc in of.items():
            oferta = ((loc.n_programas if fam == "Total" else loc[fam]).fillna(0) > 0).to_numpy()
            d = haversine_matriz(col.lat, col.lon, loc.lat[oferta], loc.lon[oferta]) <= radio
            D[esc] = ~d.any(axis=1)
        x = col[["COD_MOD", "ANEXO", "quinto"]].merge(tie[tie.familia == fam], on=["COD_MOD", "ANEXO"], how="left")
        T = {esc: (x[f"t_{esc}"].fillna(np.inf) > umbral).to_numpy() for esc in of}
        q = x.quinto.to_numpy()
        for actual in ["post", "rel"]:
            # --- por tiempo
            cats = {"licensing_t": T[actual] & ~T["pre"],
                    "topographic_t": T[actual] & T["pre"] & ~D[actual],
                    "structural_t": T[actual] & T["pre"] & D[actual]}
            tot = q[T[actual]].sum()
            for c, msk in cats.items():
                filas_t.append({"familia": fam, "oferta": actual, "origen": c, "alumnos": int(q[msk].sum()),
                                "pct_del_desierto": 100 * q[msk].sum() / max(tot, 1), "pct_quinto": 100 * q[msk].sum() / q.sum()})
            # --- por distancia, con fila propia para los que llegan en <= 90 min
            cats_d = {"structural": D[actual] & D["pre"] & T[actual],
                      "distance_only": D[actual] & ~T[actual],
                      "licensing": D[actual] & ~D["pre"] & T[actual],
                      "topographic": ~D[actual] & T[actual]}
            union = (D[actual] | T[actual])
            for c, msk in cats_d.items():
                filas_d.append({"familia": fam, "oferta": actual, "origen": c, "alumnos": int(q[msk].sum()),
                                "pct_del_desierto": 100 * q[msk].sum() / max(q[union].sum(), 1)})
    ot, od = pd.DataFrame(filas_t), pd.DataFrame(filas_d)
    guardar(ot.round(4), "origenes_por_tiempo.csv", "tablas")
    guardar(od.round(4), "origenes_por_distancia_completa.csv", "tablas")
    comp = (ot[ot.familia == "Total"].pivot_table(index="origen", columns="oferta", values="alumnos")
            .join(od[od.familia == "Total"].pivot_table(index="origen", columns="oferta", values="alumnos"), how="outer",
                  lsuffix="_tiempo", rsuffix="_distancia"))
    guardar(comp.reset_index(), "origenes_comparacion.csv", "tablas")
    print("\nCualquier carrera – orígenes por tiempo:\n", ot[ot.familia == "Total"].to_string(index=False))
    print("\nCualquier carrera – orígenes por distancia (con distance_only):\n", od[od.familia == "Total"].to_string(index=False))
    print("\nPor familia (tiempo, oferta post):\n",
          ot[ot.oferta == "post"].pivot_table(index="familia", columns="origen", values="alumnos").to_string())


if __name__ == "__main__":
    archivo = ruta("processed") / "tiempos_colegios_escenarios.csv"
    if "--solo-descomponer" in sys.argv and archivo.exists():
        tie = pd.read_csv(archivo, dtype={"COD_MOD": str, "ANEXO": str})
    else:
        tie = calcular_tiempos()
    descomponer(tie)
