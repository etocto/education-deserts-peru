"""Paso 4d – Sensibilidad de los desiertos por tiempo de viaje a los supuestos de velocidad y al umbral.

Escenarios (se lee el .pbf una sola vez y se recalculan solo los tiempos mínimos, sin el 2SFCA):
  base          velocidades de config.yaml (debe reproducir el paso 4b)
  vias_x0.75    todas las vías 25 % más lentas (terreno, estado de la vía, transporte público)
  vias_x1.25    todas las vías 25 % más rápidas
  rio_10.6      bote a 10,6 km/h: mínimo observado aguas arriba (Tenkanen et al., 2015)
  rio_18.0      bote a 18,0 km/h: máximo observado aguas abajo (Tenkanen et al., 2015)
  sin_rios      solo red vial (sin ríos ni ferris)
Umbrales: 60, 90 y 120 minutos para cada escenario.

Salidas
  datos/processed/sensibilidad_colegios.csv.gz  colegio × familia × escenario: desierto a 90 min (para el paso 4h)
  salidas/tablas/sensibilidad_tiempos.csv           % de 5.º en desierto por familia × escenario × umbral
  manuscrito/tablas/tabla_sensibilidad_tiempos.tex  tabla para el anexo del artículo

Requiere haber descargado el .pbf (paso 4b). Tiempo estimado: 15–30 min.
  python src/04d_sensibilidad_tiempos.py            todas las familias
  python src/04d_sensibilidad_tiempos.py --rapido   solo "Total" (unos minutos; no toca la tabla del manuscrito)
"""
from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from utils import CFG, FAMILIAS, FAMILIAS_EN, RAIZ, guardar, ruta

_spec = importlib.util.spec_from_file_location("t4b", RAIZ / "src" / "04b_tiempos_viaje.py")
t4b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t4b)

T = CFG["tiempos"]
R_KM = 6371.0088
FLUVIAL = {"river", "ferry"}
UMBRALES = [60, 90, 120]


def aristas(vias):
    """Nodos y tramos con su distancia (km) y tipo de vía, para recalcular tiempos sin releer el .pbf."""
    tipos = sorted(T["velocidades_kmh"])
    cod = {t: k for k, t in enumerate(tipos)}
    idx: dict[int, int] = {}
    lat, lon, a, b, tp = [], [], [], [], []
    nf, nv = set(), set()
    for tipo, nodos, coords in vias:
        marca = nf if tipo in FLUVIAL else nv
        prev = None
        for nid, (la, lo) in zip(nodos, coords):
            if nid not in idx:
                idx[nid] = len(lat)
                lat.append(la)
                lon.append(lo)
            i = idx[nid]
            marca.add(i)
            if prev is not None:
                a.append(prev); b.append(i); tp.append(cod[tipo])
            prev = i
    lat, lon = np.array(lat), np.array(lon)
    a, b, tp = np.array(a), np.array(b), np.array(tp)
    la1, lo1, la2, lo2 = map(np.radians, (lat[a], lon[a], lat[b], lon[b]))
    d = 2 * R_KM * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    fluvial, vial = np.zeros(len(lat), bool), np.zeros(len(lat), bool)
    fluvial[list(nf)] = True
    vial[list(nv)] = True
    puertos = t4b.enlaces_puerto(lat, lon, fluvial, vial)   # embarcaderos: (nodo fluvial, nodo vial, minutos)
    print(f"red: {len(lat):,} nodos, {len(a):,} tramos")
    return lat, lon, a, b, d, tp, tipos, puertos


def grafo(n, a, b, d, tp, tipos, vel, puertos):
    v = np.array([vel.get(t, np.nan) for t in tipos])[tp]
    ok = np.isfinite(v) & (v > 0)
    aa, bb, t = a[ok], b[ok], 60 * d[ok] / v[ok]
    if "river" in vel:   # embarcaderos solo si hay ríos en el escenario
        aa, bb, t = np.r_[aa, puertos[0]], np.r_[bb, puertos[1]], np.r_[t, puertos[2]]
    t = np.maximum(t, 1e-6)
    return coo_matrix((np.r_[t, t], (np.r_[aa, bb], np.r_[bb, aa])), shape=(n, n)).tocsr()


def escenarios():
    base = dict(T["velocidades_kmh"])
    esc = {"base": base,
           "vias_x0.75": {k: (v if k in FLUVIAL else 0.75 * v) for k, v in base.items()},
           "vias_x1.25": {k: (v if k in FLUVIAL else 1.25 * v) for k, v in base.items()},
           "rio_10.6": {**base, "river": 10.6},
           "rio_18.0": {**base, "river": 18.0},
           "sin_rios": {k: v for k, v in base.items() if k not in FLUVIAL}}
    return esc


def main(solo_total: bool = False):
    if not t4b.PBF.exists():
        raise FileNotFoundError("Falta el .pbf: corre primero python src/04b_tiempos_viaje.py")
    lat, lon, a, b, d, tp, tipos, puertos = aristas(t4b.leer_pbf(t4b.PBF))
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    q = col.quinto.fillna(0).to_numpy()
    nc, acc_c, _ = t4b.anclar(lat, lon, col.lat, col.lon)
    nl, acc_l, _ = t4b.anclar(lat, lon, loc.lat, loc.lon)
    familias = ["Total"] if solo_total else ["Total", *FAMILIAS]

    filas, por_colegio = [], []
    for nombre, vel in escenarios().items():
        g = grafo(len(lat), a, b, d, tp, tipos, vel, puertos)
        for fam in familias:
            oferta = (loc.n_programas if fam == "Total" else loc[fam]).fillna(0).to_numpy(float)
            sel = oferta > 0
            t_min = t4b.tiempos_desde_fuentes(g, nl[sel], acc_l[sel])[nc] + acc_c
            if not solo_total:   # clasificación por colegio (umbral 90) para medir estabilidad (paso 4h)
                por_colegio.append(pd.DataFrame({"COD_MOD": col.COD_MOD.values, "ANEXO": col.ANEXO.values,
                                                 "quinto": q, "familia": fam, "escenario": nombre,
                                                 "desierto_90": (t_min > 90).astype("int8")}))
            for u in UMBRALES:
                filas.append({"escenario": nombre, "familia": fam, "umbral_min": u,
                              "pct_desierto": 100 * np.average(t_min > u, weights=q + 1e-9),
                              "quinto_desierto": q[t_min > u].sum()})
        print(f"  {nombre}: Total a más de 90 min = "
              f"{[f['pct_desierto'] for f in filas if f['escenario'] == nombre and f['familia'] == 'Total' and f['umbral_min'] == 90][0]:.1f} %")
    res = pd.DataFrame(filas)
    if solo_total:   # la corrida rápida no sobrescribe la completa ni la tabla del manuscrito
        guardar(res.round(2), "sensibilidad_tiempos_rapido.csv", "tablas")
        print(res[res.umbral_min == 90].round(1).to_string(index=False))
        return
    guardar(res.round(2), "sensibilidad_tiempos.csv", "tablas")
    pc = pd.concat(por_colegio, ignore_index=True)
    pc.to_csv(ruta("processed") / "sensibilidad_colegios.csv.gz", index=False)
    print(f"  -> datos/processed/sensibilidad_colegios.csv.gz ({len(pc):,} filas)")

    # tabla LaTeX: filas = familias, columnas = escenarios (umbral 90) + umbrales 60/120 (escenario base)
    w = res[res.umbral_min == 90].pivot(index="familia", columns="escenario", values="pct_desierto")
    w = w[list(escenarios())]
    b_ = res[res.escenario == "base"].pivot(index="familia", columns="umbral_min", values="pct_desierto")
    w["base, 60 min"], w["base, 120 min"] = b_[60], b_[120]
    orden = [f for f in [*FAMILIAS, "Total"] if f in w.index]
    w = w.loc[orden]
    w.index = [FAMILIAS_EN.get(f, "Any field") if f != "Total" else "Any field" for f in w.index]
    w.columns = ["Base", "Roads $\\times$0.75", "Roads $\\times$1.25", "River 10.6", "River 18.0", "No rivers",
                 "Base, 60 min", "Base, 120 min"]
    tex = w.round(1).to_latex(escape=False, column_format="p{3.8cm}" + "r" * w.shape[1], float_format="%.1f")
    (ruta("manuscrito_tablas") / "tabla_sensibilidad_tiempos.tex").write_text(tex, encoding="utf-8")
    print(w.round(1).to_string())


if __name__ == "__main__":
    main(solo_total="--rapido" in sys.argv)
