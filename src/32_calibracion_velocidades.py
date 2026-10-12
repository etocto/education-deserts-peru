"""Paso 32 – Calibración de las velocidades de las vías con los tiempos observados (validación, obs. 8, N1).

La validación con 37 pares colegio-sede (Google Maps, en auto, salida lunes 7:00) mostró que el modelo ordena bien
los tiempos (Spearman 0.98) pero los subestima a cerca de la mitad. Este paso estima UN factor multiplicativo para
las velocidades de todas las vías (no para ríos, ferris, embarcaderos ni el tramo a pie, que Google no valida):

  1. Con las velocidades sin calibrar, calcula la ruta más rápida de cada par y separa su tiempo en tramo vial
     (t_via) y tramo no vial (t_otro: ríos, transbordos y caminata de acceso).
  2. Con factor f, el tiempo modelado es t_via / f + t_otro (la ruta óptima no cambia al escalar todas las vías).
  3. Elige f que minimiza el error cuadrático en logaritmos y reporta el ajuste dentro de muestra y en validación
     cruzada dejando uno fuera (LOOCV), además del acuerdo en el umbral de 90 minutos.
  4. Escribe f en config.yaml (tiempos.factor_calibracion_vias); utils.py lo aplica a todas las velocidades viales,
     de modo que los pasos 4b, 4d, 26, 31 y siguientes usan la red calibrada.

Requiere datos/manual/validacion_tiempos.csv (con tiempo_observado_min) y el .pbf.
Salidas: salidas/tablas/calibracion_velocidades.csv, salidas/tablas/calibracion_pares.csv
"""
from __future__ import annotations

import importlib.util
import re

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from red_liviana import aristas_compactas, grafo_desde
from utils import CFG, RAIZ, guardar

FLUVIAL = {"river", "ferry"}


def ajuste(f, tv, to, obs):
    m = tv / f + to
    e = np.log(m) - np.log(obs)
    return m, (e ** 2).sum()


def mejor_f(tv, to, obs, grilla=np.arange(0.30, 1.0001, 0.0025)):
    return grilla[np.argmin([ajuste(f, tv, to, obs)[1] for f in grilla])]


def metricas(m, obs):
    e = m - obs
    return {"error_medio_min": e.mean(), "error_absoluto_medio_min": np.abs(e).mean(),
            "error_relativo_mediano_pct": 100 * np.median(e / obs), "pct_dentro_30pct": 100 * np.mean(np.abs(e / obs) <= 0.3),
            "pearson": np.corrcoef(m, obs)[0, 1], "acuerdo_umbral_90min_pct": 100 * np.mean((m > 90) == (obs > 90))}


def main():
    spec = importlib.util.spec_from_file_location("m04b", RAIZ / "src" / "04b_tiempos_viaje.py")
    m4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m4)
    v = pd.read_csv(RAIZ / "datos/manual/validacion_tiempos.csv", dtype={"COD_MOD": str, "ANEXO": str})
    v = v[pd.to_numeric(v.tiempo_observado_min, errors="coerce").notna()].reset_index(drop=True)

    vel0 = CFG["tiempos"]["velocidades_kmh_sin_calibrar"]
    lat, lon, a, b, d, tp, tipos, puertos = aristas_compactas(m4.PBF)
    g = grafo_desde(lat, a, b, d, tp, tipos, vel0, puertos)
    g_otro = grafo_desde(lat, a, b, d, tp, tipos, {k: x for k, x in vel0.items() if k in FLUVIAL}, puertos)
    nc, acc_c, _ = m4.anclar(lat, lon, v.lat_colegio, v.lon_colegio)
    ns, acc_s, _ = m4.anclar(lat, lon, v.lat_sede, v.lon_sede)
    tv, to = np.zeros(len(v)), np.zeros(len(v))
    for i in range(len(v)):
        dist, pred = dijkstra(g, indices=nc[i], return_predecessors=True, limit=900)
        if not np.isfinite(dist[ns[i]]):
            tv[i] = np.nan
            continue
        k, via, otro = ns[i], 0.0, 0.0
        while k != nc[i]:
            j = pred[k]
            w = g[j, k]
            if g_otro[j, k] > 0:
                otro += w
            else:
                via += w
            k = j
        tv[i], to[i] = via, otro + acc_c[i] + acc_s[i]
    ok = np.isfinite(tv)
    v, tv, to = v[ok].reset_index(drop=True), tv[ok], to[ok]
    obs = v.tiempo_observado_min.astype(float).to_numpy()

    f = mejor_f(tv, to, obs)
    m_cal, _ = ajuste(f, tv, to, obs)
    m_sin = tv + to
    loo = np.array([ajuste(mejor_f(np.delete(tv, i), np.delete(to, i), np.delete(obs, i)), tv[i:i + 1], to[i:i + 1], obs[i:i + 1])[0][0]
                    for i in range(len(v))])
    filas = [{"modelo": "sin calibrar", "factor": 1.0, **metricas(m_sin, obs)},
             {"modelo": "calibrado (dentro de muestra)", "factor": f, **metricas(m_cal, obs)},
             {"modelo": "calibrado (LOOCV)", "factor": f, **metricas(loo, obs)}]
    r = pd.DataFrame(filas)
    r.insert(1, "pares", len(v))
    guardar(r.round(4), "calibracion_velocidades.csv", "tablas")
    pares = v[["COD_MOD", "ANEXO", "colegio", "departamento", "tiempo_observado_min"]].assign(
        t_via_sin_calibrar=tv.round(1), t_otro=to.round(1), modelado_sin_calibrar=m_sin.round(1),
        modelado_calibrado=m_cal.round(1), modelado_loocv=loo.round(1))
    guardar(pares, "calibracion_pares.csv", "tablas")
    print(r.round(3).to_string(index=False))

    # escribir el factor en config.yaml (sin tocar el resto del archivo)
    p = RAIZ / "config.yaml"
    txt = p.read_text(encoding="utf-8")
    linea = f"  factor_calibracion_vias: {f:.4f}       # paso 32: velocidades viales × factor (validación con Google Maps)"
    if "factor_calibracion_vias" in txt:
        txt = re.sub(r"  factor_calibracion_vias:.*", linea, txt)
    else:
        txt = txt.replace("  velocidades_kmh:", linea + "\n  velocidades_kmh:", 1)
    p.write_text(txt, encoding="utf-8")
    print(f"\nfactor de calibración de las vías = {f:.4f} (escrito en config.yaml)")


if __name__ == "__main__":
    main()
