"""Paso 34 – Error de los tiempos calibrados por región natural (revisión v3, obs. M3).

Reúne los 55 pares colegio-local con tiempo de Google Maps (en auto, salida lunes 07:00):
  - 37 pares de calibración (paso 32): tiempo calibrado en validación cruzada dejando uno fuera (LOOCV);
  - 18 pares de la verificación de 20 colegios (paso 33): tiempo calibrado al mismo local, fuera de muestra.
Reporta, por región natural (costa, sierra, selva) y en total: error medio y mediano relativo, error absoluto medio,
% de pares dentro de ±30 %, correlaciones de Pearson y Spearman, pares cerca del umbral (60-120 min en Google) y
acuerdo en el umbral de 90 min.

Región natural: la del paso 36 (sierra si el colegio está a >= 2,000 m; si no, costa en departamentos costeros y
selva en el resto); si el paso 36 no se ha corrido, se usa el departamento (respaldo, menos preciso).

Requiere los pasos 32 y 33 (y opcionalmente 36). Salidas: salidas/tablas/calibracion_por_region.csv,
salidas/tablas/calibracion_pares_todos.csv
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils import CFG, RAIZ, guardar, ruta

T = ruta("tablas")
M = RAIZ / "datos/manual"


def region_de(cod, dep):
    p = ruta("processed") / "relieve_colegios.csv"
    if p.exists():
        r = pd.read_csv(p, dtype={"COD_MOD": str}).drop_duplicates("COD_MOD").set_index("COD_MOD").region_natural
        out = pd.Series(cod).map(r)
        fuente = "elevación (paso 36)"
    else:
        out = pd.Series([None] * len(cod), dtype=object)
        fuente = "departamento (respaldo)"
    spec = importlib.util.spec_from_file_location("m36", RAIZ / "src" / "36_relieve_grupos.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    falta = out.isna().to_numpy()
    out = out.astype(object)
    out[falta] = m.region_sin_dem(pd.Series(dep)[falta])
    return out.to_numpy(), fuente


def metricas(obs, mod, umbral):
    e = mod - obs
    rel = 100 * e / obs
    cerca = (obs >= umbral - 30) & (obs <= umbral + 30)
    return {"pares": len(obs), "error_medio_pct": rel.mean(), "error_mediano_pct": np.median(rel),
            "error_absoluto_medio_min": np.abs(e).mean(), "pct_dentro_30pct": 100 * np.mean(np.abs(rel) <= 30),
            "pearson": np.corrcoef(mod, obs)[0, 1] if len(obs) > 2 else np.nan,
            "spearman": spearmanr(mod, obs).statistic if len(obs) > 2 else np.nan,
            "pares_60_120min": int(cerca.sum()),
            "acuerdo_umbral_90_pct": 100 * np.mean((mod > umbral) == (obs > umbral)),
            "acuerdo_umbral_90_pares_cercanos_pct": 100 * np.mean((mod[cerca] > umbral) == (obs[cerca] > umbral)) if cerca.any() else np.nan}


def main():
    umbral = CFG["tiempos"]["umbral_desierto_min"]
    cal = pd.read_csv(T / "calibracion_pares.csv", dtype={"COD_MOD": str})
    a = pd.DataFrame({"COD_MOD": cal.COD_MOD, "departamento": cal.departamento, "muestra": "calibración (LOOCV)",
                      "google_min": cal.tiempo_observado_min.astype(float), "modelado_min": cal.modelado_loocv.astype(float),
                      "modelado_sin_calibrar_min": cal.modelado_sin_calibrar.astype(float)})
    ver = pd.read_csv(M / "muestra_verificacion_manual.csv", dtype={"COD_MOD": str})
    mo = pd.read_csv(M / "verificacion_manual_modelo.csv", dtype={"COD_MOD": str})
    go = pd.read_csv(M / "verificacion_manual_google.csv", dtype={"COD_MOD": str})
    v = ver[["COD_MOD", "departamento"]].merge(mo, on="COD_MOD").merge(go, on="COD_MOD")
    v = v[v.tiempo_google_min.notna()]
    b = pd.DataFrame({"COD_MOD": v.COD_MOD, "departamento": v.departamento, "muestra": "verificación (fuera de muestra)",
                      "google_min": v.tiempo_google_min.astype(float), "modelado_min": v.tiempo_modelado_al_local_min.astype(float),
                      "modelado_sin_calibrar_min": np.nan})
    p = pd.concat([a, b], ignore_index=True)
    p["region"], fuente = region_de(p.COD_MOD.to_numpy(), p.departamento.to_numpy())
    p["error_pct"] = (100 * (p.modelado_min - p.google_min) / p.google_min).round(1)
    guardar(p, "calibracion_pares_todos.csv", "tablas")

    filas = []
    for (muestra, reg), x in [(("todas", "todas"), p), *[(("todas", r), x) for r, x in p.groupby("region")],
                              *[((mu, "todas"), x) for mu, x in p.groupby("muestra")]]:
        filas.append({"muestra": muestra, "region": reg, **metricas(x.google_min.to_numpy(), x.modelado_min.to_numpy(), umbral)})
    t = pd.DataFrame(filas)
    t["fuente_region"] = fuente
    guardar(t.round(4), "calibracion_por_region.csv", "tablas")
    print(f"región natural según {fuente}")
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
