"""Paso 37 – Sensibilidad a un factor de velocidad por región natural (revisión v3, obs. M3).

El paso 32 estima UN factor para todas las vías del país (0.4875). Los errores de validación difieren por región
(paso 34), así que aquí se repite la calibración con un factor por región natural (costa, sierra, selva) y se
recalculan los tiempos de todos los colegios:

  1. Con los 37 pares de calibración (t_via y t_otro del paso 32), estima f_costa, f_sierra y f_selva con el mismo
     criterio (mínimos cuadrados en logaritmos) y compara su ajuste con el factor único en validación cruzada
     dejando uno fuera.
  2. Construye la red tres veces (velocidades viales sin calibrar × f_región; ríos y transbordos sin cambio) y, con
     Dijkstra multi-fuente desde los locales licenciados de 2024, calcula el tiempo de cada colegio en la red de SU
     región (aproximación: toda la ruta se recorre al factor de la región del colegio).
  3. Compara con la base (factor único): % de alumnos a más de 90 min por familia, % clasificado igual y tamaño de
     los grupos de origen (estructural, asociado al licenciamiento y topográfico, con las mismas reglas de L94).

Región natural: paso 36 si existe (recomendado: correrlo antes); si no, por departamento.
Requiere los pasos 4, 4b, 32 y el .pbf. Unos 3 GB de RAM; 20-40 min.
Uso:     python src/37_factor_regional.py
Salidas: datos/processed/tiempos_colegios_factor_regional.csv
         salidas/tablas/factores_regionales.csv, sensibilidad_factor_regional.csv, origenes_factor_regional.csv
"""
from __future__ import annotations

import gc
import importlib.util

import numpy as np
import pandas as pd

from utils import CFG, FAMILIAS, RAIZ, guardar, ruta

FLUVIAL = {"river", "ferry"}


def cargar(nombre, archivo):
    spec = importlib.util.spec_from_file_location(nombre, RAIZ / "src" / archivo)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def regiones(cod, dep):
    m36 = cargar("m36", "36_relieve_grupos.py")
    p = ruta("processed") / "relieve_colegios.csv"
    out = pd.Series([None] * len(cod), dtype=object)
    fuente = "departamento (respaldo)"
    if p.exists():
        r = pd.read_csv(p, dtype={"COD_MOD": str, "ANEXO": str})
        r = r.drop_duplicates("COD_MOD").set_index("COD_MOD").region_natural
        out = pd.Series(cod).map(r).astype(object)
        fuente = "elevación (paso 36)"
    falta = out.isna().to_numpy()
    out[falta] = m36.region_sin_dem(pd.Series(dep)[falta])
    return out.to_numpy(), fuente


def ajuste_regional(m32, tv, to, obs, reg):
    f = {r: m32.mejor_f(tv[reg == r], to[reg == r], obs[reg == r]) for r in np.unique(reg)}
    mod = np.array([tv[i] / f[reg[i]] + to[i] for i in range(len(tv))])
    loo = np.empty(len(tv))
    for i in range(len(tv)):
        k = (reg == reg[i]) & (np.arange(len(tv)) != i)
        fi = m32.mejor_f(tv[k], to[k], obs[k])
        loo[i] = tv[i] / fi + to[i]
    return f, mod, loo


def main():
    m32 = cargar("m32", "32_calibracion_velocidades.py")
    m4b = cargar("m4b", "04b_tiempos_viaje.py")
    umbral = CFG["tiempos"]["umbral_desierto_min"]
    radio = CFG["accesibilidad"]["radio_desierto_km"]

    # ---- 1. factores por región con los pares de calibración
    cal = pd.read_csv(ruta("tablas") / "calibracion_pares.csv", dtype={"COD_MOD": str})
    reg_p, fuente = regiones(cal.COD_MOD.to_numpy(), cal.departamento.to_numpy())
    tv, to, obs = cal.t_via_sin_calibrar.to_numpy(float), cal.t_otro.to_numpy(float), cal.tiempo_observado_min.to_numpy(float)
    f_reg, mod_reg, loo_reg = ajuste_regional(m32, tv, to, obs, reg_p)
    f_uno = float(CFG["tiempos"]["factor_calibracion_vias"])
    filas = [{"modelo": "factor único (LOOCV)", "region": "todas", "factor": f_uno, "pares": len(obs), **m32.metricas(cal.modelado_loocv.to_numpy(float), obs)},
             {"modelo": "factor por región (LOOCV)", "region": "todas", "factor": np.nan, "pares": len(obs), **m32.metricas(loo_reg, obs)}]
    for r, f in f_reg.items():
        k = reg_p == r
        filas.append({"modelo": "factor por región (dentro de muestra)", "region": r, "factor": f, "pares": int(k.sum()),
                      **m32.metricas(mod_reg[k], obs[k])})
    fr = pd.DataFrame(filas)
    fr["fuente_region"] = fuente
    guardar(fr.round(4), "factores_regionales.csv", "tablas")
    print(f"región natural según {fuente}")
    print(fr.round(3).to_string(index=False))

    # ---- 2. tiempos de todos los colegios con el factor de su región
    from red_liviana import aristas_compactas, grafo_desde
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype=dt)
    col["quinto"] = col.quinto.fillna(0)
    col["region"], _ = regiones(col.COD_MOD.to_numpy(), col.D_DPTO.fillna("").to_numpy())
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    lat, lon, a, b, d, tp, tipos, puertos = aristas_compactas(m4b.PBF)
    nc, acc_c, _ = m4b.anclar(lat, lon, col.lat, col.lon)
    nl, acc_l, _ = m4b.anclar(lat, lon, loc.lat, loc.lon)
    vel0 = CFG["tiempos"]["velocidades_kmh_sin_calibrar"]
    t = {fam: np.full(len(col), np.nan) for fam in ["Total", *FAMILIAS]}
    for r, f in f_reg.items():
        sel_c = (col.region == r).to_numpy()
        if not sel_c.any():
            continue
        vel = {k: (v if k in FLUVIAL else v * f) for k, v in vel0.items()}
        g = grafo_desde(lat, a, b, d, tp, tipos, vel, puertos)
        for fam in t:
            sel = ((loc.n_programas if fam == "Total" else loc[fam]).fillna(0) > 0).to_numpy()
            dist = m4b.tiempos_desde_fuentes(g, nl[sel], acc_l[sel])
            t[fam][sel_c] = dist[nc[sel_c]] + acc_c[sel_c]
            print(f"  {r} (f = {f:.3f}) {fam}: listo")
        del g
        gc.collect()
    res = pd.concat([pd.DataFrame({"COD_MOD": col.COD_MOD.values, "ANEXO": col.ANEXO.values, "familia": fam,
                                   "region": col.region.values, "t_min_regional": v}) for fam, v in t.items()], ignore_index=True)
    guardar(res, "tiempos_colegios_factor_regional.csv")

    # ---- 3. comparación con la base (factor único)
    base = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt, usecols=["COD_MOD", "ANEXO", "familia", "t_min", "desierto_t"])
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt,
                      usecols=["COD_MOD", "ANEXO", "familia", "escenario", "factor_sinuosidad", "desierto"])
    acc = acc[acc.factor_sinuosidad == 1.0].pivot_table(index=["COD_MOD", "ANEXO", "familia"], columns="escenario", values="desierto").reset_index()
    x = res.merge(base, on=["COD_MOD", "ANEXO", "familia"]).merge(acc, on=["COD_MOD", "ANEXO", "familia"])
    x = x.merge(col[["COD_MOD", "ANEXO", "quinto"]], on=["COD_MOD", "ANEXO"])
    x["T_reg"] = (x.t_min_regional.fillna(np.inf) > umbral).astype(int)
    sens, ori = [], []
    for fam, z in x.groupby("familia"):
        q = z.quinto.to_numpy()
        sens.append({"familia": fam, "pct_90_base": 100 * q[z.desierto_t == 1].sum() / q.sum(),
                     "pct_90_regional": 100 * q[z.T_reg == 1].sum() / q.sum(),
                     "pct_clasificado_igual": 100 * q[z.desierto_t == z.T_reg].sum() / q.sum()})
        for nombre, T in [("base", z.desierto_t == 1), ("regional", z.T_reg == 1)]:
            D, Dp = z.post == 1, z.pre == 1
            for o, msk in [("structural", D & Dp & T), ("licensing", D & ~Dp & T), ("topographic", ~D & T), ("beyond_60_within_90", D & ~T)]:
                ori.append({"familia": fam, "factor": nombre, "origen": o, "alumnos": int(q[msk.to_numpy()].sum())})
    s = pd.DataFrame(sens)
    o = pd.DataFrame(ori).pivot_table(index=["familia", "origen"], columns="factor", values="alumnos").reset_index()
    guardar(s.round(4), "sensibilidad_factor_regional.csv", "tablas")
    guardar(o, "origenes_factor_regional.csv", "tablas")
    print(s.round(1).to_string(index=False))
    print(o[o.familia == "Total"].to_string(index=False))


if __name__ == "__main__":
    main()
