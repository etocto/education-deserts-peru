"""Paso 16 – Diagnóstico de los tiempos de viaje (revisión, obs. 8).

1. Colegios no conectados a la red (tiempo infinito): número, alumnos y departamento.
2. Efecto de la red fluvial: colegios y alumnos cuya clasificación (desierto a 90 min) cambia al quitar los ríos
   o al variar la velocidad del bote, por familia (a partir de datos/processed/sensibilidad_colegios.csv.gz).
3. Validación externa de los tiempos modelados:
   - genera datos/manual/validacion_tiempos_plantilla.csv: 40 colegios (muestra estratificada por macrorregión)
     con la sede más cercana en línea recta y el tiempo modelado; los autores completan tiempo_observado_min y fuente
     (empresas de transporte, registros oficiales o un servicio de rutas) y guardan el archivo como
     datos/manual/validacion_tiempos.csv;
   - si ese archivo existe, calcula error medio, error absoluto medio, correlación y % de pares dentro de ±30 %.
   Nota: el tiempo modelado es el mínimo a cualquier sede con programas; la sede de la plantilla es la más cercana en
   línea recta, que coincide con la de menor tiempo en la gran mayoría de casos (se marca cuando la distancia por la
   red podría cambiarla).

Salidas: salidas/tablas/colegios_no_conectados.csv, salidas/tablas/efecto_red_fluvial.csv,
         salidas/tablas/validacion_tiempos_resultados.csv (si hay datos observados)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import CFG, RAIZ, guardar, haversine_matriz, ruta

AMAZONIA = ["LORETO", "UCAYALI", "MADRE DE DIOS", "AMAZONAS", "SAN MARTIN"]
MACRO = {"COSTA": ["TUMBES", "PIURA", "LAMBAYEQUE", "LA LIBERTAD", "ICA", "LIMA", "CALLAO", "TACNA", "MOQUEGUA"],
         "SELVA": AMAZONIA}


def macrorregion(dep: str) -> str:
    for k, v in MACRO.items():
        if dep in v:
            return k
    return "SIERRA"


def main():
    dtype = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dtype)
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype=dtype)
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})[["ubigeo", "departamento"]]
    t = tie[tie.familia == "Total"].merge(base, on="ubigeo", how="left")

    # 1. no conectados
    nc = t[~np.isfinite(t.t_min)]
    r = nc.groupby("departamento").agg(colegios=("COD_MOD", "size"), quinto=("quinto", "sum")).sort_values("quinto", ascending=False)
    guardar(r.reset_index(), "colegios_no_conectados.csv", "tablas")
    print(f"no conectados: {len(nc)} colegios, {int(nc.quinto.sum()):,} alumnos; "
          f"en la Amazonía: {nc.departamento.isin(AMAZONIA).sum()} colegios, {int(nc[nc.departamento.isin(AMAZONIA)].quinto.sum()):,} alumnos")
    print(r.head(8).to_string())

    # 2. efecto de la red fluvial y de las velocidades
    s = pd.read_csv(ruta("processed") / "sensibilidad_colegios.csv.gz", dtype={"COD_MOD": str, "ANEXO": str})
    w = s.pivot_table(index=["COD_MOD", "ANEXO", "familia", "quinto"], columns="escenario", values="desierto_90").reset_index()
    filas = []
    for esc in [c for c in ["sin_rios", "rio_10.6", "rio_18.0", "vias_x0.75", "vias_x1.25"] if c in w]:
        ch = w[esc] != w["base"]
        for fam, g in w.assign(ch=ch).groupby("familia"):
            filas.append({"escenario": esc, "familia": fam, "colegios_que_cambian": int(g.ch.sum()),
                          "alumnos_que_cambian": int((g.quinto * g.ch).sum())})
    e = pd.DataFrame(filas)
    guardar(e, "efecto_red_fluvial.csv", "tablas")
    print("\ncambios de clasificación respecto de la base (cualquier carrera):")
    print(e[e.familia == "Total"].to_string(index=False))

    # 3. plantilla de validación
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"]).reset_index(drop=True)
    c = col.merge(t[["COD_MOD", "ANEXO", "t_min", "departamento"]], on=["COD_MOD", "ANEXO"], how="inner")
    c = c[np.isfinite(c.t_min) & (c.coord_valida == 1) & (c.quinto > 0)].copy()
    c["macro"] = c.departamento.map(macrorregion)
    rng = np.random.default_rng(CFG["modelos"]["semilla"])
    # estratos: macrorregión × tramo de tiempo (<30, 30-90, >90 min), para cubrir viajes cortos y largos
    c["tramo"] = pd.cut(c.t_min, [0, 30, 90, np.inf], labels=["<30", "30-90", ">90"])
    m = c.sample(frac=1, random_state=int(rng.integers(1e6))).groupby(["macro", "tramo"], observed=True).head(5)
    m = m.sample(min(len(m), 40), random_state=CFG["modelos"]["semilla"])
    dist = haversine_matriz(m.lat, m.lon, loc.lat, loc.lon)
    j = dist.argmin(axis=1)
    orden = np.sort(dist, axis=1)
    plant = pd.DataFrame({"COD_MOD": m.COD_MOD.values, "ANEXO": m.ANEXO.values, "colegio": m.get("CEN_EDU", pd.Series("", index=m.index)).values,
                          "departamento": m.departamento.values, "macrorregion": m.macro.values,
                          "lat_colegio": m.lat.values, "lon_colegio": m.lon.values,
                          "universidad_destino": loc.universidad.values[j], "lat_sede": loc.lat.values[j], "lon_sede": loc.lon.values[j],
                          "dist_linea_recta_km": dist.min(axis=1).round(1),
                          "segunda_sede_a_menos_de_10km_adicionales": (orden[:, 1] - orden[:, 0] < 10),
                          "tiempo_modelado_min": m.t_min.round(0).values,
                          "tiempo_observado_min": "", "medio_transporte": "", "fuente": ""})
    p = RAIZ / "datos/manual/validacion_tiempos_plantilla.csv"
    if (RAIZ / "datos/manual/validacion_tiempos.csv").exists():
        print("ya hay tiempos observados: no se regenera la plantilla (la muestra de 40 pares queda fija)")
    else:
        plant.to_csv(p, index=False, encoding="utf-8")
    print(f"\nplantilla de validación: {p.relative_to(RAIZ)} ({len(plant)} pares)")

    obs = RAIZ / "datos/manual/validacion_tiempos.csv"
    if obs.exists():
        v = pd.read_csv(obs)
        v = v[pd.to_numeric(v.tiempo_observado_min, errors="coerce").notna()].copy()
        v["obs"] = v.tiempo_observado_min.astype(float)
        if "tiempo_modelado_par_min" in v:   # plantilla del paso 31: tiempo modelado al mismo destino
            v["tiempo_modelado_min"] = v.tiempo_modelado_par_min
        v = v[np.isfinite(v.tiempo_modelado_min.astype(float))].copy()
        v["err"] = v.tiempo_modelado_min - v.obs
        res = {"pares": len(v), "error_medio_min": v.err.mean(), "error_absoluto_medio_min": v.err.abs().mean(),
               "error_relativo_mediano_pct": 100 * (v.err / v.obs).median(),
               "pearson": v[["tiempo_modelado_min", "obs"]].corr().iloc[0, 1],
               "spearman": v[["tiempo_modelado_min", "obs"]].corr("spearman").iloc[0, 1],
               "pct_dentro_30pct": 100 * (abs(v.err / v.obs) <= 0.3).mean(),
               "acuerdo_umbral_90min_pct": 100 * ((v.tiempo_modelado_min > 90) == (v.obs > 90)).mean()}
        guardar(pd.DataFrame([res]).round(2), "validacion_tiempos_resultados.csv", "tablas")
        print(pd.Series(res).round(2).to_string())
    else:
        print("aún no hay datos/manual/validacion_tiempos.csv: complete la plantilla y vuelva a correr este paso")


if __name__ == "__main__":
    main()
