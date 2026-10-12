"""Paso 25 – Desiertos asociados al licenciamiento netos de las universidades relicenciadas (seguimiento, N2).

Seis universidades denegadas en la primera ronda obtuvieron una nueva licencia entre 2024 y 2026 (04g). Este paso
reporta los alumnos de 5.º en desiertos asociados al licenciamiento (definición por distancia, 60 km) en dos versiones:

  v2024   oferta licenciada 2024 (corte de datos abiertos)          -> cota superior del desierto de licenciamiento
  vrel    oferta 2024 + oferta 2018 de las seis relicenciadas       -> cota inferior (si conservaron sus sedes de 2018)

En ambas, "licenciamiento" = desierto con esa oferta y sin desierto con la oferta reconstruida de 2018 (pre).
Se reporta por departamento (alumnos, % de los alumnos de 5.º del departamento, perfil ponderado de los distritos:
pobreza, % rural censal, % autoidentificación indígena) y por familia (alumnos y departamento principal).

Requiere los pasos 1, 1b y 2. Salidas (salidas/tablas):
  licenciamiento_neto_departamentos.csv, licenciamiento_neto_familias.csv, licenciamiento_neto_resumen.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import CFG, FAMILIAS, guardar, haversine_matriz, ruta

RELICENCIADAS = ["053", "048", "050", "078", "090", "117"]


def desiertos(col: pd.DataFrame, loc: pd.DataFrame, radio: float) -> dict[str, np.ndarray]:
    cerca = haversine_matriz(col.lat, col.lon, loc.lat, loc.lon) <= radio
    out = {}
    for fam in ["Total", *FAMILIAS]:
        oferta = ((loc.n_programas if fam == "Total" else loc[fam]).fillna(0) > 0).to_numpy()
        out[fam] = ~(cerca[:, oferta]).any(axis=1)
    return out


def main():
    radio = CFG["accesibilidad"]["radio_desierto_km"]
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype=dt)
    col = col.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    col["quinto"] = col.quinto.fillna(0)
    post = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    den = pd.read_csv(ruta("processed") / "oferta_locales_denegadas_2018.csv", dtype={"codigo_entidad": str}).dropna(subset=["lat", "lon"])
    rel = den[den.codigo_entidad.isin(RELICENCIADAS)]
    esc = {"v2024": desiertos(col, post, radio),
           "vrel": desiertos(col, pd.concat([post, rel], ignore_index=True), radio),
           "pre": desiertos(col, pd.concat([post, den], ignore_index=True), radio)}

    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})[["ubigeo", "departamento", "pobreza_punto_medio"]]
    cen = pd.read_csv(ruta("censo2017_agregado"), dtype={"ubigeo": str})[["ubigeo", "pct_rural_censal", "pct_autoid_indigena"]]
    col = col.merge(base, on="ubigeo", how="left").merge(cen, on="ubigeo", how="left")
    for v in ["pobreza_punto_medio", "pct_rural_censal", "pct_autoid_indigena"]:
        col[v] = col[v].fillna(col[v].mean())
    col["pct_rural_censal"] *= 100
    col["pct_autoid_indigena"] *= 100
    q = col.quinto.to_numpy()
    quinto_dep = col.groupby("departamento").quinto.sum()

    def perfil(mask):
        g = col[mask]
        w = g.quinto + 1e-9
        return {"alumnos": int(g.quinto.sum()),
                "pobreza": np.average(g.pobreza_punto_medio, weights=w) if len(g) else np.nan,
                "rural": np.average(g.pct_rural_censal, weights=w) if len(g) else np.nan,
                "indigena": np.average(g.pct_autoid_indigena, weights=w) if len(g) else np.nan}

    # ---- resumen nacional y por familia
    filas_fam, filas_res = [], []
    for fam in ["Total", *FAMILIAS]:
        for v in ["v2024", "vrel"]:
            lic = esc[v][fam] & ~esc["pre"][fam]
            todos = esc[v][fam]
            g = col[lic]
            dep = g.groupby("departamento").quinto.sum().sort_values(ascending=False)
            filas_fam.append({"familia": fam, "version": v, "alumnos_licenciamiento": int(q[lic].sum()),
                              "alumnos_en_desierto": int(q[todos].sum()),
                              "pct_del_desierto": 100 * q[lic].sum() / max(q[todos].sum(), 1),
                              "pct_quinto_nacional": 100 * q[lic].sum() / q.sum(),
                              "departamento_principal": dep.index[0] if len(dep) else "",
                              "pct_en_departamento_principal": 100 * dep.iloc[0] / dep.sum() if len(dep) else np.nan,
                              "departamentos": int((dep > 0).sum())})
    t_fam = pd.DataFrame(filas_fam)
    guardar(t_fam.round(4), "licenciamiento_neto_familias.csv", "tablas")

    # ---- por departamento (cualquier carrera y por familia)
    filas_dep = []
    for fam in ["Total", *FAMILIAS]:
        for v in ["v2024", "vrel"]:
            lic = esc[v][fam] & ~esc["pre"][fam]
            for dep, g in col[lic].groupby("departamento"):
                p = perfil(lic & (col.departamento == dep).to_numpy())
                filas_dep.append({"familia": fam, "version": v, "departamento": dep, **p,
                                  "pct_quinto_departamento": 100 * p["alumnos"] / quinto_dep[dep]})
    t_dep = pd.DataFrame(filas_dep).sort_values(["familia", "version", "alumnos"], ascending=[True, True, False])
    guardar(t_dep.round(4), "licenciamiento_neto_departamentos.csv", "tablas")

    # ---- resumen comparativo (cualquier carrera): perfil nacional en cada versión
    for v in ["v2024", "vrel"]:
        lic = esc[v]["Total"] & ~esc["pre"]["Total"]
        filas_res.append({"version": v, **perfil(lic)})
    filas_res.append({"version": "todos_quinto", **perfil(np.ones(len(col), bool))})
    t_res = pd.DataFrame(filas_res)
    guardar(t_res.round(4), "licenciamiento_neto_resumen.csv", "tablas")

    print("\nCualquier carrera, por departamento (alumnos):")
    w = t_dep[t_dep.familia == "Total"].pivot_table(index="departamento", columns="version", values="alumnos", fill_value=0)
    print(w.sort_values("v2024", ascending=False).to_string())
    print("\nPor familia:")
    print(t_fam.pivot_table(index="familia", columns="version", values="alumnos_licenciamiento").to_string())
    print("\nPerfil (cualquier carrera):")
    print(t_res.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
