"""Paso 4 – Accesibilidad geográfica a la oferta universitaria de pregrado.

Para cada colegio (punto de demanda, ponderado por alumnos de 5.º grado), cada escenario y cada familia de carreras:
  dist_min_km        distancia en línea recta al local más cercano con al menos un programa de la familia
  n_prog_Rkm         programas de la familia a menos de R km (R en config.yaml)
  acc_2sfca          índice de accesibilidad 2SFCA (programas por 1 000 alumnos de 5.º, con decaimiento gaussiano)
  desierto           1 si no hay ningún programa de la familia a menos de radio_desierto_km

Escenarios:
  post  universidades licenciadas (SUNEDU, corte 2024-08-28); se calcula por familia y para el total.
  pre   post + sedes y filiales de las universidades con licencia denegada según carnés 2018 (paso 1b), por familia.

Salidas (datos/processed):
  accesibilidad_colegios.csv   formato largo: colegio × escenario × familia
  accesibilidad_distritos.csv  agregado por distrito (promedios ponderados por 5.º grado)
  accesibilidad_provincias.csv agregado por provincia
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import CFG, FAMILIAS, guardar, haversine_matriz, peso_decaimiento, ruta

A = CFG["accesibilidad"]


def dos_sfca(dist: np.ndarray, oferta: np.ndarray, demanda: np.ndarray, d0: float, tipo: str) -> np.ndarray:
    """2SFCA: paso 1, razón oferta/demanda en cada local; paso 2, suma de razones alcanzables desde cada colegio."""
    w = peso_decaimiento(dist, d0, tipo)                       # colegios × locales
    demanda_pond = (w * demanda[:, None]).sum(axis=0)          # por local
    razon = np.divide(oferta, demanda_pond, out=np.zeros_like(oferta, float), where=demanda_pond > 0)
    return (w * razon[None, :]).sum(axis=1) * 1000             # programas por 1 000 alumnos de 5.º


def calcular(col: pd.DataFrame, loc: pd.DataFrame, escenario: str, familias: list[str], factor: float) -> pd.DataFrame:
    dist = haversine_matriz(col.lat, col.lon, loc.lat, loc.lon) * factor
    demanda = col.quinto.to_numpy(float)
    filas = []
    for fam in familias:
        oferta = (loc.n_programas if fam == "Total" else loc[fam]).fillna(0).to_numpy(float)
        tiene = oferta > 0
        d_fam = np.where(tiene[None, :], dist, np.inf)
        res = pd.DataFrame({"COD_MOD": col.COD_MOD.values, "ANEXO": col.ANEXO.values, "ubigeo": col.ubigeo.values,
                            "quinto": demanda, "escenario": escenario, "familia": fam, "factor_sinuosidad": factor,
                            "dist_min_km": d_fam.min(axis=1)})
        for r in A["radios_km"]:
            res[f"n_prog_{r}km"] = ((dist <= r) * oferta[None, :]).sum(axis=1)
        res["acc_2sfca"] = dos_sfca(dist, oferta, demanda, A["d0_km"], A["tipo_decaimiento"])
        res["desierto"] = (res.dist_min_km > A["radio_desierto_km"]).astype(int)
        filas.append(res)
    return pd.concat(filas, ignore_index=True)


def agregar(df: pd.DataFrame, clave: str) -> pd.DataFrame:
    df = df.copy()
    for c in ["dist_min_km", "acc_2sfca", "desierto", *[f"n_prog_{r}km" for r in A["radios_km"]]]:
        df[c + "_x"] = df[c].replace(np.inf, np.nan) * df.quinto
    g = df.groupby([clave, "escenario", "familia", "factor_sinuosidad"])
    out = g[[c for c in df.columns if c.endswith("_x")] + ["quinto"]].sum()
    for c in [c for c in out.columns if c.endswith("_x")]:
        out[c[:-2]] = out[c] / out.quinto.replace(0, np.nan)
    out = out.drop(columns=[c for c in out.columns if c.endswith("_x")]).reset_index()
    return out.rename(columns={"desierto": "pct_quinto_en_desierto"})


def main():
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    # colegios sin coordenadas válidas: centroide de los colegios válidos de su distrito
    cen = col[col.coord_valida == 1].groupby("ubigeo")[["lat", "lon"]].mean()
    m = col.coord_valida == 0
    col.loc[m, ["lat", "lon"]] = cen.reindex(col.loc[m, "ubigeo"]).to_numpy()
    col = col.dropna(subset=["lat", "lon"])
    col["quinto"] = col.quinto.fillna(0)
    post = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    pre = pd.read_csv(ruta("processed") / "oferta_locales_pre.csv").dropna(subset=["lat", "lon"])

    res = []
    for factor in A["factor_sinuosidad"]:
        res.append(calcular(col, post, "post", ["Total", *FAMILIAS], factor))
        res.append(calcular(col, pre, "pre", ["Total", *FAMILIAS], factor))
    res = pd.concat(res, ignore_index=True)
    guardar(res, "accesibilidad_colegios.csv")
    guardar(agregar(res, "ubigeo"), "accesibilidad_distritos.csv")
    res["prov"] = res.ubigeo.str[:4]
    guardar(agregar(res, "prov"), "accesibilidad_provincias.csv")

    base = res[res.factor_sinuosidad == 1.0]
    tab = (base.assign(q_des=base.quinto * base.desierto).groupby(["escenario", "familia"])
           .agg(quinto=("quinto", "sum"), quinto_en_desierto=("q_des", "sum")))
    tab["pct"] = (100 * tab.quinto_en_desierto / tab.quinto).round(1)
    print(f"\nAlumnos de 5.º grado sin programa a menos de {A['radio_desierto_km']} km (línea recta):")
    print(tab.to_string())
    guardar(tab.reset_index(), "tabla_desiertos_resumen.csv", "tablas")


if __name__ == "__main__":
    main()
