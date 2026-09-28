"""Paso 2 – Demanda potencial: estudiantes de 5.º de secundaria por colegio y por distrito, con pobreza distrital.

Salidas (datos/processed):
  demanda_colegios.csv    colegios de secundaria (F0) activos con coordenadas y alumnos de 5.º grado
  distritos_base.csv      distritos con 5.º grado, colegios, % rural de la matrícula, pobreza 2018 y población 2020
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import guardar, ruta


def main():
    pad = pd.read_csv(ruta("escale_padron"), sep="|", dtype=str)
    mat = pd.read_csv(ruta("escale_matricula"), sep="|", dtype={"COD_MOD": str, "ANEXO": str, "CODGEO": str})
    mat["quinto"] = mat.G5_H + mat.G5_M
    mat["cuarto"] = mat.G4_H + mat.G4_M
    col = pad.merge(mat[["COD_MOD", "ANEXO", "quinto", "cuarto"]], on=["COD_MOD", "ANEXO"], how="left", validate="1:1")
    col["lat"] = pd.to_numeric(col.NLAT_IE, errors="coerce")
    col["lon"] = pd.to_numeric(col.NLONG_IE, errors="coerce")
    col["rural"] = (col.DAREACENSO.str.upper().str.startswith("RURAL")).astype(int)
    col["publica"] = col.D_GESTION.str.contains("Pública", na=False).astype(int)
    col["ubigeo"] = col.CODGEO.str.zfill(6)
    col["prov"] = col.ubigeo.str[:4]
    col["dpto"] = col.ubigeo.str[:2]
    fuera = col.lat.isna() | (col.lat == 0) | ~col.lat.between(-18.6, 0.2) | ~col.lon.between(-81.5, -68.5)
    print(f"colegios sin coordenadas válidas: {fuera.sum()} (se asignan al centroide de su distrito en el paso 4)")
    col["coord_valida"] = (~fuera).astype(int)
    # distrito de 2017 (geografía del censo y del mapa de pobreza): se usa el ubigeo oficial del colegio si existía en 2017;
    # solo los colegios de distritos creados después de 2017 se asignan por ubicación (unión espacial con los límites INEI).
    import geopandas as gpd
    lim = gpd.read_file(ruta("limites_distrital"))[["UBIGEO", "geometry"]]
    validos = set(lim.UBIGEO)
    col["ubigeo_censal"] = col.ubigeo.where(col.ubigeo.isin(validos))
    nuevos = col.ubigeo_censal.isna() & (col.coord_valida == 1)
    pts = gpd.GeoDataFrame(col.loc[nuevos, ["COD_MOD"]], geometry=gpd.points_from_xy(col.loc[nuevos, "lon"], col.loc[nuevos, "lat"]), crs=4326)
    cerca = gpd.sjoin_nearest(pts.to_crs(32718), lim.to_crs(32718), how="left")
    col.loc[nuevos, "ubigeo_censal"] = cerca.groupby(level=0).UBIGEO.first()
    print(f"colegios de distritos creados después de 2017, reasignados por ubicación: {nuevos.sum()} (en {col.loc[nuevos, 'ubigeo'].nunique()} distritos nuevos)")
    col["ubigeo_2025"] = col.ubigeo
    col["ubigeo"] = col.ubigeo_censal
    col["prov"] = col.ubigeo.str[:4]
    col["dpto"] = col.ubigeo.str[:2]
    guardar(col, "demanda_colegios.csv")

    # agregado distrital
    col["quinto_rural"] = col.quinto * col.rural
    d = col.groupby("ubigeo").agg(dpto=("dpto", "first"), prov=("prov", "first"), departamento=("D_DPTO", "first"),
                                   provincia=("D_PROV", "first"), distrito=("D_DIST", "first"),
                                   colegios=("COD_MOD", "size"), quinto=("quinto", "sum"),
                                   quinto_rural=("quinto_rural", "sum"), alumnos_sec=("TALUM", lambda s: pd.to_numeric(s).sum()),
                                   lat_media=("lat", "mean"), lon_media=("lon", "mean")).reset_index()
    d["pct_rural_quinto"] = np.where(d.quinto > 0, d.quinto_rural / d.quinto, np.nan)

    pob = pd.read_csv(ruta("pobreza"), dtype={"ubigeo": str, "sufijo": str})
    pob["ubigeo"] = pob.ubigeo.str.zfill(6)
    d = d.merge(pob[["ubigeo", "poblacion_2020", "pobreza_punto_medio", "pobreza_ic_inf", "pobreza_ic_sup", "ubicacion_pobreza"]],
                on="ubigeo", how="outer", indicator=True)
    print("cruce ESCALE × pobreza:", d._merge.value_counts().to_dict())
    d = d.drop(columns="_merge")
    d["dpto"] = d.ubigeo.str[:2]
    d["prov"] = d.ubigeo.str[:4]
    d[["colegios", "quinto", "quinto_rural"]] = d[["colegios", "quinto", "quinto_rural"]].fillna(0)
    guardar(d, "distritos_base.csv")
    print(f"5.º grado nacional: {int(col.quinto.sum()):,} en {len(col):,} colegios y {int((d.colegios > 0).sum()):,} distritos")


if __name__ == "__main__":
    main()
