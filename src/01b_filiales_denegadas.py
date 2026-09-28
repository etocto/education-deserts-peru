"""Paso 1b – Oferta de las universidades con licencia denegada antes del licenciamiento (sedes y filiales).

Fuente principal: SUNEDU, "Carnés universitarios 2018" (datos abiertos): universidad × filial × programa × carnés emitidos.
Es el único registro público, anterior a las denegatorias (2018-2020), que identifica las filiales de cada universidad
y los programas que ofrecían en cada una. Las resoluciones e informes técnicos de licenciamiento (ITL) de SUNEDU
sirven para verificar la lista (ver literatura/resoluciones_denegadas.csv).

Pasos
  1. Filtra las universidades con licencia denegada (archivo de licenciamiento institucional, corte 2024-08-28).
  2. Extrae la ciudad de cada filial del nombre (p. ej., "UAP-SEDE JULIACA" -> JULIACA) y la ubica en su distrito
     (datos/manual/filiales_ciudades.csv, verificado contra los límites INEI). "SEDE PRINCIPAL", "ÚNICA", etc. -> sede
     principal del archivo de licenciamiento.
  3. Coordenadas: mediana de los locales licenciados del mismo distrito (centro urbano); si no hay, mediana de los colegios.
  4. Clasifica los programas en las 9 familias (misma regla que el paso 1) y cuenta programas de pregrado por filial.
     Se excluyen maestrías, doctorados y segundas especialidades por su denominación.

Salida: datos/processed/oferta_locales_denegadas_2018.csv (una fila por filial) y
        datos/interim/programas_denegadas_2018.csv (detalle por programa).
Nota: los carnés 2018 son un indicador de actividad, no la matrícula total; se usan para ubicar la oferta, no su tamaño.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils import FAMILIAS, RAIZ, familia_programa, guardar, ruta, sin_tildes

CARNES = RAIZ / "datos/raw/sunedu/sunedu_carnes_universitarios_2018.csv"
POSGRADO = r"MAESTR|DOCTOR|SEGUNDA ESPECIALIDAD|ESPECIALIDAD EN|DIPLOMADO"
SEDE = r"SEDE PRINCIPAL|SEDE CENTRAL|PRINCIPAL|UNICA|NO FIAL|LOCAL CENTRAL|NO ESPECIFICADO|UNIVERSIDAD PRIVADA LEONARDO|^USBA-LIMA$|SEDE UNICA"


def ciudad(filial: str) -> str | None:
    f = sin_tildes(filial)
    if re.search(SEDE, f):
        return None
    f = re.sub(r"^[A-Z]+-", "", f)                    # quita la sigla
    f = re.sub(r"^(SEDE|FILIAL)\s+", "", f)
    f = f.split(" - ")[-1]                            # "LA LIBERTAD - TRUJILLO" / "SISE - ATE"
    return f.strip()


def main():
    c = pd.read_csv(CARNES, sep="|", encoding="latin1", dtype=str)
    c["carnes"] = pd.to_numeric(c.Cant_Carnes, errors="coerce").fillna(0)
    lic = pd.read_csv(ruta("sunedu_licenciamiento"), sep="|", dtype=str)
    den = lic[lic.ESTADO_LICENCIAMIENTO == "LICENCIA DENEGADA"].set_index("CODIGO_ENTIDAD")
    c = c[c.CODIGO.isin(den.index)].copy()
    c = c[~c.NOMBRE_PROGRAMA.fillna("").map(sin_tildes).str.contains(POSGRADO)]
    c["ciudad"] = c.NOMBRE_FILIAL.fillna("[NO ESPECIFICADO]").map(ciudad)

    mapa = pd.read_csv(RAIZ / "datos/manual/filiales_ciudades.csv")
    import geopandas as gpd
    lim = gpd.read_file(ruta("limites_distrital"))
    lim["k"] = lim.NOMBDEP.map(sin_tildes) + "|" + lim.NOMBPROV.map(sin_tildes) + "|" + lim.NOMBDIST.map(sin_tildes)
    mapa["k"] = mapa.departamento.map(sin_tildes) + "|" + mapa.provincia.map(sin_tildes) + "|" + mapa.distrito.map(sin_tildes)
    mapa = mapa.merge(lim[["k", "UBIGEO"]], on="k", how="left")
    ub_ciudad = dict(zip(mapa.ciudad.map(sin_tildes), mapa.UBIGEO))
    faltan = sorted(set(c.ciudad.dropna()) - set(ub_ciudad))
    if faltan:
        raise ValueError(f"Ciudades sin ubicar (agregar a datos/manual/filiales_ciudades.csv): {faltan}")
    # sede principal: coordenadas del archivo de licenciamiento -> distrito INEI por unión espacial
    # (el campo UBIGEO de ese archivo no usa la codificación del INEI, por eso no se usa)
    sed = den[["LATITUD", "LONGITUD"]].astype(float)
    ps = gpd.GeoDataFrame(sed, geometry=gpd.points_from_xy(sed.LONGITUD, sed.LATITUD), crs=4326)
    sed["ub_inei"] = gpd.sjoin(ps, lim[["UBIGEO", "geometry"]], how="left", predicate="within").groupby(level=0).UBIGEO.first()
    c["ubigeo"] = np.where(c.ciudad.isna(), c.CODIGO.map(sed.ub_inei), c.ciudad.map(ub_ciudad))
    c["id_local"] = c.CODIGO + "-" + c.ciudad.fillna("SEDE")

    fam = c.apply(lambda r: familia_programa(r.NOMBRE_CLASE_PROGRAMA or "", r.NOMBRE_PROGRAMA or ""), axis=1)
    c["familia"] = [f for f, _ in fam]
    guardar(c, "programas_denegadas_2018.csv", "interim")

    # coordenadas del centro urbano
    ofer = pd.read_csv(ruta("processed") / "oferta_programas_pregrado.csv", dtype=str)
    ofer["ub"] = None
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"ubigeo": str})
    loc_lic = pd.read_csv(ruta("processed") / "oferta_locales.csv")
    pts = gpd.GeoDataFrame(loc_lic, geometry=gpd.points_from_xy(loc_lic.lon, loc_lic.lat), crs=4326)
    loc_lic["ubigeo"] = gpd.sjoin(pts, lim[["UBIGEO", "geometry"]], how="left", predicate="within").groupby(level=0).UBIGEO.first()
    cen_lic = loc_lic.groupby("ubigeo")[["lat", "lon"]].median()
    cen_col = col.groupby("ubigeo")[["lat", "lon"]].median()

    base = c.groupby("id_local").agg(codigo_entidad=("CODIGO", "first"), universidad=("NOMBRE_UNIVERSIDAD", "first"),
                                    filial=("NOMBRE_FILIAL", "first"), ubigeo=("ubigeo", "first"),
                                    carnes_2018=("carnes", "sum"), n_programas=("NOMBRE_PROGRAMA", "nunique"))
    por_fam = c.groupby(["id_local", "familia"]).NOMBRE_PROGRAMA.nunique().unstack(fill_value=0)
    for f in FAMILIAS:
        if f not in por_fam:
            por_fam[f] = 0
    out = base.join(por_fam[FAMILIAS]).reset_index()
    out["lat"] = out.ubigeo.map(cen_lic.lat).fillna(out.ubigeo.map(cen_col.lat))
    out["lon"] = out.ubigeo.map(cen_lic.lon).fillna(out.ubigeo.map(cen_col.lon))
    out["origen_coord"] = np.where(out.ubigeo.isin(cen_lic.index), "locales licenciados del distrito", "colegios del distrito")
    es_sede = out.id_local.str.endswith("-SEDE")
    out.loc[es_sede, "lat"] = out.loc[es_sede, "codigo_entidad"].map(sed.LATITUD).values
    out.loc[es_sede, "lon"] = out.loc[es_sede, "codigo_entidad"].map(sed.LONGITUD).values
    out.loc[es_sede, "origen_coord"] = "coordenadas de la sede (SUNEDU)"
    out["gestion"] = "PRIVADO"
    out["escenario"] = "denegada"
    out["departamento"] = out.ubigeo.str[:2]
    guardar(out, "oferta_locales_denegadas_2018.csv")
    # escenario "pre" = locales licenciados (2024) + sedes y filiales de las universidades denegadas (2018)
    cols = ["id_local", "codigo_entidad", "universidad", "gestion", "lat", "lon", "n_programas", *FAMILIAS, "escenario"]
    pre = pd.concat([loc_lic.assign(escenario="post")[cols], out[cols]], ignore_index=True)
    guardar(pre, "oferta_locales_pre.csv")
    print(f"universidades denegadas con registro 2018: {out.codigo_entidad.nunique()} de {len(den)}")
    print(f"filiales/sedes ubicadas: {len(out)} en {out.ubigeo.nunique()} distritos; programas de pregrado: {int(out.n_programas.sum())}")
    print(f"sin registro en carnés 2018: {sorted(set(den.index) - set(out.codigo_entidad))}")


if __name__ == "__main__":
    main()
