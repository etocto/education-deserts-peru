"""Paso 35 – Sensibilidad de los desiertos asociados al licenciamiento a la ubicación de las filiales denegadas
(revisión v3, obs. M5).

El paso 1b ubica cada filial de 2018 de una universidad denegada en la mediana de los locales licenciados de su
distrito o, si no hay ninguno, en la mediana de sus colegios. Los desiertos asociados al licenciamiento aparecen
justo donde no hay oferta licenciada, así que se comprueba si cambian con otras ubicaciones:

  A. base            ubicación del paso 1b;
  B. localidad OSM   el lugar poblado de OpenStreetMap (place = city, town, village o hamlet) cuyo nombre coincide
                     con la ciudad de la filial y que está dentro de su distrito (si no hay, se mantiene la base);
  C. punto del distrito  punto representativo del polígono del distrito (INEI);
  D. ±5 km           la filial desplazada 5 km en ocho direcciones: «robusto» = asociado al licenciamiento en la
                     base y en los ocho desplazamientos; «máximo» = en alguno de ellos.
Las sedes principales mantienen las coordenadas de SUNEDU en todas las alternativas.

Para cada alternativa y familia: alumnos en desiertos asociados al licenciamiento por distancia (más de 60 km de la
oferta 2024, a 60 km o menos de la oferta reconstruida y a más de 90 min de la oferta 2024), con la oferta 2024 y
sumando las seis relicenciadas, y los de Puno.

Requiere los pasos 1b, 2, 4, 4b y el .pbf (para B). Unos minutos.
Salidas: salidas/tablas/filiales_sensibilidad.csv, filiales_ubicacion_alternativa.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import CFG, FAMILIAS, RAIZ, guardar, haversine_matriz, ruta, sin_tildes

RELICENCIADAS = ["053", "048", "050", "078", "090", "117"]
R_KM = 6371.0088


def lugares_osm(path):
    import osmium

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.f = []

        def node(self, n):
            pl = n.tags.get("place")
            if pl in ("city", "town", "village", "hamlet") and "name" in n.tags:
                self.f.append((n.tags["name"], pl, n.location.lat, n.location.lon))

    h = H()
    h.apply_file(str(path))
    return pd.DataFrame(h.f, columns=["nombre", "tipo", "lat", "lon"])


def desplazar(lat, lon, km, ang):
    dlat = km / R_KM * np.cos(ang)
    dlon = km / R_KM * np.sin(ang) / np.cos(np.radians(lat))
    return lat + np.degrees(dlat), lon + np.degrees(dlon)


def lic_distancia(col, post, den, fam, T, radio):
    """Máscaras de licenciamiento por distancia (2024 y sumando relicenciadas)."""
    of = lambda x: ((x.n_programas if fam == "Total" else x[fam]).fillna(0) > 0).to_numpy()  # noqa: E731
    p, dd = post[of(post)], den[of(den)]
    rel = dd[dd.codigo_entidad.isin(RELICENCIADAS)]
    near = lambda x: (haversine_matriz(col.lat, col.lon, x.lat, x.lon) <= radio).any(axis=1) if len(x) else np.zeros(len(col), bool)  # noqa: E731
    Dp = ~near(p)
    Dpre = ~(near(p) | near(dd))
    Drel = ~(near(p) | near(rel))
    return Dp & ~Dpre & T, Drel & ~Dpre & T


def main():
    radio = CFG["accesibilidad"]["radio_desierto_km"]
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype=dt)
    col["quinto"] = col.quinto.fillna(0)
    post = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    den = pd.read_csv(ruta("processed") / "oferta_locales_denegadas_2018.csv", dtype={"codigo_entidad": str, "ubigeo": str}).dropna(subset=["lat", "lon"]).reset_index(drop=True)
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt, usecols=["COD_MOD", "ANEXO", "familia", "desierto_t"])
    es_fil = ~den.id_local.str.endswith("-SEDE")
    den["ciudad"] = den.id_local.str.split("-", n=1).str[1]

    # ---- B: lugar poblado de OSM con el nombre de la ciudad, dentro del distrito
    import geopandas as gpd
    lim = gpd.read_file(ruta("limites_distrital"))[["UBIGEO", "geometry"]]
    pbf = RAIZ / CFG["tiempos"]["archivo_pbf"]
    alt_b = den[["lat", "lon"]].copy()
    metodo_b = pd.Series("base (sin coincidencia en OSM)", index=den.index)
    if pbf.exists():
        lug = lugares_osm(pbf)
        lug["k"] = lug.nombre.map(sin_tildes).str.upper().str.strip()
        g = gpd.GeoDataFrame(lug, geometry=gpd.points_from_xy(lug.lon, lug.lat), crs=4326)
        lug["ubigeo"] = gpd.sjoin(g, lim, how="left", predicate="within").groupby(level=0).UBIGEO.first()
        rango = {"city": 0, "town": 1, "village": 2, "hamlet": 3}
        lug["r"] = lug.tipo.map(rango)
        for i in den.index[es_fil]:
            k = sin_tildes(str(den.at[i, "ciudad"])).upper().strip()
            c = lug[(lug.k == k) & (lug.ubigeo == den.at[i, "ubigeo"])].sort_values("r")
            if len(c):
                alt_b.loc[i] = c.iloc[0][["lat", "lon"]].to_numpy(float)
                metodo_b[i] = f"OSM place={c.iloc[0].tipo}"
    else:
        print("AVISO: no se encontró el .pbf; la alternativa B queda igual a la base")
    # ---- C: punto representativo del distrito
    rep = lim.set_index("UBIGEO").to_crs(32718).geometry.representative_point().to_crs(4326)
    alt_c = den[["lat", "lon"]].copy()
    ok = es_fil & den.ubigeo.isin(rep.index)
    alt_c.loc[ok, "lat"] = den.loc[ok, "ubigeo"].map(rep.y).values
    alt_c.loc[ok, "lon"] = den.loc[ok, "ubigeo"].map(rep.x).values

    ub = den[["id_local", "universidad", "ubigeo", "origen_coord", "lat", "lon"]].copy()
    ub["metodo_osm"] = np.where(es_fil, metodo_b, "sede (SUNEDU)")
    ub["km_a_osm"] = np.diag(haversine_matriz(den.lat, den.lon, alt_b.lat, alt_b.lon)).round(2)
    ub["km_a_punto_distrito"] = np.diag(haversine_matriz(den.lat, den.lon, alt_c.lat, alt_c.lon)).round(2)
    guardar(ub, "filiales_ubicacion_alternativa.csv", "tablas")

    alternativas = {"A. base": den[["lat", "lon"]], "B. localidad OSM": alt_b, "C. punto del distrito": alt_c}
    for k, ang in enumerate(np.arange(8) * np.pi / 4):
        a = den[["lat", "lon"]].copy()
        la, lo = desplazar(den.lat.to_numpy(), den.lon.to_numpy(), 5.0, ang)
        a.loc[es_fil, "lat"], a.loc[es_fil, "lon"] = la[es_fil], lo[es_fil]
        alternativas[f"D{k}"] = a

    puno = (col.ubigeo.str[:2] == "21").to_numpy()
    q = col.quinto.to_numpy()
    filas = []
    for fam in ["Total", *FAMILIAS]:
        T = col[["COD_MOD", "ANEXO"]].merge(tie[tie.familia == fam], on=["COD_MOD", "ANEXO"], how="left").desierto_t.fillna(1).to_numpy() == 1
        res = {}
        for nombre, coords in alternativas.items():
            dd = den.copy()
            dd[["lat", "lon"]] = coords.to_numpy()
            res[nombre] = lic_distancia(col, post, dd, fam, T, radio)
        des = [k for k in res if k.startswith("D")]
        res["D. ±5 km, robusto"] = tuple(np.logical_and.reduce([res["A. base"][i], *[res[k][i] for k in des]]) for i in (0, 1))
        res["D. ±5 km, máximo"] = tuple(np.logical_or.reduce([res["A. base"][i], *[res[k][i] for k in des]]) for i in (0, 1))
        for nombre in ["A. base", "B. localidad OSM", "C. punto del distrito", "D. ±5 km, robusto", "D. ±5 km, máximo"]:
            m24, mrel = res[nombre]
            filas.append({"familia": fam, "alternativa": nombre, "lic_2024": int(q[m24].sum()), "lic_con_relicenciadas": int(q[mrel].sum()),
                          "lic_2024_puno": int(q[m24 & puno].sum()), "lic_relicenciadas_puno": int(q[mrel & puno].sum())})
    t = pd.DataFrame(filas)
    guardar(t, "filiales_sensibilidad.csv", "tablas")
    print(ub.metodo_osm.value_counts().to_string())
    print(t[t.familia.isin(["Total", "Educación", "Servicios"])].to_string(index=False))


if __name__ == "__main__":
    main()
