"""Paso 33 – Verificación manual de la muestra de 20 colegios del paso 22 (revisión, obs. 13).

Para cada colegio de datos/manual/muestra_verificacion_manual.csv se comprueba, con fuentes y cálculos
independientes del flujo principal:

  1. Coordenadas: que el colegio caiga dentro del polígono de su distrito (límites del INEI) y que el local
     universitario caiga dentro del distrito que figura en SUNEDU.
  2. Distancia: distancia geodésica en el elipsoide WGS84 (pyproj, independiente de la fórmula de haversine del
     flujo principal) al local indicado, y búsqueda exhaustiva del local licenciado más cercano y del local más
     cercano de las universidades denegadas (oferta reconstruida de 2018).
  3. Tiempo: tiempo observado en Google Maps (en auto, salida lunes 12/10/2026 7:00) del colegio al local indicado,
     anotado a mano en datos/manual/verificacion_manual_google.csv, frente al tiempo modelado al MISMO local en la
     red calibrada (opción --modelo, que reconstruye la red; unos 3 GB de RAM).
  4. Categoría: se vuelve a derivar el origen (estructural, licenciamiento, topográfico, ninguno) con las
     distancias del punto 2 y el umbral de 90 minutos, y se compara con la del flujo principal.

Salidas: datos/manual/muestra_verificacion_manual.csv (columnas verificado_* llenas)
         salidas/tablas/verificacion_manual.csv (detalle) y verificacion_manual_resumen.csv
Uso:     python src/33_verificacion_manual.py --modelo   (una vez)   y luego   python src/33_verificacion_manual.py
"""
from __future__ import annotations

import importlib.util
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod

from utils import CFG, RAIZ, guardar, ruta

M = RAIZ / "datos/manual"
GEOD = Geod(ellps="WGS84")


def km_geod(la1, lo1, la2, lo2):
    return GEOD.inv(np.asarray(lo1, float), np.asarray(la1, float), np.asarray(lo2, float), np.asarray(la2, float))[2] / 1000


def tiempos_modelo(v: pd.DataFrame) -> None:
    spec = importlib.util.spec_from_file_location("m04b", RAIZ / "src" / "04b_tiempos_viaje.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    from scipy.sparse.csgraph import dijkstra
    from red_liviana import construir_red
    g, lat, lon = construir_red(m.PBF)
    nc, acc_c, km_c = m.anclar(lat, lon, v.lat, v.lon)
    ns, acc_s, _ = m.anclar(lat, lon, v.lat_local, v.lon_local)
    t = [float(dijkstra(g, indices=nc[i], limit=1500)[ns[i]] + acc_c[i] + acc_s[i]) for i in range(len(v))]
    pd.DataFrame({"COD_MOD": v.COD_MOD, "ANEXO": v.ANEXO, "tiempo_modelado_al_local_min": np.round(t, 1),
                  "km_colegio_a_red": np.round(km_c, 2)}).to_csv(M / "verificacion_manual_modelo.csv", index=False)
    print("-> datos/manual/verificacion_manual_modelo.csv")


def main():
    dt = {"COD_MOD": str, "ANEXO": str}
    v = pd.read_csv(M / "muestra_verificacion_manual.csv", dtype=dt)
    if "--modelo" in sys.argv:
        tiempos_modelo(v)
        return
    radio, umbral = CFG["accesibilidad"]["radio_desierto_km"], CFG["tiempos"]["umbral_desierto_min"]
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={**dt, "ubigeo": str})[["COD_MOD", "ANEXO", "ubigeo"]]
    v = v.merge(col, on=["COD_MOD", "ANEXO"], how="left")

    # 1. coordenadas dentro del distrito
    dis = gpd.read_file(ruta("limites_distrital"))
    cod = next(c for c in ["UBIGEO", "ubigeo", "IDDIST", "CODDIST"] if c in dis.columns)
    pts = gpd.GeoDataFrame(v[["ubigeo"]], geometry=gpd.points_from_xy(v.lon, v.lat), crs=4326).to_crs(dis.crs)
    j = gpd.sjoin(pts, dis[[cod, "geometry"]], how="left", predicate="within")
    v["ubigeo_por_coordenadas"] = j.groupby(level=0)[cod].first().astype(str).str.zfill(6).values
    v["coord_en_distrito"] = (v.ubigeo_por_coordenadas == v.ubigeo).astype(int)
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"]).reset_index(drop=True)
    den = pd.read_csv(ruta("processed") / "oferta_locales_denegadas_2018.csv", dtype={"codigo_entidad": str}).dropna(subset=["lat", "lon"])
    pl = gpd.GeoDataFrame(v[["local_mas_cercano"]], geometry=gpd.points_from_xy(v.lon_local, v.lat_local), crs=4326).to_crs(dis.crs)
    nd = next(c for c in ["NOMBDIST", "DISTRITO", "nombdist"] if c in dis.columns)
    jl = gpd.sjoin(pl, dis[[nd, "geometry"]], how="left", predicate="within").groupby(level=0)[nd].first()
    norm = lambda s: s.str.upper().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.strip()  # noqa: E731
    dist_local = v.local_mas_cercano.str.extract(r"\(([^)]*)\)$")[0]
    v["local_en_distrito"] = (norm(jl.reindex(v.index).fillna("")) == norm(dist_local)).astype(int)

    # 2. distancias geodésicas independientes
    v["distancia_geodesica_km"] = km_geod(v.lat, v.lon, v.lat_local, v.lon_local).round(2)
    dmin, dden, jmin = [], [], []
    for _, r in v.iterrows():
        d = km_geod(np.full(len(loc), r.lat), np.full(len(loc), r.lon), loc.lat, loc.lon)
        dmin.append(d.min()); jmin.append(d.argmin())
        dden.append(km_geod(np.full(len(den), r.lat), np.full(len(den), r.lon), den.lat, den.lon).min())
    v["distancia_minima_geodesica_km"] = np.round(dmin, 2)
    v["mismo_local_mas_cercano"] = [int(abs(loc.lat[k] - a) < 1e-6 and abs(loc.lon[k] - b) < 1e-6) for k, a, b in zip(jmin, v.lat_local, v.lon_local)]
    v["distancia_denegada_2018_km"] = np.round(dden, 2)
    v["dif_distancia_km"] = (v.distancia_geodesica_km - v.distancia_km).round(2)
    v["verificado_distancia"] = ((v.dif_distancia_km.abs() <= 0.5) & (v.mismo_local_mas_cercano == 1)
                                 & ((v.distancia_minima_geodesica_km > radio) == (v.desierto_60km == 1))).astype(int)

    # 3. tiempos: modelado al mismo local y observado en Google Maps
    mo = pd.read_csv(M / "verificacion_manual_modelo.csv", dtype=dt)
    go = pd.read_csv(M / "verificacion_manual_google.csv", dtype=dt)
    v = v.merge(mo, on=["COD_MOD", "ANEXO"], how="left").merge(go[["COD_MOD", "ANEXO", "tiempo_google_min", "km_google", "ruta_google"]], on=["COD_MOD", "ANEXO"], how="left")
    v["error_rel_pct"] = (100 * (v.tiempo_modelado_al_local_min - v.tiempo_google_min) / v.tiempo_google_min).round(1)
    v["google_mas_90"] = (v.tiempo_google_min > umbral).astype(int)
    # el tiempo modelado del flujo principal es el mínimo a cualquier local: si Google llega al local más cercano
    # en <= 90 min, el colegio no debería figurar como desierto por tiempo
    v["verificado_tiempo_plausible"] = ((v.error_rel_pct.abs() <= 30) & (v.google_mas_90 >= v.desierto_90min)).astype(int).where(v.tiempo_google_min.notna())
    # sinuosidad observada: km por carretera / km en línea recta
    v["sinuosidad_google"] = (v.km_google / v.distancia_geodesica_km).round(2)

    # 4. categoría derivada de nuevo
    D, Dp, T = v.distancia_minima_geodesica_km > radio, np.minimum(v.distancia_minima_geodesica_km, v.distancia_denegada_2018_km) > radio, v.tiempo_modelado_min > umbral
    v["origen_recalculado"] = np.select([D & ~Dp, D & Dp, ~D & T], ["licensing", "structural", "topographic"], default="none")
    v["origen_con_google"] = np.where(v.tiempo_google_min.isna() & ~D, "sin ruta en Google", np.select(
        [D & ~Dp, D & Dp, ~D & (v.tiempo_google_min > umbral)], ["licensing", "structural", "topographic"], default="none"))
    v["verificado_categoria"] = (v.origen_recalculado == v.origen).astype(int)
    v["nota"] = [("; ".join(x for x in [
        "" if r.coord_en_distrito else f"coordenadas en el distrito {r.ubigeo_por_coordenadas}, no en {r.ubigeo}",
        "" if r.local_en_distrito else "local fuera del distrito declarado",
        "Google no calcula ruta en auto" if pd.isna(r.tiempo_google_min) else "",
        "" if r.origen_con_google in (r.origen, "sin ruta en Google") else f"con el tiempo de Google sería {r.origen_con_google}",
        "" if pd.isna(r.error_rel_pct) or abs(r.error_rel_pct) <= 30 else f"error de tiempo {r.error_rel_pct:+.0f}%"] if x)) for r in v.itertuples()]

    cols = ["COD_MOD", "ANEXO", "colegio", "departamento", "distrito", "origen", "coord_en_distrito", "local_en_distrito", "distancia_km",
            "distancia_geodesica_km", "mismo_local_mas_cercano", "distancia_denegada_2018_km", "tiempo_modelado_min",
            "tiempo_modelado_al_local_min", "tiempo_google_min", "km_google", "sinuosidad_google", "error_rel_pct",
            "origen_recalculado", "origen_con_google", "verificado_distancia", "verificado_tiempo_plausible", "verificado_categoria", "nota"]
    guardar(v[cols], "verificacion_manual.csv", "tablas")
    base = pd.read_csv(M / "muestra_verificacion_manual.csv", dtype=dt)
    for c in ["verificado_distancia", "verificado_tiempo_plausible", "verificado_categoria", "nota"]:
        base[c] = v[c].values
    base.to_csv(M / "muestra_verificacion_manual.csv", index=False, encoding="utf-8")

    g = v[v.tiempo_google_min.notna()]
    e = g.tiempo_modelado_al_local_min - g.tiempo_google_min
    res = pd.DataFrame([{
        "colegios": len(v), "coordenadas_en_su_distrito": int(v.coord_en_distrito.sum()), "locales_en_su_distrito": int(v.local_en_distrito.sum()),
        "distancia_dif_max_km": float(v.dif_distancia_km.abs().max()), "mismo_local_mas_cercano": int(v.mismo_local_mas_cercano.sum()),
        "distancia_verificada": int(v.verificado_distancia.sum()),
        "error_absoluto_medio_min": float(e.abs().mean()), "error_relativo_mediano_pct": float(v.error_rel_pct.median()),
        "pares_con_ruta_google": len(g), "dentro_30pct": int((g.error_rel_pct.abs() <= 30).sum()),
        "pearson": float(np.corrcoef(g.tiempo_modelado_al_local_min, g.tiempo_google_min)[0, 1]),
        "acuerdo_umbral_90": int(((g.tiempo_modelado_al_local_min > umbral) == (g.tiempo_google_min > umbral)).sum()),
        "tiempo_plausible": int(v.verificado_tiempo_plausible.sum()), "categoria_reproducida": int(v.verificado_categoria.sum()),
        "categoria_igual_con_google": int((g.origen_con_google == g.origen).sum())}])
    guardar(res.round(2), "verificacion_manual_resumen.csv", "tablas")
    print(v[["colegio", "origen", "distancia_km", "distancia_geodesica_km", "tiempo_modelado_min", "tiempo_modelado_al_local_min",
             "tiempo_google_min", "error_rel_pct", "origen_recalculado", "origen_con_google", "nota"]].to_string(index=False))
    print(res.T.to_string())


if __name__ == "__main__":
    main()
