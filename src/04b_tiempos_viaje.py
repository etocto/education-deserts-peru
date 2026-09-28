"""Paso 4b – Tiempos de viaje por la red vial (y fluvial) de OpenStreetMap en lugar de la línea recta.

Qué hace
  1. Lee el extracto de OSM del Perú (Geofabrik, formato .osm.pbf; se descarga si no existe, ~250 MB).
  2. Construye un grafo con las vías (highway) y, opcionalmente, los ríos navegables (waterway=river), asignando
     a cada tramo una velocidad según su tipo (config.yaml -> tiempos.velocidades_kmh).
  3. Une cada colegio y cada local universitario al nodo más cercano de la red; el tramo de acceso se recorre
     a pie (velocidad_acceso_kmh).
  4. Calcula, para cada colegio y cada familia de carreras, el tiempo mínimo hasta un local con esa familia
     (Dijkstra con una "super-fuente" conectada a todos los locales de la familia).
  5. Calcula un 2SFCA con tiempos (umbral t0) y marca desierto si el tiempo mínimo supera umbral_desierto_min.

Ríos: en OSM los ríos no comparten nodos con las vías, así que sin más quedarían como redes aisladas.
  - Solo se usan los ríos de los departamentos amazónicos listados en tiempos.rios_departamentos
    (en la sierra los ríos no son navegables y crearían atajos irreales).
  - Cada nodo fluvial a menos de conexion_rio_via_km de una vía se une a ella ("embarcadero") con un tiempo de
    caminata más transbordo_min de espera/transbordo.

Salidas (datos/processed)
  tiempos_colegios.csv                 colegio × familia: t_min (minutos), acc_2sfca_t, desierto_t
  accesibilidad_tiempo_distritos.csv   promedios ponderados por 5.º grado
  salidas/tablas/tabla_desiertos_tiempo.csv

Requisitos: pip install osmium scipy scikit-learn
Tiempo estimado en una laptop: 10–30 min (la lectura del .pbf es lo más lento). Se ejecuta aparte:
  python src/04b_tiempos_viaje.py
Prueba rápida con un extracto pequeño en JSON (lista de [highway, nodos, [[lat, lon], ...]]):
  python src/04b_tiempos_viaje.py --prueba ruta/extracto.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import requests
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import BallTree

from utils import CFG, FAMILIAS, RAIZ, guardar, peso_decaimiento, ruta

T = CFG["tiempos"]
PBF = RAIZ / T["archivo_pbf"]
URL_PBF = "https://download.geofabrik.de/south-america/peru-latest.osm.pbf"
R_KM = 6371.0088


# ---------------------------------------------------------------------------- lectura de OSM
def leer_pbf(path) -> list[tuple[str, list[int], list[tuple[float, float]]]]:
    """Devuelve una lista de (tipo, ids de nodos, coordenadas lat/lon) de vías y ríos útiles."""
    import osmium

    vel = T["velocidades_kmh"]
    usar_rios = T.get("incluir_rios", True)

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.w = []

        def way(self, w):
            hw = w.tags.get("highway")
            tipo = hw if hw in vel else None
            if tipo is None and usar_rios and w.tags.get("waterway") == "river":
                tipo = "river"
            if tipo is None and w.tags.get("route") == "ferry":
                tipo = "ferry"
            if tipo is None:
                return
            try:
                nodos = [n.ref for n in w.nodes]
                coords = [(n.location.lat, n.location.lon) for n in w.nodes]
            except osmium.InvalidLocationError:
                return
            self.w.append((tipo, nodos, coords))

    h = H()
    h.apply_file(str(path), locations=True, idx="flex_mem")
    return filtrar_rios(h.w)


def filtrar_rios(vias):
    """Conserva solo los ríos cuyo punto medio cae en los departamentos de tiempos.rios_departamentos."""
    deps = [d.upper() for d in T.get("rios_departamentos", [])]
    rios = [k for k, v in enumerate(vias) if v[0] == "river"]
    if not rios or not deps:
        return vias
    import geopandas as gpd
    lim = gpd.read_file(ruta("limites_departamental"))
    lim = lim[lim.NOMBDEP.str.upper().isin(deps)]
    med = [vias[k][2][len(vias[k][2]) // 2] for k in rios]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy([c[1] for c in med], [c[0] for c in med]), crs=4326)
    dentro = gpd.sjoin(pts, lim[["geometry"]], how="left", predicate="within").groupby(level=0).index_right.first().notna()
    quitar = {rios[i] for i in np.flatnonzero(~dentro.to_numpy())}
    print(f"ríos: {len(rios) - len(quitar):,} de {len(rios):,} tramos en departamentos amazónicos")
    return [v for k, v in enumerate(vias) if k not in quitar]


def enlaces_puerto(lat, lon, fluvial, vial):
    """Une cada nodo solo fluvial con el nodo vial más cercano si está a menos de conexion_rio_via_km.
    Devuelve (nodo_fluvial, nodo_vial, minutos) con caminata + transbordo."""
    from scipy.spatial import cKDTree
    f, v = np.flatnonzero(fluvial & ~vial), np.flatnonzero(vial)
    if len(f) == 0 or len(v) == 0:
        return np.array([], int), np.array([], int), np.array([])

    def xyz(i):
        la, lo = np.radians(lat[i]), np.radians(lon[i])
        return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]

    d, j = cKDTree(xyz(v)).query(xyz(f), k=1, distance_upper_bound=T.get("conexion_rio_via_km", 0.5) / R_KM)
    ok = np.isfinite(d)
    km = d[ok] * R_KM
    minutos = 60 * km / T["velocidad_acceso_kmh"] + T.get("transbordo_min", 15)
    print(f"embarcaderos (nodos fluviales unidos a una vía): {ok.sum():,}")
    return f[ok], v[j[ok]], minutos


def descargar_pbf():
    PBF.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {URL_PBF} ...")
    with requests.get(URL_PBF, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(PBF, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


# ---------------------------------------------------------------------------- grafo
def construir_grafo(vias):
    vel = T["velocidades_kmh"]
    idx: dict[int, int] = {}
    lat, lon, a, b, t = [], [], [], [], []
    nodos_fluviales, nodos_viales = set(), set()
    for tipo, nodos, coords in vias:
        v = vel[tipo]
        marca = nodos_fluviales if tipo in ("river", "ferry") else nodos_viales
        prev = None
        for nid, (la, lo) in zip(nodos, coords):
            if nid not in idx:
                idx[nid] = len(lat)
                lat.append(la)
                lon.append(lo)
            i = idx[nid]
            marca.add(i)
            if prev is not None:
                p, pla, plo = prev
                d = 2 * R_KM * np.arcsin(np.sqrt(np.sin(np.radians(la - pla) / 2) ** 2 + np.cos(np.radians(la)) *
                                                  np.cos(np.radians(pla)) * np.sin(np.radians(lo - plo) / 2) ** 2))
                a.append(p); b.append(i); t.append(60 * d / v)   # minutos
            prev = (i, la, lo)
    n = len(lat)
    lat, lon = np.array(lat), np.array(lon)
    fluvial, vial = np.zeros(n, bool), np.zeros(n, bool)
    fluvial[list(nodos_fluviales)] = True
    vial[list(nodos_viales)] = True
    pf, pv, pt = enlaces_puerto(lat, lon, fluvial, vial)
    a, b, t = np.r_[np.array(a, int), pf], np.r_[np.array(b, int), pv], np.r_[np.array(t), pt]
    t = np.maximum(t, 1e-6)   # csgraph ignora pesos cero
    g = coo_matrix((np.r_[t, t], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    print(f"grafo: {n:,} nodos, {len(t):,} tramos")
    return g, lat, lon


def anclar(lat_red, lon_red, lat, lon):
    """Nodo más cercano y minutos de acceso a pie."""
    arbol = BallTree(np.radians(np.c_[lat_red, lon_red]), metric="haversine")
    d, i = arbol.query(np.radians(np.c_[lat, lon]), k=1)
    km = d[:, 0] * R_KM
    return i[:, 0], 60 * km / T["velocidad_acceso_kmh"], km


def tiempos_desde_fuentes(g, nodos_fuente, acceso_fuente, limite=np.inf):
    """Dijkstra multi-fuente: añade un nodo virtual unido a cada fuente con su tiempo de acceso."""
    n = g.shape[0]
    gc = g.tocoo()
    filas = np.r_[gc.row, np.full(len(nodos_fuente), n)]
    cols = np.r_[gc.col, nodos_fuente]
    pesos = np.r_[gc.data, np.maximum(acceso_fuente, 1e-6)]
    gg = coo_matrix((pesos, (filas, cols)), shape=(n + 1, n + 1)).tocsr()
    return dijkstra(gg, directed=True, indices=n, limit=limite)[:n]


# ---------------------------------------------------------------------------- principal
def main(prueba: str | None = None):
    if prueba:
        vias = json.load(open(prueba))
    else:
        if not PBF.exists():
            descargar_pbf()
        vias = leer_pbf(PBF)
    g, lat_red, lon_red = construir_grafo(vias)

    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    if prueba:   # solo puntos dentro del extracto
        bb = (lat_red.min(), lat_red.max(), lon_red.min(), lon_red.max())
        dentro = lambda d: d.lat.between(bb[0], bb[1]) & d.lon.between(bb[2], bb[3])  # noqa: E731
        col, loc = col[dentro(col)].copy(), loc[dentro(loc)].copy()
        print(f"prueba: {len(col)} colegios y {len(loc)} locales dentro del extracto")
    col["quinto"] = col.quinto.fillna(0)
    nc, acc_c, km_c = anclar(lat_red, lon_red, col.lat, col.lon)
    nl, acc_l, _ = anclar(lat_red, lon_red, loc.lat, loc.lon)
    print(f"distancia mediana colegio-red: {np.median(km_c):.2f} km; colegios a más de 5 km de la red: {(km_c > 5).sum()}")

    t0, umbral = T["t0_min"], T["umbral_desierto_min"]
    filas = []
    for fam in ["Total", *FAMILIAS]:
        oferta = (loc.n_programas if fam == "Total" else loc[fam]).fillna(0).to_numpy(float)
        sel = oferta > 0
        if not sel.any():
            continue
        dist = tiempos_desde_fuentes(g, nl[sel], acc_l[sel])
        t_min = dist[nc] + acc_c
        # 2SFCA con tiempos: un Dijkstra por local (limitado a t0) para la matriz colegio × local
        W = np.zeros((len(col), sel.sum()))
        for k, (nodo, acc) in enumerate(zip(nl[sel], acc_l[sel])):
            dk = dijkstra(g, indices=nodo, limit=t0)
            W[:, k] = peso_decaimiento(dk[nc] + acc_c + acc, t0, CFG["accesibilidad"]["tipo_decaimiento"])
        dem = (W * col.quinto.to_numpy()[:, None]).sum(axis=0)
        razon = np.divide(oferta[sel], dem, out=np.zeros(sel.sum()), where=dem > 0)
        filas.append(pd.DataFrame({"COD_MOD": col.COD_MOD.values, "ANEXO": col.ANEXO.values, "ubigeo": col.ubigeo.values,
                                   "quinto": col.quinto.values, "familia": fam, "t_min": t_min,
                                   "acc_2sfca_t": (W * razon).sum(axis=1) * 1000,
                                   "desierto_t": (t_min > umbral).astype(int)}))
        print(f"  {fam}: {100 * np.average(t_min > umbral, weights=col.quinto + 1e-9):.1f} % de 5.º a más de {umbral} min")
    res = pd.concat(filas, ignore_index=True)
    sufijo = "_prueba" if prueba else ""
    guardar(res, f"tiempos_colegios{sufijo}.csv")
    res["t_x"] = res.t_min.replace(np.inf, np.nan) * res.quinto
    res["d_x"] = res.desierto_t * res.quinto
    dis = res.groupby(["ubigeo", "familia"]).agg(quinto=("quinto", "sum"), t_x=("t_x", "sum"), d_x=("d_x", "sum")).reset_index()
    dis["t_medio_min"] = dis.t_x / dis.quinto.replace(0, np.nan)
    dis["pct_quinto_en_desierto_t"] = dis.d_x / dis.quinto.replace(0, np.nan)
    guardar(dis.drop(columns=["t_x", "d_x"]), f"accesibilidad_tiempo_distritos{sufijo}.csv")
    tab = res.groupby("familia").apply(lambda d: pd.Series({"quinto": d.quinto.sum(),
                                                            "pct_desierto_t": 100 * np.average(d.desierto_t, weights=d.quinto + 1e-9),
                                                            "t_mediano_min": d.t_min.median()})).reset_index()
    guardar(tab, f"tabla_desiertos_tiempo{sufijo}.csv", "tablas")


if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--prueba" else None)
