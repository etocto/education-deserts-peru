"""Construcción de la red de OSM con poco uso de memoria (misma red que el paso 4b).

El paso 4b guarda cada vía como listas de Python (ids y tuplas de coordenadas), lo que con el extracto del Perú
puede superar los 6 GB de RAM. Este módulo produce el mismo grafo -mismos tipos de vía, velocidades, ríos
amazónicos, embarcaderos y tiempos en minutos- guardando los datos en arreglos compactos de numpy.

Uso:  from red_liviana import construir_red;  g, lat, lon = construir_red(ruta_pbf)
"""
from __future__ import annotations

import importlib.util
from array import array

import numpy as np
from scipy.sparse import coo_matrix

from utils import CFG, RAIZ, ruta

T = CFG["tiempos"]
R_KM = 6371.0088


def _m04b():
    spec = importlib.util.spec_from_file_location("m04b", RAIZ / "src" / "04b_tiempos_viaje.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def leer(path):
    import osmium

    vel = T["velocidades_kmh"]
    tipos = list(vel.keys())
    codigo = {t: i for i, t in enumerate(tipos)}
    usar_rios = T.get("incluir_rios", True)

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ids, self.lat, self.lon = array("q"), array("d"), array("d")
            self.largo, self.tipo = array("i"), array("b")

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
                nodos = [(n.ref, n.location.lat, n.location.lon) for n in w.nodes]
            except osmium.InvalidLocationError:
                return
            for nid, la, lo in nodos:
                self.ids.append(nid); self.lat.append(la); self.lon.append(lo)
            self.largo.append(len(nodos)); self.tipo.append(codigo[tipo])

    h = H()
    h.apply_file(str(path), locations=True, idx="flex_mem")
    return (np.frombuffer(h.ids, dtype=np.int64), np.frombuffer(h.lat), np.frombuffer(h.lon),
            np.frombuffer(h.largo, dtype=np.int32), np.frombuffer(h.tipo, dtype=np.int8), tipos)


def aristas_compactas(path):
    """Nodos y tramos (km y tipo de vía) de la red, en el formato de aristas() del paso 4d:
    (lat, lon, a, b, d_km, tp, tipos, puertos), con tp = índice en tipos = sorted(velocidades_kmh)."""
    ids, lat, lon, largo, tipo, tipos_lectura = leer(path)
    tipos = sorted(T["velocidades_kmh"])
    recod = np.array([tipos.index(t) for t in tipos_lectura], np.int8)
    tipo = recod[tipo]
    es_rio = np.array([t in ("river", "ferry") for t in tipos])
    inicio = np.r_[0, np.cumsum(largo)[:-1]]
    deps = [d.upper() for d in T.get("rios_departamentos", [])]
    quitar = np.zeros(len(largo), bool)
    rios = np.flatnonzero(np.array(tipos)[tipo] == "river")
    if len(rios) and deps:
        import geopandas as gpd
        lim = gpd.read_file(ruta("limites_departamental"))
        lim = lim[lim.NOMBDEP.str.upper().isin(deps)]
        med = inicio[rios] + largo[rios] // 2
        pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon[med], lat[med]), crs=4326)
        dentro = gpd.sjoin(pts, lim[["geometry"]], how="left", predicate="within").groupby(level=0).index_right.first().notna()
        quitar[rios[~dentro.to_numpy()]] = True
        print(f"ríos: {len(rios) - quitar.sum():,} de {len(rios):,} tramos en departamentos amazónicos")
    via = np.repeat(np.arange(len(largo), dtype=np.int32), largo)
    keep = ~quitar[via]
    uniq, inv = np.unique(ids, return_inverse=True)
    n = len(uniq)
    lat_red, lon_red = np.empty(n), np.empty(n)
    lat_red[inv] = lat
    lon_red[inv] = lon
    p = np.flatnonzero((via[:-1] == via[1:]) & keep[:-1])
    a, b = inv[p], inv[p + 1]
    la1, lo1, la2, lo2 = (np.radians(x) for x in (lat[p], lon[p], lat[p + 1], lon[p + 1]))
    d = 2 * R_KM * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    tp = tipo[via[p]]
    fluvial, vial = np.zeros(n, bool), np.zeros(n, bool)
    fluvial[inv[keep & es_rio[tipo[via]]]] = True
    vial[inv[keep & ~es_rio[tipo[via]]]] = True
    del ids, lat, lon, via, keep, p, la1, lo1, la2, lo2
    usados = fluvial | vial
    nuevo = np.full(n, -1, np.int64)
    nuevo[usados] = np.arange(usados.sum())
    a, b = nuevo[a], nuevo[b]
    lat_red, lon_red, fluvial, vial = lat_red[usados], lon_red[usados], fluvial[usados], vial[usados]
    puertos = _m04b().enlaces_puerto(lat_red, lon_red, fluvial, vial)
    print(f"red: {len(lat_red):,} nodos, {len(a):,} tramos")
    return lat_red, lon_red, a, b, d, tp, tipos, puertos


def grafo_desde(lat, a, b, d, tp, tipos, vel, puertos):
    """Grafo no dirigido (minutos) para un diccionario de velocidades; tipos sin velocidad se omiten."""
    v = np.array([vel.get(t, np.nan) for t in tipos])[tp]
    ok = np.isfinite(v) & (v > 0)
    aa, bb, t = a[ok], b[ok], 60 * d[ok] / v[ok]
    if "river" in vel:
        aa, bb, t = np.r_[aa, puertos[0]], np.r_[bb, puertos[1]], np.r_[t, puertos[2]]
    t = np.maximum(t, 1e-6)
    n = len(lat)
    return coo_matrix((np.r_[t, t], (np.r_[aa, bb], np.r_[bb, aa])), shape=(n, n)).tocsr()


def construir_red(path, vel=None):
    """Grafo con las velocidades de config.yaml (ya calibradas en utils) o con `vel`."""
    lat, lon, a, b, d, tp, tipos, puertos = aristas_compactas(path)
    g = grafo_desde(lat, a, b, d, tp, tipos, vel or T["velocidades_kmh"], puertos)
    print(f"grafo: {len(lat):,} nodos")
    return g, lat, lon
