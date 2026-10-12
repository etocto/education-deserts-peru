"""Paso 36 – Relieve de los colegios y validación de la etiqueta «topográfico» (revisión v3, obs. M1).

El origen «topográfico» se define como un colegio con un programa a <= 60 km en línea recta pero a más de 90 min
por la red. Para comprobar que esa brecha refleja el relieve (y no solo la red de OpenStreetMap o el factor de
velocidad), este paso mide el relieve con un modelo digital de elevación y lo compara entre grupos:

  1. Elevación: teselas «Terrarium» de AWS Terrain Tiles (públicas, sin clave; fuentes SRTM, GMTED y otras),
     nivel de zoom 10 (~150 m por píxel en el Perú). Se descargan solo las teselas necesarias y se guardan en
     datos/raw/dem/terrarium/ (unos 150-300 MB).
  2. Por colegio: elevación, desnivel (máx. - mín.) y pendiente media en un radio de 5 km.
  3. Para los colegios con un programa (cualquier carrera) a <= 60 km: perfil de elevación en línea recta hasta el
     local más cercano (200 puntos): desnivel acumulado (suma de subidas y bajadas) y altura máxima sobre el punto
     de partida y de llegada («barrera»).
  4. Región natural del colegio (para el paso 37): sierra si la elevación es >= 2,000 m; si no, costa en los
     departamentos costeros y selva en el resto.
  5. Compara, ponderando por alumnos de 5.º, los grupos topográfico, sin desierto, estructural y de licenciamiento,
     y la proporción a más de 90 min por terciles de desnivel acumulado entre los colegios a <= 60 km.

Requiere internet la primera vez (descarga de teselas) y los pasos 2, 4 y 4b.
Uso:     python src/36_relieve_grupos.py              (10-20 min la primera vez; luego usa la caché)
Salidas: datos/processed/relieve_colegios.csv
         salidas/tablas/relieve_por_origen.csv, relieve_terciles.csv
"""
from __future__ import annotations

import io
import math
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils import CFG, RAIZ, guardar, haversine_matriz, ruta

Z = 10
TESELA = 256
CACHE = Path(os.environ.get("DEM_CACHE", RAIZ / "datos/raw/dem/terrarium")) / str(Z)
warnings.filterwarnings("ignore")
URLS = ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
        "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"]
RADIO_KM = 5.0
N_PERFIL = 200
COSTA = {"TUMBES", "PIURA", "LAMBAYEQUE", "LA LIBERTAD", "ANCASH", "LIMA", "CALLAO", "ICA", "AREQUIPA", "MOQUEGUA", "TACNA"}
SELVA_DEP = {"LORETO", "UCAYALI", "MADRE DE DIOS", "SAN MARTIN"}


# ---------------------------------------------------------------------------- teselas
def px_globales(lat, lon):
    """Coordenadas de píxel globales (Web Mercator) en el nivel Z."""
    lat = np.clip(np.asarray(lat, float), -85, 85)
    n = 2 ** Z * TESELA
    x = (np.asarray(lon, float) + 180) / 360 * n
    y = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n
    return x, y


def metros_por_px(lat):
    return 156543.03392 * np.cos(np.radians(lat)) / 2 ** Z


def archivo(x, y):
    return CACHE / str(x) / f"{y}.png"


def descargar(xy):
    x, y = xy
    p = archivo(x, y)
    if p.exists():
        return True
    p.parent.mkdir(parents=True, exist_ok=True)
    for u in URLS:
        try:
            r = requests.get(u.format(z=Z, x=x, y=y), timeout=60)
            if r.status_code == 200:
                p.write_bytes(r.content)
                return True
        except requests.RequestException:
            continue
    return False


@lru_cache(maxsize=600)
def tesela(x, y):
    from PIL import Image
    p = archivo(x, y)
    if not p.exists():
        return np.full((TESELA, TESELA), np.nan, np.float32)
    a = np.asarray(Image.open(io.BytesIO(p.read_bytes())).convert("RGB"), dtype=np.float32)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


def ventana(gx, gy, r):
    """Matriz de elevación de los píxeles globales [gx-r, gx+r] x [gy-r, gy+r]."""
    x0, x1, y0, y1 = int(gx) - r, int(gx) + r, int(gy) - r, int(gy) + r
    out = np.empty((y1 - y0 + 1, x1 - x0 + 1), np.float32)
    for ty in range(y0 // TESELA, y1 // TESELA + 1):
        for tx in range(x0 // TESELA, x1 // TESELA + 1):
            t = tesela(tx, ty)
            ya, yb = max(y0, ty * TESELA), min(y1, ty * TESELA + TESELA - 1)
            xa, xb = max(x0, tx * TESELA), min(x1, tx * TESELA + TESELA - 1)
            out[ya - y0:yb - y0 + 1, xa - x0:xb - x0 + 1] = t[ya - ty * TESELA:yb - ty * TESELA + 1, xa - tx * TESELA:xb - tx * TESELA + 1]
    return out


def elevacion_puntos(lat, lon):
    gx, gy = px_globales(lat, lon)
    tx, ty = (gx // TESELA).astype(int), (gy // TESELA).astype(int)
    out = np.empty(len(gx), np.float32)
    for (a, b), idx in pd.Series(range(len(gx))).groupby([tx, ty]).groups.items():
        t = tesela(int(a), int(b))
        i = np.asarray(idx)
        out[i] = t[(gy[i] - b * TESELA).astype(int).clip(0, 255), (gx[i] - a * TESELA).astype(int).clip(0, 255)]
    return out


def teselas_necesarias(lat, lon, r_px, lat2=None, lon2=None):
    gx, gy = px_globales(lat, lon)
    s = set()
    for x, y in zip(gx, gy):
        for tx in range(int(x - r_px) // TESELA, int(x + r_px) // TESELA + 1):
            for ty in range(int(y - r_px) // TESELA, int(y + r_px) // TESELA + 1):
                s.add((tx, ty))
    if lat2 is not None:   # teselas a lo largo de los perfiles
        for a, b, c, d in zip(lat, lon, lat2, lon2):
            t = np.linspace(0, 1, 60)
            px, py = px_globales(a + t * (c - a), b + t * (d - b))
            s |= set(zip((px // TESELA).astype(int), (py // TESELA).astype(int)))
    return s


# ---------------------------------------------------------------------------- medidas
def medidas_locales(lat, lon):
    gx, gy = px_globales(lat, lon)
    filas = []
    for la, x, y in zip(lat, gx, gy):
        m = metros_por_px(la)
        r = int(math.ceil(RADIO_KM * 1000 / m))
        w = ventana(x, y, r)
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        circ = (xx ** 2 + yy ** 2) * m ** 2 <= (RADIO_KM * 1000) ** 2
        gy_, gx_ = np.gradient(w.astype(float), m)
        pend = np.degrees(np.arctan(np.hypot(gx_, gy_)))
        v = w[circ]
        filas.append((float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).any() else np.nan,
                      float(np.nanstd(v)) if np.isfinite(v).any() else np.nan,
                      float(np.nanmean(pend[circ])) if np.isfinite(pend[circ]).any() else np.nan))
    return pd.DataFrame(filas, columns=["desnivel_5km_m", "de_elevacion_5km_m", "pendiente_media_5km_grados"])


def perfiles(lat, lon, lat2, lon2):
    t = np.linspace(0, 1, N_PERFIL)
    filas = []
    for a, b, c, d in zip(lat, lon, lat2, lon2):
        e = elevacion_puntos(a + t * (c - a), b + t * (d - b)).astype(float)
        if not np.isfinite(e).all():
            filas.append((np.nan, np.nan)); continue
        filas.append((float(np.abs(np.diff(e)).sum()), float(e.max() - max(e[0], e[-1]))))
    return pd.DataFrame(filas, columns=["desnivel_acumulado_m", "barrera_m"])


def region(dep, elev):
    d = dep.str.upper().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return np.where(elev >= 2000, "sierra", np.where(d.isin(COSTA), "costa", "selva"))


def region_sin_dem(dep):
    """Respaldo sin elevación (pasos 34 y 37): por departamento."""
    d = pd.Series(dep).str.upper().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return np.where(d.isin(COSTA), "costa", np.where(d.isin(SELVA_DEP), "selva", "sierra"))


# ---------------------------------------------------------------------------- principal
def main():
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype=dt)
    col["quinto"] = col.quinto.fillna(0)
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"]).reset_index(drop=True)
    d = haversine_matriz(col.lat, col.lon, loc.lat, loc.lon)
    j = d.argmin(axis=1)
    col["km_local"] = d.min(axis=1)
    col["lat_local"], col["lon_local"] = loc.lat.values[j], loc.lon.values[j]
    radio = CFG["accesibilidad"]["radio_desierto_km"]
    cerca = col.km_local <= radio

    r_px = int(math.ceil(RADIO_KM * 1000 / metros_por_px(0))) + 1
    nec = teselas_necesarias(col.lat, col.lon, r_px)
    nec |= teselas_necesarias(col.lat[cerca].to_numpy(), col.lon[cerca].to_numpy(), 1, col.lat_local[cerca].to_numpy(), col.lon_local[cerca].to_numpy())
    faltan = [t for t in nec if not archivo(*t).exists()]
    print(f"teselas necesarias: {len(nec):,}; por descargar: {len(faltan):,}")
    if faltan:
        with ThreadPoolExecutor(8) as ex:
            ok = list(ex.map(descargar, faltan))
        if not all(ok):
            print(f"AVISO: {ok.count(False)} teselas no se pudieron descargar (quedan sin dato)")
            if ok.count(False) == len(faltan):
                sys.exit("No hay conexión con el servidor de teselas; ejecute el paso con acceso a internet.")

    # ordenar por tesela mejora el uso de la caché en memoria
    gx, gy = px_globales(col.lat, col.lon)
    orden = np.lexsort(((gy // TESELA).astype(int), (gx // TESELA).astype(int)))
    col = col.iloc[orden].reset_index(drop=True)
    cerca = col.km_local <= radio
    col["elevacion_m"] = elevacion_puntos(col.lat, col.lon)
    col["elevacion_local_m"] = elevacion_puntos(col.lat_local, col.lon_local)
    print("medidas locales (5 km)...")
    col = pd.concat([col, medidas_locales(col.lat.to_numpy(), col.lon.to_numpy())], axis=1)
    print("perfiles al local más cercano...")
    p = perfiles(col.lat[cerca].to_numpy(), col.lon[cerca].to_numpy(), col.lat_local[cerca].to_numpy(), col.lon_local[cerca].to_numpy())
    p.index = col.index[cerca]
    col = col.join(p)
    col["region_natural"] = region(col.D_DPTO.fillna(""), col.elevacion_m)
    keep = ["COD_MOD", "ANEXO", "ubigeo", "D_DPTO", "quinto", "elevacion_m", "elevacion_local_m", "desnivel_5km_m",
            "de_elevacion_5km_m", "pendiente_media_5km_grados", "km_local", "desnivel_acumulado_m", "barrera_m", "region_natural"]
    guardar(col[keep].round(2), "relieve_colegios.csv")

    # ---- grupos (cualquier carrera, oferta 2024)
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt, usecols=["COD_MOD", "ANEXO", "familia", "escenario", "factor_sinuosidad", "desierto"])
    acc = acc[(acc.familia == "Total") & (acc.factor_sinuosidad == 1.0)].pivot_table(index=["COD_MOD", "ANEXO"], columns="escenario", values="desierto").reset_index()
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt, usecols=["COD_MOD", "ANEXO", "familia", "desierto_t", "t_min"])
    tie = tie[tie.familia == "Total"].drop(columns="familia")
    g = col.merge(acc, on=["COD_MOD", "ANEXO"]).merge(tie, on=["COD_MOD", "ANEXO"])
    g = g[g.quinto > 0]
    g["origen"] = np.select([(g.post == 1) & (g.pre == 0), (g.post == 1) & (g.pre == 1), (g.post == 0) & (g.desierto_t == 1)],
                            ["licensing", "structural", "topographic"], default="none")
    filas = []
    med = ["elevacion_m", "desnivel_5km_m", "pendiente_media_5km_grados", "desnivel_acumulado_m", "barrera_m"]
    for o, x in g.groupby("origen"):
        fila = {"origen": o, "colegios": len(x), "alumnos": int(x.quinto.sum())}
        for m in med:
            ok = x[m].notna()
            fila[m] = np.average(x.loc[ok, m], weights=x.loc[ok, "quinto"]) if ok.any() else np.nan
        fila["pct_sierra"] = 100 * np.average(x.region_natural == "sierra", weights=x.quinto)
        fila["pct_selva"] = 100 * np.average(x.region_natural == "selva", weights=x.quinto)
        filas.append(fila)
    t = pd.DataFrame(filas).set_index("origen").reindex(["structural", "licensing", "topographic", "none"]).reset_index()
    guardar(t.round(4), "relieve_por_origen.csv", "tablas")
    print(t.round(1).to_string(index=False))

    # ---- entre colegios a <= 60 km: % a más de 90 min por terciles de desnivel acumulado
    c = g[(g.post == 0) & g.desnivel_acumulado_m.notna()].copy()
    c["km_tercil"] = pd.qcut(c.km_local, 3, labels=["cerca", "intermedio", "lejos"])
    # terciles de desnivel dentro de cada tercil de distancia (compara colegios a distancias parecidas)
    c["tercil"] = c.groupby("km_tercil", observed=True).desnivel_acumulado_m.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False)).map({0: "bajo", 1: "medio", 2: "alto"})
    c["tercil"] = pd.Categorical(c.tercil, ["bajo", "medio", "alto"], ordered=True)
    ter = c.groupby(["km_tercil", "tercil"], observed=True).apply(lambda x: pd.Series({
        "colegios": len(x), "alumnos": x.quinto.sum(), "desnivel_mediano_m": x.desnivel_acumulado_m.median(),
        "pct_mas_90min": 100 * np.average(x.desierto_t, weights=x.quinto)})).reset_index()
    guardar(ter.round(4), "relieve_terciles.csv", "tablas")
    print("\n% de alumnos a más de 90 min entre los colegios con un programa a <= 60 km, por distancia y desnivel:")
    print(ter.round(1).to_string(index=False))
    # modelo: P(más de 90 min) según desnivel, controlando distancia y región natural (logit ponderado por alumnos)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    c["log_km"] = np.log(c.km_local.clip(lower=0.1))
    c["desnivel_100m"] = c.desnivel_acumulado_m / 100
    m = smf.glm("desierto_t ~ desnivel_100m + log_km + C(region_natural)", data=c, family=sm.families.Binomial(),
                var_weights=c.quinto.clip(lower=1)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(c.ubigeo.str[:4])[0]})
    p = m.fittedvalues.to_numpy()
    ame = np.average(m.params["desnivel_100m"] * p * (1 - p), weights=c.quinto.clip(lower=1))
    res = pd.DataFrame([{"variable": "desnivel acumulado (por 100 m)", "coef": m.params["desnivel_100m"], "p": m.pvalues["desnivel_100m"],
                         "efecto_marginal_pp": 100 * ame, "colegios": int(m.nobs)}])
    guardar(res.round(5), "relieve_modelo.csv", "tablas")
    print("\nlogit de estar a más de 90 min (colegios a <= 60 km):"); print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
