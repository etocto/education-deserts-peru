"""Paso 10 – ¿Importan los desiertos? Asistencia a educación superior (Censo 2017) y autocorrelación espacial.

Parte 1. Asistencia (distrito)
  Variables dependientes (Censo 2017, jóvenes de 17 a 24 años):
    tasa_asiste_17_24         % que asiste a educación superior
    pct_asiste_otro_distrito  % de los que asisten que estudian fuera de su distrito (movilidad para estudiar)
  Variable de acceso: escenario "pre" (licenciadas 2024 + sedes y filiales de denegadas, carnés 2018), el más
  cercano a 2017: log(1 + distancia al programa más cercano), % de alumnos de 5.º en desierto y log(1 + programas a 60 km).
  Controles: pobreza 2018, % rural (censo), % indígena y afroperuano (15-29), % adultos sin superior, log población.
  Mínimos cuadrados ponderados por la población de 17-24 años, efectos fijos de departamento y errores agrupados
  por provincia. Coeficientes en puntos porcentuales. Asociación descriptiva, no efecto causal.

Parte 2. Autocorrelación espacial (índice de Moran, contigüidad tipo reina, 999 permutaciones)
  - residuos de Pearson de la binomial negativa principal (todas las familias, paso 5)
  - residuos del modelo de asistencia
  - número de programas a 60 km (variable original)

Parte 3. Robustez espacial de la binomial negativa principal: errores agrupados por departamento y modelo SLX.

Salidas: salidas/tablas/asistencia_modelos.csv, salidas/tablas/moran.csv, salidas/tablas/robustez_espacial_conteo.csv
"""
from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import ajustar_nb, CFG, RAIZ, guardar, ruta

warnings.filterwarnings("ignore")
S = CFG["modelos"]["semilla"]


# ---------------------------------------------------------------------------- datos
def datos():
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str, "dpto": str, "prov": str})
    cen = pd.read_csv(ruta("censo2017_agregado"), dtype={"ubigeo": str})
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.familia == "Total") & (acc.factor_sinuosidad == 1.0) & (acc.escenario == "pre")]
    d = base.merge(cen, on="ubigeo", how="inner").merge(
        acc[["ubigeo", "dist_min_km", "pct_quinto_en_desierto", "n_prog_60km"]], on="ubigeo", how="inner")
    d = d[(d.quinto > 0) & (d.pob_17_24 > 0)].copy()
    d["asiste"] = 100 * d.tasa_asiste_17_24
    d["otro_distrito"] = 100 * d.pct_asiste_otro_distrito
    d["log_dist"] = np.log1p(d.dist_min_km)
    d["desierto"] = 100 * d.pct_quinto_en_desierto
    d["log_prog"] = np.log1p(d.n_prog_60km)
    d["pobreza"] = d.pobreza_punto_medio
    d["rural"] = 100 * d.pct_rural_censal
    for c in ["pct_autoid_indigena", "pct_afro", "pct_sin_superior"]:
        d[c] = 100 * d[c]
    d["log_pob"] = np.log(d.pob_total_2017)
    return d


CONTROLES = "pobreza + rural + pct_autoid_indigena + pct_afro + pct_sin_superior + log_pob + C(dpto)"


def asistencia(d):
    filas, resid = [], None
    for y in ["asiste", "otro_distrito"]:
        for x in ["log_dist", "desierto", "log_prog"]:
            dd = d.dropna(subset=[y, x, "pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior"])
            m = smf.wls(f"{y} ~ {x} + {CONTROLES}", data=dd, weights=dd.pob_17_24).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.prov)[0]})
            filas.append({"dependiente": y, "acceso": x, "coef": m.params[x], "ee": m.bse[x], "p": m.pvalues[x],
                          "pobreza": m.params["pobreza"], "p_pobreza": m.pvalues["pobreza"],
                          "rural": m.params["rural"], "p_rural": m.pvalues["rural"],
                          "r2": m.rsquared, "n": int(m.nobs), "media_y": np.average(dd[y], weights=dd.pob_17_24)})
            if y == "asiste" and x == "log_dist":
                resid = pd.Series(np.asarray(m.resid), index=dd.ubigeo.values)
    t = pd.DataFrame(filas)
    guardar(t.round(4), "asistencia_modelos.csv", "tablas")
    print(t[["dependiente", "acceso", "coef", "ee", "p", "r2", "n", "media_y"]].round(3).to_string(index=False))
    return resid


# ---------------------------------------------------------------------------- Moran
def vecinos(ubigeos):
    import geopandas as gpd
    lim = gpd.read_file(ruta("limites_distrital"))[["UBIGEO", "geometry"]]
    lim = lim[lim.UBIGEO.isin(ubigeos)].reset_index(drop=True)
    lim["geometry"] = lim.geometry.buffer(0)
    par = gpd.sjoin(lim, lim, how="inner", predicate="intersects")
    par = par[par.index != par.index_right]
    return lim.UBIGEO.to_numpy(), par.index.to_numpy(), par.index_right.to_numpy()


def moran(valores: pd.Series, ids, i, j, perm=999):
    x = valores.reindex(ids).to_numpy(float)
    ok = np.isfinite(x)
    keep = ok[i] & ok[j]
    i, j = i[keep], j[keep]
    z = np.where(ok, x - np.nanmean(x), 0.0)
    n = ok.sum()
    grado = np.bincount(i, minlength=len(x)).astype(float)
    w = 1.0 / grado[i]                       # pesos estandarizados por fila
    S0 = w.sum()
    num = lambda zz: (w * zz[i] * zz[j]).sum()  # noqa: E731
    I = n / S0 * num(z) / (z[ok] ** 2).sum()
    rng = np.random.default_rng(S)
    sims = np.empty(perm)
    idx = np.flatnonzero(ok)
    for k in range(perm):
        zp = z.copy()
        zp[idx] = rng.permutation(z[idx])
        sims[k] = n / S0 * num(zp) / (z[ok] ** 2).sum()
    p = (1 + (np.abs(sims) >= abs(I)).sum()) / (perm + 1)
    return I, p, int(n)


def residuos_nb():
    spec = importlib.util.spec_from_file_location("m5", RAIZ / "src" / "05_modelos_conteo.py")
    m5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m5)
    d = m5.datos_distritales()
    for c in m5.EXTRA_PRINCIPAL:
        d[c] = 100 * d[c]
    dd = d.dropna(subset=["y_Total", "pobreza", "rural", *m5.EXTRA_PRINCIPAL]).copy()
    dd["y"] = dd.y_Total.round().astype(int)
    x = " + ".join(["log_quinto", "pobreza", "rural", *m5.EXTRA_PRINCIPAL, "C(dpto)"])
    m = ajustar_nb(f"y ~ {x}", dd)
    mu = m.predict(dd)
    alpha = m.params["alpha"]
    pearson = (dd.y - mu) / np.sqrt(mu + alpha * mu ** 2)
    return pd.Series(pearson.values, index=dd.ubigeo.values), pd.Series(dd.y.values, index=dd.ubigeo.values)


def robustez_espacial(ids, i, j):
    """Binomial negativa principal (todas las familias) con (a) errores agrupados por departamento y
    (b) modelo SLX: se agregan los rezagos espaciales (promedio de vecinos) de pobreza, ruralidad e indígena."""
    spec = importlib.util.spec_from_file_location("m5", RAIZ / "src" / "05_modelos_conteo.py")
    m5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m5)
    d = m5.datos_distritales()
    for c in m5.EXTRA_PRINCIPAL:
        d[c] = 100 * d[c]
    d = d.dropna(subset=["y_Total", "pobreza", "rural", *m5.EXTRA_PRINCIPAL]).copy()
    d["y"] = d.y_Total.round().astype(int)
    pos = {u: k for k, u in enumerate(ids)}
    for v in ["pobreza", "rural", "pct_autoid_indigena"]:
        x = d.set_index("ubigeo")[v].reindex(ids).to_numpy(float)
        suma = np.bincount(i, weights=np.nan_to_num(x[j]), minlength=len(ids))
        cuenta = np.bincount(i, weights=np.isfinite(x[j]).astype(float), minlength=len(ids))
        lag = np.divide(suma, cuenta, out=np.full(len(ids), np.nan), where=cuenta > 0)
        d[f"W_{v}"] = d.ubigeo.map(lambda u: lag[pos[u]] if u in pos else np.nan)
    base = " + ".join(["log_quinto", "pobreza", "rural", *m5.EXTRA_PRINCIPAL, "C(dpto)"])
    filas = []
    for nombre, x, grupo in [("principal (cluster provincia)", base, "prov"),
                             ("cluster departamento", base, "dpto"),
                             ("SLX (rezagos espaciales)", base + " + W_pobreza + W_rural + W_pct_autoid_indigena", "prov")]:
        dd = d.dropna(subset=["W_pobreza", "W_rural", "W_pct_autoid_indigena"]) if "W_" in x else d
        m = ajustar_nb(f"y ~ {x}", dd, cov_type="cluster",
                                                             cov_kwds={"groups": pd.factorize(dd[grupo])[0]})
        fila = {"modelo": nombre, "n": int(m.nobs)}
        for v in ["log_quinto", "pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior",
                  "W_pobreza", "W_rural", "W_pct_autoid_indigena"]:
            if v in m.params:
                fila[f"irr_{v}"] = np.exp(m.params[v]); fila[f"p_{v}"] = m.pvalues[v]
        filas.append(fila)
    t = pd.DataFrame(filas)
    guardar(t.round(4), "robustez_espacial_conteo.csv", "tablas")
    print(t.round(3).T.to_string())


def main():
    d = datos()
    res_asis = asistencia(d)
    res_nb, y_nb = residuos_nb()
    ids, i, j = vecinos(set(d.ubigeo) | set(res_nb.index))
    filas = []
    for nombre, serie in [("programas a 60 km (variable)", y_nb), ("residuos binomial negativa (todas las familias)", res_nb),
                          ("residuos modelo de asistencia", res_asis)]:
        I, p, n = moran(serie, ids, i, j)
        filas.append({"serie": nombre, "moran_I": I, "p_perm": p, "n": n})
    t = pd.DataFrame(filas)
    guardar(t.round(4), "moran.csv", "tablas")
    print(t.round(3).to_string(index=False))
    robustez_espacial(ids, i, j)


if __name__ == "__main__":
    main()
