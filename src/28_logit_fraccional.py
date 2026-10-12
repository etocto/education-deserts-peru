"""Paso 28 – Modelo de la proporción de alumnos en desierto por distrito (seguimiento, N6).

RQ2 pregunta si los desiertos se concentran en zonas más pobres, rurales o indígenas. El resultado principal pasa a
ser la PROPORCIÓN de alumnos de 5.º del distrito que están en desierto (0-1), no el número de programas a 60 km.
Se estima un logit fraccional (cuasi-máxima verosimilitud de Papke y Wooldridge, 1996: GLM binomial con enlace logit
y errores robustos), para cualquier carrera y por familia, en dos especificaciones:

  entre   sin efectos fijos de departamento, errores agrupados por departamento (contraste entre regiones)
  dentro  con efectos fijos de departamento, errores agrupados por provincia (contraste dentro de regiones)

Variables: pobreza 2018 (punto medio, %), % de 5.º en colegios rurales, % autoidentificación indígena, % afroperuano,
% adultos sin educación superior (censo 2017) y log de alumnos de 5.º. Se reportan efectos marginales promedio (AME)
en puntos porcentuales de la proporción en desierto por cada 10 puntos de la covariable (y por cada log-unidad de
alumnos). Robustez: ponderado por alumnos de 5.º y con el desierto por tiempo (90 min).

Nota: la pobreza distrital es una estimación de áreas pequeñas (ancho medio del IC 95 % de 19.4 pp); el error de
medición atenúa su coeficiente hacia cero, lo que se declara en Limitaciones.

Revisión v3 (obs. M4): para cualquier carrera se añaden (a) errores estándar de Conley (1999) con núcleo de Bartlett a
60 y 100 km entre centroides de los colegios del distrito, (b) el I de Moran de los residuos de Pearson con contigüidad
tipo reina (permutaciones) y (c) la estimación sin los distritos con menos de 10 alumnos de 5.º. Salida adicional:
salidas/tablas/logit_fraccional_espacial.csv

Requiere los pasos 3, 4 y 4b. Salidas: salidas/tablas/logit_fraccional_ame.csv y logit_fraccional_resumen.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import FAMILIAS, guardar, ruta

warnings.filterwarnings("ignore")
COV = ["pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior", "log_quinto"]
ESCALA = {c: 10.0 for c in COV} | {"log_quinto": 1.0}


def datos() -> pd.DataFrame:
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0)]
    y = acc.pivot_table(index="ubigeo", columns="familia", values="pct_quinto_en_desierto")
    y.columns = [f"d_{c}" for c in y.columns]
    tie = pd.read_csv(ruta("processed") / "accesibilidad_tiempo_distritos.csv", dtype={"ubigeo": str})
    yt = tie.pivot_table(index="ubigeo", columns="familia", values="pct_quinto_en_desierto_t")
    yt.columns = [f"t_{c}" for c in yt.columns]
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str, "dpto": str, "prov": str})
    cen = pd.read_csv(ruta("censo2017_agregado"), dtype={"ubigeo": str})
    d = base.merge(y.reset_index(), on="ubigeo").merge(yt.reset_index(), on="ubigeo", how="left").merge(cen, on="ubigeo", how="left")
    d = d[d.quinto > 0].copy()
    d["log_quinto"] = np.log(d.quinto)
    d["pobreza"] = d.pobreza_punto_medio
    d["rural"] = 100 * d.pct_rural_quinto
    for c in ["pct_autoid_indigena", "pct_afro", "pct_sin_superior"]:
        d[c] = 100 * d[c]
    return d.dropna(subset=COV)


def ajustar(d: pd.DataFrame, yv: str, fe: bool, pesos: bool):
    x = " + ".join([*COV, *(["C(dpto)"] if fe else [])])
    dd = d.dropna(subset=[yv]).copy()
    dd["y"] = dd[yv].clip(0, 1)
    grupos = pd.factorize(dd.prov if fe else dd.dpto)[0]
    m = smf.glm(f"y ~ {x}", data=dd, family=sm.families.Binomial(),
                freq_weights=None, var_weights=dd.quinto if pesos else None).fit(cov_type="cluster", cov_kwds={"groups": grupos})
    p = m.fittedvalues.to_numpy()
    dens = p * (1 - p)
    w = dd.quinto.to_numpy() if pesos else np.ones(len(dd))
    filas = []
    for v in COV:
        # AME y su error por el método delta (gradiente numérico respecto de todos los parámetros)
        b = m.params.to_numpy()
        X = m.model.exog
        j = list(m.params.index).index(v)

        def ame(beta):
            pp = 1 / (1 + np.exp(-X @ beta))
            return np.average(beta[j] * pp * (1 - pp), weights=w)

        g = np.array([(ame(b + e) - ame(b - e)) / (2 * 1e-6) for e in np.eye(len(b)) * 1e-6])
        se = float(np.sqrt(g @ m.cov_params().to_numpy() @ g))
        k = ESCALA[v] * 100   # pp de la proporción por 10 pp (o por log-unidad)
        filas.append({"variable": v, "coef": m.params[v], "p": m.pvalues[v], "ame_pp": k * ame(b), "ame_ee_pp": k * se,
                      "n": int(m.nobs), "media_y": np.average(dd.y, weights=w)})
    return filas, float(np.average(dens, weights=w))


def ame_con_v(m, V, w=None):
    """AME (pp por 10 pp) y error estándar por el método delta con una matriz de covarianza V."""
    b = m.params.to_numpy()
    X = m.model.exog
    w = np.ones(len(X)) if w is None else w
    out = {}
    for v in COV:
        j = list(m.params.index).index(v)

        def ame(beta):
            pp = 1 / (1 + np.exp(-X @ beta))
            return np.average(beta[j] * pp * (1 - pp), weights=w)

        g = np.array([(ame(b + e) - ame(b - e)) / (2 * 1e-6) for e in np.eye(len(b)) * 1e-6])
        se = float(np.sqrt(g @ V @ g))
        k = ESCALA[v] * 100
        out[v] = (k * ame(b), k * se)
    return out


def espacial(d: pd.DataFrame):
    """Conley, I de Moran de los residuos y exclusión de distritos pequeños (cualquier carrera)."""
    from scipy.stats import norm

    from utils import haversine_matriz
    filas = []
    for medida, yv in [("distancia", "d_Total"), ("tiempo", "t_Total")]:
        for fe in [False, True]:
            x = " + ".join([*COV, *(["C(dpto)"] if fe else [])])
            dd = d.dropna(subset=[yv, "lat_media", "lon_media"]).reset_index(drop=True).copy()
            dd["y"] = dd[yv].clip(0, 1)
            m = smf.glm(f"y ~ {x}", data=dd, family=sm.families.Binomial()).fit()
            X = m.model.exog
            p = m.fittedvalues.to_numpy()
            s = X * (dd.y.to_numpy() - p)[:, None]
            Hi = np.linalg.pinv((X * (p * (1 - p))[:, None]).T @ X)
            D = haversine_matriz(dd.lat_media, dd.lon_media, dd.lat_media, dd.lon_media)
            esp = "dentro" if fe else "entre"
            grupos = pd.factorize(dd.prov if fe else dd.dpto)[0]
            mc = smf.glm(f"y ~ {x}", data=dd, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": grupos})
            for etiqueta, V in [("agrupado (principal)", mc.cov_params().to_numpy()),
                                *[(f"Conley {c} km", Hi @ (s.T @ np.clip(1 - D / c, 0, None) @ s) @ Hi) for c in (60, 100)]]:
                for v, (a, se) in ame_con_v(m, V).items():
                    filas.append({"medida": medida, "especificacion": esp, "errores": etiqueta, "muestra": "todos",
                                  "variable": v, "ame_pp": a, "ee_pp": se, "p": 2 * norm.sf(abs(a / se)), "n": len(dd)})
            # I de Moran de los residuos de Pearson (contigüidad tipo reina)
            try:
                import geopandas as gpd
                from libpysal.weights import KNN, Queen, attach_islands
                lim = gpd.read_file(ruta("limites_distrital"))[["UBIGEO", "geometry"]]
                lim = lim[lim.UBIGEO.isin(dd.ubigeo)].set_index("UBIGEO").loc[dd.ubigeo].reset_index()
                w = Queen.from_dataframe(lim, use_index=False)
                if w.islands:
                    c = lim.to_crs(32718).geometry.centroid
                    w = attach_islands(w, KNN.from_array(np.c_[c.x, c.y], k=1))
                w.transform = "r"
                W = w.sparse
                z = m.resid_pearson.to_numpy() - m.resid_pearson.mean()
                moran = lambda v: float(v @ (W @ v) / (v @ v))  # noqa: E731  (W estandarizada por filas: S0 = n)
                i_obs = moran(z)
                rng = np.random.default_rng(2026)
                perm = np.array([moran(rng.permutation(z)) for _ in range(999)])
                p_sim = (1 + (perm >= i_obs).sum()) / 1000
                filas.append({"medida": medida, "especificacion": esp, "errores": "I de Moran de los residuos", "muestra": "todos",
                              "variable": "residuos", "ame_pp": i_obs, "ee_pp": np.nan, "p": p_sim, "n": len(dd)})
            except ImportError as e:
                print(f"I de Moran omitido ({e}); instale: pip install libpysal")
            # sin distritos pequeños
            ds = dd[dd.quinto >= 10]
            ms = smf.glm(f"y ~ {x}", data=ds, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": pd.factorize(ds.prov if fe else ds.dpto)[0]})
            for v, (a, se) in ame_con_v(ms, ms.cov_params().to_numpy()).items():
                filas.append({"medida": medida, "especificacion": esp, "errores": "agrupado (principal)", "muestra": ">= 10 alumnos",
                              "variable": v, "ame_pp": a, "ee_pp": se, "p": 2 * norm.sf(abs(a / se)), "n": len(ds)})
    t = pd.DataFrame(filas)
    guardar(t.round(5), "logit_fraccional_espacial.csv", "tablas")
    print("\nCualquier carrera: errores alternativos, I de Moran y sin distritos pequeños:")
    print(t[t.variable.isin(["pobreza", "rural", "pct_autoid_indigena", "residuos"])].round(3).to_string(index=False))


def main():
    d = datos()
    filas = []
    for fam in ["Total", *FAMILIAS]:
        for medida, pref in [("distancia", "d_"), ("tiempo", "t_")]:
            for fe in [False, True]:
                for pesos in [False, True]:
                    if medida == "tiempo" and pesos:
                        continue
                    yv = f"{pref}{fam}"
                    if yv not in d or d[yv].nunique() < 2:
                        continue
                    res, _ = ajustar(d, yv, fe, pesos)
                    for r in res:
                        filas.append({"familia": fam, "medida": medida, "especificacion": "dentro" if fe else "entre",
                                      "ponderado": pesos, **r})
    t = pd.DataFrame(filas)
    guardar(t.round(4), "logit_fraccional_ame.csv", "tablas")
    principal = t[(t.medida == "distancia") & (~t.ponderado)]
    res = principal.assign(celda=principal.apply(lambda r: f"{r.ame_pp:.2f}{'***' if r.p < .01 else '**' if r.p < .05 else '*' if r.p < .1 else ''}", axis=1)
                           ).pivot_table(index=["familia", "especificacion"], columns="variable", values="celda", aggfunc="first")[COV]
    guardar(res.reset_index(), "logit_fraccional_resumen.csv", "tablas")
    print("AME (pp de la proporción en desierto por 10 pp de la covariable; log_quinto por log-unidad), distancia, sin ponderar:")
    print(res.to_string())
    espacial(d)
    print("\nCualquier carrera, todas las variantes:")
    print(t[t.familia == "Total"][["medida", "especificacion", "ponderado", "variable", "ame_pp", "ame_ee_pp", "p", "n"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
