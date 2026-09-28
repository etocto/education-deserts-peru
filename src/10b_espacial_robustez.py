"""Paso 10b – Robustez espacial y de sesgo por variables omitidas (responde a los revisores 2 y 5).

1. Matriz de pesos espaciales W: contigüidad tipo reina (libpysal.weights.Queen), estandarizada por filas; los
   distritos sin vecinos contiguos (islas) se unen a su vecino más cercano. Se reportan sus características.
2. Modelos espaciales autorregresivos (spreg, máxima verosimilitud) para log(1 + programas a 60 km), con las mismas
   covariables y efectos fijos de departamento que la binomial negativa:
     - SAR / rezago espacial (ML_Lag): reporta rho (coeficiente de rezago espacial) y los coeficientes.
     - SEM / error espacial (ML_Error): reporta lambda.
3. Errores estándar alternativos para la binomial negativa (todas las familias) y el modelo de asistencia:
   agrupados por provincia, por departamento, en dos vías (provincia y departamento) y de Conley (1999) con núcleo
   de Bartlett a 60 y 100 km entre centroides de distritos.
4. Sensibilidad a variables omitidas en el modelo de asistencia (Oster, 2019): delta necesario para anular el
   efecto y coeficiente corregido (beta*) con delta = 1 y R2max = min(1, 1,3 × R2).

Requisitos: pip install libpysal spreg
Salidas: salidas/tablas/pesos_espaciales.csv, modelos_espaciales_sar_sem.csv, errores_estandar_alternativos.csv,
         oster_asistencia.csv
"""
from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import RAIZ, guardar, haversine_matriz, ruta

warnings.filterwarnings("ignore")


def cargar(modulo, archivo):
    spec = importlib.util.spec_from_file_location(modulo, RAIZ / "src" / archivo)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m5 = cargar("m5", "05_modelos_conteo.py")
m10 = cargar("m10", "10_asistencia_moran.py")
COV = ["log_quinto", "pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior"]


# ---------------------------------------------------------------------------- W
def matriz_w(ubigeos):
    import geopandas as gpd
    from libpysal.weights import Queen, attach_islands, KNN
    lim = gpd.read_file(ruta("limites_distrital"))[["UBIGEO", "geometry"]]
    lim = lim[lim.UBIGEO.isin(ubigeos)].sort_values("UBIGEO").reset_index(drop=True)
    w = Queen.from_dataframe(lim, ids=lim.UBIGEO.tolist(), use_index=False)
    islas = list(w.islands)
    if islas:
        cent = lim.to_crs(32718).geometry.centroid
        knn = KNN.from_array(np.c_[cent.x, cent.y], k=1, ids=lim.UBIGEO.tolist())
        w = attach_islands(w, knn)
    w.transform = "r"
    card = pd.Series(w.cardinalities)
    info = {"n": w.n, "vecinos_media": card.mean(), "vecinos_min": card.min(), "vecinos_max": card.max(),
            "islas_unidas_al_vecino_mas_cercano": len(islas), "pct_no_cero": w.pct_nonzero}
    guardar(pd.DataFrame([info]).round(3), "pesos_espaciales.csv", "tablas")
    print("W:", {k: round(v, 3) if isinstance(v, float) else v for k, v in info.items()})
    return w


# ---------------------------------------------------------------------------- SAR / SEM
def sar_sem(d, w):
    from spreg import ML_Error, ML_Lag, OLS
    dd = d.set_index("ubigeo").loc[w.id_order]
    y = np.log1p(dd.y_Total.to_numpy())[:, None]
    fe = pd.get_dummies(dd.dpto, prefix="dpto", drop_first=True).astype(float)
    X = pd.concat([dd[COV], fe], axis=1)
    nombres = list(X.columns)
    filas = []
    ols = OLS(y, X.to_numpy(), w=w, spat_diag=True, moran=True, name_x=nombres, name_y="log_prog60")
    lag = ML_Lag(y, X.to_numpy(), w=w, name_x=nombres, name_y="log_prog60")
    err = ML_Error(y, X.to_numpy(), w=w, name_x=nombres, name_y="log_prog60")
    for nombre, m, extra in [("OLS", ols, {}), ("SAR (ML_Lag)", lag, {"rho": float(lag.rho)}),
                             ("SEM (ML_Error)", err, {"lambda": float(err.lam)})]:
        fila = {"modelo": nombre, **extra}
        b = np.asarray(m.betas).ravel()
        se = np.asarray(m.std_err).ravel()
        names = list(m.name_x)
        for v in ["pobreza", "rural", "pct_autoid_indigena", "log_quinto"]:
            k = names.index(v)
            fila[f"b_{v}"], fila[f"se_{v}"] = b[k], se[k]
            fila[f"p_{v}"] = float(np.asarray(m.z_stat[k] if hasattr(m, "z_stat") else m.t_stat[k])[1])
        if hasattr(m, "z_stat") and nombre == "SAR (ML_Lag)":
            fila["p_rho"] = float(m.z_stat[-1][1])
        if nombre == "OLS":
            fila["moran_residuos"] = float(m.moran_res[0])
            fila["LM_lag"], fila["LM_error"] = float(m.lm_lag[0]), float(m.lm_error[0])
            fila["robust_LM_lag"], fila["robust_LM_error"] = float(m.rlm_lag[0]), float(m.rlm_error[0])
        filas.append(fila)
    t = pd.DataFrame(filas)
    guardar(t.round(4), "modelos_espaciales_sar_sem.csv", "tablas")
    print(t.round(3).T.to_string())


# ---------------------------------------------------------------------------- Conley
def conley(X, s, H_inv, coords, corte_km):
    """V = H^-1 (sum_ij K(d_ij) s_i s_j') H^-1, núcleo de Bartlett K = max(0, 1 - d/corte)."""
    D = haversine_matriz(coords[:, 0], coords[:, 1], coords[:, 0], coords[:, 1])
    K = np.clip(1 - D / corte_km, 0, None)
    M = s.T @ K @ s
    return H_inv @ M @ H_inv


def errores_alternativos(d_nb, d_as):
    filas = []   # lat_media / lon_media (centroide de los colegios del distrito) ya vienen de distritos_base
    # --- binomial negativa (todas las familias), efectos fijos de departamento
    dd = d_nb.dropna(subset=["lat_media"]).reset_index(drop=True)
    f = "y ~ " + " + ".join(COV + ["C(dpto)"])
    for etiqueta, kw in [("provincia", dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.prov)[0]})),
                         ("departamento", dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.dpto)[0]})),
                         ("dos vías (provincia y departamento)", dict(cov_type="cluster", cov_kwds={"groups": np.c_[pd.factorize(dd.prov)[0], pd.factorize(dd.dpto)[0]]}))]:
        try:
            m = smf.negativebinomial(f, data=dd).fit(disp=0, maxiter=300, **kw)
            for v in ["pobreza", "rural", "pct_autoid_indigena"]:
                filas.append({"modelo": "binomial negativa", "errores": etiqueta, "variable": v,
                              "coef": m.params[v], "irr": np.exp(m.params[v]), "ee": m.bse[v], "p": m.pvalues[v]})
        except Exception as e:  # noqa: BLE001
            print(f"NB {etiqueta}: {e}")
    m = smf.negativebinomial(f, data=dd).fit(disp=0, maxiter=300)
    alpha = m.params["alpha"]
    Xdf = m.model.exog
    names = m.model.exog_names
    mu = m.predict()
    s = Xdf * ((dd.y.to_numpy() - mu) / (1 + alpha * mu))[:, None]
    H = (Xdf * (mu / (1 + alpha * mu))[:, None]).T @ Xdf
    Hi = np.linalg.pinv(H)
    coords = dd[["lat_media", "lon_media"]].to_numpy()
    for corte in (60, 100):
        V = conley(Xdf, s, Hi, coords, corte)
        for v in ["pobreza", "rural", "pct_autoid_indigena"]:
            k = names.index(v)
            ee = np.sqrt(V[k, k]); z = m.params[v] / ee
            from scipy.stats import norm
            filas.append({"modelo": "binomial negativa", "errores": f"Conley {corte} km", "variable": v,
                          "coef": m.params[v], "irr": np.exp(m.params[v]), "ee": ee, "p": 2 * norm.sf(abs(z))})
    # --- asistencia (WLS), medida de acceso: % en desierto y log distancia
    da = d_as.dropna(subset=["lat_media"]).reset_index(drop=True)
    for x in ["desierto", "log_dist"]:
        dx = da.dropna(subset=[x, "asiste", "pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior"]).reset_index(drop=True)
        fx = f"asiste ~ {x} + {m10.CONTROLES}"
        for etiqueta, kw in [("provincia", dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(dx.prov)[0]})),
                             ("departamento", dict(cov_type="cluster", cov_kwds={"groups": pd.factorize(dx.dpto)[0]})),
                             ("dos vías (provincia y departamento)", dict(cov_type="cluster", cov_kwds={"groups": np.c_[pd.factorize(dx.prov)[0], pd.factorize(dx.dpto)[0]]}))]:
            r = smf.wls(fx, data=dx, weights=dx.pob_17_24).fit(**kw)
            filas.append({"modelo": "asistencia (WLS)", "errores": etiqueta, "variable": x,
                          "coef": r.params[x], "irr": np.nan, "ee": r.bse[x], "p": r.pvalues[x]})
        r = smf.wls(fx, data=dx, weights=dx.pob_17_24).fit()
        X = r.model.exog; wts = dx.pob_17_24.to_numpy()
        s = X * (wts * r.resid.to_numpy())[:, None]
        Hi = np.linalg.pinv((X * wts[:, None]).T @ X)
        k = r.model.exog_names.index(x)
        for corte in (60, 100):
            V = conley(X, s, Hi, dx[["lat_media", "lon_media"]].to_numpy(), corte)
            ee = np.sqrt(V[k, k])
            from scipy.stats import norm
            filas.append({"modelo": "asistencia (WLS)", "errores": f"Conley {corte} km", "variable": x,
                          "coef": r.params[x], "irr": np.nan, "ee": ee, "p": 2 * norm.sf(abs(r.params[x] / ee))})
    t = pd.DataFrame(filas)
    guardar(t.round(5), "errores_estandar_alternativos.csv", "tablas")
    print(t.round(4).to_string(index=False))


# ---------------------------------------------------------------------------- Oster
def oster(d_as):
    filas = []
    for x in ["desierto", "log_dist", "log_prog"]:
        dx = d_as.dropna(subset=[x, "asiste", "pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior"])
        corto = smf.wls(f"asiste ~ {x} + C(dpto)", data=dx, weights=dx.pob_17_24).fit()
        largo = smf.wls(f"asiste ~ {x} + {m10.CONTROLES}", data=dx, weights=dx.pob_17_24).fit()
        b0, r0, b1, r1 = corto.params[x], corto.rsquared, largo.params[x], largo.rsquared
        rmax = min(1.0, 1.3 * r1)
        delta = b1 * (r1 - r0) / ((b0 - b1) * (rmax - r1)) if (b0 - b1) * (rmax - r1) != 0 else np.nan
        bstar = b1 - (b0 - b1) * (rmax - r1) / (r1 - r0)
        filas.append({"acceso": x, "beta_sin_controles": b0, "R2_sin_controles": r0, "beta_con_controles": b1,
                      "R2_con_controles": r1, "R2max": rmax, "delta_para_beta_0": delta, "beta_corregido_delta1": bstar})
    t = pd.DataFrame(filas)
    guardar(t.round(4), "oster_asistencia.csv", "tablas")
    print(t.round(3).to_string(index=False))


def main():
    d_nb = m5.datos_distritales()
    for c in m5.EXTRA_PRINCIPAL:
        d_nb[c] = 100 * d_nb[c]
    d_nb = d_nb.dropna(subset=["y_Total", *COV]).copy()
    d_nb["y"] = d_nb.y_Total.round().astype(int)
    w = matriz_w(set(d_nb.ubigeo))
    d_nb = d_nb[d_nb.ubigeo.isin(w.id_order)]
    sar_sem(d_nb, w)
    d_as = m10.datos()
    errores_alternativos(d_nb, d_as)
    oster(d_as)


if __name__ == "__main__":
    main()
