"""Paso 5 – Réplica del diseño de Hillman (2016): modelos de conteo de la oferta cercana.

Unidad: distrito (con alumnos de 5.º grado). Variable dependiente: número de programas de pregrado de cada familia
a menos de 60 km (promedio ponderado de los colegios del distrito, redondeado a entero).
Regresión binomial negativa con efectos fijos de departamento y errores agrupados por provincia; se reportan IRR.

Covariables: log de alumnos de 5.º, pobreza monetaria 2018 (punto medio), % de 5.º en colegios rurales y,
si existe datos/interim/censo2017_distrital.csv, % lengua originaria, % autoidentificación indígena y % sin superior.

Salidas: salidas/tablas/modelos_conteo_irr.csv y manuscrito/tablas/tabla_irr.tex
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)

from utils import FAMILIAS, FAMILIAS_EN, guardar, ruta


def datos_distritales(radio: int = 60) -> pd.DataFrame:
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0)]
    y = acc.pivot_table(index="ubigeo", columns="familia", values=f"n_prog_{radio}km")
    y.columns = [f"y_{c}" for c in y.columns]
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str, "dpto": str, "prov": str})
    d = base.merge(y.reset_index(), on="ubigeo", how="inner")
    cen = ruta("censo2017_agregado")
    if cen.exists():
        d = d.merge(pd.read_csv(cen, dtype={"ubigeo": str}), on="ubigeo", how="left")
    d = d[d.quinto > 0].copy()
    d["log_quinto"] = np.log(d.quinto)
    d["pobreza"] = d.pobreza_punto_medio
    d["rural"] = 100 * d.pct_rural_quinto
    return d


EXTRA_PRINCIPAL = ["pct_autoid_indigena", "pct_afro", "pct_sin_superior"]
EXTRA_ROBUSTEZ = ["pct_lengua_originaria", "pct_afro", "pct_sin_superior"]  # lengua materna en lugar de autoidentificación


def main(extra_vars=EXTRA_PRINCIPAL, sufijo="", efectos_fijos=True):
    d = datos_distritales()
    extra = [c for c in extra_vars if c in d]
    for c in extra:
        d[c] = 100 * d[c]
    x = " + ".join(["log_quinto", "pobreza", "rural", *extra, *(["C(dpto)"] if efectos_fijos else [])])
    filas = []
    for fam in ["Total", *FAMILIAS]:
        yv = f"y_{fam}"
        dd = d.dropna(subset=[yv, "pobreza", "rural", *extra]).copy()
        dd["y"] = dd[yv].round().astype(int)
        try:
            m = smf.negativebinomial(f"y ~ {x}", data=dd).fit(disp=0, maxiter=300, cov_type="cluster",
                                                                 cov_kwds={"groups": pd.factorize(dd.prov)[0]})
        except Exception as e:  # noqa: BLE001  (p. ej., familia sin variación)
            print(f"{fam}: no convergió ({e}); se usa Poisson con errores robustos")
            m = smf.glm(f"y ~ {x}", data=dd, family=sm.families.Poisson()).fit(cov_type="cluster",
                                                                               cov_kwds={"groups": pd.factorize(dd.prov)[0]})
        for v in ["log_quinto", "pobreza", "rural", *extra]:
            filas.append({"familia": fam, "variable": v, "irr": np.exp(m.params[v]), "ee": m.bse[v],
                          "p": m.pvalues[v], "n": int(m.nobs)})
    t = pd.DataFrame(filas)
    guardar(t, f"modelos_conteo_irr{sufijo}.csv", "tablas")
    ancho = t.assign(celda=t.apply(lambda r: f"{r.irr:.3f}{'***' if r.p < .01 else '**' if r.p < .05 else '*' if r.p < .1 else ''}", axis=1)
                     ).pivot(index="familia", columns="variable", values="celda")
    ancho.index = [FAMILIAS_EN.get(i, i) for i in ancho.index]
    ancho = ancho.rename(columns={"log_quinto": "Log students", "pobreza": "Poverty", "rural": "Rural",
                                  "pct_lengua_originaria": "Indig. tongue",
                                  "pct_autoid_indigena": "Indigenous",
                                  "pct_sin_superior": "No higher ed.",
                                  "pct_afro": "Afro-Peruvian"})
    orden = [c for c in ["Log students", "Poverty", "Rural", "Indigenous", "Indig. tongue", "Afro-Peruvian", "No higher ed."] if c in ancho]
    ancho = ancho[orden]
    ancho.index.name = "Field"
    ancho.columns.name = None
    (ruta("manuscrito_tablas") / f"tabla_irr{sufijo}.tex").write_text(ancho.to_latex(escape=True, column_format="p{3.8cm}" + "r" * ancho.shape[1]), encoding="utf-8")
    print(ancho.to_string())


if __name__ == "__main__":
    main()
    main(EXTRA_ROBUSTEZ, "_robustez_lengua")
    main(EXTRA_PRINCIPAL, "_sin_efectos_fijos", efectos_fijos=False)  # contraste entre departamentos
