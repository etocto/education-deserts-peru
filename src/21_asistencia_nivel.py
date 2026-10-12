"""Paso 21 – Asistencia a educación superior por nivel y definición del resultado (revisión, obs. 1).

El paso 10 usó como resultado tasa_asiste_17_24 = asistentes a CUALQUIER institución educativa / población de 17-24
años (C5P14 = 1, incluye secundaria). Aquí se define el resultado por nivel que cursa (C5P13NIV, archivo
cpv2017_asiste_nivel_17_24.csv):
  universitaria      superior universitaria incompleta o completa + maestría/doctorado
  no_universitaria   superior no universitaria incompleta o completa
  superior           universitaria + no universitaria
  cualquiera         definición del paso 10 (para comparar)
Numerador: jóvenes de 17-24 años que asisten a ese nivel, residentes en el distrito (residencia habitual, censo 2017).
Denominador: población de 17-24 años residente en el distrito.
Se reestima el modelo del paso 10 (WLS, efectos fijos de departamento, errores por provincia) para cada resultado con
la medida "% de alumnos de 5.º en desierto".

LIMITACIÓN que este paso NO resuelve: el resultado sigue medido por distrito de residencia; quien migró para estudiar
cuenta en el distrito de destino. La versión por distrito de ORIGEN requiere una consulta nueva al REDATAM con la
residencia habitual de hace 5 años (ver docs/REDATAM_asistencia_origen.md) y queda pendiente.

Salida: salidas/tablas/asistencia_por_nivel.csv
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from utils import RAIZ, guardar

R = RAIZ / "datos/raw/censo2017/redatam"


def main():
    m10 = importlib.import_module("10_asistencia_moran")
    d = m10.datos()
    niv = pd.read_csv(R / "cpv2017_asiste_nivel_17_24.csv", dtype={"ubigeo": str}).set_index("ubigeo")
    univ = niv[["Superior universitaria incompleta", "Superior universitaria completa", "Maestría / Doctorado"]].sum(axis=1)
    nouniv = niv[["Superior no universitaria incompleta", "Superior no universitaria completa"]].sum(axis=1)
    d = d.set_index("ubigeo")
    den = d.pob_17_24
    d["universitaria"] = 100 * univ.reindex(d.index) / den
    d["no_universitaria"] = 100 * nouniv.reindex(d.index) / den
    d["superior"] = d.universitaria + d.no_universitaria
    d["cualquiera"] = d.asiste
    d = d.reset_index()
    filas = []
    for y in ["cualquiera", "superior", "universitaria", "no_universitaria"]:
        dd = d.dropna(subset=[y, "desierto", "pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior"])
        m = smf.wls(f"{y} ~ desierto + {m10.CONTROLES}", data=dd, weights=dd.pob_17_24).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.prov)[0]})
        filas.append({"resultado": y, "media_ponderada": np.average(dd[y], weights=dd.pob_17_24),
                      "coef_desierto_pp_por_punto": m.params["desierto"], "ee": m.bse["desierto"], "p": m.pvalues["desierto"],
                      "brecha_100_vs_0_pp": 100 * m.params["desierto"], "r2": m.rsquared, "n": int(m.nobs)})
    t = pd.DataFrame(filas)
    guardar(t.round(4), "asistencia_por_nivel.csv", "tablas")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
