"""Paso 3 – Covariables distritales del Censo 2017 (INEI, REDATAM CPV2017DI).

Lee las listas de áreas por distrito de datos/raw/censo2017/redatam/ (ver LEEME_fuente.txt) y calcula:
  pct_lengua_originaria   % de 15-29 años cuya lengua materna es quechua, aimara u otra lengua indígena u originaria
  pct_autoid_indigena     % de 15-29 años que se autoidentifica como quechua, aimara, nativo amazónico u otro pueblo indígena
  pct_afro                % de 15-29 años que se autoidentifica como afroperuano
  pct_sin_superior        % de 25+ años sin educación superior (equivalente a "% < BA" de Hillman, 2016)
  pct_rural_censal        % de la población en área rural censal
  pob_17_24               población de 17 a 24 años
  tasa_asiste_17_24       % de 17-24 años que asiste a alguna institución educativa
  pct_asiste_otro_distrito % de los que asisten cuya institución está en otro distrito (movilidad)
Los porcentajes excluyen de los denominadores las respuestas "No sabe / No responde" solo en autoidentificación.
Salida: datos/interim/censo2017_distrital.csv
"""
from __future__ import annotations

import pandas as pd

from utils import RAIZ, guardar

R = RAIZ / "datos/raw/censo2017/redatam"
NO_ORIGINARIAS = {"Castellano", "Portugués", "Otra lengua extranjera", "Lengua de señas peruanas", "No escucha, ni habla", "Total"}
AUTOID_IND = ["Quechua", "Aimara", "Nativo o indígena de la amazonía", "Parte de otro pueblo indígena u originario"]
SUPERIOR = ["Superior no universitaria incompleta", "Superior no universitaria completa",
            "Superior universitaria incompleta", "Superior universitaria completa", "Maestría / Doctorado"]


def leer(nombre):
    return pd.read_csv(R / f"cpv2017_{nombre}.csv", dtype={"ubigeo": str}).set_index("ubigeo")


def main():
    len_ = leer("lengua_15_29")
    orig = [c for c in len_.columns[2:] if c not in NO_ORIGINARIAS]
    aut = leer("autoid_15_29")
    afro = [c for c in aut.columns if c.startswith("Negro")]
    nsnr = [c for c in aut.columns if c.startswith("No sabe")]
    niv = leer("nivel_25mas")
    area = leer("area_total")
    a1724 = leer("area_17_24")
    asis = leer("asiste_lugar_17_24")
    out = pd.DataFrame(index=len_.index)
    out["pob_15_29"] = len_.Total
    out["pct_lengua_originaria"] = len_[orig].sum(axis=1) / len_.Total
    den = aut.Total - aut[nsnr].sum(axis=1)
    out["pct_autoid_indigena"] = aut[AUTOID_IND].sum(axis=1) / den
    out["pct_afro"] = aut[afro].sum(axis=1) / den
    out["pct_sin_superior"] = 1 - niv[SUPERIOR].sum(axis=1) / niv.Total
    out["pct_rural_censal"] = area["Rural censal"] / area.Total
    out["pob_total_2017"] = area.Total
    out["pob_17_24"] = a1724.Total
    out["tasa_asiste_17_24"] = asis.Total / a1724.Total
    out["pct_asiste_otro_distrito"] = asis["En otro distrito"] / asis.Total
    out = out.reset_index()
    guardar(out, "censo2017_distrital.csv", "interim")
    print(out.describe().T[["mean", "min", "max"]].round(3).to_string())
    print(f"lenguas originarias consideradas: {len(orig)}")


if __name__ == "__main__":
    main()
