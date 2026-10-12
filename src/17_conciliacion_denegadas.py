"""Paso 17 – Conciliación de las universidades con licencia denegada y oferta 2018 de las licenciadas (revisión, obs. 4).

1. Tabla de conciliación: las 45 universidades con licencia denegada en el archivo de SUNEDU (corte 2024-08-28),
   si tienen registro en los carnés 2018, número de sedes/filiales y programas ubicados, y si obtuvieron una nueva
   licencia entre 2024 y 2026. Las columnas fecha_resolucion, fecha_cese_efectivo y fuente quedan para completar
   con las resoluciones de SUNEDU (literatura/resoluciones_denegadas.csv lista las que ya se descargaron).
2. Dirección del sesgo del escenario "antes": para cada universidad LICENCIADA compara el número de filiales con
   carnés en 2018 con el número de locales en la oferta licenciada de 2024. Si una universidad licenciada cerró
   filiales durante el proceso, esa oferta no está en el escenario "antes" y los desiertos atribuibles al
   licenciamiento quedan subestimados (y los estructurales sobrestimados).

Salidas: salidas/tablas/conciliacion_denegadas.csv, salidas/tablas/filiales_licenciadas_2018_vs_2024.csv
"""
from __future__ import annotations

import re

import pandas as pd

from utils import RAIZ, guardar, ruta, sin_tildes

RELICENCIADAS = ["053", "048", "050", "078", "090", "117"]   # mismas que el paso 4g
SEDE = r"SEDE PRINCIPAL|SEDE CENTRAL|PRINCIPAL|UNICA|NO FIAL|LOCAL CENTRAL|NO ESPECIFICADO|SEDE UNICA"


def main():
    lic = pd.read_csv(ruta("sunedu_licenciamiento"), sep="|", dtype=str)
    den = lic[lic.ESTADO_LICENCIAMIENTO == "LICENCIA DENEGADA"]
    loc_d = pd.read_csv(ruta("processed") / "oferta_locales_denegadas_2018.csv", dtype={"codigo_entidad": str})
    agg = loc_d.groupby("codigo_entidad").agg(sedes_filiales_2018=("id_local", "size"), programas_2018=("n_programas", "sum"),
                                              carnes_2018=("carnes_2018", "sum"), distritos=("ubigeo", "nunique"))
    t = den[["CODIGO_ENTIDAD", "NOMBRE", "DEPARTAMENTO", "DISTRITO"]].rename(columns={"CODIGO_ENTIDAD": "codigo_entidad"})
    t = t.merge(agg, left_on="codigo_entidad", right_index=True, how="left")
    t["registro_carnes_2018"] = t.sedes_filiales_2018.notna()
    t["nueva_licencia_2024_2026"] = t.codigo_entidad.isin(RELICENCIADAS)
    res = pd.read_csv(RAIZ / "literatura/resoluciones_denegadas.csv")
    clave = lambda s: re.sub(r"[^A-Z ]", "", sin_tildes(s)).replace("UNIVERSIDAD", "").split()  # noqa: E731
    def buscar(n):
        k = set(clave(n))
        hits = [r.contenido for r in res.itertuples() if len(k & set(clave(r.universidad))) >= max(2, len(set(clave(r.universidad))) - 1)]
        return "; ".join(hits)
    t["resolucion_descargada"] = t.NOMBRE.map(buscar)
    t["fecha_resolucion"] = ""
    t["fecha_cese_efectivo"] = ""
    t["fuente"] = ""
    t = t.sort_values(["registro_carnes_2018", "carnes_2018"], ascending=[True, False])
    guardar(t, "conciliacion_denegadas.csv", "tablas")
    print(f"denegadas: {len(t)}; con registro 2018: {int(t.registro_carnes_2018.sum())}; "
          f"nueva licencia 2024-2026: {int(t.nueva_licencia_2024_2026.sum())}")
    print("sin registro en carnés 2018:", t.loc[~t.registro_carnes_2018, ["codigo_entidad", "NOMBRE", "DEPARTAMENTO"]].to_string(index=False))
    print("con resolución ya descargada:", int((t.resolucion_descargada != "").sum()))

    # 2. licenciadas: filiales 2018 vs locales 2024
    c = pd.read_csv(RAIZ / "datos/raw/sunedu/sunedu_carnes_universitarios_2018.csv", sep="|", encoding="latin1", dtype=str)
    ok = lic[lic.ESTADO_LICENCIAMIENTO == "LICENCIA OTORGADA"].CODIGO_ENTIDAD
    c = c[c.CODIGO.isin(ok)].copy()
    c["es_sede"] = c.NOMBRE_FILIAL.fillna("[NO ESPECIFICADO]").map(sin_tildes).str.contains(SEDE)
    f18 = c[~c.es_sede].groupby("CODIGO").NOMBRE_FILIAL.nunique().rename("filiales_2018")
    dep18 = c[~c.es_sede].groupby("CODIGO").DEPARTAMENTO_FILIAL.nunique().rename("departamentos_filiales_2018")
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv", dtype={"codigo_entidad": str})
    l24 = loc.groupby("codigo_entidad").agg(locales_2024=("id_local", "size"), departamentos_2024=("departamento", "nunique"))
    comp = pd.concat([f18, dep18], axis=1).join(l24, how="outer").fillna(0).astype(int)
    comp["universidad"] = comp.index.map(lic.set_index("CODIGO_ENTIDAD").NOMBRE)
    comp["menos_departamentos_en_2024"] = comp.departamentos_2024 < comp.departamentos_filiales_2018
    guardar(comp.reset_index(names="codigo_entidad"), "filiales_licenciadas_2018_vs_2024.csv", "tablas")
    print(f"\nlicenciadas con filiales en 2018: {(comp.filiales_2018 > 0).sum()}; "
          f"con presencia en menos departamentos en 2024 que filiales en 2018: {int(comp.menos_departamentos_en_2024.sum())}")
    print(comp[comp.menos_departamentos_en_2024].sort_values("departamentos_filiales_2018", ascending=False)
          [["universidad", "departamentos_filiales_2018", "departamentos_2024"]].head(15).to_string())


if __name__ == "__main__":
    main()
