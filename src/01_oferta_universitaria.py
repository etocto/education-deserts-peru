"""Paso 1 – Oferta: locales universitarios de pregrado con sus programas, familia de carrera y gestión.

Salidas (datos/processed):
  oferta_programas_pregrado.csv   una fila por programa-local (universidades licenciadas, corte 2024-08-28)
  oferta_locales.csv              una fila por local con número de programas por familia (escenario post)
  oferta_locales_pre.csv          locales post + sedes principales de universidades con licencia denegada (escenario pre)
  control_familias_regla.csv      programas clasificados por palabra clave (clase N2 = NINGUNO) para revisión manual
"""
from __future__ import annotations

import pandas as pd

from utils import FAMILIAS, RAIZ, familia_programa, guardar, ruta


def main():
    p = pd.read_csv(ruta("sunedu_programas"), sep="|", dtype=str, encoding="latin1")
    p = p[p.TIPO_NIVEL_ACADEMICO == "PREGRADO"].copy()
    fam = p.apply(lambda r: familia_programa(r.NOMBRE_CLASE_PROGRAMA_N2, r.DENOMINACION_PROGRAMA), axis=1)
    p["familia"] = [f for f, _ in fam]
    p["familia_origen"] = [o for _, o in fam]
    # correcciones manuales revisadas (datos/manual/correcciones_familia.csv)
    corr = pd.read_csv(RAIZ / "datos/manual/correcciones_familia.csv")
    mapa = dict(zip(corr.denominacion_programa, corr.familia_corregida))
    m = p.DENOMINACION_PROGRAMA.isin(mapa)
    p.loc[m, "familia"] = p.loc[m, "DENOMINACION_PROGRAMA"].map(mapa)
    p.loc[m, "familia_origen"] = "manual"
    excl = p.familia == "EXCLUIR"
    print(f"  programas corregidos a mano: {m.sum()} | excluidos (complementación, no admiten egresados de secundaria): {excl.sum()}")
    p = p[~excl].copy()
    p["lat"] = pd.to_numeric(p.LATITUD_UBICACION, errors="coerce")
    p["lon"] = pd.to_numeric(p.LONGITUD_UBICACION, errors="coerce")
    p["gestion"] = p.TIPO_GESTION.str.upper().str.replace("Ú", "U")
    p["id_local"] = p.CODIGO_ENTIDAD + "-" + p.CODIGO_LOCAL
    guardar(p, "oferta_programas_pregrado.csv")
    guardar(p[p.familia_origen == "regla"][["NOMBRE", "DENOMINACION_PROGRAMA", "NOMBRE_CLASE_PROGRAMA_N2", "familia"]]
            .drop_duplicates(), "control_familias_regla.csv", "interim")

    # locales (escenario post)
    base = p.groupby("id_local").agg(codigo_entidad=("CODIGO_ENTIDAD", "first"), universidad=("NOMBRE", "first"),
                                     gestion=("gestion", "first"), departamento=("DEPARTAMENTO_LOCAL", "first"),
                                     provincia=("PROVINCIA_LOCAL", "first"), distrito=("DISTRITO_LOCAL", "first"),
                                     lat=("lat", "first"), lon=("lon", "first"), n_programas=("DENOMINACION_PROGRAMA", "size"))
    por_fam = p.pivot_table(index="id_local", columns="familia", values="DENOMINACION_PROGRAMA", aggfunc="count", fill_value=0)
    for f in FAMILIAS:
        if f not in por_fam:
            por_fam[f] = 0
    loc = base.join(por_fam[FAMILIAS]).reset_index()
    loc["escenario"] = "post"
    guardar(loc, "oferta_locales.csv")

    # escenario pre: se agregan las sedes principales de las universidades con licencia denegada.
    # Supuesto documentado: se desconoce su oferta por programa; se registran como local con oferta "general"
    # (n_programas = mediana de programas por local de las licenciadas) y sin desagregación por familia.
    lic = pd.read_csv(ruta("sunedu_licenciamiento"), sep="|", dtype=str)
    den = lic[lic.ESTADO_LICENCIAMIENTO == "LICENCIA DENEGADA"].copy()
    den = pd.DataFrame({
        "id_local": den.CODIGO_ENTIDAD + "-SEDE", "codigo_entidad": den.CODIGO_ENTIDAD, "universidad": den.NOMBRE,
        "gestion": den.TIPO_GESTION.str.upper().str.replace("Ú", "U"), "departamento": den.DEPARTAMENTO,
        "provincia": den.PROVINCIA, "distrito": den.DISTRITO,
        "lat": pd.to_numeric(den.LATITUD, errors="coerce"), "lon": pd.to_numeric(den.LONGITUD, errors="coerce"),
        "n_programas": int(loc.n_programas.median())})
    for f in FAMILIAS:
        den[f] = pd.NA
    den["escenario"] = "denegada"
    pre = pd.concat([loc, den], ignore_index=True)
    guardar(pre, "oferta_locales_pre.csv")

    print("\nResumen:")
    print(f"  programas de pregrado: {len(p):,} | locales: {len(loc):,} | universidades: {p.CODIGO_ENTIDAD.nunique()}")
    print(f"  clasificados por regla: {(p.familia_origen == 'regla').sum()} | a mano: {(p.familia_origen == 'manual').sum()} | sin clasificar: {(p.familia == 'Sin clasificar').sum()}")
    print(f"  universidades con licencia denegada agregadas al escenario pre: {len(den)} "
          f"(con coordenadas: {den.lat.notna().sum()})")
    print(p.familia.value_counts().to_string())


if __name__ == "__main__":
    main()
