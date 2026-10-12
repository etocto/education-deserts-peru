"""Paso 4e – Desiertos por tipo de gestión: ¿qué pasa si solo se cuenta la oferta pública o solo la privada?

Para cada familia de carreras calcula el % de alumnos de 5.º grado sin ningún programa a menos de 60 km
(línea recta, oferta licenciada 2024) con (a) toda la oferta, (b) solo universidades públicas y (c) solo privadas,
y el % de alumnos cuya única opción cercana es privada (hay privada a 60 km pero no pública).
También compara la pobreza media (ponderada por alumnos) de los que solo tienen opción privada.

Además: ¿cubren los institutos de educación superior no universitaria (padrón ESCALE 2025) los desiertos
universitarios? Para los alumnos en desierto se calcula el % con un instituto a menos de 60 km:
cualquier instituto, pedagógico (para el desierto de Educación) y escuela de formación artística (para Artes).

Salidas: salidas/tablas/desiertos_por_gestion.csv, salidas/tablas/institutos_en_desiertos.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import CFG, FAMILIAS, FAMILIAS_EN, guardar, haversine_matriz, ruta


def main():
    radio = CFG["accesibilidad"]["radio_desierto_km"]
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    col["quinto"] = col.quinto.fillna(0)
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    col = col.merge(base[["ubigeo", "pobreza_punto_medio"]], on="ubigeo", how="left")
    loc = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    cerca = haversine_matriz(col.lat, col.lon, loc.lat, loc.lon) <= radio
    q = col.quinto.to_numpy()
    pub = (loc.gestion == "PUBLICO").to_numpy()
    filas = []
    for fam in ["Total", *FAMILIAS]:
        oferta = ((loc.n_programas if fam == "Total" else loc[fam]).fillna(0) > 0).to_numpy()
        hay_pub = (cerca & (oferta & pub)).any(axis=1)
        hay_priv = (cerca & (oferta & ~pub)).any(axis=1)
        solo_priv = hay_priv & ~hay_pub
        pob = lambda m: np.average(col.pobreza_punto_medio[m], weights=q[m]) if q[m].sum() > 0 else np.nan  # noqa: E731
        ok = col.pobreza_punto_medio.notna().to_numpy()
        filas.append({"familia": fam,
                      "pct_desierto_total": 100 * q[~(hay_pub | hay_priv)].sum() / q.sum(),
                      "pct_sin_publica": 100 * q[~hay_pub].sum() / q.sum(),
                      "pct_sin_privada": 100 * q[~hay_priv].sum() / q.sum(),
                      "pct_solo_privada": 100 * q[solo_priv].sum() / q.sum(),
                      "pobreza_solo_privada": pob(solo_priv & ok),
                      "pobreza_con_publica": pob(hay_pub & ok)})
    t = pd.DataFrame(filas)
    guardar(t.round(4), "desiertos_por_gestion.csv", "tablas")
    print(t.round(1).to_string(index=False))

    orden = [*FAMILIAS, "Total"]
    x = t.set_index("familia").loc[orden, ["pct_desierto_total", "pct_sin_publica", "pct_sin_privada", "pct_solo_privada"]]
    x.index = [FAMILIAS_EN.get(f, "Any field") for f in x.index]
    x.columns = ["All supply", "Public only", "Private only", "Only a private option"]
    tex = x.round(1).to_latex(column_format="p{4.2cm}rrrr", float_format="%.1f")
    (ruta("manuscrito_tablas") / "tabla_gestion.tex").write_text(tex, encoding="utf-8")
    institutos(col, cerca, loc, radio)


def institutos(col, cerca_uni, loc, radio):
    ins = pd.read_csv(ruta("escale_superior_no_univ"), sep="|", dtype=str)
    ins = ins.assign(lat=pd.to_numeric(ins.NLAT_IE, errors="coerce"), lon=pd.to_numeric(ins.NLONG_IE, errors="coerce")).dropna(subset=["lat", "lon"])
    d = haversine_matriz(col.lat, col.lon, ins.lat, ins.lon) <= radio
    q = col.quinto.to_numpy()
    tipo = ins.D_NIV_MOD.to_numpy()
    publico = ins.D_GESTION.str.startswith("Pública").to_numpy()
    filas = []
    for fam, tipos in [("Total", None), ("Educación", ["Instituto Superior Pedagógico"]),
                       ("Artes y humanidades", ["Escuela Formación Artística"])]:
        oferta = ((loc.n_programas if fam == "Total" else loc[fam]).fillna(0) > 0).to_numpy()
        desierto = ~(cerca_uni & oferta).any(axis=1)
        sel = np.ones(len(ins), bool) if tipos is None else np.isin(tipo, tipos)
        con = (d & sel).any(axis=1)
        con_pub = (d & sel & publico).any(axis=1)
        filas.append({"familia": fam, "quinto_desierto": q[desierto].sum(),
                      "pct_con_instituto": 100 * q[desierto & con].sum() / q[desierto].sum(),
                      "pct_con_instituto_publico": 100 * q[desierto & con_pub].sum() / q[desierto].sum()})
    t = pd.DataFrame(filas)
    guardar(t.round(4), "institutos_en_desiertos.csv", "tablas")
    print(t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
