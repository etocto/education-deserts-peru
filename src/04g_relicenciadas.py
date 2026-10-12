"""Paso 4g – Escenario con las universidades relicenciadas.

Seis universidades a las que se denegó la licencia en la primera ronda obtuvieron una nueva licencia entre 2024 y
2026 (posterior al corte de datos abiertos de SUNEDU, 28-08-2024): Católica Los Ángeles de Chimbote (053),
José Carlos Mariátegui (048), Científica del Perú (050), Peruana de las Américas (078), Privada de Trujillo (090)
e Interamericana para el Desarrollo (117).

Como no hay datos abiertos de sus sedes y programas actuales, se agrega su oferta de 2018 (carnés) a la oferta
licenciada de 2024. Es una COTA SUPERIOR: su oferta relicenciada no puede ser más amplia que la de 2018 si
cerraron filiales o programas. Con el escenario post (cota inferior) acota el sistema actual.

Salida: salidas/tablas/escenario_relicenciadas.csv (% de alumnos de 5.º en desierto por familia en post,
post + relicenciadas y pre) y distritos que salen del desierto total.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import CFG, FAMILIAS, guardar, haversine_matriz, ruta

RELICENCIADAS = ["053", "048", "050", "078", "090", "117"]


def main():
    radio = CFG["accesibilidad"]["radio_desierto_km"]
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype={"COD_MOD": str, "ANEXO": str, "ubigeo": str})
    col["quinto"] = col.quinto.fillna(0)
    q = col.quinto.to_numpy()
    post = pd.read_csv(ruta("processed") / "oferta_locales.csv").dropna(subset=["lat", "lon"])
    den = pd.read_csv(ruta("processed") / "oferta_locales_denegadas_2018.csv", dtype={"codigo_entidad": str}).dropna(subset=["lat", "lon"])
    rel = den[den.codigo_entidad.isin(RELICENCIADAS)]
    print(f"relicenciadas: {rel.codigo_entidad.nunique()} universidades, {len(rel)} sedes/filiales, {int(rel.n_programas.sum())} programas (2018)")
    escenarios = {"post": post, "post_relicenciadas": pd.concat([post, rel], ignore_index=True),
                  "pre": pd.concat([post, den], ignore_index=True)}
    filas, desierto_total = [], {}
    for nombre, loc in escenarios.items():
        cerca = haversine_matriz(col.lat, col.lon, loc.lat, loc.lon) <= radio
        for fam in ["Total", *FAMILIAS]:
            oferta = ((loc.n_programas if fam == "Total" else loc[fam]).fillna(0) > 0).to_numpy()
            des = ~(cerca & oferta).any(axis=1)
            filas.append({"escenario": nombre, "familia": fam, "pct_desierto": 100 * q[des].sum() / q.sum(),
                          "quinto_desierto": q[des].sum()})
            if fam == "Total":
                desierto_total[nombre] = des
    t = pd.DataFrame(filas).pivot(index="familia", columns="escenario", values="pct_desierto")[list(escenarios)]
    t["recuperado_pp"] = t.post - t.post_relicenciadas
    t["share_perdida_recuperada"] = 100 * t.recuperado_pp / (t.post - t.pre).replace(0, np.nan)
    guardar(t.round(4).reset_index(), "escenario_relicenciadas.csv", "tablas")
    print(t.round(1).to_string())
    salen = desierto_total["post"] & ~desierto_total["post_relicenciadas"]
    print(f"alumnos que salen del desierto total con las relicenciadas: {int(q[salen].sum())} "
          f"en {col.ubigeo[salen].nunique()} distritos")


if __name__ == "__main__":
    main()
