"""Paso 30 – Cifras citadas en el texto del artículo que no figuran en sus tablas (seguimiento, N8).

Cada cifra se calcula aquí a partir de las salidas sin redondear y se redondea una sola vez (mitad hacia arriba):
porcentajes a un decimal y conteos a enteros. El resultado alimenta la última tabla del material suplementario, de
modo que cada número del texto tiene una tabla de respaldo y un archivo de origen en el repositorio.

Requiere los pasos 4, 4b, 4c, 4e, 4g, 15, 16, 25, 27 y 29. Salida: salidas/tablas/cifras_texto.csv
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from utils import guardar, ruta

T = ruta("tablas")
AMAZONIA = ["LORETO", "UCAYALI", "MADRE DE DIOS", "AMAZONAS", "SAN MARTIN"]


def pct(x) -> str:
    return str(Decimal(str(round(float(x), 6))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def ent(x) -> str:
    return f"{int(Decimal(str(round(float(x), 6))).quantize(Decimal('1'), rounding=ROUND_HALF_UP)):,}"


def main():
    filas = []
    add = lambda sec, cifra, valor, fuente: filas.append({"seccion": sec, "cifra": cifra, "valor": valor, "fuente": fuente})  # noqa: E731
    dt = {"COD_MOD": str, "ANEXO": str, "ubigeo": str}

    # --- 4.1 Datos (revisión v3, obs. m4): inventario de unidades y tabla de flujo (STROBE 13)
    col = pd.read_csv(ruta("processed") / "demanda_colegios.csv", dtype=dt)
    q5 = pd.to_numeric(col.quinto, errors="coerce").fillna(0)
    base_d = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    prog = pd.read_csv(ruta("processed") / "oferta_programas_pregrado.csv", dtype=str)
    locs = pd.read_csv(ruta("processed") / "oferta_locales.csv")
    ins = pd.read_csv(ruta("escale_superior_no_univ"), sep="|", dtype=str)
    flujo = [
        ("Secondary schools", "Active regular secondary schools with coordinates (2025 Census of Education)", len(col)),
        ("Secondary schools", "... with final-year enrolment", int((q5 > 0).sum())),
        ("Secondary schools", "Final-year students", int(q5.sum())),
        ("Secondary schools", "Schools in districts created after 2017 (reassigned by location)", int((col.ubigeo_censal != col.ubigeo_2025).sum()) if "ubigeo_censal" in col else np.nan),
        ("Districts", "Districts (2017 geography) with final-year students", int((base_d.quinto > 0).sum())),
        ("University supply", "Undergraduate programmes of licensed universities open to secondary graduates", len(prog)),
        ("University supply", "... classified by SUNEDU class / by keyword / corrected manually", " / ".join(f"{int((prog.familia_origen == k).sum()):,}" for k in ["N2", "regla", "manual"])),
        ("University supply", "Campuses with coordinates (public / private)", f"{len(locs)} ({int((locs.gestion == 'PUBLICO').sum())} / {int((locs.gestion == 'PRIVADO').sum())})"),
        ("Non-university supply", "Higher education institutes (technological / pedagogical / arts)", f"{len(ins)} ({int((ins.D_NIV_MOD == 'Instituto Superior Tecnológico').sum())} / {int((ins.D_NIV_MOD == 'Instituto Superior Pedagógico').sum())} / {int((ins.D_NIV_MOD == 'Escuela Formación Artística').sum())})"),
        ("Non-university supply", "... public", int(ins.D_GESTION.str.startswith("Pública").sum())),
    ]
    den0 = pd.read_csv(ruta("processed") / "oferta_locales_denegadas_2018.csv", dtype={"codigo_entidad": str, "ubigeo": str})
    flujo += [("Denied universities (2018)", "Universities with a denied licence in the open data (28 August 2024)", 45),
              ("Denied universities (2018)", "... with records in the 2018 identity-card register", int(den0.codigo_entidad.nunique())),
              ("Denied universities (2018)", "Headquarters and branches located (districts)", f"{len(den0)} ({den0.ubigeo.nunique()})"),
              ("Denied universities (2018)", "... headquarters / branches (two universities have only branches in the register)",
               f"{int(den0.id_local.str.endswith('-SEDE').sum())} / {int((~den0.id_local.str.endswith('-SEDE')).sum())}"),
              ("Denied universities (2018)", "Undergraduate programmes at those sites", int(den0.n_programas.sum()))]
    fl = pd.DataFrame(flujo, columns=["bloque", "unidad", "n"])
    guardar(fl, "flujo_unidades.csv", "tablas")
    add("4.1", "Active regular secondary schools with coordinates", ent(len(col)), "demanda_colegios.csv")
    add("4.1", "Undergraduate programmes of licensed universities", ent(len(prog)), "oferta_programas_pregrado.csv")
    add("4.1", "... without an official field class: assigned by keyword / manually",
        f"{int((prog.familia_origen == 'regla').sum())} / {int((prog.familia_origen == 'manual').sum())}", "oferta_programas_pregrado.csv")
    add("4.1", "Campuses of licensed universities (public / private)", f"{len(locs)} ({int((locs.gestion == 'PUBLICO').sum())} / {int((locs.gestion == 'PRIVADO').sum())})", "oferta_locales.csv")
    add("4.1", "Higher education institutes (technological / pedagogical / arts; public)",
        f"{len(ins)} ({int((ins.D_NIV_MOD == 'Instituto Superior Tecnológico').sum())} / {int((ins.D_NIV_MOD == 'Instituto Superior Pedagógico').sum())} / {int((ins.D_NIV_MOD == 'Escuela Formación Artística').sum())}; {int(ins.D_GESTION.str.startswith('Pública').sum())})", "padron superior no universitaria (ESCALE)")
    cal = ruta("tablas") / "calibracion_pares.csv"
    if cal.exists():
        from scipy.stats import spearmanr
        c = pd.read_csv(cal)
        add("4.2", "Spearman correlation, uncalibrated modelled vs. Google Maps times (37 pairs)",
            f"{spearmanr(c.modelado_sin_calibrar, c.tiempo_observado_min).statistic:.2f}", "calibracion_pares.csv")

    # --- 3. Contexto y datos: oferta de las denegadas en 2018
    den = pd.read_csv(ruta("processed") / "oferta_locales_denegadas_2018.csv", dtype={"codigo_entidad": str, "ubigeo": str})
    add("4.3", "Headquarters and branches of denied universities located (2018)", ent(len(den)), "oferta_locales_denegadas_2018.csv")
    add("4.3", "Denied universities with 2018 records", ent(den.codigo_entidad.nunique()), "oferta_locales_denegadas_2018.csv")
    add("4.3", "Districts with those sites", ent(den.ubigeo.nunique()), "oferta_locales_denegadas_2018.csv")
    add("4.3", "Undergraduate programmes at those sites (2018)", ent(den.n_programas.sum()), "oferta_locales_denegadas_2018.csv")
    fil = pd.read_csv(T / "filiales_licenciadas_2018_vs_2024.csv")
    con = fil[fil.filiales_2018 > 0]
    add("4.3", "Licensed universities with branches in 2018", ent(len(con)), "filiales_licenciadas_2018_vs_2024.csv")
    add("4.3", "... of which in fewer departments in 2024", ent(con.menos_departamentos_en_2024.sum()), "filiales_licenciadas_2018_vs_2024.csv")

    # --- 5.1 Umbral y tiempos
    acc = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt, usecols=["COD_MOD", "ANEXO", "familia", "escenario", "factor_sinuosidad", "dist_min_km", "quinto"])
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0) & (acc.familia == "Total")]
    tie = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt)
    tie = tie[tie.familia == "Total"]
    g = acc.merge(tie[["COD_MOD", "ANEXO", "t_min"]], on=["COD_MOD", "ANEXO"])
    g = g[g.dist_min_km.between(55, 65) & np.isfinite(g.t_min)]
    add("4.2", "Median modelled travel time, connected schools whose nearest programme is 55--65 km away (minutes)", ent(g.t_min.median()), "accesibilidad_colegios.csv; tiempos_colegios.csv")
    nc = pd.read_csv(T / "colegios_no_conectados.csv")
    am = nc[nc.departamento.isin(AMAZONIA)]
    td = pd.read_csv(T / "topografico_departamentos.csv").set_index("departamento")
    add("5.1", "Topographic-desert students in Loreto, unconnected schools", ent(td.loc["LORETO", "alumnos_no_conectados"]), "topografico_departamentos.csv")
    for d in td.sort_values("alumnos", ascending=False).index[:1]:   # solo el departamento citado en el texto
        add("5.1", f"Topographic-desert students in {d.title().replace('Ancash', 'Áncash')}", ent(td.loc[d, "alumnos"]), "topografico_departamentos.csv")

    # --- 5.3 Orígenes por departamento
    k = ["COD_MOD", "ANEXO", "ubigeo", "familia"]
    a = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt, usecols=[*k, "quinto", "escenario", "factor_sinuosidad", "desierto"])
    a = a[(a.factor_sinuosidad == 1.0) & (a.familia == "Total")].pivot_table(index=k + ["quinto"], columns="escenario", values="desierto").reset_index()
    t2 = pd.read_csv(ruta("processed") / "tiempos_colegios.csv", dtype=dt, usecols=[*k, "desierto_t"])
    d = a.merge(t2, on=k).merge(pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})[["ubigeo", "departamento"]], on="ubigeo")
    D, Dp, Tt = d.post == 1, d.pre == 1, d.desierto_t == 1
    d["structural"] = D & Dp & Tt
    d["lic_ambos"] = D & ~Dp & Tt
    por_dep = d.groupby("departamento").apply(lambda x: 100 * x.quinto[x.structural].sum() / x.quinto.sum())
    for dep_ in ["MADRE DE DIOS", "LORETO"]:
        add("5.3", f"Structural deserts, share of final-year students in {dep_.title().replace(' De ', ' de ')} (%)", pct(por_dep[dep_]), "accesibilidad_colegios.csv; tiempos_colegios.csv")

    # --- 5.3 Efecto del licenciamiento
    f = pd.read_csv(T / "efecto_licenciamiento_familias.csv").set_index("familia")
    add("5.3", "Fewer programmes within 60 km, licensed vs. reconstructed, average student (%)", pct(f.loc["Total", "caida_pct_oferta_60km"]), "efecto_licenciamiento_familias.csv")
    add("5.3", "Students with at least one programme within 60 km present only in the reconstructed scenario (%)", pct(f.loc["Total", "pct_quinto_con_perdida"]), "efecto_licenciamiento_familias.csv")
    dp = pd.read_csv(T / "efecto_licenciamiento_departamentos.csv").set_index("departamento")
    for dep_ in ["MOQUEGUA", "PUNO"]:
        add("5.3", f"Lower nearby supply in the licensed scenario, {dep_.title()} (%)", pct(dp.loc[dep_, "caida_pct"]), "efecto_licenciamiento_departamentos.csv")
    nd = pd.read_csv(T / "efecto_licenciamiento_distritos.csv", dtype={"ubigeo": str})
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    base = base[base.quinto > 0]
    add("5.3", "Districts with schools that have no licensed programme within 60 km but had one in the reconstructed scenario", ent(len(nd)), "efecto_licenciamiento_distritos.csv")
    add("5.3", "Mean poverty of those districts (%)", pct(nd.pobreza_punto_medio.mean()), "efecto_licenciamiento_distritos.csv")
    add("5.3", "Mean poverty of the other districts (%)", pct(base[~base.ubigeo.isin(nd.ubigeo)].pobreza_punto_medio.mean()), "distritos_base.csv")
    r = pd.read_csv(T / "escenario_relicenciadas.csv").set_index("familia")
    rel = den[den.codigo_entidad.isin(["053", "048", "050", "078", "090", "117"])]
    add("5.3", "Campuses of the six relicensed universities in 2018", ent(len(rel)), "oferta_locales_denegadas_2018.csv")
    add("5.3", "Programmes of the six relicensed universities in 2018", ent(rel.n_programas.sum()), "oferta_locales_denegadas_2018.csv")

    # --- 6.2 Región natural de los desiertos de educación y ciencias naturales (paso 36)
    rel = ruta("processed") / "relieve_colegios.csv"
    if rel.exists():
        r = pd.read_csv(rel, dtype=dt, usecols=["COD_MOD", "ANEXO", "region_natural"])
        a2 = pd.read_csv(ruta("processed") / "accesibilidad_colegios.csv", dtype=dt, usecols=["COD_MOD", "ANEXO", "familia", "escenario", "factor_sinuosidad", "desierto", "quinto"])
        a2 = a2[(a2.escenario == "post") & (a2.factor_sinuosidad == 1.0)].merge(r, on=["COD_MOD", "ANEXO"])
        for fam, en in [("Educación", "education"), ("Ciencias naturales, matemáticas e informática", "natural sciences and ICT")]:
            z = a2[(a2.familia == fam) & (a2.desierto == 1)]
            add("6.2", f"Students in {en} deserts (60 km) in highland or Amazonian schools (%)", pct(100 * z.quinto[z.region_natural != "costa"].sum() / z.quinto.sum()), "accesibilidad_colegios.csv; relieve_colegios.csv")
        z = a2[a2.familia == "Total"]
        add("6.2", "All final-year students in highland or Amazonian schools (%)", pct(100 * z.quinto[z.region_natural != "costa"].sum() / z.quinto.sum()), "relieve_colegios.csv")

    # --- 5.4 Institutos
    i = pd.read_csv(T / "institutos_en_desiertos.csv").set_index("familia")
    add("5.4", "Students in any-field deserts (60 km) with a higher education institute within 60 km (%)", pct(i.loc["Total", "pct_con_instituto"]), "institutos_en_desiertos.csv")
    add("5.4", "... with a public institute (%)", pct(i.loc["Total", "pct_con_instituto_publico"]), "institutos_en_desiertos.csv")
    add("5.4", "Students in education deserts (60 km) with a pedagogical institute within 60 km (%)", pct(i.loc["Educación", "pct_con_instituto"]), "institutos_en_desiertos.csv")
    add("5.4", "... with a public pedagogical institute (%)", pct(i.loc["Educación", "pct_con_instituto_publico"]), "institutos_en_desiertos.csv")
    add("5.4", "Students in arts deserts (60 km) with an arts school within 60 km (%)", pct(i.loc["Artes y humanidades", "pct_con_instituto"]), "institutos_en_desiertos.csv")

    t = pd.DataFrame(filas)
    clave = t.seccion.map(lambda x: 100 * int(x.split(".")[0]) + int(x.split(".")[1]))
    t = t.iloc[np.argsort(clave.to_numpy(), kind="stable")].reset_index(drop=True)   # en el orden del artículo
    guardar(t, "cifras_texto.csv", "tablas")
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
