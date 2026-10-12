"""Paso 24 – Tablas LaTeX del manuscrito y del material suplementario a partir de salidas/tablas (versión revisada).

Cada tabla se escribe como un fragmento .tex en manuscrito_hep/tablas/ y se incluye con \\input{} en main.tex o
supplementary.tex, de modo que ninguna cifra se transcribe a mano. Formato numérico: separador de miles con coma
(4,781) y un decimal para porcentajes.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from utils import FAMILIAS, FAMILIAS_EN, RAIZ, ruta

T = ruta("tablas")
DEST = RAIZ / "manuscrito_hep/tablas"
ORDEN = ["Total", *FAMILIAS]
EN = {**FAMILIAS_EN, "Total": "Any field"}


DEP_EN = {"ANCASH": "Áncash", "APURIMAC": "Apurímac", "HUANUCO": "Huánuco", "JUNIN": "Junín", "SAN MARTIN": "San Martín",
          "MADRE DE DIOS": "Madre de Dios", "LA LIBERTAD": "La Libertad"}


def dep(x):
    x = str(x).upper()
    return DEP_EN.get(x, x.title())


def n(x):
    return "--" if pd.isna(x) else f"{x:,.0f}"


def p1(x):
    return "--" if pd.isna(x) else f"{x + 1e-9:.1f}"


def p3(x):
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def escribir(nombre, filas, cabecera, alineacion, nota=None, primera_regla=1, ajustar=True):
    """Requiere \\usepackage{adjustbox} y {makecell} en el documento.
    filas: lista de listas de celdas ya formateadas; primera_regla: tras cuántas filas va \\midrule (0 = ninguna)."""
    alineacion = alineacion.replace("p{", r">{\raggedright\arraybackslash}p{")
    lin = ([r"\begin{adjustbox}{max width=\textwidth}"] if ajustar else [r"\setlength{\tabcolsep}{4pt}"]) + [r"\begin{tabular}{" + alineacion + "}", r"\toprule", " & ".join(cabecera) + r" \\", r"\midrule"]
    for i, f in enumerate(filas):
        lin.append(" & ".join(f) + r" \\")
        if primera_regla and i + 1 == primera_regla:
            lin.append(r"\midrule")
    lin += [r"\bottomrule", r"\end{tabular}"] + ([r"\end{adjustbox}"] if ajustar else [])
    if nota:
        lin.append(r"\par\smallskip\parbox{\linewidth}{\footnotesize " + nota + "}")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / f"{nombre}.tex").write_text("\n".join(lin) + "\n", encoding="utf-8")
    print("  ->", nombre)


def main():
    tot = 537451
    # ---------------- Tabla 2: campos, distancia, tiempo y tabla cruzada
    c = pd.read_csv(T / "cruce_distancia_tiempo.csv").set_index("familia").reindex(ORDEN)
    cal = pd.read_csv(T / "calibracion_velocidades.csv").set_index("modelo")
    fac = cal.loc["calibrado (LOOCV)", "factor"]
    assert (c.distance_only == 0).all(), "hay alumnos a más de 60 km pero a 90 min o menos: restaurar la columna"
    escribir("tab2_campos", [[EN[f], p1(r.pct_distance), p1(r.pct_time), n(r.time_only)]
                             for f, r in c.iterrows()],
             [r"\textbf{Field}", r"\textbf{\makecell{None within\\60 km (\%)}}", r"\textbf{\makecell{None within\\90 min (\%)}}",
              r"\textbf{\makecell{Within 60 km,\\beyond 90 min}}"],
             r"p{3.9cm}rrr", ajustar=False, nota=
             rf"Last column: students within 60 km but beyond 90 min; no student is beyond 60 km but within 90 min. Students are final-year secondary students (n = 537,451). Travel times on the OpenStreetMap road network with road speeds calibrated against Google Maps travel times (factor {fac:.2f}; Supplementary Table [[N:tab:calibracion]]) and navigable rivers in five Amazonian departments; the 235 schools (2,881 students) not connected to the network count as beyond 90 min. ICT: information and communication technology.")

    # ---------------- Tabla 3: sensibilidad al radio
    s = pd.read_csv(T / "sensibilidad_radio.csv").set_index("familia").reindex(ORDEN)
    cal = pd.read_csv(T / "calibracion_radio_tiempo.csv").set_index("familia").reindex(ORDEN)
    escribir("tab3_radio", [[EN[f], *[p1(r[str(k)]) for k in [30, 45, 60, 75, 90]], f"{int(cal.loc[f, 'radio_equivalente_km'])}"]
                            for f, r in s.iterrows()],
             [r"\textbf{Field}", r"\textbf{30 km}", r"\textbf{45 km}", r"\textbf{60 km}", r"\textbf{75 km}", r"\textbf{90 km}",
              r"\textbf{\makecell{Radius\\equal to\\90 min}}"],
             r"p{3.9cm}rrrrrr", ajustar=False, nota=r"Share of final-year students (\%) with no licensed programme of the field within each straight-line radius. Last column: radius at which the straight-line desert share equals the 90-minute travel-time share.")

    # ---------------- Tabla 4: proveedores
    g = pd.read_csv(T / "desiertos_por_gestion.csv").set_index("familia").reindex(ORDEN)
    escribir("tab4_proveedores", [[EN[f], p1(r.pct_desierto_total), p1(r.pct_sin_publica), p1(r.pct_sin_privada), p1(r.pct_solo_privada)]
                                  for f, r in g.iterrows()],
             [r"\textbf{Field}", r"\textbf{All supply}", r"\textbf{Public only}", r"\textbf{Private only}", r"\textbf{\makecell{Only a private\\option nearby}}"],
             r"p{3.9cm}rrrr", ajustar=False, nota=r"Share of final-year students (\%) with no programme within 60 km counting all licensed campuses, only public or only private ones; last column: share with a private but no public programme of the field within 60 km.")

    # ---------------- Tabla 4 del artículo (tab5_origenes): orígenes en categorías excluyentes + licenciamiento neto
    u = pd.read_csv(T / "origenes_unificados.csv")
    ln = pd.read_csv(T / "licenciamiento_neto_familias.csv")
    ut = u[u.familia == "Total"].set_index("origen")
    etiqueta = {"structural": "Structural", "licensing": "Licensing-related", "topographic": "Topographic",
                "distance_only": "Beyond 60 km, within 90 min", "fuera_de_desierto": "Not in a desert"}
    filas = [[r"\multicolumn{6}{l}{\textit{A. Any field: mutually exclusive groups (licensed supply, 2024)}}"]]
    for cat in ["structural", "licensing", "topographic", "distance_only"]:
        r = ut.loc[cat]
        if r.alumnos == 0:
            continue
        filas.append([r"\quad " + etiqueta[cat], n(r.alumnos), p1(r.pct_quinto), p1(r.poverty), p1(r.rural), p1(r.indigenous)])
    r = ut.loc["fuera_de_desierto"]
    filas.append([r"\quad " + etiqueta["fuera_de_desierto"], n(r.alumnos), p1(100 * r.alumnos / tot), p1(r.poverty), p1(r.rural), p1(r.indigenous)])
    filas.append([r"\multicolumn{6}{l}{\textit{B. Licensing-related by distance: 2024 supply / adding six relicensed universities}}"])
    filas.append(["", r"\textit{Students}", r"\textit{Share (\%)}", r"\multicolumn{3}{r}{\textit{Main department (\%)}}"])
    for fam in ["Total", "Servicios", "Educación", "Ciencias naturales, matemáticas e informática", "Ciencias sociales, periodismo y derecho"]:
        a24 = ln[(ln.familia == fam) & (ln.version == "v2024")].iloc[0]
        arel = ln[(ln.familia == fam) & (ln.version == "vrel")].iloc[0]
        corto = {"Ciencias naturales, matemáticas e informática": "Natural sciences and ICT", "Ciencias sociales, periodismo y derecho": "Social sciences and law"}
        filas.append([r"\quad " + corto.get(fam, EN[fam]), f"{n(a24.alumnos_licenciamiento)} / {n(arel.alumnos_licenciamiento)}",
                      f"{p1(a24.pct_quinto_nacional)} / {p1(arel.pct_quinto_nacional)}",
                      r"\multicolumn{3}{r}{" + dep(arel.departamento_principal) + f" ({p1(arel.pct_en_departamento_principal)})}}"])
    escribir("tab5_origenes", filas,
             [r"\textbf{Group}", r"\textbf{Students}", r"\textbf{\makecell{Share of all\\students (\%)}}", r"\textbf{Poverty}", r"\textbf{Rural}", r"\textbf{Indigenous}"],
             r"p{4.2cm}rrrrr", ajustar=False, primera_regla=0, nota=
             r"Panel A: groups defined in Section 4.3. Poverty, rural and Indigenous: averages of the districts where students live, weighted by students (\%); they describe districts, not individuals. Panel B: licensing-related by distance (60 km) before / after adding the 2018 supply of the six universities relicensed in 2024--2026, an upper bound of their current supply; main department in the second version (Supplementary Tables [[N:tab:lic_dep]]--[[N:tab:lic_fam]]).")

    # ================= SUPLEMENTO
    o = pd.read_csv(T / "sensibilidad_radio_origenes.csv")
    o = o[o.familia == "Total"].set_index("radio_km")
    escribir("tabS_origenes_radio", [[f"{k} km", n(r.structural), n(r.topographic), n(r.licensing)] for k, r in o.iterrows()],
             [r"\textbf{Radius}", r"\textbf{Structural}", r"\textbf{Topographic}", r"\textbf{Licensing-related}"], "lrrr",
             r"Final-year students by origin of the any-field desert for each straight-line radius, with groups defined by distance: structural, beyond the radius with both licensed and reconstructed supply; licensing-related, beyond the radius only with licensed supply; topographic, within the radius but beyond 90 min. Unlike Section 4.3 of the article, structural and licensing-related do not require being beyond 90 min, so at radii below 60 km they include students within 90 min of a programme.", primera_regla=0)

    e = pd.read_csv(T / "efecto_red_fluvial.csv")
    e = e[e.familia == "Total"].set_index("escenario")
    lbl = {"sin_rios": "Roads only (no river network)", "rio_10.6": "Boat speed 10.6 km/h", "rio_18.0": "Boat speed 18.0 km/h",
           "vias_x0.75": r"Road speeds $\times$0.75", "vias_x1.25": r"Road speeds $\times$1.25"}
    escribir("tabS_red", [[lbl[k], n(r.colegios_que_cambian), n(r.alumnos_que_cambian)] for k, r in e.iterrows()],
             [r"\textbf{Scenario (vs. baseline)}", r"\textbf{Schools reclassified}", r"\textbf{Students reclassified}"], "lrr",
             r"Schools and final-year students whose any-field classification at 90 min changes relative to the baseline network (roads at assumed speeds plus rivers at 12 km/h).", primera_regla=0)

    nc = pd.read_csv(T / "colegios_no_conectados.csv")
    filas = [[dep(r.departamento), n(r.colegios), n(r.quinto)] for r in nc.head(6).itertuples()]
    filas.append(["Other departments", n(nc.colegios[6:].sum()), n(nc.quinto[6:].sum())])
    filas.append([r"\textbf{Total}", n(nc.colegios.sum()), n(nc.quinto.sum())])
    escribir("tabS_no_conectados", filas, [r"\textbf{Department}", r"\textbf{Schools}", r"\textbf{Final-year students}"], "lrr",
             primera_regla=len(filas) - 1)

    def minus(x):
        x = " " + x.title().replace(" S.A.C.", "").replace(" S.A.", "") + " "
        for _ in range(2):
            x = re.sub(r" (De|Del|La|Las|El|Y|Para|E) ", lambda m: " " + m.group(1).lower() + " ", x)
        return x.strip().replace(" Ii", " II")
    cdn = pd.read_csv(RAIZ / "datos/manual/denegadas_resoluciones.csv", dtype=str)
    cdn = cdn.sort_values("NOMBRE")
    def fecha(x):
        return "--" if pd.isna(x) else pd.Timestamp(x).strftime("%d/%m/%Y")
    def rcd(x):
        return "--" if pd.isna(x) else x.replace("-SUNEDU/CD", "").replace("-SUNEDU-CD", "")
    filas = [[minus(r.NOMBRE), dep(r.DEPARTAMENTO),
              "--" if pd.isna(r.sedes_filiales_2018) else n(float(r.sedes_filiales_2018)),
              "--" if pd.isna(r.programas_2018) else n(float(r.programas_2018)),
              f"{rcd(r.rcd_denegatoria)} ({fecha(r.fecha_publicacion)})",
              "--" if pd.isna(r.rcd_nueva_licencia) else f"{rcd(r.rcd_nueva_licencia)} ({pd.Timestamp(r.fecha_nueva_licencia).year})",
              fecha(r.fin_cese_reportado)] for r in cdn.itertuples()]
    escribir("tabS_conciliacion", filas, [r"\textbf{University}", r"\textbf{Dept.}", r"\textbf{\makecell{Sites\\2018}}", r"\textbf{\makecell{Progr.\\2018}}",
                                          r"\textbf{\makecell{Denial resolution\\(published)}}", r"\textbf{\makecell{New licence\\(year)}}",
                                          r"\textbf{\makecell{End of\\cessation}}"],
             r"p{5.4cm}lrrlll",
             r"Universities with status `licence denied' in SUNEDU's open licensing file (cut-off 28 August 2024). Sites and programmes: headquarters and branches with university identity cards issued in 2018; `--': no record. Denial resolution: Resolution of SUNEDU's Board of Directors (Consejo Directivo) and publication date, from SUNEDU's list of denied universities (consulted 12 October 2026) or, for the six universities licensed again, from \emph{El Peruano} and SUNEDU. New licence: resolution granting a new institutional licence. End of cessation: date reported by SUNEDU for the associative universities (SUNEDU report on universities in cessation, November 2025); for the others, the cessation period could not exceed two years from the semester after notification (Board Resolution 111-2018-SUNEDU/CD), later extended for specific cases (Board Resolutions 017-2023, 030-2024 and 0035-2025-SUNEDU/CD). Sources by university: repository file \path{datos/manual/denegadas_resoluciones.csv}.", primera_regla=0)

    d = pd.read_csv(T / "descriptivos_distritales.csv")
    escribir("tabS_descriptivos", [[r.Variable.replace("%", r"\%").replace("-", "--") if "ages" in r.Variable else r.Variable.replace("%", r"\%"),
                                    f"{r.Mean:,.2f}", f"{r.SD:,.2f}", f"{r.Min:,.2f}", f"{r.Max:,.2f}", n(r.n)] for r in d.itertuples()],
             [r"\textbf{Variable}", r"\textbf{Mean}", r"\textbf{SD}", r"\textbf{Min}", r"\textbf{Max}", r"\textbf{n}"], r"p{7cm}rrrrr",
             r"Unit: district with final-year students (2017 census geography). Programmes within 60 km: final-year-weighted mean of school counts in the district, rounded to an integer (dependent variable of the count models).", primera_regla=1)

    f = pd.read_csv(T / "datos_faltantes.csv")
    escribir("tabS_faltantes", [[r.Item, n(r.Units), n(r._3)] for r in f.itertuples()],
             [r"\textbf{Item}", r"\textbf{Units}", r"\textbf{Final-year students}"], r"p{9cm}rr", primera_regla=0)

    sk = pd.read_csv(T / "tipologia_silueta_k.csv")
    est = pd.read_csv(T / "tipologia_estabilidad.csv").iloc[0]
    escribir("tabS_silueta", [[r"Silhouette coefficient"] + [f"{r.silueta:.3f}" for r in sk.itertuples()]],
             [r"\textbf{Number of clusters ($k$)}"] + [rf"\textbf{{{int(r.k)}}}" for r in sk.itertuples()], "l" + "r" * len(sk),
             rf"Standardized variables, 50 initialisations. Stability of $k = 6$ over 100 subsamples of 80\% of districts: mean adjusted Rand index {est.ari_media:.3f} (5th percentile {est.ari_p5:.3f}).",
             primera_regla=0)

    a = pd.read_csv(T / "asistencia_por_nivel.csv").set_index("resultado")
    lab = {"cualquiera": "Any educational institution (includes secondary schooling)", "superior": "Higher education (university or non-university)",
           "universitaria": "University", "no_universitaria": "Non-university higher education"}
    escribir("tabS_asistencia", [[lab[k], p1(r.media_ponderada), p1(r.brecha_100_vs_0_pp), f"{100 * r.ee:.1f}", p3(r.p)] for k, r in a.iterrows()],
             [r"\textbf{Outcome: share of youth aged 17--24 attending}", r"\textbf{\makecell{Weighted\\mean (\%)}}", r"\textbf{\makecell{Gap, 100\%\\vs. 0\% (pp)}}",
              r"\textbf{SE}", r"\textbf{p}"], r"p{7cm}rrrr",
             r"Weighted least squares, one model per outcome; 1,827 districts; weights: population aged 17--24; department fixed effects; standard errors clustered by province. Exposure: share of final-year students in any-field deserts (scenario including the 2018 supply of denied universities). Outcomes are measured by place of usual residence in the 2017 Census. pp: percentage points; SE: standard error.",
             primera_regla=1)

    rf = pd.read_csv(RAIZ / "salidas/modelos/reforms_pliegues.csv").groupby("modelo")[["r2", "mae"]].agg(["mean", "std"])
    hp = pd.read_csv(RAIZ / "salidas/modelos/reforms_hiperparametros.csv").set_index("modelo")
    escribir("tabS_reforms", [[m, hp.loc[m, "hiperparametros"].replace("_", r"\_").replace("{", "").replace("}", "").replace("'", ""),
                               f"{rf.loc[m, ('r2', 'mean')]:.3f} ({rf.loc[m, ('r2', 'std')]:.3f})", f"{rf.loc[m, ('mae', 'mean')]:.3f} ({rf.loc[m, ('mae', 'std')]:.3f})"]
                              for m in ["Linear", "RandomForest", "XGBoost"] if m in rf.index],
             [r"\textbf{Model}", r"\textbf{Hyperparameters (fixed a priori)}", r"\textbf{$R^2$, mean (SD)}", r"\textbf{MAE, mean (SD)}"], r"lp{6.5cm}rr",
             r"Outcome: log(1 + distance to the nearest programme, km); 1,827 districts; eight predictors; five-fold cross-validation grouped by department (GroupKFold); metrics are means and standard deviations across the five test folds. No hyperparameter search was performed. MAE: mean absolute error.", primera_regla=0)
    imp = pd.read_csv(RAIZ / "salidas/modelos/reforms_importancia_grupos.csv")
    escribir("tabS_importancia", [[r.grupo, f"{r.mean:.3f}", f"{r.std:.3f}"] for r in imp.itertuples()],
             [r"\textbf{Predictor group (permuted jointly)}", r"\textbf{\makecell{Mean drop\\in $R^2$}}", r"\textbf{SD across folds}"], r"p{7.5cm}rr",
             r"Random forest; permutation importance computed on held-out folds (20 permutations per fold), permuting correlated predictors as one block.", primera_regla=0)

    pdv = pd.read_csv(T / "perfil_desiertos_vs_departamento.csv")
    pdv = pdv[pdv.origen == "licensing"].set_index("familia").reindex(ORDEN).dropna(subset=["alumnos"])
    escribir("tabS_perfil", [[EN[f], n(r.alumnos), dep(r.departamento_principal), p1(r.pct_en_departamento_principal), p1(r.poverty), p1(r.poverty_esperado_dep),
                              p1(r.indigenous), p1(r.indigenous_esperado_dep)] for f, r in pdv.iterrows()],
             [r"\textbf{Field}", r"\textbf{Students}", r"\textbf{\makecell{Main\\department}}", r"\textbf{\makecell{Share in main\\dept. (\%)}}",
              r"\textbf{Poverty}", r"\textbf{\makecell{Poverty,\\departments}}", r"\textbf{Indigenous}", r"\textbf{\makecell{Indigenous,\\departments}}"],
             r"p{4.2cm}rlrrrrr",
             r"Licensing-related deserts. Observed profile (student-weighted district averages, \%) and profile expected from the composition of all final-year students in the same departments, each department weighted by the group's students there.")
    tablas_fase1(tot)
    print("tablas escritas en", DEST.relative_to(RAIZ))


ETQ_VAR = {"pobreza": "Poverty", "rural": "Rural schools", "pct_autoid_indigena": "Indigenous", "pct_afro": "Afro-Peruvian",
           "pct_sin_superior": "Adults without higher education", "log_quinto": "Log final-year students"}


def estrellas(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def tablas_fase1(tot):
    """Tablas añadidas en la fase 1 del seguimiento (N2, N3, N4, N6, N8)."""
    # S1 regenerada: tiempos, umbrales y estabilidad de la clasificación
    st = pd.read_csv(T / "sensibilidad_tiempos.csv")
    st = st[st.escenario == "base"].pivot_table(index="familia", columns="umbral_min", values="pct_desierto").reindex(ORDEN)
    rb = pd.read_csv(T / "robustez_tiempos.csv").set_index("familia").reindex(ORDEN)
    escribir("tabS_sens_tiempo", [[EN[f], p1(st.loc[f, 60]), p1(st.loc[f, 90]), p1(st.loc[f, 120]),
                                   f"{p1(rb.loc[f, 'pct_min'])}--{p1(rb.loc[f, 'pct_max'])}", p1(rb.loc[f, "pct_estable"])] for f in ORDEN],
             [r"\textbf{Field}", r"\textbf{60 min}", r"\textbf{90 min}", r"\textbf{120 min}", r"\textbf{\makecell{90 min, range\\across speeds}}",
              r"\textbf{\makecell{Classified\\identically (\%)}}"], r"p{4.2cm}rrrrr",
             r"Columns 2--4: share of final-year students (\%) beyond each travel-time threshold at base speeds. Column 5: range of the 90-minute share across six speed scenarios: base, road speeds $\times$0.75 and $\times$1.25, boats at 10.6 and 18.0 km/h, and roads only (no river network). Column 6: share of students classified identically (in or out of a 90-minute desert) in all six scenarios.")

    # calibración de velocidades (paso 32)
    cv = pd.read_csv(T / "calibracion_velocidades.csv")
    etq = {"sin calibrar": "Uncalibrated speeds", "calibrado (dentro de muestra)": "Calibrated, in sample", "calibrado (LOOCV)": "Calibrated, leave-one-out"}
    escribir("tabS_calibracion", [[etq[r.modelo], f"{r.factor:.3f}", f"{r.error_medio_min:.1f}", f"{r.error_absoluto_medio_min:.1f}",
                                   p1(r.error_relativo_mediano_pct), p1(r.pct_dentro_30pct), f"{r.pearson:.3f}", p1(r.acuerdo_umbral_90min_pct)] for r in cv.itertuples()],
             [r"\textbf{Model}", r"\textbf{\makecell{Road speed\\factor}}", r"\textbf{\makecell{Mean\\error (min)}}", r"\textbf{\makecell{Mean abs.\\error (min)}}",
              r"\textbf{\makecell{Median rel.\\error (\%)}}", r"\textbf{\makecell{Within\\30\% (\%)}}", r"\textbf{Pearson}", r"\textbf{\makecell{Same class\\at 90 min (\%)}}"],
             r"lrrrrrrr", rf"{int(cv.pares.iloc[0])} school--campus pairs (stratified sample of 40; three without a road route in Google Maps are excluded). Observed: travel time by car in Google Maps, departure Monday 07:00 (consulted 11 October 2026). Modelled: same school and campus on the OpenStreetMap network. The factor multiplies all road speeds (not river speeds, transfers or walking) and minimizes squared errors of log times; leave-one-out estimates the factor without each pair in turn.", primera_regla=0)

    # verificación manual de 20 colegios (paso 33)
    vm = T / "verificacion_manual.csv"
    if vm.exists():
        v = pd.read_csv(vm, dtype={"COD_MOD": str})
        o = {"structural": "Structural", "licensing": "Licensing", "topographic": "Topographic", "none": "None", "sin ruta en Google": "--"}
        orden = {"licensing": 0, "structural": 1, "topographic": 2, "none": 3}
        v = v.assign(_o=v.origen.map(orden)).sort_values(["_o", "departamento", "distrito"])
        nom = lambda x: x.title().replace(" De ", " de ").replace(" Del ", " del ").replace("Ancash", "Áncash").replace("Junin", "Junín").replace("Huanuco", "Huánuco").replace("San Martin", "San Martín").replace("Curimana", "Curimaná").replace("Huasmin", "Huasmín")  # noqa: E731
        fl = lambda x, d=0: "--" if pd.isna(x) else f"{x:,.{d}f}"  # noqa: E731
        escribir("tabS_verificacion", [[nom(r.distrito) + " (" + nom(r.departamento) + ")", o[r.origen], f"{r.distancia_km:.1f}", f"{r.distancia_geodesica_km:.1f}",
                                        "yes" if r.mismo_local_mas_cercano else "no", fl(r.tiempo_modelado_al_local_min), fl(r.tiempo_google_min),
                                        "--" if pd.isna(r.error_rel_pct) else f"{r.error_rel_pct:+.0f}".replace("-", "$-$"), o[r.origen_recalculado], o[r.origen_con_google]]
                                       for r in v.itertuples()],
                 [r"\textbf{School district (department)}", r"\textbf{Origin}", r"\textbf{\makecell{Distance\\(km)}}", r"\textbf{\makecell{Geodesic\\check (km)}}",
                  r"\textbf{\makecell{Same nearest\\campus}}", r"\textbf{\makecell{Modelled\\time (min)}}", r"\textbf{\makecell{Google\\Maps (min)}}",
                  r"\textbf{\makecell{Error\\(\%)}}", r"\textbf{\makecell{Origin\\recomputed}}", r"\textbf{\makecell{Origin with\\Google time}}"],
                 r"p{4.4cm}lrrcrrrll", "Stratified sample of 20 schools (five per origin, any field; fixed seed), none of them in the calibration sample. Distance: straight-line distance to the nearest licensed campus in the analysis; geodesic check: distance on the WGS84 ellipsoid recomputed independently, with an exhaustive search for the nearest licensed campus and the nearest 2018 campus of a denied university. Modelled time: calibrated network time to that same campus. Google Maps: time by car, departure Monday 12 October 2026 at 07:00 (midpoint when a range is shown); two schools have no car route in Google Maps (Sechura, whose structural origin depends only on distance, and Moyobamba, whose topographic origin cannot be checked against Google Maps). Origin recomputed: from the geodesic distances and the modelled travel time; origin with Google time: the same, using the Google Maps time.", primera_regla=0)

    # N4: topográficos y colegios no conectados
    tc = pd.read_csv(T / "topografico_conectividad.csv")
    tc = tc[tc.origen == "topographic"]
    filas = []
    for f in ORDEN:
        a = tc[(tc.familia == f) & (tc.version == "todos")].iloc[0]
        b = tc[(tc.familia == f) & (tc.version == "sin_no_conectados")].iloc[0]
        c = tc[(tc.familia == f) & (tc.version == "solo_no_conectados")].iloc[0]
        filas.append([EN[f], n(a.alumnos), n(c.alumnos), p1(100 * c.alumnos / a.alumnos), p1(a.poverty), p1(b.poverty), p1(a.rural), p1(b.rural)])
    escribir("tabS_topografico", filas,
             [r"\textbf{Field}", r"\textbf{\makecell{Topographic\\students}}", r"\textbf{\makecell{In unconnected\\schools}}", r"\textbf{\%}",
              r"\textbf{\makecell{Poverty,\\all}}", r"\textbf{\makecell{Poverty, excl.\\unconnected}}", r"\textbf{\makecell{Rural,\\all}}",
              r"\textbf{\makecell{Rural, excl.\\unconnected}}"], r"p{3.8cm}rrrrrrr",
             r"Topographic deserts: programme within 60 km in straight line but beyond 90 min. Unconnected schools have no link to the OpenStreetMap network and count as beyond 90 min. Poverty and rural: averages of the districts where students live, weighted by students (\%).")

    # N3: grupos excluyentes por familia
    u = pd.read_csv(T / "origenes_unificados.csv")
    filas = []
    for f in ORDEN:
        x = u[u.familia == f].set_index("origen")
        assert x.loc["distance_only", "alumnos"] == 0
        filas.append([EN[f], *[n(x.loc[c, "alumnos"]) for c in ["structural", "licensing", "topographic"]], n(x.loc["union", "alumnos"])])
    escribir("tabS_origenes_completo", filas,
             [r"\textbf{Field}", r"\textbf{Structural}", r"\textbf{\makecell{Licensing-\\related}}", r"\textbf{Topographic}", r"\textbf{Total}"],
             r"p{3.8cm}rrrr", r"Final-year students in mutually exclusive groups (defined in Section 4.3 of the article). Licensed supply, 2024. The fourth group (beyond 60 km but within 90 min) is empty for every field.")

    # N3: descomposición por tiempo (requiere el paso 26)
    ot_p = T / "origenes_por_tiempo.csv"
    if ot_p.exists():
        ot = pd.read_csv(ot_p)
        filas = []
        for f in ORDEN:
            x = ot[ot.familia == f]
            g = lambda o, c: x[(x.oferta == o) & (x.origen == c)].alumnos.iloc[0]  # noqa: E731
            filas.append([EN[f], *[n(g(o, c)) for o in ["post", "rel"] for c in ["structural_t", "licensing_t", "topographic_t"]]])
        escribir("tabS_origenes_tiempo", filas,
                 [r" & \multicolumn{3}{c}{\textbf{Licensed supply, 2024}} & \multicolumn{3}{c}{\textbf{Adding six relicensed}} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7} \textbf{Field}", r"\textbf{Struct.}", r"\textbf{Licens.}", r"\textbf{Topogr.}", r"\textbf{Struct.}", r"\textbf{Licens.}", r"\textbf{Topogr.}"],
                 r"p{3.8cm}rrrrrr", r"Origins defined with travel time to both supply scenarios (90 min); the 60 km radius separates structural from topographic. Columns 2--4: licensed supply 2024; columns 5--7: adding the 2018 supply of the six relicensed universities. Licensing-related: beyond 90 min with licensed supply, within 90 min of the reconstructed supply; topographic: beyond 90 min in both scenarios but within 60 km in straight line; structural: beyond 90 min in both scenarios and beyond 60 km.")
    else:
        (DEST / "tabS_origenes_tiempo.tex").write_text(r"\pending{Run src/26\_tiempos\_oferta\_2018.py and src/24\_tablas\_manuscrito.py to fill this table.}" + "\n", encoding="utf-8")

    # N6: logit fraccional
    lf = pd.read_csv(T / "logit_fraccional_ame.csv")
    prin = lf[(lf.medida == "distancia") & (~lf.ponderado)]
    filas = []
    for f in ORDEN:
        for esp, etq in [("entre", "between"), ("dentro", "within")]:
            x = prin[(prin.familia == f) & (prin.especificacion == esp)].set_index("variable")
            filas.append([(EN[f] if esp == "entre" else "") , etq,
                          *[f"{x.loc[v, 'ame_pp']:.2f}{estrellas(x.loc[v, 'p'])}" for v in ["pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior"]]])
    escribir("tabS_logit", filas,
             [r"\textbf{Field}", r"\textbf{Model}", r"\textbf{Poverty}", r"\textbf{Rural}", r"\textbf{Indigenous}", r"\textbf{\makecell{Afro-\\Peruvian}}",
              r"\textbf{\makecell{No higher\\education}}"], r"p{3.6cm}lrrrrr",
             r"Fractional logit (quasi-maximum likelihood; \citealp{Papke1996}) of the share of the district's final-year students with no programme of the field within 60 km; 1,827 districts. Average marginal effects in percentage points of the share per 10-point increase in the covariate. All models include log final-year students. Between: no department fixed effects, standard errors clustered by department; within: department fixed effects, standard errors clustered by province. *** $p<0.01$, ** $p<0.05$, * $p<0.1$.",
             primera_regla=0)
    sen = lf[(lf.familia == "Total")]
    filas = []
    for (med, pond), etq in [(("distancia", False), "Distance, unweighted (main)"), (("distancia", True), "Distance, weighted by students"),
                             (("tiempo", False), "Travel time (90 min), unweighted")]:
        for esp, e2 in [("entre", "between"), ("dentro", "within")]:
            x = sen[(sen.medida == med) & (sen.ponderado == pond) & (sen.especificacion == esp)].set_index("variable")
            filas.append([etq if esp == "entre" else "", e2, *[f"{x.loc[v, 'ame_pp']:.2f}{estrellas(x.loc[v, 'p'])}" for v in ["pobreza", "rural", "pct_autoid_indigena", "pct_afro", "pct_sin_superior"]]])
    escribir("tabS_logit_sens", filas,
             [r"\textbf{Specification}", r"\textbf{Model}", r"\textbf{Poverty}", r"\textbf{Rural}", r"\textbf{Indigenous}", r"\textbf{\makecell{Afro-\\Peruvian}}",
              r"\textbf{\makecell{No higher\\education}}"], r"p{4.2cm}lrrrrr",
             r"Any field. As Table~\ref{tab:logit}, with the outcome weighted by final-year students or defined with travel time.", primera_regla=0)

    # N2: licenciamiento neto por departamento y por familia
    dd = pd.read_csv(T / "licenciamiento_neto_departamentos.csv")
    dd = dd[dd.familia == "Total"]
    w = dd.pivot_table(index="departamento", columns="version", values="alumnos", fill_value=0).sort_values("v2024", ascending=False)
    pr = dd[dd.version == "v2024"].set_index("departamento")
    escribir("tabS_lic_dep", [[dep(d), n(r.v2024), n(r.vrel), p1(pr.loc[d, "pct_quinto_departamento"]), p1(pr.loc[d, "pobreza"]), p1(pr.loc[d, "rural"]),
                               p1(pr.loc[d, "indigena"])] for d, r in w.iterrows()],
             [r"\textbf{Department}", r"\textbf{\makecell{Students,\\2024 supply}}", r"\textbf{\makecell{Adding\\relicensed}}", r"\textbf{\makecell{Share of dept.\\students (\%)}}",
              r"\textbf{Poverty}", r"\textbf{Rural}", r"\textbf{Indigenous}"], r"lrrrrrr",
             r"Any field; licensing-related by distance (60 km). Second column: adding the 2018 headquarters and branches of the six universities relicensed in 2024--2026 (codes 048, 050, 053, 078, 090 and 117). Profile columns refer to the 2024 version (averages of the districts where students live, weighted by students, \%).", primera_regla=0)
    ln = pd.read_csv(T / "licenciamiento_neto_familias.csv")
    filas = []
    for f in ORDEN:
        a = ln[(ln.familia == f) & (ln.version == "v2024")].iloc[0]
        b = ln[(ln.familia == f) & (ln.version == "vrel")].iloc[0]
        filas.append([EN[f], n(a.alumnos_licenciamiento), "--" if not a.departamento_principal or pd.isna(a.departamento_principal) else dep(a.departamento_principal),
                      n(b.alumnos_licenciamiento), "--" if not b.departamento_principal or pd.isna(b.departamento_principal) else dep(b.departamento_principal),
                      p1(b.pct_en_departamento_principal) if b.alumnos_licenciamiento > 0 else "--"])
    escribir("tabS_lic_fam", filas,
             [r"\textbf{Field}", r"\textbf{\makecell{Students,\\2024 supply}}", r"\textbf{\makecell{Main\\department}}", r"\textbf{\makecell{Adding\\relicensed}}",
              r"\textbf{\makecell{Main\\department}}", r"\textbf{\makecell{Share in main\\dept. (\%)}}"], r"p{4.2cm}rlrlr",
             r"Licensing-related by distance (60 km), with licensed supply 2024 and adding the 2018 supply of the six relicensed universities.")

    tablas_v3()

    # N8: cifras citadas en el texto (paso 30) -> filas de un longtable
    cf = T / "cifras_texto.csv"
    if cf.exists():
        c = pd.read_csv(cf, dtype=str)
        lin = [r"\begingroup\footnotesize", r"\begin{longtable}{@{}p{0.9cm}>{\raggedright\arraybackslash}p{7.8cm}r>{\raggedright\arraybackslash}p{4.8cm}@{}}", r"\caption{Figures cited in the text and their source}\label{tab:cifras}\\", r"\toprule", r"\textbf{Sec.} & \textbf{Figure cited in the article} & \textbf{Value} & \textbf{Output table (repository)} \\", r"\midrule", r"\endfirsthead",
               r"\toprule", r"\textbf{Sec.} & \textbf{Figure cited in the article} & \textbf{Value} & \textbf{Output table (repository)} \\", r"\midrule", r"\endhead",
               r"\bottomrule", r"\multicolumn{4}{p{0.95\textwidth}}{\footnotesize Values computed from unrounded outputs and rounded once, half up: percentages to one decimal and counts to integers. Section numbers refer to the article.}\\", r"\endlastfoot"]
        for r in c.itertuples():
            lin.append(" & ".join([r.seccion, r.cifra.replace("%", r"\%").replace("...", r"\ldots"), r.valor, r.fuente.replace("_", r"\_\allowbreak{}").replace("; ", r"; \newline ")]) + r" \\")
        lin += [r"\end{longtable}", r"\endgroup"]
        (DEST / "tabS_cifras.tex").write_text("\n".join(lin) + "\n", encoding="utf-8")
        print("  -> tabS_cifras")


def tablas_v3():
    """Tablas de la revisión v3 (pasos 28, 30, 34-38)."""
    reg_en = {"costa": "Coast", "sierra": "Highlands", "selva": "Amazon", "todas": "All"}
    # M3: error de calibración por región y factores regionales
    cr, fr = T / "calibracion_por_region.csv", T / "factores_regionales.csv"
    if cr.exists() and fr.exists():
        c = pd.read_csv(cr)
        c = c[c.muestra == "todas"].set_index("region").reindex(["todas", "costa", "sierra", "selva"])
        f = pd.read_csv(fr)
        fac = f[f.modelo.str.contains("dentro")].set_index("region").factor
        filas = [[reg_en[r], n(x.pares), f"{x.error_medio_pct:+.1f}".replace("-", "$-$"), f"{x.error_absoluto_medio_min:.1f}", p1(x.pct_dentro_30pct),
                  f"{x.spearman:.2f}", n(x.pares_60_120min), "0.487" if r == "todas" else f"{fac[r]:.3f}"] for r, x in c.iterrows()]
        loo = f.set_index("modelo")
        nota = (r"Pairs: the 37 calibration pairs (calibrated time in leave-one-out cross-validation) and the 18 verification pairs of Table~\ref{tab:verificacion} (out of sample). "
                r"Region: highlands if the school is at 2,000 m or more; otherwise coast in coastal departments and Amazon elsewhere. Error: modelled minus Google Maps time, as \% of the latter. "
                rf"Factor: road speed factor estimated on the calibration pairs of each region. In leave-one-out cross-validation, region-specific factors give a mean absolute error of {loo.loc['factor por región (LOOCV)', 'error_absoluto_medio_min']:.1f} min, "
                rf"against {loo.loc['factor único (LOOCV)', 'error_absoluto_medio_min']:.1f} min with the single factor.")
        escribir("tabS_calibracion_region", filas,
                 [r"\textbf{Region}", r"\textbf{Pairs}", r"\textbf{\makecell{Mean rel.\\error (\%)}}", r"\textbf{\makecell{Mean abs.\\error (min)}}",
                  r"\textbf{\makecell{Within\\30\% (\%)}}", r"\textbf{Spearman}", r"\textbf{\makecell{Pairs at\\60--120 min}}", r"\textbf{\makecell{Road speed\\factor}}"],
                 "lrrrrrrr", nota)
    sf, of = T / "sensibilidad_factor_regional.csv", T / "origenes_factor_regional.csv"
    if sf.exists() and of.exists():
        a = pd.read_csv(sf).set_index("familia").reindex(ORDEN)
        o = pd.read_csv(of)
        o = o[o.familia == "Total"].set_index("origen")
        filas = [[EN[k], p1(x.pct_90_base), p1(x.pct_90_regional), p1(x.pct_clasificado_igual)] for k, x in a.iterrows()]
        nota = (r"Share of final-year students beyond 90 min with the single road speed factor (base) and with a factor for the region of each school (Table~\ref{tab:calibracion_region}); last column: students classified identically. "
                rf"Any-field groups (base / regional): structural {n(o.loc['structural', 'base'])} / {n(o.loc['structural', 'regional'])}; licensing-related {n(o.loc['licensing', 'base'])} / {n(o.loc['licensing', 'regional'])}; "
                rf"topographic {n(o.loc['topographic', 'base'])} / {n(o.loc['topographic', 'regional'])}. Approximation: each school's whole route is travelled at the factor of its region.")
        escribir("tabS_factor_regional", filas, [r"\textbf{Field}", r"\textbf{\makecell{Beyond 90 min,\\base (\%)}}", r"\textbf{\makecell{Beyond 90 min,\\regional (\%)}}",
                                                  r"\textbf{\makecell{Classified\\identically (\%)}}"], r"p{4.6cm}rrr", nota)
    # M1: relieve
    rp, rt, rm = T / "relieve_por_origen.csv", T / "relieve_terciles.csv", T / "relieve_modelo.csv"
    if rp.exists():
        x = pd.read_csv(rp).set_index("origen")
        lab = {"structural": "Structural", "licensing": "Licensing-related", "topographic": "Topographic", "none": "Not in a desert"}
        filas = [[lab[k], n(r.alumnos), n(r.elevacion_m), n(r.desnivel_5km_m), f"{r.pendiente_media_5km_grados:.1f}", p1(r.pct_sierra), p1(r.pct_selva)] for k, r in x.iterrows()]
        escribir("tabS_relieve", filas, [r"\textbf{Group (any field)}", r"\textbf{Students}", r"\textbf{\makecell{Elevation\\(m)}}", r"\textbf{\makecell{Relief within\\5 km (m)}}",
                                          r"\textbf{\makecell{Mean slope\\within 5 km (degrees)}}", r"\textbf{\makecell{Highlands\\(\%)}}", r"\textbf{\makecell{Amazon\\(\%)}}"], "lrrrrrr",
                 r"Averages over schools weighted by final-year students. Elevation from AWS Terrain Tiles (Terrarium, zoom 10, about 150 m per pixel; sources include SRTM). Relief: maximum minus minimum elevation within 5 km of the school. Groups as in Table~3 of the article.", primera_regla=0)
    if rt.exists():
        t = pd.read_csv(rt)
        km = {"cerca": "Closest third", "intermedio": "Middle third", "lejos": "Farthest third"}
        tr = {"bajo": "Low", "medio": "Middle", "alto": "High"}
        filas = [[km[r.km_tercil], tr[r.tercil], n(r.alumnos), p1(r.pct_mas_90min)] for r in t.itertuples()]
        m = pd.read_csv(rm).iloc[0] if rm.exists() else None
        nota = (r"Schools with a programme of any field within 60 km in straight line, grouped by thirds of the distance to the nearest campus and, within each, by thirds of the terrain crossed "
                r"(sum of elevation changes along 200 points of the straight line to that campus; a relative index, inflated by elevation noise). ")
        if m is not None:
            nota += rf"Logit of being beyond 90 min on terrain crossed, log distance and region (12,615 schools): {m.efecto_marginal_pp:.2f} percentage points per 100 m, p < 0.001."
        escribir("tabS_relieve_terciles", filas, [r"\textbf{Distance to campus}", r"\textbf{Terrain crossed}", r"\textbf{Students}", r"\textbf{\makecell{Beyond\\90 min (\%)}}"],
                 "llrr", nota, primera_regla=0)
    # M5: ubicación de las filiales
    fs = T / "filiales_sensibilidad.csv"
    if fs.exists():
        x = pd.read_csv(fs)
        alt = {"A. base": "Base (Section 4.3)", "B. localidad OSM": "OpenStreetMap locality", "C. punto del distrito": "District representative point",
               "D. ±5 km, robusto": r"$\pm$5 km, in all eight shifts", "D. ±5 km, máximo": r"$\pm$5 km, in any shift"}
        filas = []
        for fam in ["Total", "Educación", "Servicios"]:
            for k, v in alt.items():
                r = x[(x.familia == fam) & (x.alternativa == k)].iloc[0]
                filas.append([EN[fam] if k == "A. base" else "", v, n(r.lic_2024), n(r.lic_con_relicenciadas), n(r.lic_2024_puno)])
        ub = pd.read_csv(T / "filiales_ubicacion_alternativa.csv")
        n_osm = int(ub.metodo_osm.str.startswith("OSM").sum())
        n_fil = int((ub.metodo_osm != "sede (SUNEDU)").sum())
        lejos = ub[ub.metodo_osm != "sede (SUNEDU)"].drop_duplicates("ubigeo").sort_values("km_a_punto_distrito", ascending=False)
        km_pu = lejos.loc[lejos.ubigeo.astype(str).str.zfill(6) == "250101", "km_a_punto_distrito"].max()
        km_pm = lejos.loc[lejos.ubigeo.astype(str).str.zfill(6) == "170101", "km_a_punto_distrito"].max()
        escribir("tabS_filiales", filas, [r"\textbf{Field}", r"\textbf{Branch location}", r"\textbf{\makecell{Licensing-related,\\2024 supply}}",
                                           r"\textbf{\makecell{Adding six\\relicensed}}", r"\textbf{\makecell{Puno,\\2024 supply}}"], "llrrr",
                 rf"Final-year students in licensing-related deserts by distance under alternative locations of the 2018 branches of denied universities (headquarters keep SUNEDU's coordinates). OpenStreetMap locality: populated place with the branch's city name inside its district ({n_osm} of {n_fil} branches matched; the rest keep the base location). $\pm$5 km: branches shifted 5 km in eight directions. District representative point: it falls far from the branch's city in large Amazonian districts ({km_pu:.0f} km from Pucallpa and {km_pm:.0f} km from Puerto Maldonado), so it is not a plausible location there; the fall in services with relicensed universities under this alternative reflects that displacement.", primera_regla=0)
    # M4: logit fraccional, errores espaciales
    le = T / "logit_fraccional_espacial.csv"
    if le.exists():
        x = pd.read_csv(le)
        filas = []
        for med, mlab in [("distancia", "Distance (60 km)"), ("tiempo", "Travel time (90 min)")]:
            for esp in ["entre", "dentro"]:
                z = x[(x.medida == med) & (x.especificacion == esp)]
                fila = [mlab if esp == "entre" else "", "between" if esp == "entre" else "within"]
                for err, mu in [("agrupado (principal)", "todos"), ("Conley 60 km", "todos"), ("Conley 100 km", "todos"), ("agrupado (principal)", ">= 10 alumnos")]:
                    for v in ["pobreza", "rural"]:
                        r = z[(z.errores == err) & (z.muestra == mu) & (z.variable == v)].iloc[0]
                        fila.append(f"{r.ame_pp:.2f}{estrellas(r.p)}")
                mo = z[z.errores == "I de Moran de los residuos"].iloc[0]
                fila.append(f"{mo.ame_pp:.2f}")
                filas.append(fila)
        escribir("tabS_logit_espacial", filas,
                 [r" & & \multicolumn{2}{c}{\textbf{Clustered}} & \multicolumn{2}{c}{\textbf{Conley 60 km}} & \multicolumn{2}{c}{\textbf{Conley 100 km}} & \multicolumn{2}{c}{\textbf{$\geq$10 students}} & \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10} \textbf{Outcome}",
                  r"\textbf{Model}", "Pov.", "Rural", "Pov.", "Rural", "Pov.", "Rural", "Pov.", "Rural", r"\textbf{\makecell{Moran's I\\residuals}}"],
                 "llrrrrrrrrr",
                 r"Any field; average marginal effects in percentage points per 10-point increase, as in Table~\ref{tab:logit}. Clustered: standard errors clustered by department (between) or province (within). Conley: Bartlett kernel with the cutoff shown, between district centroids. $\geq$10 students: excluding the districts with fewer than 10 final-year students (1,651 districts remain). Moran's I of the Pearson residuals with queen contiguity (999 permutations; p = 0.001 in all models). *** p $<$ 0.01, ** p $<$ 0.05, * p $<$ 0.1.", primera_regla=0)
    # m1: correspondencia de campos
    cc = T / "correspondencia_campos.csv"
    if cc.exists():
        x = pd.read_csv(cc)
        filas, prev = [], None
        for r in x.itertuples():
            filas.append([r.familia if r.familia != prev else "", r.clase_N2_SUNEDU.replace("Ninguno (asignado por palabra clave)", "None (assigned by keyword)"),
                          r.cine_f_2013.replace("—", "--"), n(r.programas)])
            prev = r.familia
        escribir("tabS_campos", filas, [r"\textbf{Field (this study)}", r"\textbf{SUNEDU class (level 2)}", r"\textbf{ISCED-F 2013 field}", r"\textbf{Programmes}"],
                 r"p{3.6cm}p{5.2cm}p{5.4cm}r",
                 r"Nine fields adapted from the broad fields of ISCED-F 2013. Differences: natural sciences (05) and ICT (06) are merged; law (042) is grouped with social sciences rather than business; environment (052) is grouped with agriculture; hygiene and occupational health services (102) are grouped with health. Services comprise personal services (101: hotels and restaurants, travel and tourism, sports, domestic and beauty services) and transport services (104). SUNEDU class names translated from Spanish. Programmes whose SUNEDU class is `none' are assigned by keywords in the programme name (rules in \path{src/utils.py}).", primera_regla=0)
    # m4: flujo de unidades
    fl = T / "flujo_unidades.csv"
    if fl.exists():
        x = pd.read_csv(fl, dtype=str)
        filas, prev = [], None
        for r in x.itertuples():
            v = r.n
            try:
                v = n(float(v))
            except ValueError:
                pass
            filas.append([r.bloque if r.bloque != prev else "", r.unidad.replace("...", r"\ldots"), v])
            prev = r.bloque
        escribir("tabS_flujo", filas, [r"\textbf{Block}", r"\textbf{Units}", r"\textbf{Number}"], r"p{3.4cm}p{9cm}r",
                 r"Units of analysis from the sources to the analytic sample (STROBE item 13). All schools and campuses have valid coordinates; no district with final-year students lacks covariates (Table~\ref{tab:faltantes}).", primera_regla=0)

if __name__ == "__main__":
    main()
