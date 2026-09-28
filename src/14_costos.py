"""Paso 14 – Costo-efectividad ilustrativo de las nuevas sedes del modelo de localización (responde al revisor 4).

No es un análisis costo-beneficio completo (requeriría supuestos de retornos salariales); es una estimación
transparente y reproducible del costo anual y del costo por estudiante adicional, con rangos.

Para cada familia y número de sedes p (paso 8):
  cubiertos         alumnos de 5.º en desierto que quedarían a menos de 60 km (por cohorte)
  ingresantes_adic  cubiertos × efecto de salir del desierto sobre la asistencia (paso 10: coeficiente del % en
                    desierto × 100 = 6,8 pp; escenario prudente con el coeficiente corregido de Oster, paso 10b)
  matricula_adic    ingresantes_adic × duración de la carrera × retención (estado estacionario)
  costo_anual       p × inversión por sede × factor de recuperación de capital + matrícula_adic × gasto por alumno
  costo_por_estudiante_adicional_anual = costo_anual / matricula_adic
Los supuestos (inversión por sede, gasto por alumno, tasa de descuento, vida útil, retención) están en
config.yaml -> costos, con sus fuentes. Se reportan combinaciones bajo/alto.

Salida: salidas/tablas/costos_localizacion.csv
"""
from __future__ import annotations

import itertools

import pandas as pd

from utils import CFG, guardar, ruta

C = CFG["costos"]


def factor_recuperacion(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def main():
    loc = pd.read_csv(ruta("tablas") / "localizacion_optima.csv")
    res = loc.groupby(["familia", "p"]).agg(cubiertos=("quinto_cubierto", "first"),
                                           en_desierto=("quinto_en_desierto", "first")).reset_index()
    asis = pd.read_csv(ruta("tablas") / "asistencia_modelos.csv")
    efecto = abs(asis[(asis.dependiente == "asiste") & (asis.acceso == "desierto")].coef.iloc[0]) * 100   # pp
    efectos = {"modelo": efecto}
    f_oster = ruta("tablas") / "oster_asistencia.csv"
    if f_oster.exists():
        o = pd.read_csv(f_oster)
        efectos["corregido_Oster"] = abs(o[o.acceso == "desierto"].beta_corregido_delta1.iloc[0]) * 100
    crf = factor_recuperacion(C["tasa_descuento"], C["vida_util_anios"])
    filas = []
    for (_, r), (nom_ef, ef), capex, gasto in itertools.product(res.iterrows(), efectos.items(),
                                                                  C["inversion_sede_soles"], C["gasto_corriente_alumno_anual"]):
        ingresantes = r.cubiertos * ef / 100
        matricula = ingresantes * C["duracion_carrera_anios"] * C["retencion"]
        capital = r.p * capex * crf
        corriente = matricula * gasto
        filas.append({"familia": r.familia, "p": r.p, "cubiertos": r.cubiertos, "efecto_pp": ef, "escenario_efecto": nom_ef,
                      "inversion_sede": capex, "gasto_alumno": gasto, "ingresantes_adic_anual": ingresantes,
                      "matricula_adic": matricula, "costo_anual_millones": (capital + corriente) / 1e6,
                      "inversion_total_millones": r.p * capex / 1e6,
                      "costo_por_estudiante_adicional_anual": (capital + corriente) / matricula})
    t = pd.DataFrame(filas)
    guardar(t.round(2), "costos_localizacion.csv", "tablas")
    resumen = t[t.p == t.p.max()].groupby(["familia", "escenario_efecto"]).agg(
        cubiertos=("cubiertos", "first"), ingresantes=("ingresantes_adic_anual", "first"),
        inversion_min=("inversion_total_millones", "min"), inversion_max=("inversion_total_millones", "max"),
        costo_est_min=("costo_por_estudiante_adicional_anual", "min"), costo_est_max=("costo_por_estudiante_adicional_anual", "max"))
    print(f"Factor de recuperación de capital: {crf:.4f}  |  efectos (pp): {efectos}")
    print(resumen.round(0).to_string())


if __name__ == "__main__":
    main()
