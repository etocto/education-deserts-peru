"""Paso 38 – Correspondencia entre las nueve familias, las clases N2 de SUNEDU y la CINE-F 2013 (revisión v3, obs. m1).

El artículo agrupa los programas en nueve familias «adaptadas de» los campos amplios de la CINE-F 2013 (ISCED-F
2013, UNESCO-UIS). Este paso documenta la correspondencia: para cada familia, las clases de segundo nivel (N2) de
SUNEDU que la forman, el campo amplio o específico de la CINE-F al que corresponden, el número de programas y las
diferencias con la CINE-F (une 05 y 06, separa derecho de 04 y medio ambiente de 05, añade higiene y salud
ocupacional a salud). Las clases N2 = «NINGUNO» se asignan por palabra clave (src/utils.py).

Requiere el paso 1. Salida: salidas/tablas/correspondencia_campos.csv
"""
from __future__ import annotations

import pandas as pd

from utils import FAMILIAS_EN, FAMILIAS_N2, guardar, ruta, sin_tildes

CINE = {  # clase N2 de SUNEDU -> campo de la CINE-F 2013 (código y nombre en inglés)
    "EDUCACION INICIAL Y PRIMARIA": "011 Education", "EDUCACION SECUNDARIA": "011 Education",
    "OTROS PROGRAMAS EN EDUCACION": "011 Education", "EDUCACION FISICA": "011 Education",
    "EDUCACION ESPECIAL": "011 Education", "EDUCACION ARTISTICA": "011 Education",
    "EDUCACION SUPERIOR Y TECNOLOGICA": "011 Education",
    "ARTE": "021 Arts", "HUMANIDADES": "022 Humanities (except languages)", "IDIOMAS": "023 Languages",
    "CIENCIAS SOCIALES Y DEL COMPORTAMIENTO": "031 Social and behavioural sciences",
    "PERIODISMO E INFORMACION": "032 Journalism and information", "DERECHO": "042 Law",
    "ADMINISTRACION Y COMERCIO": "041 Business and administration",
    "CIENCIAS FISICAS Y QUIMICAS": "053 Physical sciences", "CIENCIAS BIOLOGICAS Y AFINES": "051 Biological and related sciences",
    "MATEMATICAS Y ESTADISTICA": "054 Mathematics and statistics", "INFORMATICA": "061 Information and communication technologies",
    "INGENIERIA Y PROFESIONES AFINES": "071 Engineering and engineering trades", "INDUSTRIA Y PRODUCCION": "072 Manufacturing and processing",
    "ARQUITECTURA Y CONSTRUCCION": "073 Architecture and construction",
    "AGRICULTURA": "081 Agriculture", "SILVICULTURA": "082 Forestry", "PESCA": "083 Fisheries", "VETERINARIA": "084 Veterinary",
    "MEDIO AMBIENTE": "052 Environment", "SALUD": "091 Health", "BIENESTAR": "092 Welfare",
    "SERVICIOS DE HIGIENE Y SALUD OCUPACIONAL": "102 Hygiene and occupational health services",
    "SERVICIOS PERSONALES": "101 Personal services", "SERVICIOS DE TRANSPORTE": "104 Transport services",
}
CAMPO_AMPLIO = {"01": "01 Education", "02": "02 Arts and humanities", "03": "03 Social sciences, journalism and information",
                "04": "04 Business, administration and law", "05": "05 Natural sciences, mathematics and statistics",
                "06": "06 Information and communication technologies", "07": "07 Engineering, manufacturing and construction",
                "08": "08 Agriculture, forestry, fisheries and veterinary", "09": "09 Health and welfare", "10": "10 Services"}


def main():
    p = pd.read_csv(ruta("processed") / "oferta_programas_pregrado.csv", dtype=str)
    p["n2"] = p.NOMBRE_CLASE_PROGRAMA_N2.map(sin_tildes)
    n = p.groupby(["familia", "n2"]).size()
    filas = []
    for fam, clases in FAMILIAS_N2.items():
        for c in clases:
            cine = CINE[c]
            filas.append({"familia": FAMILIAS_EN[fam], "clase_N2_SUNEDU": c.capitalize(), "cine_f_2013": cine,
                          "campo_amplio_cine": CAMPO_AMPLIO[cine[:2]], "programas": int(n.get((fam, c), 0))})
        filas.append({"familia": FAMILIAS_EN[fam], "clase_N2_SUNEDU": "Ninguno (asignado por palabra clave)", "cine_f_2013": "—",
                      "campo_amplio_cine": "—", "programas": int(n.get((fam, "NINGUNO"), 0))})
    t = pd.DataFrame(filas)
    guardar(t, "correspondencia_campos.csv", "tablas")
    print(t.to_string(index=False))
    print("\nprogramas por familia:", t.groupby("familia", sort=False).programas.sum().to_dict(), "total:", t.programas.sum())


if __name__ == "__main__":
    main()
