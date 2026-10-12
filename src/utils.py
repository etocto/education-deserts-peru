"""Utilidades comunes: rutas, lectura de datos, distancias y familias de carreras."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(RAIZ / "config.yaml", encoding="utf-8"))

# Calibración de velocidades (paso 32): las vías se multiplican por tiempos.factor_calibracion_vias; ríos y ferris no.
_T = CFG.get("tiempos", {})
_T["velocidades_kmh_sin_calibrar"] = dict(_T.get("velocidades_kmh", {}))
_F = float(_T.get("factor_calibracion_vias", 1.0))
_T["velocidades_kmh"] = {k: (v if k in ("river", "ferry") else v * _F) for k, v in _T["velocidades_kmh_sin_calibrar"].items()}


def ruta(clave: str) -> Path:
    """Ruta absoluta de un archivo o carpeta definido en config.yaml."""
    if clave in CFG["archivos"]:
        return RAIZ / CFG["archivos"][clave]
    p = RAIZ / CFG["rutas"][clave]
    p.mkdir(parents=True, exist_ok=True)
    return p


def sin_tildes(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()


# ---------------------------------------------------------------------------- distancias
R_TIERRA_KM = 6371.0088


def haversine_matriz(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Matriz de distancias en km entre los puntos (lat1, lon1) y (lat2, lon2)."""
    la1, lo1 = np.radians(np.asarray(lat1, float))[:, None], np.radians(np.asarray(lon1, float))[:, None]
    la2, lo2 = np.radians(np.asarray(lat2, float))[None, :], np.radians(np.asarray(lon2, float))[None, :]
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def peso_decaimiento(d: np.ndarray, d0: float, tipo: str = "gaussiano") -> np.ndarray:
    """Función de decaimiento del 2SFCA (Luo y Wang, 2003; Dai, 2010)."""
    if tipo == "escalon":
        return (d <= d0).astype(float)
    w = (np.exp(-0.5 * (d / d0) ** 2) - np.exp(-0.5)) / (1 - np.exp(-0.5))
    return np.where(d <= d0, w, 0.0)


# ---------------------------------------------------------------------------- familias de carreras
# Agrupación de las clases N2 de SUNEDU en 9 familias (inspirada en los campos amplios de la CINE-F 2013).
FAMILIAS_N2 = {
    "Educación": ["EDUCACION INICIAL Y PRIMARIA", "EDUCACION SECUNDARIA", "OTROS PROGRAMAS EN EDUCACION",
                  "EDUCACION FISICA", "EDUCACION ESPECIAL", "EDUCACION ARTISTICA", "EDUCACION SUPERIOR Y TECNOLOGICA"],
    "Artes y humanidades": ["ARTE", "HUMANIDADES", "IDIOMAS"],
    "Ciencias sociales, periodismo y derecho": ["CIENCIAS SOCIALES Y DEL COMPORTAMIENTO", "PERIODISMO E INFORMACION", "DERECHO"],
    "Administración y negocios": ["ADMINISTRACION Y COMERCIO"],
    "Ciencias naturales, matemáticas e informática": ["CIENCIAS FISICAS Y QUIMICAS", "CIENCIAS BIOLOGICAS Y AFINES",
                                                        "MATEMATICAS Y ESTADISTICA", "INFORMATICA"],
    "Ingeniería, industria y construcción": ["INGENIERIA Y PROFESIONES AFINES", "INDUSTRIA Y PRODUCCION", "ARQUITECTURA Y CONSTRUCCION"],
    "Agropecuaria y ambiente": ["AGRICULTURA", "SILVICULTURA", "PESCA", "VETERINARIA", "MEDIO AMBIENTE"],
    "Salud y bienestar": ["SALUD", "BIENESTAR", "SERVICIOS DE HIGIENE Y SALUD OCUPACIONAL"],
    "Servicios": ["SERVICIOS PERSONALES", "SERVICIOS DE TRANSPORTE"],
}
_N2_A_FAMILIA = {n2: fam for fam, lista in FAMILIAS_N2.items() for n2 in lista}

# Reglas por palabra clave para los programas con clase N2 = "NINGUNO" (se registran en un archivo de control).
REGLAS_NOMBRE = [
    (r"EDUCACION|PEDAGOG|DOCENCIA", "Educación"),
    (r"MEDICINA|ENFERMER|OBSTETR|ODONTOLOG|FARMAC|NUTRICI|PSICOLOG|TECNOLOGIA MEDICA|SALUD|ESTOMATOLOG|BIOQUIMICA", "Salud y bienestar"),
    (r"DERECHO|CIENCIA POLITICA|SOCIOLOG|ANTROPOLOG|ECONOM|COMUNICACI|PERIODISMO|TRABAJO SOCIAL|RELACIONES INTERNAC", "Ciencias sociales, periodismo y derecho"),
    (r"ADMINISTRAC|CONTAB|NEGOCIOS|MARKETING|COMERCI|FINANZ|GESTION|TURISMO|HOTELER", "Administración y negocios"),
    (r"SISTEMAS|INFORMATIC|SOFTWARE|COMPUTAC|DATOS|ESTADIST|MATEMAT|FISICA|QUIMICA|BIOLOG", "Ciencias naturales, matemáticas e informática"),
    (r"INGENIER|ARQUITECT|INDUSTRIAL|CIVIL|MECANIC|ELECTR|MINAS|METALURG|PETROL|TEXTIL", "Ingeniería, industria y construcción"),
    (r"AGRONOM|AGRICOL|AGROIND|ZOOTEC|VETERIN|FORESTAL|PESQUER|ACUICULT|AMBIENTAL", "Agropecuaria y ambiente"),
    (r"ARTE|MUSICA|DISENO|TRADUCC|LINGUIST|LITERAT|FILOSOF|HISTORIA|TEOLOG", "Artes y humanidades"),
]


def familia_programa(n2: str, nombre: str) -> tuple[str, str]:
    """Devuelve (familia, origen de la asignación: 'N2' o 'regla')."""
    n2s = sin_tildes(n2)
    if n2s in _N2_A_FAMILIA:
        return _N2_A_FAMILIA[n2s], "N2"
    nom = sin_tildes(nombre)
    for patron, fam in REGLAS_NOMBRE:
        if re.search(patron, nom):
            return fam, "regla"
    return "Sin clasificar", "regla"


FAMILIAS = list(FAMILIAS_N2.keys())

# Nombres en inglés para figuras y tablas del manuscrito
FAMILIAS_EN = {
    "Educación": "Education", "Artes y humanidades": "Arts and humanities",
    "Ciencias sociales, periodismo y derecho": "Social sciences, journalism and law",
    "Administración y negocios": "Business and administration",
    "Ciencias naturales, matemáticas e informática": "Natural sciences, mathematics and ICT",
    "Ingeniería, industria y construcción": "Engineering, manufacturing and construction",
    "Agropecuaria y ambiente": "Agriculture and environment", "Salud y bienestar": "Health and welfare",
    "Servicios": "Services", "Total": "All fields",
}


def guardar(df: pd.DataFrame, nombre: str, carpeta: str = "processed") -> Path:
    p = ruta(carpeta) / nombre
    df.to_csv(p, index=False, encoding="utf-8")
    print(f"  -> {p.relative_to(RAIZ)} ({len(df):,} filas)")
    return p
