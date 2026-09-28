"""Ejecuta el pipeline completo en orden. Uso:
    python run_all.py            # todos los pasos, salvo 4b y 4d
    python run_all.py 4 5        # solo los pasos indicados
    python run_all.py 4b 4d      # opcional: recalcula los tiempos de viaje con un extracto nuevo de OSM
                                 # (descarga ~250 MB y tarda 30-60 min; los resultados del artículo ya están
                                 #  en datos/processed/tiempos_colegios.csv y sensibilidad_colegios.csv.gz)
El paso 3 (Censo 2017) se omite solo si no hay microdatos en datos/raw/censo2017.
"""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
PASOS = {
    "0": "00_descargar_datos.py", "1": "01_oferta_universitaria.py", "2": "02_demanda_escolar.py",
    "1b": "01b_filiales_denegadas.py", "3": "03_censo2017_covariables.py", "4": "04_accesibilidad.py",
    "4b": "04b_tiempos_viaje.py", "4d": "04d_sensibilidad_tiempos.py",
    "4c": "04c_efecto_licenciamiento.py", "4e": "04e_gestion.py", "4g": "04g_relicenciadas.py",
    "4h": "04h_robustez_tiempos.py", "5": "05_modelos_conteo.py", "6": "06_ml_explicable.py",
    "7": "07_tipologia_clusters.py", "8": "08_localizacion_optima.py", "9": "09_mapas_figuras.py",
    "10": "10_asistencia_moran.py", "10b": "10b_espacial_robustez.py", "11": "11_figuras_modelos.py",
    "12": "12_resumen_grafico.py", "13": "13_taxonomia_desiertos.py", "14": "14_costos.py",
}
OPCIONALES = {"0", "4b", "4d"}   # no se ejecutan por defecto

if __name__ == "__main__":
    elegidos = sys.argv[1:] or [k for k in PASOS if k not in OPCIONALES]
    sys.path.insert(0, str(SRC))
    for k in elegidos:
        print(f"\n===== Paso {k}: {PASOS[k]} =====")
        sys.argv = [PASOS[k]]   # cada script corre con sus parámetros por defecto
        runpy.run_path(str(SRC / PASOS[k]), run_name="__main__")
