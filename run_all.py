"""Ejecuta el pipeline completo en orden. Uso:
    python run_all.py            # todos los pasos, salvo 4b y 4d
    python run_all.py 4 5        # solo los pasos indicados
    python run_all.py 26 24      # fase 1: tiempos a la oferta 2018 (N3) y tablas
    python run_all.py 36 34 37 35 38 28 30 24   # revisión v3: relieve, calibración por región, filiales, campos
    python rehacer_tiempos_v4.py # revisión v4: rehace todos los tiempos tras corregir la red y compara cifras
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
    # pasos de la revisión (octubre 2026)
    "6b": "06b_ml_reforms.py", "15": "15_sensibilidad_umbral.py", "16": "16_diagnostico_tiempos.py",
    "17": "17_conciliacion_denegadas.py", "18": "18_perfil_desiertos_licenciamiento.py", "19": "19_descriptivos.py",
    "20": "20_tipologia_k.py", "21": "21_asistencia_nivel.py", "22": "22_muestra_verificacion.py",
    "23": "23_figuras_revision.py",
    # fase 1 del seguimiento (N2, N3, N4, N6, N8); 26 recalcula tiempos con la red de OSM (opcional, 10-30 min)
    "25": "25_licenciamiento_neto.py", "26": "26_tiempos_oferta_2018.py", "27": "27_topografico_conectividad.py",
    "28": "28_logit_fraccional.py", "29": "29_origenes_unificados.py", "30": "30_cifras_texto.py",
    "31": "31_validacion_pares.py", "32": "32_calibracion_velocidades.py", "33": "33_verificacion_manual.py",   # validación y calibración de tiempos
    # revisión v3 (M1, M3, M4, M5, m1, m4): 36 descarga teselas de elevación (internet) y 37 recalcula tiempos (20-40 min)
    "36": "36_relieve_grupos.py", "34": "34_error_calibracion_region.py", "37": "37_factor_regional.py",
    "35": "35_sensibilidad_filiales.py", "38": "38_correspondencia_campos.py",
    "24": "24_tablas_manuscrito.py",   # al final: escribe las tablas LaTeX a partir de todas las salidas
}
OPCIONALES = {"0", "4b", "4d", "22", "26", "31", "32", "36", "37"}   # no se ejecutan por defecto (22: la muestra ya verificada no se vuelve a sortear)

if __name__ == "__main__":
    elegidos = sys.argv[1:] or [k for k in PASOS if k not in OPCIONALES]
    sys.path.insert(0, str(SRC))
    for k in elegidos:
        print(f"\n===== Paso {k}: {PASOS[k]} =====")
        sys.argv = [PASOS[k]]   # cada script corre con sus parámetros por defecto
        runpy.run_path(str(SRC / PASOS[k]), run_name="__main__")
