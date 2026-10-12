"""Revisión v4 (obs. M2): rehace todos los tiempos de viaje tras corregir la suma de duplicados en la red y compara
las cifras con las de la versión anterior.

Qué se corrigió: coo_matrix(...).tocsr() suma las entradas repetidas. Cuando dos locales se anclaban al mismo nodo, su
tiempo de acceso a pie se sumaba (y lo mismo con dos tramos paralelos entre los mismos nodos), de modo que añadir oferta
podía alargar un tiempo. Ahora utils.matriz_minimos conserva el mínimo (pruebas en tests/test_utils.py).

Uso (desde la raíz del repositorio; 2-3 horas, unos 3 GB de RAM; el paso 36 usa las teselas de elevación ya
descargadas en datos/raw/dem o en DEM_CACHE):

    python rehacer_tiempos_v4.py            # respalda, recalcula, compara
    python rehacer_tiempos_v4.py --solo-flujo  # si los tiempos ya se rehicieron: flujo por defecto y comparación
    python rehacer_tiempos_v4.py --comparar # solo vuelve a comparar con el respaldo

Salidas: salidas_antes_v4/ (respaldo de la versión anterior) y salidas/cambios_v4.csv (cada cifra que cambia),
más el resumen en pantalla. Enviar salidas/cambios_v4.csv para actualizar el texto del artículo.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
ANTES = R / "salidas_antes_v4"
PROC = ["tiempos_colegios.csv", "tiempos_colegios_escenarios.csv", "tiempos_colegios_factor_regional.csv",
        "relieve_colegios.csv", "accesibilidad_colegios.csv"]

# 32 recalibra el factor (escribe config.yaml); 4b, 4d, 26 y 37 rehacen los tiempos con la red corregida;
# 33 --modelo recalcula el tiempo a los locales de la muestra de verificación (no se vuelve a sortear: paso 22 fuera);
# luego el flujo completo por defecto (sin 22) y, al final, las tablas LaTeX.
PASOS = [["32"], ["4b"], ["4d"], ["26"], ["37"], ["33", "--modelo"], ["36"]]


def correr(paso, *args):
    sys.path.insert(0, str(R))
    from run_all import PASOS as P  # noqa: N811
    script = R / "src" / P[paso]
    print(f"\n===== Paso {paso}: {script.name} {' '.join(args)} =====", flush=True)
    subprocess.run([sys.executable, str(script), *args], cwd=R, check=True)


def respaldo():
    if ANTES.exists():
        print(f"El respaldo {ANTES.name}/ ya existe; no se sobrescribe.")
        return
    shutil.copytree(R / "salidas" / "tablas", ANTES / "tablas")
    (ANTES / "processed").mkdir(parents=True)
    for f in PROC:
        p = R / "datos" / "processed" / f
        if p.exists():
            shutil.copy2(p, ANTES / "processed" / f)
    shutil.copy2(R / "config.yaml", ANTES / "config.yaml")
    print(f"Respaldo en {ANTES.name}/")


def comparar():
    filas = []
    for p in sorted((ANTES / "tablas").glob("*.csv")):
        q = R / "salidas" / "tablas" / p.name
        if not q.exists():
            continue
        try:
            a, b = pd.read_csv(p), pd.read_csv(q)
        except Exception:  # noqa: BLE001
            continue
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            filas.append({"archivo": p.name, "fila": "", "columna": "(forma)", "antes": str(a.shape), "despues": str(b.shape)})
            continue
        for c in a.columns:
            x, y = a[c], b[c]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                dif = ~np.isclose(x.astype(float), y.astype(float), rtol=1e-6, atol=1e-9, equal_nan=True)
            else:
                dif = x.astype(str) != y.astype(str)
            for i in np.flatnonzero(np.asarray(dif)):
                etiqueta = " | ".join(str(a.iloc[i, k]) for k in range(min(2, a.shape[1])) if a.columns[k] != c)
                filas.append({"archivo": p.name, "fila": etiqueta, "columna": c, "antes": x.iloc[i], "despues": y.iloc[i]})
    t = pd.DataFrame(filas)
    t.to_csv(R / "salidas" / "cambios_v4.csv", index=False, encoding="utf-8")
    print(f"\n{len(t):,} celdas cambian en {t.archivo.nunique() if len(t) else 0} tablas -> salidas/cambios_v4.csv")
    if len(t):
        print(t.groupby("archivo").size().sort_values(ascending=False).head(25).to_string())
        c = t[t.archivo == "cifras_texto.csv"]
        if len(c):
            print("\nCifras citadas en el texto que cambian:")
            print(c.to_string(index=False))
    viejo = (ANTES / "config.yaml").read_text(encoding="utf-8")
    nuevo = (R / "config.yaml").read_text(encoding="utf-8")
    f = lambda s: [ln.strip() for ln in s.splitlines() if "factor_calibracion_vias" in ln]  # noqa: E731
    print("\nFactor de calibración antes / después:", f(viejo), f(nuevo))


MODULOS = ["pandas", "numpy", "yaml", "requests", "dbfread", "statsmodels", "sklearn", "scipy", "xgboost", "shap",
           "geopandas", "shapely", "pyproj", "matplotlib", "PIL", "pulp", "osmium", "libpysal", "spreg", "jinja2"]


def verificar_paquetes():
    import importlib.util
    faltan = [m for m in MODULOS if importlib.util.find_spec(m) is None]
    if faltan:
        sys.exit(f"Faltan paquetes de Python: {', '.join(faltan)}.\nInstalar con:  pip install -r requirements.txt")


def main():
    if "--comparar" not in sys.argv:
        verificar_paquetes()
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=R, check=True)
        respaldo()
        if "--solo-flujo" not in sys.argv:   # --solo-flujo: los tiempos ya se rehicieron; solo el flujo y la comparación
            for p in PASOS:
                correr(*p)
        subprocess.run([sys.executable, str(R / "run_all.py")], cwd=R, check=True)   # flujo por defecto, sin 22; termina en 24
    comparar()


if __name__ == "__main__":
    main()
