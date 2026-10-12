"""Pruebas unitarias de las funciones de distancia, decaimiento, clasificación de desiertos y familias.
Uso (desde la raíz del repositorio):  python -m pytest -q tests
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import FAMILIAS, familia_programa, haversine_matriz, peso_decaimiento  # noqa: E402


def test_haversine_un_grado_de_latitud():
    d = haversine_matriz([0.0], [-75.0], [1.0], [-75.0])
    assert abs(d[0, 0] - 111.19) < 0.05


def test_haversine_lima_cusco():
    # Plaza de Armas de Lima (-12.0464, -77.0300) y del Cusco (-13.5164, -71.9785): ~ 575 km en línea recta
    d = haversine_matriz([-12.0464], [-77.0300], [-13.5164], [-71.9785])[0, 0]
    assert 565 < d < 585


def test_haversine_simetria_y_cero():
    lat, lon = np.array([-5.0, -15.0]), np.array([-80.0, -70.0])
    d = haversine_matriz(lat, lon, lat, lon)
    assert np.allclose(np.diag(d), 0) and np.isclose(d[0, 1], d[1, 0])


def test_decaimiento():
    d = np.array([0.0, 30.0, 60.0, 61.0])
    w = peso_decaimiento(d, 60.0)
    assert np.isclose(w[0], 1) and np.isclose(w[2], 0) and w[3] == 0 and 0 < w[1] < 1
    assert np.array_equal(peso_decaimiento(d, 60.0, "escalon"), np.array([1, 1, 1, 0]))


def test_clasificacion_desierto_y_origen():
    # misma regla que los pasos 4 y 13: desierto si la distancia mínima supera 60 km
    calc = importlib.import_module("04_accesibilidad").calcular
    col = pd.DataFrame({"COD_MOD": ["a", "b"], "ANEXO": ["0", "0"], "ubigeo": ["010101", "010101"],
                        "lat": [0.0, 0.0], "lon": [0.0, 1.0], "quinto": [10, 20]})
    loc = pd.DataFrame({"lat": [0.0], "lon": [0.3], "n_programas": [3], **{f: [1 if f == FAMILIAS[0] else 0] for f in FAMILIAS}})
    r = calc(col, loc, "post", ["Total", FAMILIAS[0], FAMILIAS[1]], 1.0)
    tot = r[r.familia == "Total"].set_index("COD_MOD")
    assert tot.loc["a", "desierto"] == 0 and tot.loc["b", "desierto"] == 1      # 33 km y 78 km
    assert np.isinf(r[r.familia == FAMILIAS[1]].dist_min_km).all()              # familia sin oferta: desierto
    assert (r[r.familia == FAMILIAS[1]].desierto == 1).all()


def test_familias():
    assert familia_programa("SALUD", "MEDICINA HUMANA") == ("Salud y bienestar", "N2")
    assert familia_programa("NINGUNO", "EDUCACION PRIMARIA INTERCULTURAL")[0] == "Educación"


def test_grupos_excluyentes_fase1():
    """Los cuatro grupos del paso 29 son excluyentes y cubren a todo alumno que cumple al menos un criterio."""
    import itertools
    for D, Dp, T in itertools.product([0, 1], repeat=3):
        if Dp and not D:
            continue   # la oferta reconstruida contiene a la licenciada: sin desierto post no hay desierto pre
        D_, Dp_, T_ = bool(D), bool(Dp), bool(T)
        grupos = [D_ and Dp_ and T_, D_ and not Dp_ and T_, (not D_) and T_, D_ and not T_]
        assert sum(grupos) == (1 if (D_ or T_) else 0)
