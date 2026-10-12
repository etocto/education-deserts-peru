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


# ---------------------------------------------------------------- revisión v4 (obs. M2): duplicados en la red
def _m04b():
    spec = importlib.util.spec_from_file_location("m04b", Path(__file__).resolve().parents[1] / "src" / "04b_tiempos_viaje.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_matriz_minimos_conserva_el_tramo_mas_corto():
    from utils import matriz_minimos
    g = matriz_minimos([0, 0, 1], [1, 1, 2], [5.0, 3.0, 2.0], 3)
    assert g[0, 1] == 3.0 and g[1, 2] == 2.0


def test_dos_locales_en_el_mismo_nodo_no_suman_acceso():
    from utils import matriz_minimos
    m = _m04b()
    g = matriz_minimos([0, 1], [1, 0], [10.0, 10.0], 2)
    uno = m.tiempos_desde_fuentes(g, np.array([0]), np.array([2.0]))
    dos = m.tiempos_desde_fuentes(g, np.array([0, 0]), np.array([2.0, 3.0]))
    assert np.allclose(uno, dos) and np.isclose(dos[1], 12.0)


def test_mas_oferta_nunca_alarga_el_tiempo():
    from utils import matriz_minimos
    m = _m04b()
    rng = np.random.default_rng(1)
    n = 60
    a, b = rng.integers(0, n, 300), rng.integers(0, n, 300)
    w = rng.uniform(1, 9, 300)
    g = matriz_minimos(np.r_[a, b], np.r_[b, a], np.r_[w, w], n)
    post = rng.choice(n, 5, replace=False)
    rel = np.r_[post, rng.choice(n, 5)]          # puede repetir nodos de post
    pre = np.r_[rel, rng.choice(n, 10)]
    acc = lambda k: np.full(len(k), 1.5)        # noqa: E731
    t_post, t_rel, t_pre = (m.tiempos_desde_fuentes(g, k, acc(k)) for k in (post, rel, pre))
    assert np.all(t_pre <= t_rel + 1e-9) and np.all(t_rel <= t_post + 1e-9)
