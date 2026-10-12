"""Paso 20 – Elección de k en la tipología y estabilidad de los grupos (revisión, obs. 18).

Mismas variables que el paso 7 (% de 5.º en desierto por familia, estandarizadas con media 0 y varianza 1).
1. Coeficiente de silueta para k = 2, ..., 10 (k-means, 50 inicios, semilla de config.yaml).
2. Estabilidad del k elegido en el paso 7: 100 submuestras del 80 % de los distritos; en cada una se reajusta k-means y
   se compara la partición con la de la muestra completa (índice de Rand ajustado, ARI, sobre los distritos comunes).

Salida: salidas/tablas/tipologia_silueta_k.csv, salidas/tablas/tipologia_estabilidad.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import CFG, FAMILIAS, guardar, ruta

S = CFG["modelos"]["semilla"]


def main(k_elegido: int = 6, reps: int = 100):
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0) & acc.familia.isin(FAMILIAS)]
    Z = StandardScaler().fit_transform(acc.pivot_table(index="ubigeo", columns="familia", values="pct_quinto_en_desierto").dropna())
    sil = pd.DataFrame([{"k": k, "silueta": silhouette_score(Z, KMeans(k, n_init=50, random_state=S).fit_predict(Z))}
                        for k in range(2, 11)])
    guardar(sil.round(3), "tipologia_silueta_k.csv", "tablas")
    print(sil.round(3).to_string(index=False))
    ref = KMeans(k_elegido, n_init=50, random_state=S).fit_predict(Z)
    rng = np.random.default_rng(S)
    ari = []
    for _ in range(reps):
        idx = rng.choice(len(Z), int(0.8 * len(Z)), replace=False)
        lab = KMeans(k_elegido, n_init=20, random_state=int(rng.integers(1e6))).fit_predict(Z[idx])
        ari.append(adjusted_rand_score(ref[idx], lab))
    e = pd.DataFrame([{"k": k_elegido, "reps": reps, "ari_media": np.mean(ari), "ari_p5": np.percentile(ari, 5), "ari_min": np.min(ari)}])
    guardar(e.round(3), "tipologia_estabilidad.csv", "tablas")
    print(e.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
