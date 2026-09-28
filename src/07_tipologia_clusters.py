"""Paso 7 – Tipología de distritos según su acceso por familia de carreras (desierto, oasis y situaciones intermedias).

Variables: % de alumnos de 5.º en desierto para cada una de las 9 familias (escenario post, línea recta).
Método: estandarización, PCA para describir los ejes y k-means (k elegido por silueta entre los valores de config).
Se reporta el perfil medio de cada grupo con pobreza y ruralidad para nombrar los tipos.

Salidas: salidas/tablas/tipologia_perfiles.csv, datos/processed/tipologia_distritos.csv, salidas/figuras/pca_tipologia.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import CFG, FAMILIAS, guardar, ruta


def main():
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.factor_sinuosidad == 1.0) & acc.familia.isin(FAMILIAS)]
    w = acc.pivot_table(index="ubigeo", columns="familia", values="pct_quinto_en_desierto").dropna()
    Z = StandardScaler().fit_transform(w)
    sil = {k: silhouette_score(Z, KMeans(k, n_init=20, random_state=CFG["modelos"]["semilla"]).fit_predict(Z))
           for k in CFG["modelos"]["kmeans_k"]}
    k = max(sil, key=sil.get)
    print("silueta:", {k_: round(v, 3) for k_, v in sil.items()}, "-> k =", k)
    km = KMeans(k, n_init=50, random_state=CFG["modelos"]["semilla"]).fit(Z)
    w["grupo"] = km.labels_
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str})
    t = w.reset_index().merge(base[["ubigeo", "quinto", "pobreza_punto_medio", "pct_rural_quinto"]], on="ubigeo")
    guardar(t, "tipologia_distritos.csv")
    perf = t.groupby("grupo").agg(distritos=("ubigeo", "size"), quinto=("quinto", "sum"),
                                  pobreza=("pobreza_punto_medio", "mean"), rural=("pct_rural_quinto", "mean"),
                                  **{f: (f, "mean") for f in FAMILIAS}).round(3)
    guardar(perf.reset_index(), "tipologia_perfiles.csv", "tablas")
    print(perf.T.to_string())

    pc = PCA(2).fit(Z)
    xy = pc.transform(Z)
    plt.figure(figsize=(6, 5))
    for g in range(k):
        plt.scatter(xy[w.grupo == g, 0], xy[w.grupo == g, 1], s=6, label=f"Grupo {g}")
    plt.xlabel(f"CP1 ({pc.explained_variance_ratio_[0]:.0%})")
    plt.ylabel(f"CP2 ({pc.explained_variance_ratio_[1]:.0%})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(ruta("figuras") / "pca_tipologia.png", dpi=600)


if __name__ == "__main__":
    main()
