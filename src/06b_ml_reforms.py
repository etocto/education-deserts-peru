"""Paso 6b – Reporte REFORMS del componente de aprendizaje automático (revisión, obs. 7). Material suplementario.

Mismo objetivo, predictores y validación que el paso 6: log(1 + distancia al programa más cercano), 8 predictores
distritales, GroupKFold con 5 pliegues por departamento (el modelo se evalúa en departamentos que no vio).
Reporta:
  1. hiperparámetros de cada modelo (fijos, sin búsqueda: valores de config.yaml y del paso 6);
  2. R2 y MAE por pliegue, media y desviación estándar, y prueba de Wilcoxon pareada por pliegue (bosque vs. lineal);
  3. importancia por permutación AGRUPANDO predictores muy correlacionados (tamaño: log alumnos + log población;
     etnicidad: autoidentificación indígena + lengua originaria), calculada en los pliegues de prueba.

Salidas: salidas/modelos/reforms_hiperparametros.csv, reforms_pliegues.csv, reforms_importancia_grupos.csv
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from utils import CFG, guardar, ruta

S = CFG["modelos"]["semilla"]
GRUPOS = {"Size (log students, log population)": ["log_quinto", "log_pob"], "Poverty": ["pobreza_punto_medio"],
          "Rurality": ["pct_rural_quinto"], "Indigenous (self-identification, mother tongue)": ["pct_autoid_indigena", "pct_lengua_originaria"],
          "Afro-Peruvian": ["pct_afro"], "Adults without higher education": ["pct_sin_superior"]}


def datos():
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.familia == "Total") & (acc.factor_sinuosidad == 1.0)]
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str, "dpto": str})
    d = base.merge(acc[["ubigeo", "dist_min_km"]], on="ubigeo").merge(pd.read_csv(ruta("censo2017_agregado"), dtype={"ubigeo": str}), on="ubigeo", how="left")
    d = d[d.quinto > 0].copy()
    d["log_quinto"], d["log_pob"] = np.log(d.quinto), np.log(d.poblacion_2020)
    feats = [c for g in GRUPOS.values() for c in g]
    d = d.dropna(subset=feats + ["dist_min_km"])
    return d[feats], np.log1p(d.dist_min_km), d.dpto


def main(reps: int = 20):
    X, y, g = datos()
    modelos = {"Linear": LinearRegression(),
               "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=S, n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        modelos["XGBoost"] = XGBRegressor(**CFG["modelos"]["xgboost"], random_state=S)
    except Exception:  # noqa: BLE001
        pass
    hp = [{"modelo": n, "hiperparametros": str({k: v for k, v in m.get_params().items()
                                                 if k in ["n_estimators", "min_samples_leaf", "max_depth", "learning_rate", "subsample", "colsample_bytree", "fit_intercept"]}),
           "busqueda": "none (fixed a priori)", "semilla": S} for n, m in modelos.items()]
    guardar(pd.DataFrame(hp), "reforms_hiperparametros.csv", "modelos")
    filas, imp = [], []
    rng = np.random.default_rng(S)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        for n, m in modelos.items():
            m.fit(X.iloc[tr], y.iloc[tr])
            p = m.predict(X.iloc[te])
            filas.append({"pliegue": k + 1, "departamentos_prueba": g.iloc[te].nunique(), "n_prueba": len(te), "modelo": n,
                          "r2": r2_score(y.iloc[te], p), "mae": mean_absolute_error(y.iloc[te], p)})
            if n == "RandomForest":
                base_r2 = r2_score(y.iloc[te], p)
                for nom, cols in GRUPOS.items():
                    caidas = []
                    for _ in range(reps):
                        Xp = X.iloc[te].copy()
                        perm = rng.permutation(len(te))
                        Xp[cols] = Xp[cols].to_numpy()[perm]          # permuta el bloque completo (conserva la correlación interna)
                        caidas.append(base_r2 - r2_score(y.iloc[te], m.predict(Xp)))
                    imp.append({"pliegue": k + 1, "grupo": nom, "caida_r2": np.mean(caidas)})
    f = pd.DataFrame(filas)
    guardar(f.round(4), "reforms_pliegues.csv", "modelos")
    res = f.groupby("modelo")[["r2", "mae"]].agg(["mean", "std"]).round(3)
    print(res.to_string())
    a, b = f[f.modelo == "RandomForest"].r2.to_numpy(), f[f.modelo == "Linear"].r2.to_numpy()
    print(f"Wilcoxon pareado por pliegue (bosque vs lineal, R2): p = {wilcoxon(a, b).pvalue:.3f} (5 pliegues: potencia baja)")
    i = pd.DataFrame(imp).groupby("grupo").caida_r2.agg(["mean", "std"]).sort_values("mean", ascending=False)
    guardar(i.round(4).reset_index(), "reforms_importancia_grupos.csv", "modelos")
    print(i.round(3).to_string())


if __name__ == "__main__":
    main()
