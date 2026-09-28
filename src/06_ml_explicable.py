"""Paso 6 – Machine learning interpretable: qué explica la accesibilidad distrital, con relaciones no lineales.

Objetivo (OBJETIVO abajo): log(1 + distancia media al programa más cercano) del distrito, escenario post.
Alternativa: "acc_2sfca" (índice 2SFCA); en la prueba inicial fue mucho menos predecible con estas covariables.
Modelos: regresión lineal (referencia), Random Forest y XGBoost, con validación cruzada espacial
(GroupKFold por departamento: el modelo se evalúa en departamentos que no vio al entrenar).
Interpretación: valores SHAP del mejor modelo (importancia global, dependencia e interacciones).

Nota metodológica: el propósito es explicar (no predecir) la distribución de la oferta, en la línea de
Litmeyer y Hennemann (2024). Por eso las variables son características del territorio, no de la oferta misma.

Salidas: salidas/modelos/metricas_cv.csv, salidas/figuras/shap_*.png, salidas/modelos/shap_valores.csv
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from utils import CFG, guardar, ruta

S = CFG["modelos"]["semilla"]
OBJETIVO = "dist_min_km"   # o "acc_2sfca"
ETIQUETAS = {   # nombres en inglés para las figuras del manuscrito
    "log_quinto": "Final-year students (log)",
    "log_pob": "District population (log)",
    "pobreza_punto_medio": "Poverty rate (%)",
    "pct_rural_quinto": "Rural share of final-year students",
    "pct_lengua_originaria": "Indigenous mother tongue (15-29)",
    "pct_autoid_indigena": "Indigenous self-identification",
    "pct_afro": "Afro-Peruvian self-identification",
    "pct_sin_superior": "Adults without higher education",
}


def main():
    acc = pd.read_csv(ruta("processed") / "accesibilidad_distritos.csv", dtype={"ubigeo": str})
    acc = acc[(acc.escenario == "post") & (acc.familia == "Total") & (acc.factor_sinuosidad == 1.0)]
    base = pd.read_csv(ruta("processed") / "distritos_base.csv", dtype={"ubigeo": str, "dpto": str})
    d = base.merge(acc[["ubigeo", OBJETIVO]], on="ubigeo")
    cen = ruta("censo2017_agregado")
    if cen.exists():
        d = d.merge(pd.read_csv(cen, dtype={"ubigeo": str}), on="ubigeo", how="left")
    d = d[d.quinto > 0].copy()
    d["log_quinto"] = np.log(d.quinto)
    d["log_pob"] = np.log(d.poblacion_2020)
    feats = [c for c in ["log_quinto", "log_pob", "pobreza_punto_medio", "pct_rural_quinto",
                         "pct_lengua_originaria", "pct_autoid_indigena", "pct_afro", "pct_sin_superior"] if c in d]
    d = d.dropna(subset=feats + [OBJETIVO])
    X, y, g = d[feats], np.log1p(d[OBJETIVO]), d.dpto

    modelos = {"Lineal": LinearRegression(),
               "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=S, n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        modelos["XGBoost"] = XGBRegressor(**CFG["modelos"]["xgboost"], random_state=S)
    except Exception as e:   # ImportError o XGBoostError (en Mac falta libomp: brew install libomp)
        print(f"XGBoost no disponible ({type(e).__name__}); se continúa con lineal y Random Forest.")
        print("  En Mac se soluciona con: brew install libomp")

    cv = GroupKFold(n_splits=5)
    met = []
    for nombre, m in modelos.items():
        pred = np.zeros(len(y))
        for tr, te in cv.split(X, y, g):
            m.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = m.predict(X.iloc[te])
        met.append({"modelo": nombre, "r2_cv": r2_score(y, pred), "mae_cv": mean_absolute_error(y, pred), "n": len(y)})
    met = pd.DataFrame(met)
    guardar(met, "metricas_cv.csv", "modelos")
    print(met.to_string(index=False))

    mejor = met.sort_values("r2_cv", ascending=False).modelo.iloc[0]
    if mejor == "Lineal":
        mejor = "XGBoost" if "XGBoost" in modelos else "RandomForest"
    m = modelos[mejor].fit(X, y)
    try:
        import shap
    except ImportError:
        print("shap no instalado: pip install shap")
        return
    ex = shap.TreeExplainer(m)
    sv = ex.shap_values(X)
    guardar(pd.DataFrame(sv, columns=feats).assign(ubigeo=d.ubigeo.values), "shap_valores.csv", "modelos")
    Xf = X.rename(columns=ETIQUETAS)
    shap.summary_plot(sv, Xf, show=False)
    plt.xlabel("SHAP value (impact on log distance to the nearest program)")
    plt.tight_layout()
    plt.savefig(ruta("figuras") / f"shap_resumen_{mejor}.png", dpi=600)
    plt.savefig(ruta("manuscrito_figuras") / f"shap_resumen_{mejor}.png", dpi=600)
    plt.close()
    for f in feats[:4]:
        shap.dependence_plot(ETIQUETAS.get(f, f), sv, Xf, show=False)
        plt.tight_layout()
        plt.savefig(ruta("figuras") / f"shap_dependencia_{f}.png", dpi=600)
        plt.close()
    print(f"SHAP calculado con {mejor}")


if __name__ == "__main__":
    main()
