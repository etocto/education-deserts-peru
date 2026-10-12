"""Paso 31 – Plantilla de validación de tiempos con el tiempo modelado del PAR exacto colegio-sede (obs. 8, N1).

La plantilla del paso 16 trae el tiempo mínimo del colegio a CUALQUIER sede, pero la sede de destino es la más
cercana en línea recta. Para comparar el mismo viaje, este paso calcula en la misma red de OSM (src/red_liviana.py)
el tiempo modelado del colegio a ESA sede, y añade enlaces de Google Maps para consultar el tiempo observado.

Entrada:  datos/manual/validacion_tiempos_plantilla.csv (paso 16) y datos/raw/osm/peru-latest.osm.pbf
Salidas:  datos/manual/validacion_tiempos_plantilla_pares.csv  y  .xlsx (para llenar a mano)
Uso:      python src/31_validacion_pares.py      (unos 3 GB de RAM; 5-10 min)
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

from red_liviana import construir_red
from utils import CFG, RAIZ


def main():
    spec = importlib.util.spec_from_file_location("m04b", RAIZ / "src" / "04b_tiempos_viaje.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    obs = RAIZ / "datos/manual/validacion_tiempos.csv"
    if obs.exists():   # actualizar el tiempo modelado del par en el archivo ya llenado (p. ej., tras calibrar)
        p = pd.read_csv(obs, dtype={"COD_MOD": str, "ANEXO": str})
        g, lat_red, lon_red = construir_red(m.PBF)
        nc, acc_c, _ = m.anclar(lat_red, lon_red, p.lat_colegio, p.lon_colegio)
        ns, acc_s, _ = m.anclar(lat_red, lon_red, p.lat_sede, p.lon_sede)
        p["tiempo_modelado_par_min"] = [round(float(dijkstra(g, indices=nc[i], limit=1200)[ns[i]] + acc_c[i] + acc_s[i]), 0)
                                        for i in range(len(p))]
        p.to_csv(obs, index=False, encoding="utf-8")
        print(p[["colegio", "tiempo_modelado_par_min", "tiempo_observado_min"]].to_string(index=False))
        return
    p = pd.read_csv(RAIZ / "datos/manual/validacion_tiempos_plantilla.csv", dtype={"COD_MOD": str, "ANEXO": str})
    g, lat_red, lon_red = construir_red(m.PBF)
    nc, acc_c, km_c = m.anclar(lat_red, lon_red, p.lat_colegio, p.lon_colegio)
    ns, acc_s, _ = m.anclar(lat_red, lon_red, p.lat_sede, p.lon_sede)
    t = []
    for i in range(len(p)):
        d = dijkstra(g, indices=nc[i], limit=600)
        t.append(d[ns[i]] + acc_c[i] + acc_s[i])
        print(f"  par {i + 1}/{len(p)}: {t[-1]:.0f} min")
    p.insert(p.columns.get_loc("tiempo_modelado_min") + 1, "tiempo_modelado_par_min", np.round(t, 0))
    p = p.rename(columns={"tiempo_modelado_min": "tiempo_modelado_min_cualquier_sede"})
    p["km_colegio_a_red"] = km_c.round(2)
    p["google_maps"] = [f"https://www.google.com/maps/dir/?api=1&origin={a},{b}&destination={c},{d}&travelmode=driving"
                        for a, b, c, d in zip(p.lat_colegio, p.lon_colegio, p.lat_sede, p.lon_sede)]
    for c in ["tiempo_observado_min", "medio_transporte", "fuente", "fecha_consulta", "observaciones"]:
        p[c] = p.get(c, "")
    sal = RAIZ / "datos/manual/validacion_tiempos_plantilla_pares"
    p.to_csv(f"{sal}.csv", index=False, encoding="utf-8")
    p.to_excel(f"{sal}.xlsx", index=False)
    print(f"-> {sal}.csv / .xlsx")
    print(p[["colegio", "departamento", "dist_linea_recta_km", "tiempo_modelado_min_cualquier_sede", "tiempo_modelado_par_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
