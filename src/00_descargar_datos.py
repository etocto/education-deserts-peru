"""Paso 0 – Descarga (o verificación) de los datos públicos nacionales.

Los archivos de datos/raw ya vienen incluidos en el proyecto (descargados el 2026-09-27).
Este script sirve para reproducir la descarga desde cero en una computadora con acceso normal a internet:
  python src/00_descargar_datos.py            -> solo verifica que existan
  python src/00_descargar_datos.py --forzar   -> vuelve a descargar y regenerar los extractos

Nota: algunos portales del Estado peruano bloquean descargas automatizadas o cambian sus URL.
Si una descarga falla, el script indica la página oficial para bajar el archivo a mano.
"""
from __future__ import annotations

import io
import sys
import zipfile

import pandas as pd
import requests

from utils import RAIZ, ruta

FUENTES = {
    "sunedu_programas": "https://www.datosabiertos.gob.pe/sites/default/files/Programas%20de%20Universidades_8.csv",
    "sunedu_licenciamiento": "https://www.datosabiertos.gob.pe/sites/default/files/Licenciamiento%20Institucional_18.csv",
    "escale_padron_zip": "https://escale.minedu.gob.pe/documents/10156/10632985/00_Padron.zip",
    "escale_matricula_zip": "https://escale.minedu.gob.pe/documents/10156/10632985/01_Matricula_01.zip",
    "pobreza_pdf": "https://www.inei.gob.pe/media/MenuRecursivo/publicaciones_digitales/Est/Lib1718/Libro.pdf",
    "limites_inei": "https://arcgis.inei.gob.pe:6443/arcgis/rest/services/ENAHO/ENAHO002_AGS_CONGLOMERADOS/MapServer/2/query",
}
PAGINAS = {
    "sunedu": "https://www.datosabiertos.gob.pe/dataset/sunedu-programas-acad%C3%A9micos",
    "escale": "https://escale.minedu.gob.pe/uee/-/document_library_display/GMv7/view/10632985",
    "pobreza": "https://www.gob.pe/institucion/inei/informes-publicaciones/3204872-mapa-de-pobreza-provincial-y-distrital-2018",
    "censo2017": "https://censos2017.inei.gob.pe/redatam/",
}
CABECERA = {"User-Agent": "Mozilla/5.0 (investigacion academica)"}


def bajar(url: str) -> bytes:
    r = requests.get(url, headers=CABECERA, timeout=300)
    r.raise_for_status()
    return r.content


def dbf_a_df(contenido: bytes) -> pd.DataFrame:
    from dbfread import DBF  # pip install dbfread
    tmp = RAIZ / "datos/interim/_tmp.dbf"
    tmp.write_bytes(contenido)
    df = pd.DataFrame(iter(DBF(tmp, encoding="latin1", char_decode_errors="replace")))
    tmp.unlink()
    return df


def regenerar_escale():
    pad = zipfile.ZipFile(io.BytesIO(bajar(FUENTES["escale_padron_zip"])))
    df = dbf_a_df(pad.read([n for n in pad.namelist() if n.lower().endswith(".dbf")][0]))
    cols = ["COD_MOD", "ANEXO", "CODLOCAL", "CEN_EDU", "NIV_MOD", "D_FORMA", "GESTION", "D_GESTION", "GES_DEP", "D_GES_DEP",
            "CODCP_INEI", "AREA_CENSO", "DAREACENSO", "CODGEO", "D_DPTO", "D_PROV", "D_DIST", "NLAT_IE", "NLONG_IE", "ESTADO",
            "D_ESTADO", "ZFRONTERA", "VRAEM", "TALUM", "TALUM_HOM", "TALUM_MUJ", "TDOC", "TSEC", "COD_CAR", "D_COD_CAR"]
    df[df.NIV_MOD == "F0"][cols].to_csv(ruta("escale_padron"), sep="|", index=False)
    mat = zipfile.ZipFile(io.BytesIO(bajar(FUENTES["escale_matricula_zip"])))
    m = dbf_a_df(mat.read([n for n in mat.namelist() if n.lower().endswith(".dbf")][0]))
    m = m[m.NIV_MOD == "F0"]
    dcols = [f"D{i:02d}" for i in range(1, 11)]
    m[dcols] = m[dcols].apply(pd.to_numeric, errors="coerce").fillna(0)
    g = m.groupby(["COD_MOD", "ANEXO", "CODGEO"], as_index=False)[dcols].sum()
    g.columns = ["COD_MOD", "ANEXO", "CODGEO"] + [f"G{k}_{s}" for k in range(1, 6) for s in ("H", "M")]
    g.to_csv(ruta("escale_matricula"), sep="|", index=False)


def descargar_limites_inei():
    """Límite distrital oficial del INEI (CPV 2017) desde su servicio ArcGIS; genera también provincias y departamentos."""
    import json
    import geopandas as gpd
    from shapely.geometry import Polygon, mapping
    feats = []
    for off in range(0, 2000, 500):
        params = {"where": "1=1", "outFields": "UBIGEO,CCDD,CCPP,CCDI,NOMBDEP,NOMBPROV,NOMBDIST,CAPITAL,FUENTE",
                  "returnGeometry": "true", "outSR": 4326, "maxAllowableOffset": 0.003, "geometryPrecision": 4,
                  "resultOffset": off, "resultRecordCount": 500, "orderByFields": "OBJECTID", "f": "json"}
        feats += requests.get(FUENTES["limites_inei"], params=params, headers=CABECERA, timeout=300).json()["features"]
    out = []
    for ft in feats:
        polys = sorted([p for p in (Polygon(r).buffer(0) for r in ft["geometry"]["rings"] if len(r) >= 4) if not p.is_empty],
                       key=lambda p: -p.area)
        geom = None
        for p in polys:   # anillos contenidos en otro = huecos
            geom = p if geom is None else (geom.difference(p) if geom.buffer(1e-9).contains(p) else geom.union(p))
        out.append({"type": "Feature", "properties": ft["attributes"], "geometry": mapping(geom)})
    json.dump({"type": "FeatureCollection", "features": out}, open(ruta("limites_distrital"), "w"))
    g = gpd.read_file(ruta("limites_distrital"))
    g.dissolve(by=g.UBIGEO.str[:4]).reset_index(names="IDPROV")[["IDPROV", "NOMBDEP", "NOMBPROV", "geometry"]].to_file(ruta("limites_provincial"), driver="GeoJSON")
    g.dissolve(by=g.UBIGEO.str[:2]).reset_index(names="IDDPTO")[["IDDPTO", "NOMBDEP", "geometry"]].to_file(ruta("limites_departamental"), driver="GeoJSON")


def main(forzar: bool):
    esperados = ["sunedu_programas", "sunedu_licenciamiento", "escale_padron", "escale_matricula",
                 "escale_superior_no_univ", "pobreza", "limites_distrital", "limites_provincial", "limites_departamental"]
    faltan = [k for k in esperados if not ruta(k).exists()]
    for k in esperados:
        print(f"{'OK   ' if k not in faltan else 'FALTA'} {ruta(k).relative_to(RAIZ)}")
    if not forzar and not faltan:
        print("\nTodos los datos están presentes. Use --forzar para volver a descargarlos.")
        return
    if forzar or "limites_distrital" in faltan:
        try:
            descargar_limites_inei()
            print("descargados límites INEI")
        except Exception as e:  # noqa: BLE001
            print(f"No se pudieron descargar los límites del INEI: {e}")
    for clave in ["sunedu_programas", "sunedu_licenciamiento"]:
        if forzar or clave in faltan:
            try:
                ruta(clave).write_bytes(bajar(FUENTES[clave]))
                print("descargado", clave)
            except Exception as e:  # noqa: BLE001
                print(f"No se pudo descargar {clave}: {e}\n  Descárguelo a mano desde {PAGINAS.get(clave.split('_')[0], FUENTES[clave])}")
    if forzar or "escale_padron" in faltan or "escale_matricula" in faltan:
        try:
            regenerar_escale()
            print("regenerados extractos ESCALE")
        except Exception as e:  # noqa: BLE001
            print(f"No se pudo regenerar ESCALE: {e}\n  Página oficial: {PAGINAS['escale']}")
    if "pobreza" in faltan:
        print(f"El mapa de pobreza se extrajo del PDF oficial (Anexo 2). Página: {PAGINAS['pobreza']}")
    print("\nCenso 2017: tablas REDATAM en datos/raw/censo2017/redatam (programa REDATAM en su LEEME_fuente.txt)")


if __name__ == "__main__":
    main("--forzar" in sys.argv)
