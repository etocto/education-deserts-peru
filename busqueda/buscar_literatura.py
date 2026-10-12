"""Kit de búsqueda bibliográfica para levantar las observaciones del manuscrito (Higher Education Policy).

Tres modos (todos en Python, sin instalar nada raro: pip install requests pyyaml pandas openpyxl):

  1. OpenAlex (gratis, sin clave). Corre las búsquedas libres de consultas.yaml y descarga título, autores, año,
     revista, DOI, resumen y citas:
        python buscar_literatura.py openalex --correo tu_correo@upeu.edu.pe
     (el correo entra en el "polite pool" de OpenAlex; es opcional)

  2. Scopus por API (opcional; requiere clave de https://dev.elsevier.com y estar en la red de la UPeU o tener token
     institucional). Corre las cadenas TITLE-ABS-KEY de consultas.yaml:
        set SCOPUS_API_KEY=xxxx      (Windows)   |   export SCOPUS_API_KEY=xxxx   (Mac)
        python buscar_literatura.py scopus --max 200
     El resumen completo no siempre lo entrega la API básica; en ese caso usa el modo 3.

  3. Exportes manuales (recomendado para Scopus y WoS). Copia la cadena de cada bloque en la búsqueda avanzada,
     exporta los resultados con título, autores, año, fuente, DOI, resumen y palabras clave
       - Scopus: CSV            -> guárdalo como exportes/scopus_<BLOQUE>.csv   (p. ej. scopus_A_novedad.csv)
       - WoS: "Tab delimited"   -> guárdalo como exportes/wos_<BLOQUE>.txt      (o Excel .xls/.xlsx)
       Si un bloque supera el máximo por exporte (WoS: 1,000), guarda las partes como wos_<BLOQUE>-2.txt, -3, etc.
     y procesa todo:
        python buscar_literatura.py procesar

Salida: resultados/literatura_consolidada.xlsx (una hoja por bloque y una hoja "todo"), sin duplicados (por DOI o por
título normalizado), con una puntuación de pertinencia por palabras clave para ordenar la lectura, y
resultados/registro_busqueda.csv (bloque, base, cadena, fecha, registros) que sirve para la Sección S9 del suplemento.
Envíame los dos archivos: con ellos selecciono los trabajos, te indico cuáles leer completos y redacto los cambios.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
import yaml

RAIZ = Path(__file__).resolve().parent
CONSULTAS = yaml.safe_load(open(RAIZ / "consultas.yaml", encoding="utf-8"))["bloques"]
EXP, RES = RAIZ / "exportes", RAIZ / "resultados"
COLS = ["bloque", "base", "titulo", "autores", "anio", "revista", "doi", "resumen", "palabras_clave", "citas", "url"]
# términos que suben la prioridad de lectura (presencia en título o resumen)
PESOS = {r"desert|oasis|oases|desierto": 3, r"field of study|fields of study|discipline|major|carrera|programa": 2,
         r"peru|latin america|am[eé]rica latina|chile|colombia|mexic|brazil|brasil|ecuador|uruguay": 2,
         r"travel time|commut|distance|distancia|tiempo de viaje|proximity": 2,
         r"closure|licens|licenciamiento|accreditation|quality assurance|sunedu": 2,
         r"higher education policy": 1, r"access|acceso|equity|equidad": 1}


def norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", s).strip()


def registrar(log, bloque, base, cadena, n):
    log.append({"bloque": bloque, "base": base, "cadena": " ".join(str(cadena).split()), "fecha": dt.date.today().isoformat(), "registros": n})


# ------------------------------------------------------------------ OpenAlex
def resumen_openalex(inv) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def openalex(args):
    filas, log = [], []
    sesion = requests.Session()
    fuente_ids = {}
    for bloque, q in CONSULTAS.items():
        filtro = "from_publication_date:2000-01-01"
        if q.get("openalex_fuente"):
            nombre = q["openalex_fuente"]
            if nombre not in fuente_ids:
                r = sesion.get("https://api.openalex.org/sources", params={"search": nombre, "mailto": args.correo}, timeout=30).json()
                fuente_ids[nombre] = next((s["id"].split("/")[-1] for s in r.get("results", []) if s["display_name"].lower() == nombre.lower()), None)
            if fuente_ids[nombre]:
                filtro += f",primary_location.source.id:{fuente_ids[nombre]}"
        for texto in q.get("openalex", []):
            params = {"search": texto, "filter": filtro, "per-page": 50, "mailto": args.correo,
                      "select": "id,doi,title,authorships,publication_year,primary_location,abstract_inverted_index,cited_by_count,keywords"}
            n = 0
            for pagina in range(1, args.paginas + 1):
                params["page"] = pagina
                r = sesion.get("https://api.openalex.org/works", params=params, timeout=60)
                if r.status_code != 200:
                    print(f"  {bloque} '{texto}': error {r.status_code}")
                    break
                res = r.json().get("results", [])
                for w in res:
                    loc = w.get("primary_location") or {}
                    filas.append({"bloque": bloque, "base": "OpenAlex", "titulo": w.get("title"),
                                  "autores": "; ".join(a["author"]["display_name"] for a in w.get("authorships", [])[:6]),
                                  "anio": w.get("publication_year"), "revista": ((loc.get("source") or {}).get("display_name")),
                                  "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                                  "resumen": resumen_openalex(w.get("abstract_inverted_index")),
                                  "palabras_clave": "; ".join(k.get("display_name", "") for k in w.get("keywords", []) or []),
                                  "citas": w.get("cited_by_count"), "url": w.get("id")})
                n += len(res)
                if len(res) < 50:
                    break
                time.sleep(0.2)
            registrar(log, bloque, "OpenAlex", texto + (f" [fuente: {q['openalex_fuente']}]" if q.get("openalex_fuente") else ""), n)
            print(f"  {bloque}: '{texto}' -> {n}")
    guardar(pd.DataFrame(filas, columns=COLS), log, "openalex")


# ------------------------------------------------------------------ Scopus API
def scopus(args):
    clave = os.environ.get("SCOPUS_API_KEY")
    if not clave:
        sys.exit("Falta SCOPUS_API_KEY (clave de https://dev.elsevier.com). Alternativa: exporte manual y modo 'procesar'.")
    cab = {"X-ELS-APIKey": clave, "Accept": "application/json"}
    if os.environ.get("SCOPUS_INSTTOKEN"):
        cab["X-ELS-Insttoken"] = os.environ["SCOPUS_INSTTOKEN"]
    filas, log = [], []
    for bloque, q in CONSULTAS.items():
        cadena = " ".join(q["scopus"].split())
        n, inicio, total = 0, 0, None
        while total is None or (inicio < total and n < args.max):
            r = requests.get("https://api.elsevier.com/content/search/scopus", headers=cab, timeout=60,
                             params={"query": cadena, "start": inicio, "count": 25, "view": "COMPLETE" if args.completo else "STANDARD"})
            if r.status_code != 200:
                print(f"  {bloque}: error {r.status_code} {r.text[:200]}")
                break
            sr = r.json()["search-results"]
            total = int(sr.get("opensearch:totalResults", 0))
            for e in sr.get("entry", []):
                if "error" in e:
                    continue
                filas.append({"bloque": bloque, "base": "Scopus", "titulo": e.get("dc:title"), "autores": e.get("dc:creator"),
                              "anio": (e.get("prism:coverDate") or "")[:4], "revista": e.get("prism:publicationName"),
                              "doi": e.get("prism:doi", ""), "resumen": e.get("dc:description", ""),
                              "palabras_clave": e.get("authkeywords", ""), "citas": e.get("citedby-count"),
                              "url": next((l["@href"] for l in e.get("link", []) if l.get("@ref") == "scopus"), "")})
                n += 1
            inicio += 25
            time.sleep(0.15)
        registrar(log, bloque, "Scopus (API)", cadena, total or 0)
        print(f"  {bloque}: {total} resultados en Scopus; descargados {n}")
    guardar(pd.DataFrame(filas, columns=COLS), log, "scopus")


# ------------------------------------------------------------------ exportes manuales
def leer_scopus_csv(p: Path, bloque: str) -> pd.DataFrame:
    d = pd.read_csv(p, encoding="utf-8-sig")
    g = lambda c: d[c] if c in d else pd.Series([""] * len(d), index=d.index)  # noqa: E731
    return pd.DataFrame({"bloque": bloque, "base": "Scopus", "titulo": g("Title"), "autores": g("Authors"), "anio": g("Year"),
                         "revista": g("Source title"), "doi": g("DOI"), "resumen": g("Abstract"),
                         "palabras_clave": g("Author Keywords"), "citas": g("Cited by"), "url": g("Link")})


def _leer_tabla_wos(p: Path) -> pd.DataFrame:
    """WoS exporta 'Tab delimited' en UTF-8, UTF-16 o Latin-1 según la versión; se prueban los tres."""
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            d = pd.read_csv(p, sep="\t", encoding=enc, quoting=3, dtype=str, index_col=False)
        except (UnicodeError, pd.errors.ParserError):
            continue
        d.columns = [str(c).strip().strip('"').lstrip("\ufeff") for c in d.columns]
        if "TI" in d.columns or "Article Title" in d.columns:
            return d
    return pd.DataFrame()


def _leer_texto_plano_wos(p: Path) -> pd.DataFrame:
    """Formato 'Plain text file' de WoS (líneas con etiquetas FN/PT/AU/TI ... ER)."""
    texto = None
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            texto = p.read_text(encoding=enc)
            if "\nTI " in texto or texto.startswith("FN "):
                break
        except UnicodeError:
            continue
    if not texto:
        return pd.DataFrame()
    regs, act, etq = [], {}, None
    for linea in texto.splitlines():
        if linea.startswith("ER"):
            regs.append(act); act, etq = {}, None
        elif linea[:2].strip() and linea[2:3] == " ":
            etq = linea[:2]
            act[etq] = (act.get(etq, "") + "; " if etq in act else "") + linea[3:].strip()
        elif linea.startswith("   ") and etq:
            sep = "; " if etq in ("AU", "AF", "DE", "ID") else " "
            act[etq] += sep + linea.strip()
    return pd.DataFrame(regs)


def leer_wos(p: Path, bloque: str) -> pd.DataFrame:
    if p.suffix.lower() in (".xls", ".xlsx"):
        d = pd.read_excel(p, dtype=str)
        m = {"TI": "Article Title", "AU": "Authors", "PY": "Publication Year", "SO": "Source Title", "DI": "DOI",
             "AB": "Abstract", "DE": "Author Keywords", "TC": "Times Cited, All Databases"}
        d = d.rename(columns={v: k for k, v in m.items()})
    else:
        d = _leer_tabla_wos(p)
        if d.empty:
            d = _leer_texto_plano_wos(p)
    if d.empty or "TI" not in d.columns:
        primera = p.read_bytes()[:200]
        print(f"  AVISO: no se reconoció el formato de {p.name}; se omite. Inicio del archivo: {primera!r}")
        return pd.DataFrame(columns=COLS)
    g = lambda c: d[c] if c in d else pd.Series([""] * len(d), index=d.index)  # noqa: E731
    return pd.DataFrame({"bloque": bloque, "base": "WoS", "titulo": g("TI"), "autores": g("AU"), "anio": g("PY"),
                         "revista": g("SO"), "doi": g("DI"), "resumen": g("AB"), "palabras_clave": g("DE"),
                         "citas": g("TC"), "url": ""})


def procesar(args):
    partes, log = [], []
    for p in sorted(EXP.glob("*")):
        nombre = p.name.strip()
        if nombre != p.name:                      # nombre con espacios al inicio o al final: se corrige
            p = p.rename(p.with_name(nombre))
            print(f"  renombrado (tenía espacios): {nombre}")
        m = re.match(r"(scopus|wos)_(\w+?)(?:-\d+)?\.(csv|txt|xls|xlsx)$", nombre, re.I)
        if not m:
            print(f"  ignorado (nombre no reconocido): {p.name}")
            continue
        base, bloque = m.group(1).lower(), m.group(2)
        if bloque not in CONSULTAS:
            print(f"  ignorado (bloque desconocido '{bloque}'): {p.name}")
            continue
        d = leer_scopus_csv(p, bloque) if base == "scopus" else leer_wos(p, bloque)
        partes.append(d)
        registrar(log, bloque, "Scopus" if base == "scopus" else "Web of Science",
                  CONSULTAS[bloque]["scopus" if base == "scopus" else "wos"], len(d))
        print(f"  {p.name}: {len(d)} registros")
    previos = RES / "literatura_cruda_openalex.csv"
    if previos.exists():
        partes.append(pd.read_csv(previos))
        print("  + resultados de OpenAlex ya descargados")
    if not partes:
        sys.exit("No hay exportes en la carpeta exportes/ (ver instrucciones al inicio del archivo).")
    guardar(pd.concat(partes, ignore_index=True), log, "exportes")


# ------------------------------------------------------------------ consolidación
def puntaje(fila) -> int:
    t = norm(f"{fila.titulo} {fila.resumen} {fila.revista}")
    return sum(w for pat, w in PESOS.items() if re.search(pat, t))


def guardar(df: pd.DataFrame, log: list, origen: str):
    RES.mkdir(exist_ok=True)
    df = df.copy()
    if origen == "openalex":
        df.to_csv(RES / "literatura_cruda_openalex.csv", index=False, encoding="utf-8")
    df["doi"] = df.doi.fillna("").astype(str).str.lower().str.strip()
    df["clave"] = df.doi.where(df.doi != "", df.titulo.map(norm).str[:120])
    df["puntaje"] = df.apply(puntaje, axis=1)
    agg = (df.sort_values("puntaje", ascending=False)
             .groupby(["bloque", "clave"], as_index=False)
             .agg({**{c: "first" for c in COLS if c not in ("bloque", "base")}, "base": lambda s: "; ".join(sorted(set(s))), "puntaje": "max"}))
    agg = agg.sort_values(["bloque", "puntaje", "citas"], ascending=[True, False, False])
    agg["decision_autor"] = ""   # INCLUIR / EXCLUIR / LEER COMPLETO
    agg["nota"] = ""
    salida = RES / "literatura_consolidada.xlsx"
    with pd.ExcelWriter(salida) as w:
        agg.drop(columns="clave").to_excel(w, sheet_name="todo", index=False)
        for b, g in agg.groupby("bloque"):
            g.drop(columns="clave").to_excel(w, sheet_name=b[:31], index=False)
    reg = RES / "registro_busqueda.csv"
    lg = pd.DataFrame(log)
    if reg.exists():
        lg = pd.concat([pd.read_csv(reg), lg], ignore_index=True).drop_duplicates()
    lg.to_csv(reg, index=False, encoding="utf-8")
    print(f"\n{len(df)} registros -> {len(agg)} únicos por bloque. Archivos: {salida.name}, {reg.name} (en resultados/)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="modo", required=True)
    a = sub.add_parser("openalex"); a.add_argument("--correo", default=None); a.add_argument("--paginas", type=int, default=2)
    b = sub.add_parser("scopus"); b.add_argument("--max", type=int, default=200); b.add_argument("--completo", action="store_true")
    sub.add_parser("procesar")
    args = ap.parse_args()
    {"openalex": openalex, "scopus": scopus, "procesar": procesar}[args.modo](args)


if __name__ == "__main__":
    main()
