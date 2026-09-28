# Publicar el proyecto en GitHub y obtener el DOI en Zenodo

Repositorio propuesto: `https://github.com/etocto/education-deserts-peru` (público; Zenodo solo archiva repositorios públicos).

Qué se sube: código (`src/`, `run_all.py`, `config.yaml`, `requirements.txt`), datos oficiales de `datos/raw` (~15 MB), correcciones manuales, tiempos de viaje procesados y `salidas/`.
Qué NO se sube (lo excluye `.gitignore`): `.venv`, `NOTAS_INTERNAS.md` (el README anterior, con pendientes), el extracto de OSM (245 MB), los demás archivos de `datos/processed` (se regeneran), el manuscrito y sus PDF (se agregan cuando el artículo sea aceptado), y los PDF y exportaciones de Scopus (tienen derechos de terceros).

## 1. Crear el repositorio vacío en GitHub

En https://github.com/new: nombre `education-deserts-peru`, **Public**, sin README, sin .gitignore y sin licencia (ya están en el proyecto).

## 2. Configurar git una sola vez

Usa el correo asociado a tu cuenta de GitHub:

```
git config --global user.name "Esteban Tocto-Cano"
git config --global user.email "TU_CORREO_DE_GITHUB"
```

## 3. Primer commit y revisión antes de subir

Desde la carpeta del proyecto:

```
git init
git add .
git status
```

Revisa que la lista no incluya `.venv`, `manuscrito_mdpi`, `peru-latest.osm.pbf` ni PDF. Para ver el tamaño total que se subirá:

```
git count-objects -vH
```

Debe ser menor de 60 MB. Luego:

```
git commit -m "Pipeline reproducible: desiertos universitarios en el Perú (v1.0.0)"
git branch -M main
git remote add origin https://github.com/etocto/education-deserts-peru.git
git push -u origin main
```

GitHub pedirá autenticación: usa un *personal access token* (Settings → Developer settings → Personal access tokens) en lugar de la contraseña, o instala GitHub CLI (`brew install gh`, luego `gh auth login`).

Nota: la carpeta está en Google Drive. Git funciona ahí, pero no edites el proyecto desde dos computadoras a la vez, porque Drive puede sincronizar a medias la carpeta oculta `.git`.

## 4. Conectar Zenodo y obtener el DOI

1. Entra a https://zenodo.org con "Log in with GitHub".
2. En tu perfil: **GitHub** → activa el interruptor de `etocto/education-deserts-peru`.
3. En GitHub: **Releases** → **Create a new release** → tag `v1.0.0` → título `v1.0.0 — submitted version` → **Publish release**.
4. En unos minutos Zenodo archiva el release y asigna el DOI (usa los metadatos de `.zenodo.json`).
5. Copia el DOI en:
   - `manuscrito_mdpi/main.tex`, en la Data Availability Statement (reemplaza `[DOI to be inserted before submission]`);
   - `README.md`, en la sección Citation (Zenodo te da también el código del *badge*).

Zenodo entrega dos DOI: uno de la versión (v1.0.0) y uno "concept" que siempre apunta a la última versión. En el artículo cita el de la versión, que corresponde exactamente a los resultados publicados.

## 5. Después de la revisión por pares

Si los revisores piden cambios en el código, haz commit, crea el release `v1.1.0` y Zenodo generará un DOI nuevo para esa versión. Cuando el artículo sea aceptado, puedes quitar `manuscrito_mdpi/` del `.gitignore` y publicar la versión final.
