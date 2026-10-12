# Pasos añadidos en la revisión (octubre 2026)

Todos usan los archivos que generan los pasos 1-4 y los tiempos ya guardados en `datos/processed/`; ninguno descarga OSM.

```bash
python run_all.py 1 2 1b 3 4 4c 4e 13      # regenera la base (unos minutos)
python run_all.py 15 16 17 18 19 20 21 22 23 6b
python -m pytest -q tests                  # pruebas unitarias
python -m pip freeze > requirements-lock.txt   # fija las versiones exactas del entorno con el que se corrió
```

| Paso | Observación | Qué produce |
|---|---|---|
| 15 | 2 | sensibilidad a 30-90 km por familia y por origen; tabla cruzada 60 km × 90 min; radio equivalente a 90 min |
| 16 | 8 | colegios no conectados; efecto de la red fluvial; plantilla de validación de tiempos (40 pares) |
| 17 | 4 | conciliación de las 45 denegadas; filiales 2018 vs. locales 2024 de las licenciadas |
| 18 | 5 | perfil de los desiertos por origen y familia frente a sus departamentos |
| 19 | 12, STROBE 12c/14 | estadísticos descriptivos distritales; datos faltantes |
| 20 | 18 | silueta para k = 2-10 y estabilidad (ARI) de la tipología |
| 21 | 1 | asistencia por nivel (universitaria, no universitaria, superior) con su numerador y denominador |
| 22 | 13 | muestra de 20 colegios para verificación manual |
| 23 | 17 | figuras regeneradas (mapa con límites, escala y % ; barras con valores) |
| 6b | 7 | hiperparámetros, métricas por pliegue e importancia por permutación agrupada (REFORMS) |

Pendientes que requieren datos que no están en el repositorio:
- `datos/manual/validacion_tiempos.csv`: completar la plantilla del paso 16 con tiempos observados y volver a correr el paso 16.
- Asistencia por distrito de origen: ver `docs/REDATAM_asistencia_origen.md`.
- Fechas de resolución y de cese efectivo en `salidas/tablas/conciliacion_denegadas.csv` (resoluciones de SUNEDU).
- Archivar una versión (p. ej., v1.1.0) en Zenodo y citar el DOI con el número de versión.

# Fase 1 del seguimiento (octubre 2026): N2, N3, N4, N6, N8

```bash
python run_all.py 4c 4e 4g 13 15 18 19 25 27 28 29 30 23 11 24   # rápidos (menos de 2 min en total)
python run_all.py 26 24        # N3: tiempos a la oferta 2018 en la red de OSM (lee el .pbf; unos 3 GB de RAM con src/red_liviana.py) y tablas
```

| Paso | Observación | Qué produce |
|---|---|---|
| 25 | N2 | desiertos de licenciamiento con la oferta 2024 y sumando la oferta 2018 de las seis relicenciadas (cota), por departamento y familia |
| 26 | N3 | tiempo mínimo a las ofertas post, post + relicenciadas y reconstruida; orígenes definidos solo por tiempo; orígenes por distancia con la fila «a más de 60 km pero a menos de 90 min» |
| 27 | N4 | perfil de los desiertos topográficos con y sin los colegios no conectados a la red; topográficos por departamento |
| 28 | N6 | logit fraccional (Papke y Wooldridge) de la proporción de alumnos en desierto por distrito: entre y dentro de departamentos, ponderado y por tiempo; efectos marginales promedio |
| 29 | N3, N14 | orígenes en cuatro grupos excluyentes (estructural, licenciamiento, topográfico, solo por distancia) con perfil distrital y perfil esperado por departamento |
| 30 | N8 | todas las cifras del texto que no están en una tabla, calculadas sin redondear y redondeadas una vez (mitad hacia arriba) |
| 23, 11 | N15 | Figura 2 con barra de «cualquier carrera» y las tres ofertas; ortografía británica; mapa de tipología con escala |
| 24 | N7, N8, N13 | tablas nuevas S5-S7, S11-S12, S21-S22 y S27; S1 con % de clasificación estable; S24 bien formada |

Regla de redondeo (N8): los pasos guardan las tablas con 4 decimales; solo el paso 24 (y 30) redondea, una vez.
Tras correr el paso 26, vuelva a correr el paso 24: la Tabla S7 se llena sola.

# Validación y calibración de los tiempos de viaje (octubre 2026)

1. `datos/manual/validacion_tiempos.csv`: 40 pares colegio-sede (muestra estratificada del paso 16) con el tiempo en auto
   de Google Maps (salida lunes 12-oct-2026, 07:00). Tres pares sin ruta vial (Loreto, Madre de Dios).
2. `python run_all.py 32`: estima el factor de las velocidades viales (0.4875) y lo escribe en `config.yaml`
   (`tiempos.factor_calibracion_vias`); `src/utils.py` lo aplica a todas las vías (no a ríos, ferris ni caminata).
3. Con la red calibrada: `python run_all.py 4b 4d 4h 16 26 31 13 15 18 19 22 25 27 28 29 30 24`.
   Los tiempos sin calibrar quedan en `datos/processed/*_sin_calibrar.csv`.
Resultados: sin calibrar, error absoluto medio 67 min y acuerdo en 90 min de 78 %; calibrado (validación cruzada
dejando uno fuera), 21 min, 84 % de pares dentro de ±30 % y 100 % de acuerdo en el umbral de 90 min.

# Verificación manual de 20 colegios (obs. 13; octubre 2026)

1. Muestra del paso 22 (`datos/manual/muestra_verificacion_manual.csv`; 5 colegios por origen, semilla fija, ninguno en la
   muestra de calibración).
2. `python src/33_verificacion_manual.py --modelo`: tiempo calibrado al MISMO local (unos 3 GB de RAM, ~5 min) →
   `datos/manual/verificacion_manual_modelo.csv`.
3. `datos/manual/verificacion_manual_google.csv`: tiempo en auto de Google Maps (salida lunes 12-oct-2026, 07:00; punto medio
   del rango), consultado a mano. Dos colegios sin ruta en auto (Sechura, Moyobamba).
4. `python run_all.py 33 24`: coordenadas dentro del distrito (límites INEI), distancia geodésica WGS84 con búsqueda exhaustiva
   del local más cercano, origen recalculado → `salidas/tablas/verificacion_manual*.csv` y Tabla S2.
Resultados: 20/20 colegios en su distrito; distancia con diferencia ≤ 0.5 km y mismo local más cercano en 20/20; origen
reproducido en 20/20; con ruta de Google (18): error relativo mediano 0.4 %, 15 dentro de ±30 %, Pearson 0.96 y 18/18 del
mismo lado del umbral de 90 min. Tres locales con distrito de SUNEDU distinto al de sus coordenadas (1.7-11.4 km).

# Revisión v3 (12 de octubre de 2026): análisis para levantar M1, M3, M4, M5, m1 y m4

Orden (desde la carpeta del repositorio):

    python run_all.py 36 34 37 35 38 28 30 24

| Paso | Script | Obs. | Qué hace | Requisitos | Tiempo |
|---|---|---|---|---|---|
| 36 | 36_relieve_grupos.py | M1 | Elevación, desnivel y pendiente a 5 km de cada colegio; perfil en línea recta al local más cercano; compara grupos de origen; % a más de 90 min por terciles de desnivel; región natural | Internet (teselas Terrarium de AWS, ~100 MB, se guardan en datos/raw/dem); pip install pillow | 10-20 min la 1.ª vez |
| 34 | 34_error_calibracion_region.py | M3 | Error de los tiempos calibrados por región natural con los 55 pares (37 LOOCV + 18 verificación); pares cerca de 90 min; Spearman | Pasos 32, 33 (y 36) | segundos |
| 37 | 37_factor_regional.py | M3 | Factor de velocidad por región, tiempos de todos los colegios y comparación de desiertos y orígenes | .pbf; ~3 GB RAM | 20-40 min |
| 35 | 35_sensibilidad_filiales.py | M5 | Desiertos de licenciamiento con filiales en la localidad de OSM, en el punto del distrito y desplazadas ±5 km | .pbf | 2-3 min |
| 38 | 38_correspondencia_campos.py | m1 | Correspondencia familias - clases N2 de SUNEDU - CINE-F 2013 | paso 1 | segundos |
| 28 | 28_logit_fraccional.py | M4 | Añade errores de Conley (60 y 100 km), I de Moran de los residuos y estimación sin distritos con menos de 10 alumnos | pip install libpysal | 1-2 min |
| 30 | 30_cifras_texto.py | m4 | Añade inventario de unidades (tabla de flujo STROBE) y Spearman a la Tabla S29 | — | segundos |
| 24 | 24_tablas_manuscrito.py | — | Regenera las tablas LaTeX (las tablas nuevas se agregan al redactar) | — | segundos |

Si 36 no puede descargar las teselas, 34 y 37 usan la región por departamento (respaldo) y lo indican en la salida.
Salidas: salidas/tablas/relieve_por_origen.csv, relieve_terciles.csv, relieve_modelo.csv, calibracion_por_region.csv,
factores_regionales.csv, sensibilidad_factor_regional.csv, origenes_factor_regional.csv, filiales_sensibilidad.csv,
filiales_ubicacion_alternativa.csv, correspondencia_campos.csv, logit_fraccional_espacial.csv, flujo_unidades.csv.

## Revisión v4 (12 de octubre de 2026)

- **M2, corrección de la red.** `coo_matrix(...).tocsr()` sumaba las entradas repetidas: dos locales anclados al mismo nodo sumaban su acceso a pie y dos tramos paralelos sumaban su tiempo, de modo que añadir oferta podía alargar un tiempo (5,108 filas con t_pre > t_rel). `utils.matriz_minimos` conserva el mínimo; se usa en `04b`, `04d` y `red_liviana`. Pruebas en `tests/test_utils.py`.
- **Rehacer los tiempos:** `python rehacer_tiempos_v4.py` (2-3 horas). Respalda las salidas en `salidas_antes_v4/`, corre 32, 4b, 4d, 26, 37, 33 --modelo y 36, luego el flujo por defecto (termina en 24), y escribe `salidas/cambios_v4.csv` con cada cifra que cambia.
- El paso 22 ya no se ejecuta por defecto ni sobrescribe la muestra verificada a mano (usar `--forzar`).
- `30_cifras_texto.py`: Tabla S37 en el orden del artículo, denominadores «(60 km)», definición de los 64 distritos, sedes y filiales en el flujo, sin la fila 2,342.
- `24_tablas_manuscrito.py`: nota de la Tabla S2 (punto representativo de distrito) y de la Tabla S7 (seis escenarios); nombres de archivo con corte en la Tabla S37.
- `utils.py`: figuras PDF de matplotlib con fuentes TrueType.
