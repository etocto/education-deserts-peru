# Asistencia a educación superior por distrito de ORIGEN (pendiente, obs. 1 de la revisión)

El resultado de los pasos 10 y 21 está medido por distrito de **residencia habitual** en el Censo 2017. Un joven que se
mudó para estudiar cuenta en el distrito de destino, por lo que un distrito sin universidad tiene menos asistentes
residentes aunque sus jóvenes sí estudien. Para medir la asistencia por distrito de **origen** hace falta una consulta
nueva al REDATAM en línea del INEI (base CPV2017DI).

## Consulta

1. En el diccionario de la base, ubicar la variable de **lugar de residencia hace cinco años** (distrito). Anotar su
   nombre exacto y sus categorías. Confirmar que está disponible a nivel de distrito; si solo hay departamento o
   provincia, usar el nivel disponible y declararlo.
2. Universo: personas de 17 a 24 años (`POBLACIO.C5P41 >= 17 AND POBLACIO.C5P41 <= 24`).
3. Tabla cruzada: distrito de residencia hace cinco años × nivel que cursa (`C5P13NIV`, solo quienes asisten,
   `C5P14 = 1`), y por separado el total de 17-24 años por distrito de residencia hace cinco años.
4. Guardar como `datos/raw/censo2017/redatam/cpv2017_asiste_nivel_17_24_origen5.csv` y
   `cpv2017_pob_17_24_origen5.csv`, con la misma estructura que los archivos actuales (ubigeo, ccdi, nombre, categorías, Total).
5. Contraste adicional: repetir con el universo de 17-18 años (recién egresados, que aún no han migrado en su mayoría).

## Uso

Con esos archivos, en `src/21_asistencia_nivel.py` reemplazar el numerador y el denominador por los de origen y
reestimar. Reportar ambos resultados (residencia y origen). Limitación que permanece: quien tenía 12-19 años en 2012
pudo haberse mudado con su familia por otras razones.
