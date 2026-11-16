# Generador de datos sintéticos de urgencias

Este repositorio genera una base de datos sintética de urgencias compatible con el diccionario de variables de urgencias.

## Archivos incluidos

- `generar_datos_sinteticos.py`: genera la base sintética en formato TSV.
- `variables_urgencia.csv`: definiciones de columnas, tipos y catalogos.

## Uso

```bash
python generar_datos_sinteticos.py
python generar_datos_sinteticos.py -n 20000 --salida prueba.txt --semilla 7
```

## Salida

La ejecución crea un archivo tabulado con columnas como:

- `urg_folio`
- `urg_fecha_hora`
- `tri_fecha_hora`
- `niv_cod`
- `sex_cod`
- `pac_edad`
- `pac_edad_tipo`
- `mot_glosa`
- `dest_glosa`
- `pron_glosa`

## Nota

Los datos son inventados y están diseñados para pruebas de EDA, validación de limpieza y análisis exploratorio, no para sacar conclusiones clínicas.
