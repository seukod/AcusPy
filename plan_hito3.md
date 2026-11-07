# AcusPy — Plan Hito 3

1 nov 2026 · Seuko y Niik

## Resumen

El Hito 3 debe responder si la mejora del PRE-DSP en el bajo se sostiene en todo BabySlakh, y explicar qué hace (o no hace) el POST-DSP. El trabajo se divide en cuatro notebooks Marimo (archivos `.py` en `notebooks/hito_03/`) más un notebook 00 compartido que fija datos, métricas y formato de resultados para que todo sea comparable.

| Notebook | Pregunta que responde | Depende de | Prioridad |
| --- | --- | --- | --- |
| `00_setup_comun` | Datos, métricas y formato de salida comunes | — | Base |
| `01_benchmark_bajo_babyslakh` | ¿La mejora del PRE-DSP se mantiene en todos los tracks? | 00 | 1 |
| `02_post_dsp` | ¿Por qué el POST-DSP no actúa y cómo redefinirlo? | 00 | 2 |
| `03_analisis_errores` | ¿Qué tipo de errores comete cada condición? | 00, 01 | 3 |
| `04_htdemucs_6s` (opcional) | ¿El pipeline se extiende a guitarra y piano? | 00, 01 | 4 |

## Reparto

Propuesta: Niik toma lo que es cómputo pesado (benchmark completo y htdemucs_6s, idealmente en Patagón) y Seuko toma lo que es diagnóstico y análisis (POST-DSP y errores). El notebook 00 se hace juntos en la primera sesión.

| Persona | Notebooks | Por qué |
| --- | --- | --- |
| Niik | `01_benchmark_bajo_babyslakh`, `04_htdemucs_6s` | Ambos corren separación + transcripción sobre muchos tracks; mismo entorno y cola en Patagón |
| Seuko | `02_post_dsp`, `03_analisis_errores` | Ambos trabajan sobre salidas ya generadas; se pueden empezar con los tracks del Hito 2 mientras corre el 01 |
| Ambos | `00_setup_comun`, integración final, slides | Acordar el contrato de datos evita rehacer código |

Riesgo principal: el 03 depende de las salidas del 01. Para no bloquear, Seuko arranca el 03 con los tracks ya procesados en el Hito 2 y lo vuelve a correr cuando el 01 termine.

## Detalle por notebook

### 00 — Setup común (ambos)

Objetivo: que 01–04 lean y escriban lo mismo.

- Listar todos los tracks de BabySlakh que tienen bajo y fijar ese listado en `results/hito3/catalogo_bajo.csv`.
- Mover a `src/` las funciones ya usadas en el Hito 2: separación, PRE-DSP, transcripción, POST-DSP, evaluación.
- Fijar métricas de evaluación nota a nota con `mir_eval.transcription`: F1 de onset (tolerancia 50 ms) y F1 de onset+offset.
- Definir el formato de resultados (ver Reglas de trabajo).

Entregable: `notebooks/hito_03/00_setup_comun.py` + `src/` + listado de tracks.

**Hecho (commit 40f75a1).** Lógica en `src/hito3.py`, resultados en `results/hito3/`. Lo que fijó:

- La unidad es track × grupo Demucs; la referencia une los MIDI de todos los instrumentos del grupo (3 tracks tienen dos bajos).
- Bajo: 20 tracks, **19 evaluables** (Track00017 no tiene MIDI del bajo). El 01 corre sobre 19, no 20.
- La referencia del bajo se transpone −12 semitonos, como en el Hito 2.
- Uso en 01/04: `list_track_groups("bass")` + `run_conditions(...)` por track, y `write_rows(...)` al CSV.

### 01 — Benchmark 2×2 del bajo en todo BabySlakh (Niik)

Pregunta: ¿la mejora del PRE-DSP observada en el Hito 2 se mantiene en el dataset completo?

1. Correr las 4 condiciones (PRE on/off × POST on/off) en todos los tracks del listado.
2. Guardar las notas transcritas (MIDI) y las métricas por track y condición.
3. Comparar pareado por track: diferencia de F1 PRE vs sin PRE, % de tracks que mejoran, media ± intervalo de confianza.
4. Test de Wilcoxon pareado para decir si la diferencia es significativa.
5. Figuras: boxplot de F1 por condición y histograma de la diferencia por track.

Entregable: `results/hito3/benchmark_bajo.csv`, MIDIs en `results/hito3/midi/`, 2–3 figuras con su interpretación escrita.

### 02 — POST-DSP (Seuko)

Pregunta: ¿por qué el POST-DSP no cambia el resultado, y vale la pena redefinirlo?

1. Medir cuánto actúa hoy: notas eliminadas/modificadas, energía removida, diferencia entre entrada y salida.
2. Revisar sus parámetros contra la distribución real de los datos (umbrales que nunca se alcanzan, rangos de frecuencia fuera del bajo).
3. Decidir: si se puede corregir, proponer una versión nueva y probarla en una muestra de ~10 tracks; si no, documentar por qué no actúa.
4. Si hay versión nueva, pasársela a Niik para re-correr la columna POST del 2×2.

Entregable: notebook con la evidencia y la decisión escrita en el README (formato "pensábamos… observamos… decidimos…").

### 03 — Análisis de errores (Seuko)

Pregunta: ¿qué tipo de error domina en cada condición y cuál corrige el PRE-DSP?

| Tipo de error | Cómo detectarlo |
| --- | --- |
| Octava | Onset coincide pero la altura difiere ±12 semitonos |
| Nota fantasma | Nota predicha sin pareja en la referencia (falso positivo); revisar si es corta o débil |
| Inicio desplazado | Altura coincide pero el onset cae fuera de los 50 ms; histograma del desfase |
| Omisión | Nota de referencia sin pareja (falso negativo); revisar si son notas cortas o graves |

- Contar cada tipo por track y condición, y comparar con y sin PRE-DSP.
- Escoger 2–3 ejemplos ilustrativos (piano-roll referencia vs predicción) para las slides.

Entregable: `results/hito3/errores.csv` + figuras + ejemplos.

### 04 — htdemucs_6s, guitarra y piano (Niik, opcional)

Solo si el 01 está cerrado antes del 20 de noviembre.

- Repetir el benchmark con `htdemucs_6s`, que separa guitarra y piano además de bajo.
- Empezar por un subconjunto de tracks que tengan esos instrumentos.
- Ojo: piano y guitarra son polifónicos, así que los resultados no son directamente comparables con el bajo; reportarlo como extensión exploratoria.

Entregable: tabla de métricas por instrumento y una conclusión corta.

## Cronograma

Cuatro semanas hasta el Hito 3 (fines de noviembre); la última queda solo para integrar y preparar la presentación.

### Semana 1 — 2 al 8 nov

- [ ] Ambos: sesión para cerrar `00_setup_comun` y el formato de resultados
- [ ] Niik: preparar entorno en Patagón y correr el 01 en un subconjunto de prueba
- [ ] Seuko: 02, medir cuánto actúa hoy el POST-DSP

### Semana 2 — 9 al 15 nov

- [ ] Niik: 01 completo en todos los tracks + comparación estadística
- [ ] Seuko: 02, decisión (redefinir o documentar) y versión nueva si aplica
- [ ] Seuko: arrancar el 03 con los tracks del Hito 2

### Semana 3 — 16 al 22 nov

- [ ] Niik: re-correr la columna POST si el 02 trae filtro nuevo
- [ ] Seuko: 03 completo sobre los resultados del 01
- [ ] Niik: 04 opcional, solo si el 01 ya está cerrado

### Semana 4 — 23 nov al Hito 3

- [ ] Ambos: figuras finales, README actualizado, conclusiones y limitaciones
- [ ] Ambos: slides y ensayo con tiempo (10 min)

## Reglas de trabajo

La pauta del Hito 2 marcó la trazabilidad en GitHub; estas reglas apuntan a que el historial muestre el proceso, no solo el final.

- **Marimo:** cada notebook es un `.py` de Marimo en `notebooks/hito_03/`, igual que los del Hito 2. Al ser texto plano, los diffs y las revisiones de PR se leen bien; los controles reactivos sirven para afinar parámetros en el 02. Para correr en lote (01 y 04, o en Patagón) se ejecutan como script: `python notebooks/hito_03/01_benchmark_bajo_babyslakh.py`.
- **Ramas:** una por notebook (`nb01-benchmark`, `nb02-post-dsp`, …); merge a `main` con PR que el otro revisa.
- **Commits:** uno por unidad de trabajo real ("nb02: mide notas eliminadas por POST-DSP"), no un commit grande antes del hito.
- **Notebooks livianos:** la lógica va en `src/`; el notebook llama funciones, muestra figuras e interpreta.
- **Datos:** BabySlakh y audios separados fuera del repo (`.gitignore`); solo CSV de resultados y figuras se versionan.
- **Decisiones:** cada notebook termina con una celda "Qué observamos / qué decidimos" que luego pasa al README.
- **IA:** anotar qué herramienta se usó y para qué, para la declaración del Hito 3.

Formato común de resultados (`RESULT_COLUMNS` en `src/hito3.py`), una fila por track × grupo × condición. `write_rows` rechaza columnas fuera de esta lista:

```csv
track_id,grupo,stems_ref,inst_class,modelo_demucs,condicion,pre_dsp,post_dsp,inicio_s,duracion_s,ref_transpose,P_no,R_no,F_no,P,R,F,Acc,n_ref,n_pred,estado
```
