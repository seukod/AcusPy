# AcusPy

**Separación de fuentes, procesamiento de audio y transcripción musical a MIDI.**

AcusPy es un proyecto experimental que combina **HT-Demucs**, **DSP** y **Basic Pitch** para estudiar cómo cambia la transcripción de una mezcla musical al procesar sus pistas separadas. Incluye un comparador interactivo en **Marimo** para escuchar resultados, inspeccionar espectrogramas y evaluar los MIDI contra referencias conocidas.

La pregunta que guía el proyecto es: **¿cuánto mejora —o empeora— la transcripción al aplicar DSP antes y después de la inferencia?**

## Qué puedes hacer

- Separar una mezcla en cuatro stems: `vocals`, `drums`, `bass` y `other`.
- Comparar varias variantes DSP por stem, cada una con una cadena de etapas independiente.
- Aplicar filtros paso alto, paso bajo, noise gate y transformaciones personalizadas.
- Recuperar y escuchar el audio original de cada stem y sus salidas intermedias.
- Transcribir audio a MIDI y filtrar notas por duración, tesitura o velocidad.
- Comparar formas de onda, STFT, Mel, log-Mel y piano rolls.
- Evaluar las transcripciones con `F_no`, `F`, `Acc`, precisión y recall.
- Exportar audio WAV, MIDI, parámetros del experimento y tablas de resultados.

El proyecto funciona localmente y está en desarrollo. El DSP se evalúa como una hipótesis experimental: sus ajustes pueden mejorar o degradar el resultado según el instrumento y el audio.

## Inicio rápido

Se recomienda **Python 3.10**, la versión configurada en `.python-version`, y [uv](https://docs.astral.sh/uv/) para preparar el entorno. Los comandos siguientes están pensados para macOS y Linux.

```bash
git clone https://github.com/seukod/AcusPy.git
cd AcusPy

uv venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/marimo edit notebooks/07_demucs_dsp_marimo.py
```

Marimo mostrará la dirección local en la terminal. Para abrir el comparador como aplicación, sin el editor de código:

```bash
.venv/bin/marimo run notebooks/07_demucs_dsp_marimo.py
```

Para detener el servidor, presiona **Ctrl+C**. La primera separación con Demucs necesita acceso a internet para descargar los pesos del modelo.

## Usar el comparador de Marimo

El punto de entrada principal es [07_demucs_dsp_marimo.py](notebooks/07_demucs_dsp_marimo.py).

1. **Selecciona un audio** con el explorador de archivos y define el inicio y la duración del fragmento. Empieza con unos 10 segundos.
2. **Elige el dispositivo** y pulsa **Separar con HT-Demucs**.
3. **Selecciona un stem** y las variantes que quieres comparar.
4. **Ajusta el DSP** y escucha las etapas intermedias. Los gráficos se actualizan sin repetir la separación.
5. **Transcribe las variantes** con Basic Pitch para comparar sus MIDI crudos y filtrados.
6. **Opcionalmente, evalúa el benchmark** seleccionando uno o varios MIDI de referencia y pulsando **Evaluar benchmark**.
7. **Exporta los resultados** desde los controles de audio/MIDI o descarga la tabla desde Marimo.

Las variantes incluidas en el notebook son:

| Variante | Procesamiento |
| --- | --- |
| `raw` | Stem mono sin DSP. |
| `filtered` | Paso alto → paso bajo → noise gate → Basic Pitch → filtro de duración de notas. |
| `alternative` | Paso alto → paso bajo → Basic Pitch → filtro de duración de notas. |

Los valores iniciales son puntos de partida para experimentar, no presets óptimos para todos los instrumentos. Cambiar el DSP invalida los MIDI anteriores; cambiar la representación espectral actualiza solo los gráficos. Cambiar la referencia MIDI invalida únicamente el benchmark.

### Dispositivos disponibles

| Opción | Hardware para la separación con Demucs |
| --- | --- |
| `cpu` | Procesador del equipo; ejecución ARM nativa en un Mac M1 con Python `arm64`. |
| `mps` | GPU de Apple Silicon, cuando PyTorch la detecta. |
| `cuda` | GPU NVIDIA compatible, cuando PyTorch la detecta. |

CPU es el valor predeterminado y el selector muestra solo los dispositivos disponibles. MPS se ha probado con HT-Demucs en un Mac M1; no garantiza mayor velocidad para todos los fragmentos. La selección afecta a **Demucs**; **Basic Pitch** mantiene su backend independiente.

## Cómo funciona el pipeline

```text
Audio a su frecuencia original
  → conversión a estéreo y 44.1 kHz
  → HT-Demucs: vocals / drums / bass / other
  → copia mono del stem a 22.05 kHz
  → variantes DSP con etapas recuperables
  → Basic Pitch
  → filtrado de notas
  → MIDI y evaluación opcional
```

La separación se ejecuta una vez y puede reutilizarse para explorar distintas cadenas DSP. Por ejemplo, una entrada de BabySlakh a 16 kHz se remuestrea a 44.1 kHz para Demucs y a 22.05 kHz para el flujo de transcripción. Aumentar la frecuencia de muestreo no recupera información ausente del audio original.

Los espectrogramas comparten la referencia de energía del stem sin DSP, incluso si se oculta la variante `raw`. La vista inicial es **log-Mel**; Mel agrupa frecuencias y la escala logarítmica facilita inspeccionar cambios de energía. Esta representación es una ayuda visual: no modifica la entrada de Basic Pitch ni mejora por sí sola las métricas.

## Datos y referencias MIDI

Puedes separar y transcribir tus propios audios sin descargar un dataset. Para obtener métricas necesitas **MIDI de referencia alineados temporalmente con el audio**.

Los notebooks de evaluación utilizan **BabySlakh**, una versión reducida de Slakh2100. El dataset se descarga por separado desde [Zenodo](https://zenodo.org/records/4603870) y no está incluido en el repositorio. Organízalo así:

```text
data/
└── babyslakh_16k/
    └── Track00001/
        ├── mix.wav
        ├── metadata.yaml
        ├── stems/
        │   └── S03.wav
        └── MIDI/
            └── S03.mid
```

`S03` es un ejemplo del primer track; los instrumentos se identifican mediante `metadata.yaml`. El comparador muestra una tabla de archivos e instrumentos cuando encuentra esos metadatos.

La referencia debe corresponder al stem evaluado: para `bass`, selecciona los MIDI del bajo; para `other`, combina los MIDI de los instrumentos que integran esa agrupación. El MIDI completo de la mezcla no es una referencia adecuada para un único stem.

## Benchmark

El notebook `07` reutiliza los criterios de evaluación del `06` mediante [mir_eval](https://mir-eval.readthedocs.io/stable/api/transcription.html).

| Métrica | Qué evalúa |
| --- | --- |
| `P_no`, `R_no`, `F_no` | Precisión, recall y F-score de notas que coinciden en altura e inicio, sin exigir coincidencia en su final. |
| `P`, `R`, `F` | Las mismas métricas, exigiendo también coincidencia en el final de la nota. |
| `Acc` | Coincidencia de alturas activas por frame: `TP / (TP + FP + FN)`. |
| `Δ F_no`, `Δ F`, `Δ Acc` | Diferencia frente al MIDI del modelo de la variante `raw`. |

Las métricas van de **0 a 1**; los deltas pueden ser negativos si el resultado empeora frente a `raw`. `F_no` es un F-score; **F0** designa la frecuencia fundamental y es un concepto distinto.

La evaluación utiliza tolerancias de **50 ms** en el inicio y **50 cents** en la altura. Para el final de nota, usa el máximo entre **20 % de su duración** y **50 ms**. La resolución por frames es de **10 ms**, contando una sola vez las alturas simultáneas duplicadas.

La referencia se recorta al mismo fragmento del audio, se desplaza al inicio cero y se compara con los MIDI disponibles sin volver a ejecutar los modelos. La tabla distingue las salidas del modelo de las posteriores al filtrado de notas.

Si no hay notas de referencia no percusivas dentro del fragmento, se muestra **sin referencia evaluable** y las métricas quedan sin calcular. Si no se transcribió `raw`, se calculan las métricas absolutas y los deltas quedan vacíos.

## Uso desde Python

Ejecuta el siguiente ejemplo desde la raíz del repositorio, con el entorno del proyecto:

```python
from pedalboard import HighpassFilter
from src import (
    AudioPreProcessor,
    DemucsTranscriber,
    DSPVariant,
    NotePostProcessor,
    available_devices,
)

print(available_devices())  # Por ejemplo: ["cpu", "mps"] en Apple Silicon.
engine = DemucsTranscriber(device="cpu")
separation = engine.separate("mi_audio.wav")

variants = engine.process_variants(separation, {
    "bass": [
        DSPVariant("raw"),
        DSPVariant(
            "filtered",
            stages=[AudioPreProcessor(
                name="highpass",
                board=[HighpassFilter(cutoff_frequency_hz=35)],
            )],
            note_postprocessor=NotePostProcessor(min_duration_sec=0.05),
        ),
    ],
})

# Acceso al stem separado y a una etapa intermedia.
bass = separation.stems["bass"]                # bass.audio, bass.sr
filtered = variants["bass"]["filtered"].stages["highpass"]

transcriptions = engine.transcribe_variants(variants)
engine.export(separation, transcriptions, "outputs/mi_experimento")
```

Para volver a ajustar el DSP, llama a `process_variants()` con la misma separación. Cada rama comienza en una copia del stem mono a 22.05 kHz. Las transformaciones personalizadas usan la firma `(audio, sr) -> audio` y deben conservar la forma, duración y frecuencia de muestreo.

La API también acepta arrays NumPy: `engine.separate(audio, sr=16000)`. Usa `(muestras,)` para mono y `(canales, muestras)` para estéreo; en este caso, `sr` es obligatorio. Para transcribir audio sin separación, puedes utilizar `BasicPitchTranscriber` directamente.

## Salidas y exportación

El notebook crea una carpeta nueva por exportación dentro de `outputs/07_demucs/`:

```text
outputs/07_demucs/<experimento>/
├── input.wav
├── demucs_input.wav
├── stems/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── variants/<stem>/<variante>/
│   ├── 00_input_22050.wav
│   ├── 01_<etapa>.wav
│   ├── raw_midi.mid
│   └── midi.mid
├── manifest.json
├── source.json
└── dsp_settings.json
```

Los WAV se guardan en formato **FLOAT**, conservando la amplitud sin normalizar cada etapa por separado. Los MIDI se incluyen cuando se ha ejecutado la transcripción; sus tiempos son relativos al fragmento seleccionado. `manifest.json` registra las frecuencias, canales y duraciones; los otros JSON del notebook guardan la selección de audio y los parámetros DSP.

Desde Python, `engine.export()` genera los audios, los MIDI disponibles y el manifiesto en la carpeta indicada. Las salidas del comparador, el dataset y las sesiones de Marimo están excluidos de Git. La tabla del benchmark se descarga por separado desde Marimo.

## Organización del repositorio

| Ruta | Contenido |
| --- | --- |
| [src/transcriber.py](src/transcriber.py) | Basic Pitch, preprocesamiento de audio, filtrado de notas y presets por instrumento. |
| [src/demucs_transcriber.py](src/demucs_transcriber.py) | Separación, variantes DSP por stem, resultados intermedios y exportación. |
| [src/benchmark.py](src/benchmark.py) | Carga y recorte de referencias, métricas y comparación de variantes. |
| [src/spectral.py](src/spectral.py) | Representaciones STFT, Mel y log-Mel con escalas compartidas. |
| [notebooks/01_auditoria_datos.ipynb](notebooks/01_auditoria_datos.ipynb) | Exploración y auditoría del dataset. |
| [notebooks/02_benchmark_basic_pitch.ipynb](notebooks/02_benchmark_basic_pitch.ipynb) | Experimentos iniciales con Basic Pitch y DSP. |
| [notebooks/06_benchmark_dsp.ipynb](notebooks/06_benchmark_dsp.ipynb) | Benchmark por instrumento sobre pistas aisladas de BabySlakh. |
| [notebooks/07_demucs_dsp_marimo.py](notebooks/07_demucs_dsp_marimo.py) | Comparador interactivo de separación, DSP, transcripción y evaluación. |
| [tests/](tests/) | Pruebas de orquestación, métricas y representaciones espectrales. |

## Pruebas

Desde la raíz del proyecto:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/marimo check --strict notebooks/07_demucs_dsp_marimo.py
```

Las pruebas automatizadas de orquestación simulan la inferencia para no descargar pesos. Cubren remuestreo, canales, etapas independientes, exportación, detección de dispositivos, referencias MIDI, tolerancias de evaluación y escalas espectrales. Las pruebas con modelos reales requieren los pesos de Demucs y un backend de Basic Pitch compatible con el equipo.

## Límites actuales y próximos pasos

- `htdemucs` separa cuatro fuentes. `other` sigue siendo una mezcla de instrumentos; esta integración no aísla guitarra o piano como stems independientes.
- La batería puede separarse, escucharse y procesarse, pero el notebook excluye `drums` de la transcripción y el benchmark de notas.
- Los errores de octava cuentan como errores de altura en las métricas actuales. Escuchar menos notas o ver un espectrograma más limpio no demuestra por sí solo una mejora.
- El postprocesador de notas actual reconstruye el MIDI y no conserva los pitch bends ni todos los metadatos del resultado crudo.
- La salida musical implementada es MIDI. La generación de partituras, la cuantización a escalas y el procesamiento en tiempo real quedan como trabajo futuro.
- Los siguientes experimentos se centran en ajustar el DSP por instrumento y medir su efecto en más fragmentos y tracks, antes de extraer conclusiones generales.

## Tecnologías y referencias

- [Demucs](https://github.com/facebookresearch/demucs): separación de fuentes con el modelo HT-Demucs.
- [Basic Pitch](https://github.com/spotify/basic-pitch): transcripción de audio a MIDI.
- [Pedalboard](https://github.com/spotify/pedalboard): efectos y procesamiento de audio.
- [librosa](https://librosa.org/): carga, remuestreo y análisis espectral.
- [pretty_midi](https://github.com/craffel/pretty-midi): lectura y escritura de MIDI.
- [mir_eval](https://mir-eval.readthedocs.io/): evaluación de transcripción musical.
- [Marimo](https://docs.marimo.io/): notebooks reactivos y comparador interactivo.
- [BabySlakh / Slakh2100](https://zenodo.org/records/4603870): datos de audio y MIDI para los experimentos.

## Uso de herramientas de IA

Se han utilizado herramientas de IA como apoyo a la formulación, documentación e implementación del proyecto. Las decisiones, interpretaciones y conclusiones de la investigación son responsabilidad del autor.
