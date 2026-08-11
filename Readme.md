# Proyecto Acus — Transcripción y separación de audio con IA

## Ideas

- Trabajar con pasajes sonoros?

- Transcripción Musical Automática Audio a MIDI 
    - Tomar una grabación de audio (por ejemplo, una melodía tocada en guitarra o piano) y usar un modelo de IA que detecte las frecuencias fundamentales ($f_0$), las notas exactas y el tiempo en que se tocan, generando un archivo MIDI
    - Dataset: MAPS Dataset o URMP (audio polifónico etiquetado nota por nota con MIDI).
    - Se podría usar la arquitectura de Basic Pitch o reentrenarla con otro dataset
    - Resuelve un problema clásico de procesamiento de señal (detección de pitch y armónicos)

## Pipeline

El proyecto usa dos modelos de IA como etapas de un pipeline de limpieza y análisis de audio:

```
Canciones/*.mp3
      │
      ▼  1. Demucs (Meta) — separación de pistas / limpieza de la mezcla
separated/htdemucs/<canción>/{vocals,drums,bass,other}.wav
      │
      ▼  2. Basic Pitch (Spotify) — transcripción audio → MIDI
MIDI/*_basic_pitch.mid
      │
      ▼  3. Scripts de análisis (pretty_midi)
Notas, rangos, tempos, cromagrama...
```

La separación de stems actúa como filtro de limpieza: al aislar cada instrumento antes de transcribir, Basic Pitch detecta las notas con mucha menos interferencia de los demás instrumentos de la mezcla.

## Instalación

El repo incluye un `.venv` (Python 3.10) con todo instalado. Para recrearlo desde cero:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install basic-pitch demucs pretty_midi numpy
```

## 1. Separación de pistas con Demucs (Meta)

Repo: https://github.com/facebookresearch/demucs

Separa una canción en 4 stems (voces, batería, bajo y otros) con el modelo `htdemucs`:

```bash
.venv/bin/demucs -n htdemucs -o separated "Canciones/<canción>.mp3"
```

Resultado:

```
separated/htdemucs/<canción>/
├── vocals.wav
├── drums.wav
├── bass.wav
└── other.wav
```

Opciones útiles:

```bash
# Solo separar voz del resto (más rápido)
.venv/bin/demucs --two-stems vocals -o separated "Canciones/<canción>.mp3"

# Forzar CPU si hay problemas con la GPU
.venv/bin/demucs -d cpu -o separated "Canciones/<canción>.mp3"

# Salida en mp3 en vez de wav
.venv/bin/demucs --mp3 -o separated "Canciones/<canción>.mp3"

# Listar todos los modelos disponibles
.venv/bin/demucs --list-models
```

## 2. Transcripción a MIDI con Basic Pitch (Spotify)

Repo: https://github.com/spotify/basic-pitch

Transcribe un archivo de audio a MIDI. El primer argumento es la carpeta de salida:

```bash
.venv/bin/basic-pitch MIDI "separated/htdemucs/<canción>/bass.wav"
```

Genera `MIDI/bass_basic_pitch.mid` (el sufijo `_basic_pitch` se agrega automáticamente).

Transcribir todos los stems de una canción de una vez:

```bash
.venv/bin/basic-pitch MIDI separated/htdemucs/<canción>/*.wav
```

Filtros de limpieza de la transcripción (umbrales):

```bash
# Más estricto: menos notas fantasma (útil con stems ruidosos)
.venv/bin/basic-pitch MIDI --onset-threshold 0.6 --frame-threshold 0.4 \
    --minimum-note-length 0.1 "separated/htdemucs/<canción>/bass.wav"

# Restringir el rango de frecuencias (p.ej. solo graves para el bajo, en Hz)
.venv/bin/basic-pitch MIDI --minimum-frequency 30 --maximum-frequency 300 \
    "separated/htdemucs/<canción>/bass.wav"
```

## 3. Análisis del MIDI

Con el MIDI generado, los scripts en `Scripts/` extraen información musical:

```bash
# Nota por nota: pitch, inicio, fin, velocity
.venv/bin/python Scripts/analizador.py

# Resumen completo: rango, duración, tempo, cromagrama
.venv/bin/python Scripts/analisis_completo.py
```

(Ajustar la ruta del archivo `.mid` dentro de cada script según la canción procesada.)

## Estructura del proyecto

```
├── Canciones/    # Audio de entrada (mp3)
├── separated/    # Stems separados por Demucs
├── MIDI/         # MIDIs generados por Basic Pitch
└── Scripts/      # Análisis de los MIDIs con pretty_midi
```

## Referencias

- Basic Pitch (Spotify): https://github.com/spotify/basic-pitch
- Demucs (Meta): https://github.com/facebookresearch/demucs
- pretty_midi: https://github.com/craffel/pretty-midi
