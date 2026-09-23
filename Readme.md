# Proyecto Acus — Transcripción y separación de audio con IA



## Idea del proyecto
El objetivo de este proyecto es desarrollar un *pipeline* computacional que brinde la facilidad de transcribir canciones arbitrarias. Al ingresar una canción, el sistema debe ser capaz de procesarla, extraer sus notas musicales (hacer "ingeniería inversa") y generar una partitura o archivo MIDI. A futuro, esta tecnología podría integrarse en una **aplicación web interactiva** para enseñar a tocar instrumentos al unísono.

> [!NOTE]
> **Consideración sobre el Marco Legal:**
> Existe una limitante respecto al marco legal relacionado con la procedencia de los archivos de audio de los usuarios (por ejemplo, descargas no oficiales o piratería). Sin embargo, este es un aspecto externo que no afecta el alcance técnico ni los objetivos académicos de este proyecto, cuyo foco central es la implementación tecnológica de Inteligencia Artificial y Procesamiento de Señales Digitales.

## Pregunta principal -> BUSCAR METRICAS DE FIDELIDAD
**¿Qué nivel de fidelidad se puede alcanzar al transcribir automáticamente instrumentos de una mezcla a formato MIDI utilizando IA (Demucs + Basic Pitch), y de qué manera el procesamiento de señal (DSP) de las pistas aisladas permite reducir los errores y notas fantasma?**

## Motivación
El problema que se busca resolver es la falta de herramientas automatizadas, accesibles y precisas para transcribir canciones arbitrarias. Nos interesa investigar no solo la capacidad de los modelos actuales de IA, sino cómo los conocimientos en acústica y DSP pueden mejorar directamente los resultados de estos modelos, limpiando artefactos y refinando la transcripción.

## Datos
Para desarrollar y probar nuestro pipeline, utilizaremos audios de prueba con las siguientes características:
- **Fuente:** Archivos de audio locales (canciones comerciales, grabaciones propias) y posiblemente algún dataset estándar de separación de fuentes (ej. MUSDB18) si requerimos validar el error con pistas aisladas reales.
- **Tipo de datos:** Archivos de audio (WAV/MP3).
- **Etiquetas disponibles:** Inicialmente sin etiquetas (exploratorio). Si usamos datasets como MUSDB18 o partituras MIDI existentes, se usarán como "ground truth" para comparar.
- **Aspectos a investigar:** Necesitamos definir con exactitud qué audios usaremos como línea base para medir si nuestro DSP mejora o empeora la transcripción cruda de la IA.

## Alcance inicial
Para este semestre, el alcance se define en:
- **Objetivo inicial:** Construir el pipeline localmente para lograr aislar 1 o 2 instrumentos específicos (ej. bajo, guitarra) de una mezcla, aplicarles limpieza por DSP (ecualización, noise gate), y transcribirlos a MIDI. Comparar el resultado "crudo" de la IA vs el resultado "limpio" con DSP.
- **Si existe tiempo:** Investigar la viabilidad de empaquetar esto en una aplicación web interactiva o medir latencias para un uso en tiempo real.

## Pipeline provisional
Para lograr la transcripción de una mezcla de audio compleja a notas musicales legibles, el proyecto plantea el siguiente *pipeline*:

- [x] **1. Separación de Fuentes (Demucs):** Utilizar el modelo **Demucs** de Meta para aislar pistas específicas (como la guitarra, voces o bajo) separándolas del resto de la canción.
- [ ] **2. Procesamiento de Señal (DSP):** Antes de intentar detectar las notas, la pista aislada es procesada con ecualización (filtros paso alto y paso bajo) y compuertas de ruido (Noise Gate). Esto limpia las frecuencias residuales de otros instrumentos.
- [ ] **3. Extracción de Notas (Basic Pitch):** Una vez que el audio está purificado, se utiliza **Basic Pitch** (de Spotify) para analizar las frecuencias fundamentales y transcribir el audio a MIDI.
- [ ] **4. Post-procesamiento Musical:** Filtrar mediante código (`pretty_midi`) las notas extremadamente cortas (fantasmas), resolver solapamientos y forzar (cuantizar) el resultado a una escala musical.

## Posibles dificultades
- *Tiempo real vs Latencia:* Separar stems con Demucs en tiempo real es computacionalmente muy costoso.
- *Ajuste de DSP:* Definir qué frecuencias de corte (cutoff) y umbrales (thresholds) de la compuerta de ruido son ideales sin cortar el *sustain* real del instrumento.
- *Cuantización:* Programar un algoritmo para forzar/cuantizar las notas crudas a una escala musical teórica de manera algorítmica.

## Estado actual
- Entorno de desarrollo creado e investigación inicial completada.
- Definición de la pregunta principal de investigación.
- Dependencias base instaladas (`basic-pitch`, `demucs`, `pedalboard`).

## Próximos pasos
1. Descargar archivos de audio de prueba.
2. Pasar un audio de prueba por Demucs para aislar un instrumento (ej. Guitarra).
3. Pasar la pista aislada cruda por Basic Pitch y generar el MIDI base.
4. Experimentar con `pedalboard` para aplicar DSP a la pista aislada antes de transcribirla.

---

## Instalación

El repositorio utiliza Python 3.10 y `uv` para gestionar el entorno y las dependencias:

```bash
# 1. Crear el entorno virtual
uv venv --python 3.10

# 2. Instalar las dependencias del proyecto
uv pip install --python .venv/bin/python -r requirements.txt
```

En VS Code, selecciona `.venv/bin/python` como intérprete y kernel del notebook.

### Dataset Requerido (Slakh)
Para poder probar y evaluar la fidelidad de las transcripciones, este proyecto **requiere** el dataset **BabySlakh** (una versión reducida de Slakh2100). 
Debes descargarlo desde [Zenodo (Registro 4603870)](https://zenodo.org/records/4603870) y descomprimirlo en la raíz del proyecto (asegurando que exista la carpeta `babyslakh_16k/`).

## Referencias

- **Basic Pitch (Spotify):** Sistema eficiente para la Transcripción Musical Automática (Audio a MIDI). [Repositorio](https://github.com/spotify/basic-pitch)
- **Demucs (Meta):** Arquitectura de separación de fuentes de música en stems. [Repositorio](https://github.com/facebookresearch/demucs)
- **pretty_midi:** Herramienta para manipulación MIDI. [Repositorio](https://github.com/craffel/pretty-midi)
- **pedalboard (Spotify):** Librería para aplicar efectos de estudio (DSP). [Repositorio](https://github.com/spotify/pedalboard)
- **Slakh / BabySlakh Dataset:** Dataset renderizado con pares de audio y MIDI alineados, requerido para evaluación. [Zenodo](https://zenodo.org/records/4603870)

---
## Declaración de Uso de inteligencia artificial

La reestructuración y definición metodológica de este documento fue desarrollada con la asistencia de Gemini 3.1. 
Las ideas y conclusiones son del autor.


GPT-5.6 luna**