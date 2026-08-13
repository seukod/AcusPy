# Proyecto Acus — Transcripción y separación de audio con IA

## Ideas y Objetivos

El objetivo principal de este proyecto es desarrollar una **aplicación web interactiva** que brinde a los usuarios la facilidad de aprender a tocar cualquier canción que tengan almacenada en su computadora. 

El problema que se busca resolver es la falta de herramientas automatizadas y accesibles para transcribir canciones arbitrarias. Al subir una canción a la plataforma, el sistema debe ser capaz de procesarla, extraer sus notas musicales y enseñarle al usuario cómo tocarla mientras el instrumento suena al unísono.

> [!NOTE]
> **Consideración sobre el Marco Legal:**
> Existe una limitante respecto al marco legal relacionado con la procedencia de los archivos de audio de los usuarios (por ejemplo, descargas no oficiales o piratería). Sin embargo, este es un aspecto externo que no afecta el alcance técnico ni los objetivos académicos de este proyecto, cuyo foco central es la implementación tecnológica de Inteligencia Artificial y Procesamiento de Señales Digitales.

## Metodología

Para lograr la transcripción de una mezcla de audio compleja a notas musicales legibles, el proyecto plantea un *pipeline* basado en modelos de IA y procesamiento de señales (DSP):

- [x] **1. Separación de Fuentes (Demucs):** El primer gran desafío en la transcripción automática de música es la interferencia entre instrumentos. Se utiliza el modelo **Demucs** de Meta para aislar pistas específicas (como la guitarra, voces o bajo) separándolas del resto de la canción.
- [ ] **2. Procesamiento de Señal (DSP) (Por investigar):** Antes de intentar detectar las notas, la pista aislada es procesada con ecualización (filtros paso alto y paso bajo) y compuertas de ruido (Noise Gate). Esto limpia las frecuencias residuales de otros instrumentos que hayan quedado tras la separación.
  - [ ] *Tarea:* Investigar y definir qué frecuencias de corte (cutoff) son las ideales para filtrar instrumentos específicos (ej. guitarra vs. bajo).
  - [ ] *Tarea:* Experimentar con diferentes umbrales (thresholds) de la compuerta de ruido para asegurar que no se corten notas largas abruptamente (sustain).
- [ ] **3. Extracción de Notas (Basic Pitch) (Por investigar):** Una vez que el audio está purificado, se utiliza **Basic Pitch** (de Spotify). Este modelo analiza las frecuencias fundamentales y transcribe el audio crudo a un archivo MIDI.
  - [ ] *Tarea:* Investigar la viabilidad técnica y latencia de ejecutar la conversión de audio a MIDI en **tiempo real** (mientras el usuario toca el instrumento en vivo).
- [ ] **4. Post-procesamiento Musical (Por investigar):** La transcripción directa suele contener errores (notas fantasma o duraciones irreales). La última fase del pipeline filtra mediante código las notas extremadamente cortas, resuelve solapamientos y fuerza (cuantiza) el resultado para que pertenezca a una escala musical teórica previamente definida, obteniendo un MIDI final limpio.
  - [ ] *Tarea:* Investigar cómo programar un algoritmo para forzar/cuantizar las notas crudas a una escala musical específica (ej. Do Mayor) usando `pretty_midi`.
  - [ ] *Tarea:* Definir bajo qué umbral de duración (milisegundos) una nota debe considerarse "ruido fantasma" y ser eliminada del MIDI final.

## [Placeholder] Arquitectura de la Aplicación Web
*(Espacio reservado para documentar el diseño y la integración técnica entre el backend de procesamiento de audio y la interfaz web).*

## [Placeholder] Preguntas y Reflexiones
*(Espacio reservado para documentar dudas, ideas o problemas que vayan surgiendo durante el desarrollo del ramo).*
- *¿Sería posible implementar la conversión en tiempo real de manera eficiente? (Considerando que separar stems con Demucs en tiempo real es costoso, pero procesar la señal limpia directo a Basic Pitch podría ser más ligero).*
- *¿Qué herramientas o librerías en Python son las más eficientes para aplicar Procesamiento de Señal (DSP) (como ecualización y compuertas de ruido) a las pistas aisladas?*
- *¿Cuál sería la ruta técnica (arquitectura de software) para escalar este pipeline hecho en Python y convertirlo en un plugin de audio (VST3/AU) utilizable en DAWs (Ableton, FL Studio, etc.)?*

## Instalación

El repositorio utiliza un entorno en Python 3.10. Para inicializar el proyecto y descargar las librerías necesarias para que funcionen los modelos (`basic-pitch` y `demucs`), sigue estos pasos:

```bash
# 1. Crear y activar el entorno virtual
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Instalar los modelos de IA y herramientas de manipulación de audio
pip install basic-pitch demucs pretty_midi numpy pedalboard soundfile
```

## Referencias

- **Basic Pitch (Spotify):** Sistema eficiente para la Transcripción Musical Automática (Audio a MIDI). [Repositorio](https://github.com/spotify/basic-pitch)
- **Demucs (Meta):** Arquitectura de separación de fuentes de música en stems. [Repositorio](https://github.com/facebookresearch/demucs)
- **pretty_midi:** Herramienta para la manipulación y análisis de datos en formato MIDI usando Python. [Repositorio](https://github.com/craffel/pretty-midi)
- **pedalboard (Spotify):** Librería para aplicar efectos de estudio y manipulación de señal en Python. [Repositorio](https://github.com/spotify/pedalboard)
