# Ideas

- Trabajar con pasajes sonoros?

- Transcripción Musical Automática Audio a MIDI 
    - Tomar una grabación de audio (por ejemplo, una melodía tocada en guitarra o piano) y usar un modelo de IA que detecte las frecuencias fundamentales ($f_0$), las notas exactas y el tiempo en que se tocan, generando un archivo MIDI
    - Dataset: MAPS Dataset o URMP (audio polifónico etiquetado nota por nota con MIDI).

    - ``repo de referencia`` https://github.com/spotify/basic-pitch

    - El modelo ligero de estimación de pitch y transcripción a MIDI desarrollado por Spotify en Python

    - se podría usar la arquitectura o reentrenarlo con otro dataset

    - Segun la ia : resuelve un problema clásico de procesamiento de señal (detección de pitch y armónicos)

// TODO probar el repo de referencia