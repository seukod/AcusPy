# Checklist para Presentación PPT: Proyecto Acus

Esta checklist está estructurada para una audiencia de **estudiantes y compañeros (nivel intermedio)**. El enfoque está en lo visual, lo auditivo y en explicar los conceptos clave sin abrumar con matemáticas densas.

## 1. Introducción y Problemática
- [ ] **Título del Proyecto:** Presentar el nombre y el objetivo general (Transcripción y separación de audio con IA).
- [ ] **La Problemática:** Explicar lo difícil que es transcribir música de oído o aislar instrumentos de una mezcla comercial (ej. tratar de sacar los acordes de guitarra de una canción llena de ruido).
- [ ] **Nuestra Pregunta de Investigación (La Idea Central):** Mostrar a la clase el eje del proyecto: *"¿Qué nivel de fidelidad se puede alcanzar al transcribir un instrumento de una mezcla a MIDI con IA, y cómo el procesamiento de señal (DSP) puede reducir las notas fantasma?"*. Explicar brevemente el pipeline: `Audio -> Separación (Demucs) -> Limpieza (DSP) -> Transcripción (Basic Pitch) -> MIDI`.

## 2. El Estado del Arte (Basic Pitch)
- [ ] **¿Qué es Basic Pitch?** Breve introducción al modelo de Spotify.
- [ ] **El Paper:** Mencionar que se basan en el paper oficial ([arXiv:2203.09893](https://arxiv.org/pdf/2203.09893)) y que buscan replicar parte de su validación.
- [ ] **Explicación de Métricas (Simples y Visuales):**
  - **F1-Score (F y Fno):** ¿Acertó la nota a tiempo? (Explicar brevemente qué es un Falso Positivo vs Falso Negativo en música, y que *Fno* evalúa solo el inicio, mientras que *F* evalúa inicio y duración).
  - **Multi-Pitch Accuracy (Acc):** ¿Acertó las frecuencias de manera general?

> [!TIP]
> Usa íconos o analogías (ej. tiro al blanco) para explicar el F1-Score sin mostrar la fórmula matemática completa.

## 3. Demostración Práctica (El Testeo con BabySlakh)
- [ ] **Introducir BabySlakh:** Explicar rápidamente qué es el dataset y por qué se usa (tiene la mezcla y el "Ground Truth" perfecto en MIDI).
- [ ] **Muestra de Audio ("Antes y Después"):**
  - Reproducir 5-10 segundos del audio de entrada.
  - Reproducir el audio generado a partir del MIDI de Basic Pitch.
- [ ] **Gráfico Comparativo (Piano Roll):**
  - Mostrar un Piano Roll superpuesto.
  - **Verde:** Notas que Basic Pitch acertó (True Positives).
  - **Rojo:** Notas fantasma que Basic Pitch inventó (False Positives).
  - **Azul/Gris:** Notas que Basic Pitch ignoró (False Negatives).
- [ ] **Tabla de Benchmarking (Estilo Heatmap):**
  - Presentar una tabla visual inspirada en la del paper, pero simplificada y enfocada **exclusivamente en BabySlakh**.
  - **Columnas:** Tus métricas elegidas (ej. `Acc` [Accuracy], `Fno` [F1-Score sin offset], `F` [F1-Score con offset]).
  - **Filas:** Las aproximaciones que están testeando (ej. *Basic Pitch Crudo* vs *Basic Pitch + Nuestro DSP*).
  - **Estilo Visual:** Usar un degradado de color (como el verde del paper) donde el verde más fuerte resalte el mejor puntaje. Así la audiencia capta el resultado en 1 segundo sin tener que leer cada número.

## 4. Dejando la puerta abierta (Futura Investigación)
- [ ] **El problema detectado:** Señalar en el Piano Roll anterior que Basic Pitch a veces se equivoca porque el audio tiene ruido de otros instrumentos o frecuencias indeseadas.
- [ ] **Planteamiento de la Hipótesis:** 
  > *"Proyectamos que al limpiar las pistas aisladas utilizando procesamiento de señales (DSP) —como filtros de ecualización y noise gates— reduciremos los falsos positivos (notas fantasma) de la IA, mejorando significativamente sus métricas de rendimiento (Acc, Fno, F)."*
- [ ] **Cierre:** "Este será nuestro siguiente paso en la investigación".

---
### 🛠️ Tareas técnicas previas a armar el PPT
Para que tengas el material listo para las diapositivas, necesitas generar:
- [ ] Un script que calcule las métricas usando la librería `mir_eval`.
- [ ] Extraer un fragmento `.wav` del original y un `.wav` del MIDI generado.
- [ ] Generar la imagen del Piano Roll superpuesto (puedes modificar el script del notebook `05_exploracion_babyslakh.ipynb` para graficar dos matrices juntas con matplotlib).
