import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import librosa
    import librosa.display
    import pedalboard
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import basic_pitch

    import anywidget
    import traitlets
    import io
    import soundfile as sf


    return (
        Path,
        anywidget,
        basic_pitch,
        io,
        librosa,
        mo,
        np,
        pedalboard,
        plt,
        sf,
        traitlets,
    )


@app.cell
def _(Path):
    ruta_audio = Path("data/babyslakh_16k/Track00001/mix.wav")
    return (ruta_audio,)


@app.cell
def _(librosa, ruta_audio):
    audio_original, sr = librosa.load(ruta_audio, sr=22050, mono=True)
    return audio_original, sr


@app.cell
def _(audio_original, mo, sr):
    mo.audio(audio_original, rate=sr)
    return


@app.cell
def _(audio_original, librosa, np):
    espectro_original = np.abs(
        librosa.stft(audio_original, n_fft=2048, hop_length=512)
    )

    referencia_db = max(float(espectro_original.max()), 1e-5)

    original_db = librosa.amplitude_to_db(
        espectro_original,
        ref=referencia_db,
        top_db=None,
    )
    return original_db, referencia_db


@app.cell
def _(audio_original, sr):
    len(audio_original) / sr
    return


@app.cell
def _(basic_pitch):
    basic_pitch.ICASSP_2022_MODEL_PATH
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Filtro corte graves
    """)
    return


@app.cell
def _(mo):
    corte_graves = mo.ui.slider(
        start=20,
        stop=1000,
        step=5,
        value=35,
        label="Frecuencia de corte del paso alto (Hz)",
        show_value=True,
    )
    corte_graves
    return (corte_graves,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Filtro corte agudos
    """)
    return


@app.cell
def _(mo):
    corte_agudos = mo.ui.slider(
        start=500,
        stop=10000,
        step=100,
        value=5000,
        label="Frecuencia de corte del paso bajo (Hz)",
        show_value=True,
    )
    corte_agudos
    return (corte_agudos,)


@app.cell
def _(audio_original, corte_agudos, corte_graves, pedalboard, sr):
    cadena_filtros = pedalboard.Pedalboard([
        pedalboard.HighpassFilter(
            cutoff_frequency_hz=corte_graves.value
        ),
        pedalboard.LowpassFilter(
            cutoff_frequency_hz=corte_agudos.value
        ),
    ])

    audio_procesado = cadena_filtros(audio_original, sr)
    return (audio_procesado,)


@app.cell
def _(audio_procesado, mo, sr):
    mo.audio(audio_procesado, rate=sr)
    return


@app.cell
def _(audio_procesado, librosa, np, referencia_db):
    espectro_procesado = np.abs(
        librosa.stft(audio_procesado, n_fft=2048, hop_length=512)
    )

    procesado_db = librosa.amplitude_to_db(
        espectro_procesado,
        ref=referencia_db,
        top_db=None,
    )
    return (procesado_db,)


@app.cell
def _(librosa, original_db, plt, procesado_db, sr):
    fig, ejes = plt.subplots(
        1, 2,
        figsize=(12, 4),
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for _eje, _datos, _titulo in zip(
        ejes,
        [original_db, procesado_db],
        ["Original", "Con filtro"],
    ):
        _imagen = librosa.display.specshow(
            _datos,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="log",
            cmap="magma",
            vmin=-80,
            vmax=0,
            ax=_eje,
        )
        _eje.set_title(_titulo)

    fig.colorbar(
        _imagen,
        ax=ejes,
        label="dB respecto al máximo del original",
    )

    fig
    return


@app.cell
def _(anywidget, traitlets):
    class WavesurferWidget(anywidget.AnyWidget):
            """Interactive audio waveform widget using wavesurfer.js v7.

            Attributes:
                audio_data: Audio file as bytes (converted to data URL in JS)
                regions: List of {id, start, end, label, color, drag, resize}
                current_time: Current playback position in seconds
                is_playing: Whether audio is currently playing
                selected_region_id: ID of currently selected region
                duration: Total audio duration in seconds (set by JS)
                waveform_color: Color of the waveform
                progress_color: Color of the played portion
                height: Height of waveform in pixels
            """

            # Audio data (Python → JavaScript)
            audio_data = traitlets.Bytes(default_value=b"").tag(sync=True)

            # Regions (bidirectional)
            regions = traitlets.List(trait=traitlets.Dict()).tag(sync=True)

            # Playback state (bidirectional)
            current_time = traitlets.Float(default_value=0.0).tag(sync=True)
            is_playing = traitlets.Bool(default_value=False).tag(sync=True)

            # Selection state (bidirectional)
            selected_region_id = traitlets.Unicode(default_value="").tag(sync=True)

            # Audio metadata (JavaScript → Python)
            duration = traitlets.Float(default_value=0.0).tag(sync=True)

            # Appearance (Python → JavaScript)
            waveform_color = traitlets.Unicode(default_value="#4F4A85").tag(sync=True)
            progress_color = traitlets.Unicode(default_value="#383351").tag(sync=True)
            height = traitlets.Int(default_value=128).tag(sync=True)

            # Playback speed (Python ↔ JavaScript)
            playback_speed = traitlets.Float(default_value=1.0).tag(sync=True)

            _esm = """
            import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js';
            import RegionsPlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js';
            import TimelinePlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/timeline.esm.js';

            function render({ model, el }) {
              const container = document.createElement('div');
              container.style.width = '100%';
              container.style.fontFamily = 'sans-serif';

              const waveformDiv = document.createElement('div');
              waveformDiv.id = 'waveform-' + Math.random().toString(36).substr(2, 9);

              const controls = document.createElement('div');
              controls.style.marginTop = '10px';
              controls.style.display = 'flex';
              controls.style.gap = '10px';
              controls.style.alignItems = 'center';

              const playPauseBtn = document.createElement('button');
              playPauseBtn.textContent = '▶ Play';
              playPauseBtn.style.padding = '8px 16px';
              playPauseBtn.style.cursor = 'pointer';
              playPauseBtn.style.fontSize = '14px';

              const speedLabel = document.createElement('span');
              speedLabel.textContent = 'Speed: ';
              speedLabel.style.cssText = 'font-size:14px;margin-left:8px;';

              const speedSelect = document.createElement('select');
              speedSelect.style.cssText = 'padding:4px 8px;cursor:pointer;font-size:14px;border:1px solid #ccc;border-radius:4px;background:#fff;';
              const speeds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
              speeds.forEach(speed => {
                const option = document.createElement('option');
                option.value = speed;
                option.textContent = `${speed}x`;
                if (speed === 1.0) option.selected = true;
                speedSelect.appendChild(option);
              });

              const timeDisplay = document.createElement('span');
              timeDisplay.style.fontFamily = 'monospace';
              timeDisplay.style.fontSize = '14px';
              timeDisplay.textContent = '0.000 / 0.000';

              const regionInfo = document.createElement('div');
              regionInfo.style.marginTop = '10px';
              regionInfo.style.padding = '8px';
              regionInfo.style.backgroundColor = '#f0f0f0';
              regionInfo.style.borderRadius = '4px';
              regionInfo.style.fontSize = '13px';
              regionInfo.textContent = 'Click a region to select it';

              controls.appendChild(playPauseBtn);
              controls.appendChild(speedLabel);
              controls.appendChild(speedSelect);
              controls.appendChild(timeDisplay);
              const filaOnda = document.createElement('div');
    filaOnda.style.display = 'flex';
    filaOnda.style.alignItems = 'flex-start';

    const ejeY = document.createElement('div');
    ejeY.style.cssText = `
        position: relative;
        width: 55px;
        flex-shrink: 0;
        border-right: 1px solid #999;
        font-size: 12px;
        color: #555;
    `;
    ejeY.style.height = model.get('height') + 'px';
    ejeY.setAttribute('aria-label', 'Amplitud de −1 a 1');

    [1, 0.5, 0, -0.5, -1].forEach((valor) => {
        const etiqueta = document.createElement('span');
        etiqueta.textContent = valor.toFixed(1);
        etiqueta.style.position = 'absolute';
        etiqueta.style.right = '6px';
        etiqueta.style.top = ((1 - valor) / 2 * 100) + '%';

        const ajuste = valor === 1 ? '0%' :
                       valor === -1 ? '-100%' : '-50%';
        etiqueta.style.transform = `translateY(${ajuste})`;

        ejeY.appendChild(etiqueta);
    });

    waveformDiv.style.flex = '1';
    waveformDiv.style.minWidth = '0';

    filaOnda.appendChild(ejeY);
    filaOnda.appendChild(waveformDiv);
    container.appendChild(filaOnda);
              container.appendChild(controls);
              container.appendChild(regionInfo);
              el.appendChild(container);

              const regionsPlugin = RegionsPlugin.create();
              const timelinePlugin = TimelinePlugin.create({
                height: 20,
                timeInterval: 0.5,
                primaryLabelInterval: 5,
                style: { fontSize: '10px', color: '#6B7280' },
              });

              const wavesurfer = WaveSurfer.create({
                container: waveformDiv,
                waveColor: model.get('waveform_color'),
                progressColor: model.get('progress_color'),
                height: model.get('height'),
                normalize: false,
                barHeight: 1,
                plugins: [regionsPlugin, timelinePlugin],
              });

              const audioData = model.get('audio_data');
              if (audioData && audioData.byteLength > 0) {
                const blob = new Blob([audioData], { type: 'audio/mpeg' });
                const url = URL.createObjectURL(blob);
                wavesurfer.load(url);
              }

              wavesurfer.on('ready', () => {
                const dur = wavesurfer.getDuration();
                model.set('duration', dur);
                model.save_changes();
                timeDisplay.textContent = `0.000 / ${dur.toFixed(3)}`;

                // Set initial playback speed
                const initialSpeed = model.get('playback_speed');
                if (initialSpeed && initialSpeed !== 1.0) {
                  wavesurfer.setPlaybackRate(initialSpeed);
                  speedSelect.value = initialSpeed;
                }

                const regions = model.get('regions') || [];
                regions.forEach((region) => {
                  regionsPlugin.addRegion({
                    id: region.id,
                    start: region.start,
                    end: region.end,
                    color: region.color || 'rgba(0, 123, 255, 0.3)',
                    drag: region.drag !== false,
                    resize: region.resize !== false,
                    content: region.label || '',
                  });
                });
              });

              wavesurfer.on('timeupdate', (currentTime) => {
                const dur = wavesurfer.getDuration();
                timeDisplay.textContent = `${currentTime.toFixed(3)} / ${dur.toFixed(3)}`;
              });

              wavesurfer.on('play', () => {
                model.set('is_playing', true);
                model.save_changes();
                playPauseBtn.textContent = '⏸ Pause';
              });

              wavesurfer.on('pause', () => {
                model.set('is_playing', false);
                model.set('current_time', wavesurfer.getCurrentTime());
                model.save_changes();
                playPauseBtn.textContent = '▶ Play';
              });

              playPauseBtn.addEventListener('click', () => {
                wavesurfer.playPause();
              });

              speedSelect.addEventListener('change', () => {
                const speed = parseFloat(speedSelect.value);
                wavesurfer.setPlaybackRate(speed);
                model.set('playback_speed', speed);
                model.save_changes();
              });

              regionsPlugin.on('region-updated', (region) => {
                const regions = model.get('regions').map((r) => {
                  if (r.id === region.id) {
                    return { ...r, start: region.start, end: region.end };
                  }
                  return r;
                });
                model.set('regions', regions);
                model.save_changes();
              });

              regionsPlugin.on('region-clicked', (region, e) => {
                regionsPlugin.getRegions().forEach((r) => {
                  r.element.style.opacity = r.id === region.id ? '0.6' : '0.3';
                });

                model.set('selected_region_id', region.id);
                model.save_changes();

                regionInfo.innerHTML = `
                  <strong>Selected:</strong> ${region.content || region.id}<br>
                  <strong>Time:</strong> ${region.start.toFixed(3)}s - ${region.end.toFixed(3)}s
                  (duration: ${(region.end - region.start).toFixed(3)}s)
                `;

                e.stopPropagation();
              });

              model.on('change:audio_data', () => {
                const audioData = model.get('audio_data');
                if (audioData && audioData.byteLength > 0) {
                  const blob = new Blob([audioData], { type: 'audio/mpeg' });
                  const url = URL.createObjectURL(blob);
                  wavesurfer.load(url);
                }
              });

              model.on('change:regions', () => {
                regionsPlugin.clearRegions();
                const regions = model.get('regions') || [];
                regions.forEach((region) => {
                  regionsPlugin.addRegion({
                    id: region.id,
                    start: region.start,
                    end: region.end,
                    color: region.color || 'rgba(0, 123, 255, 0.3)',
                    drag: region.drag !== false,
                    resize: region.resize !== false,
                    content: region.label || '',
                  });
                });
              });

              model.on('change:current_time', () => {
                wavesurfer.setTime(model.get('current_time'));
              });

              model.on('change:playback_speed', () => {
                const speed = model.get('playback_speed');
                wavesurfer.setPlaybackRate(speed);
                speedSelect.value = speed;
              });

              return () => wavesurfer.destroy();
            }

            export default { render };
            """

            _css = """
            .wavesurfer-region {
              transition: opacity 0.2s ease;
            }
            """



    return (WavesurferWidget,)


@app.cell
def _(io, sf):
    def convertir_a_wav(senal, frecuencia):
        buffer = io.BytesIO()
        sf.write(
            buffer, senal, frecuencia,
            format="WAV", subtype="PCM_16",
        )
        return buffer.getvalue()

    return (convertir_a_wav,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visor original
    """)
    return


@app.cell
def _(WavesurferWidget, audio_original, convertir_a_wav, mo, sr):
    visor_original = mo.ui.anywidget(
        WavesurferWidget(
            audio_data=convertir_a_wav(audio_original, sr),
            regions=[],
            height=450,
        )
    )
    return (visor_original,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visor con pre procesamiento
    """)
    return


@app.cell
def _(WavesurferWidget, audio_procesado, convertir_a_wav, mo, sr):
    visor_procesado = mo.ui.anywidget(
        WavesurferWidget(
            audio_data=convertir_a_wav(audio_procesado, sr),
            regions=[],
            height=450,
        )
    )
    return (visor_procesado,)


@app.cell
def _(corte_graves, mo, visor_original, visor_procesado):
    mo.hstack(
        [
            mo.vstack([mo.md("### Original"), visor_original]),
            mo.vstack([mo.md(f"### Con filtro paso alto: {corte_graves.value} Hz"), visor_procesado]),
        ],
        widths="equal",
    )
    return


if __name__ == "__main__":
    app.run()
