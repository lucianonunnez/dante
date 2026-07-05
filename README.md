# TRIBE v2 · Predicción de actividad cerebral a partir de video

Pipeline de investigación que estima **actividad cortical (fMRI sintético) segundo a segundo** a partir de un video, alinea el resultado con la transcripción del audio y genera un **informe de atención** con los momentos de mayor y menor activación.

> Proyecto experimental sobre modelos multimodales aplicados a neurociencia. Corre en Google Colab con GPU.

## ¿Qué hace?

Dado un video (`mp4/mov/mkv/webm/avi`), el notebook produce:

- **Predicción de fMRI** por segundo de video usando el modelo **TRIBE v2**.
- **Transcripción con timestamps** a nivel palabra (WhisperX).
- **Energía de audio** por segundo, para cruzar estímulo sonoro con respuesta cortical.
- **Informe de atención**: timeline de activación (norma L2 y por hemisferio) con detección de **picos y valles**, alineados con las palabras que se dicen en ese instante.
- **Render del cerebro** en los momentos clave y (opcional) un video animado de la activación.
- **Exportación** de todos los resultados a `.npz` + informe en **Markdown y PDF**.

## Stack

| Área | Tecnologías |
|------|-------------|
| Modelo | TRIBE v2, Llama-3.2 (HuggingFace) |
| Audio / NLP | WhisperX, transcripción con timestamps |
| Datos / cómputo | Python, NumPy, SciPy |
| Visualización | Matplotlib, Nilearn (render cortical) |
| Reportes | ReportLab (PDF), Markdown |
| Entorno | Google Colab (GPU T4 / L4 / A100) |

## Cómo usarlo

1. Abrí `out/tribe_colab_v2_mejorado.ipynb` en **Google Colab**.
2. Activá la GPU: *Entorno de ejecución → Cambiar tipo de entorno → GPU*.
3. Cargá tu token de HuggingFace como secreto `HF_TOKEN` (hay que aceptar los términos de [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)).
4. Ejecutá las celdas en orden. La celda 1 pide **reiniciar el runtime**: hacelo y seguí desde la celda 2.
5. Subí tu video cuando te lo pida.
6. Descargá el `.npz` y el informe en PDF al final.

## Estructura

```
out/
├── tribe_colab_v2_mejorado.ipynb   # Notebook principal (Colab)
└── build_notebook.py               # Script que genera el notebook de forma reproducible
```

El notebook se genera de forma reproducible desde `build_notebook.py`:

```bash
python out/build_notebook.py
```

## Estado

Prototipo funcional de investigación. Pensado para explorar la relación entre estímulos audiovisuales y activación cortical predicha.
