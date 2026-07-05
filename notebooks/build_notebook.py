"""Build improved TRIBE v2 Colab notebook."""
import json
from pathlib import Path

def md(text, cell_id=None):
    cell = {
        "cell_type": "markdown",
        "metadata": {"id": cell_id} if cell_id else {},
        "source": text.split("\n") if "\n" in text else [text],
    }
    # Normalize: each line should end with \n except the last
    src = text.splitlines(keepends=True)
    cell["source"] = src
    return cell

def code(text, cell_id=None):
    cell = {
        "cell_type": "code",
        "metadata": {"id": cell_id} if cell_id else {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }
    return cell


cells = []

# ============================================================
# Intro
# ============================================================
cells.append(md("""# TRIBE v2 en Colab - RMN de videos aburridos

Notebook para correr **TRIBE v2 con GPU en Colab** y obtener:
- Predicción de actividad cerebral (fMRI sintético) por segundo de video
- Transcripción con timestamps (WhisperX)
- **Informe de atención**: segundos de mayor y menor activación cortical alineados con el texto
- Video animado del cerebro + exportable `.npz` para la app local

**Pasos:**
1. Activá la GPU: **Entorno de ejecución → Cambiar tipo de entorno → GPU A100 / L4 / T4** (cualquiera sirve, A100 va más rápido).
2. Poné tu token de HuggingFace en `Secrets` (icono de llave, izquierda) con nombre **HF_TOKEN**. Si no, te lo pide interactivamente.
   - Necesitás haber aceptado los términos de Llama-3.2 en https://huggingface.co/meta-llama/Llama-3.2-1B
3. Corré las celdas en orden. La celda 1 **te va a pedir reiniciar el runtime** — hacelo (Runtime → Restart) y seguí desde la celda 2.
4. Cuando te lo pida, subí tu video (mp4/mov/mkv/webm/avi).
5. Al final descargá el `.npz` + el informe en PDF, o recibilo por mail.
""", "intro"))

# ============================================================
# 1. Install
# ============================================================
cells.append(md("""## 1. Instalar dependencias

Tarda ~4-6 minutos la primera vez. **Después de esta celda, Colab te va a pedir reiniciar el runtime: hacelo** (Runtime → Restart session) y volvé directo a la **celda 2**, no re-ejecutes esta.
""", "sec-install-md"))

cells.append(code("""# Instalación robusta. Deja que pip resuelva los pins de tribev2.
# Estrategia: instalar tribev2 primero (pulls numpy/scipy correctos),
# después nilearn, y avisar para restart. No desinstalamos a mano para
# no romper Colab a mitad de instalación si algo se cancela.

import sys, subprocess, os

def pip(*args):
    print(">>> pip", " ".join(args))
    r = subprocess.run([sys.executable, "-m", "pip", *args],
                       capture_output=False)
    if r.returncode != 0:
        raise SystemExit(f"pip {args} fallo con codigo {r.returncode}")

# 1) Forzar numpy 2.2.6 (lo que pide tribev2). Reinstala scipy compatible.
pip("install", "-q", "--upgrade", "numpy==2.2.6")
pip("install", "-q", "--upgrade", "scipy>=1.13.0,<1.17", "--no-deps")

# 2) tribev2 + extras de plotting
pip("install", "-q",
    "tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git")

# 3) nilearn (para el cerebro) + imageio-ffmpeg (para el video animado)
pip("install", "-q", "nilearn", "imageio[ffmpeg]")

# 4) reportlab para el PDF del informe
pip("install", "-q", "reportlab")

print("\\n" + "="*60)
print("INSTALACION OK")
print("="*60)
print("AHORA: Runtime -> Restart session, y volve directo a la CELDA 2.")
print("(Si no reinicias, numpy/scipy van a fallar con import errors.)")
""", "install-cell"))

# ============================================================
# 2. HF login
# ============================================================
cells.append(md("""## 2. Login HuggingFace

Poné tu token en `Secrets` (icono llave izquierda) con nombre `HF_TOKEN`. Si no, te lo va a pedir.
""", "sec-hf-md"))

cells.append(code("""import os

HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    pass

if not HF_TOKEN:
    import getpass
    HF_TOKEN = getpass.getpass("Pega tu HF_TOKEN (no se va a mostrar): ").strip()

assert HF_TOKEN, "HF_TOKEN vacio. Configuralo en Secrets o ingresalo."
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
print("HF_TOKEN configurado:", HF_TOKEN[:8] + "...")
""", "hf-cell"))

# ============================================================
# 3. Cargar modelo
# ============================================================
cells.append(md("""## 3. Cargar el modelo TRIBE

La primera vez baja ~1 GB de checkpoint. Tarda 2-3 minutos. Detecta automáticamente A100 / L4 / T4 / CPU.
""", "sec-model-md"))

cells.append(code("""import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from pathlib import Path
from tribev2.demo_utils import TribeModel
from tribev2.plotting import PlotBrain

CACHE_FOLDER = Path("./cache")
CACHE_FOLDER.mkdir(exist_ok=True)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU detectada: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    if "A100" in gpu_name:
        print("  -> A100: vas a poder procesar videos largos (>5min) sin problema.")
    elif "L4" in gpu_name:
        print("  -> L4: bien para videos medianos (~2-3 min).")
    elif "T4" in gpu_name:
        print("  -> T4: ok para videos cortos (~30-90s).")
else:
    print("WARN: sin GPU. Va a andar pero lentisimo. Activa GPU en Runtime -> Change runtime type.")

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_FOLDER)
plotter = PlotBrain(mesh="fsaverage5")
print("\\nModelo y plotter listos.")
""", "model-cell"))

# ============================================================
# 4. Upload video
# ============================================================
cells.append(md("""## 4. Subí tu video

Apretá el botón de abajo y seleccioná el video desde tu PC. Mp4 / mov / mkv / webm / avi funcionan.
""", "sec-upload-md"))

cells.append(code("""from google.colab import files
import shutil
from pathlib import Path

uploaded = files.upload()
assert len(uploaded) == 1, "Sube un solo archivo."
filename = list(uploaded.keys())[0]
video_path = CACHE_FOLDER / filename
if Path(filename).exists():
    shutil.move(filename, video_path)
print("Video guardado en:", video_path, f"({video_path.stat().st_size/1e6:.1f} MB)")
""", "upload-cell"))

# ============================================================
# 5. Process
# ============================================================
cells.append(md("""## 5. Procesar el video con TRIBE

Extrae audio + transcribe (WhisperX) + extrae features (LLaMA / V-JEPA2 / Wav2Vec) + predice fMRI.
- T4: ~1-3 min para 30-60s de video
- L4 / A100: ~30s-1min
""", "sec-process-md"))

cells.append(code("""import time
t0 = time.time()

df = model.get_events_dataframe(video_path=video_path)
print(f"Eventos detectados: {len(df)}  (tipos: {sorted(df['type'].unique().tolist())})")
display(df.head(8)[["type", "start", "duration", "filepath", "text"]])

preds, segments = model.predict(events=df)
print(f"\\nPrediccion shape: {preds.shape}  (timesteps, vertices)")
print(f"Tiempo total: {time.time()-t0:.1f}s")
""", "process-cell"))

# ============================================================
# 6. Visualize brain
# ============================================================
cells.append(md("""## 6. Visualizar en el cerebro (primeros 15s)

Cada panel es un segundo, con el frame del estímulo abajo.
""", "sec-viz-md"))

cells.append(code("""import matplotlib.pyplot as plt

n_timesteps = min(15, preds.shape[0])
fig = plotter.plot_timesteps(
    preds[:n_timesteps],
    segments=segments[:n_timesteps],
    cmap="fire", norm_percentile=99, vmin=.6,
    alpha_cmap=(0, .2), show_stimuli=True,
)
plt.show()
""", "viz-cell"))

# ============================================================
# 7. Transcript + words
# ============================================================
cells.append(md("""## 7. Extraer transcripción + palabras con timestamps

De `df` sacamos las palabras detectadas por WhisperX y la transcripción completa.
""", "sec-words-md"))

cells.append(code("""import numpy as np

word_rows = df[df["type"] == "Word"].reset_index(drop=True)
words = [
    {"text": str(r["text"]),
     "start": float(r["start"]),
     "end": float(r["start"]) + float(r["duration"])}
    for _, r in word_rows.iterrows()
]
sentence_rows = df[df["type"].isin(["Text", "Sentence"])]
transcript = " ".join(str(t) for t in sentence_rows["text"].dropna().tolist()).strip()
duration = float(df[df["type"] == "Video"]["duration"].iloc[0]) if (df["type"] == "Video").any() else float(preds.shape[0])

print(f"Palabras: {len(words)}")
print(f"Transcripcion: {transcript[:300]}{'...' if len(transcript) > 300 else ''}")
print(f"Duracion: {duration:.1f}s")
""", "words-cell"))

# ============================================================
# 8. Audio energy
# ============================================================
cells.append(md("""## 8. Calcular energía de audio por segundo
""", "sec-audio-md"))

cells.append(code("""import wave
n_seconds = int(np.ceil(duration))

# El audio lo extrajo TRIBE
candidates = list(CACHE_FOLDER.glob("*.wav"))
assert candidates, "No se encontro el .wav extraido por TRIBE"
audio_path = candidates[0]

with wave.open(str(audio_path), "rb") as wf:
    sr = wf.getframerate()
    n_channels = wf.getnchannels()
    raw = wf.readframes(wf.getnframes())
samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
if n_channels > 1:
    samples = samples.reshape(-1, n_channels).mean(axis=1)

audio_energy = np.zeros(n_seconds, dtype=np.float32)
for i in range(n_seconds):
    chunk = samples[i*sr:(i+1)*sr]
    if chunk.size:
        audio_energy[i] = float(np.sqrt(np.mean(chunk*chunk)))
if audio_energy.max() > 0:
    audio_energy /= audio_energy.max()
print("Audio energy shape:", audio_energy.shape,
      "(picos en", np.argsort(audio_energy)[-5:][::-1].tolist(), ")")
""", "audio-cell"))

# ============================================================
# 9. INFORME DE ATENCION  ← NUEVO
# ============================================================
cells.append(md("""## 9. Informe de atención por segmento

Calcula la magnitud de actividad cortical por segundo (la "intensidad de atención" del cerebro modelado),
identifica los **picos altos** y **valles bajos**, y los alinea con las palabras del transcript.

> Nota: TRIBE no mide *atención* en sentido psicológico — predice activación BOLD agregada en córtex.
> Lo que llamamos "atención" acá es la magnitud `||predicción||` por segundo, que tiende a ser alta cuando
> el modelo predice respuesta cortical intensa (estímulos salientes / lingüísticamente densos).
""", "sec-attn-md"))

cells.append(code("""# --- Calcular metricas de atencion por segundo ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Magnitud por segundo: norma L2 de las predicciones (mas robusta que la media absoluta)
# preds shape: (n_seconds, n_vertices)
attn_l2   = np.linalg.norm(preds, axis=1)
attn_mean = preds.mean(axis=1)         # mas suave, sensible al signo
attn_abs  = np.abs(preds).mean(axis=1) # similar a L2 pero normalizada

# Normalizar a 0-1 para reportes
def _norm01(x):
    x = np.asarray(x, dtype=np.float32)
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

attn_score = _norm01(attn_l2)  # nuestra metrica principal
n_t = len(attn_score)

# Tambien por hemisferio (los primeros 10242 vertices de fsaverage5 son LH, despues RH)
HALF = preds.shape[1] // 2
attn_lh = _norm01(np.linalg.norm(preds[:, :HALF], axis=1))
attn_rh = _norm01(np.linalg.norm(preds[:, HALF:], axis=1))

print(f"Atencion calculada para {n_t} segundos.")
print(f"  rango L2: {attn_l2.min():.3f} - {attn_l2.max():.3f}")
print(f"  score 0-1: media={attn_score.mean():.3f}, std={attn_score.std():.3f}")
""", "attn-compute"))

cells.append(code("""# --- Helper: que palabras caen en cada segundo ---
def words_in_interval(start_s, end_s, words_list):
    out = []
    for w in words_list:
        # overlap > 0
        if w["end"] > start_s and w["start"] < end_s:
            out.append(w["text"])
    return out

def segment_text(t_sec, window=1.0):
    return " ".join(words_in_interval(t_sec, t_sec + window, words)).strip()

# --- Encontrar picos (top-K) y valles (bot-K) ---
TOP_K = 5
# Para evitar que los top-K esten todos pegados, requeris separacion minima
def topk_with_gap(scores, k, min_gap=2):
    order = np.argsort(scores)[::-1]
    chosen = []
    for idx in order:
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
            if len(chosen) == k:
                break
    return chosen

def botk_with_gap(scores, k, min_gap=2):
    order = np.argsort(scores)
    chosen = []
    for idx in order:
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(int(idx))
            if len(chosen) == k:
                break
    return chosen

peaks   = topk_with_gap(attn_score, TOP_K, min_gap=2)
valleys = botk_with_gap(attn_score, TOP_K, min_gap=2)

print(f"Top {TOP_K} segundos de MAYOR atencion:")
for rank, t in enumerate(peaks, 1):
    txt = segment_text(t) or "(silencio)"
    print(f"  {rank}. t={t:>3}s  score={attn_score[t]:.3f}  audio={audio_energy[t]:.2f}  -> {txt!r}")

print(f"\\nTop {TOP_K} segundos de MENOR atencion:")
for rank, t in enumerate(valleys, 1):
    txt = segment_text(t) or "(silencio)"
    print(f"  {rank}. t={t:>3}s  score={attn_score[t]:.3f}  audio={audio_energy[t]:.2f}  -> {txt!r}")
""", "attn-peaks"))

cells.append(code("""# --- Plot: timeline de atencion con picos/valles + audio ---
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
xs = np.arange(n_t)
ax.fill_between(xs, attn_score, alpha=0.35, color="#e63946", label="Atencion (cortex total, L2 normalizada)")
ax.plot(xs, attn_lh, "--", color="#1d3557", alpha=0.7, linewidth=1.2, label="Hemisferio izq.")
ax.plot(xs, attn_rh, "--", color="#2a9d8f", alpha=0.7, linewidth=1.2, label="Hemisferio der.")

for rank, t in enumerate(peaks, 1):
    ax.axvline(t, color="#e63946", alpha=0.6, linewidth=1)
    ax.annotate(f"P{rank}", xy=(t, attn_score[t]), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9,
                color="#e63946", fontweight="bold")
for rank, t in enumerate(valleys, 1):
    ax.axvline(t, color="#457b9d", alpha=0.4, linewidth=1, linestyle=":")
    ax.annotate(f"V{rank}", xy=(t, attn_score[t]), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=9,
                color="#457b9d", fontweight="bold")

ax.set_ylabel("Atencion cortical (0-1)")
ax.set_title("Linea de tiempo de atencion - P=picos, V=valles")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.15)

ax2 = axes[1]
ax2.bar(xs, audio_energy, color="#264653", alpha=0.7, width=0.9)
ax2.set_ylabel("Energia audio")
ax2.set_xlabel("Tiempo (s)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
timeline_png = CACHE_FOLDER / f"{video_path.stem}_timeline.png"
plt.savefig(timeline_png, dpi=120, bbox_inches="tight", facecolor="white")
plt.show()
print("Timeline guardado:", timeline_png)
""", "attn-plot"))

cells.append(code("""# --- Renderizar el cerebro en los picos y valles, lado a lado ---
import matplotlib.pyplot as plt

def render_pick(t_idx):
    fig = plotter.plot_timesteps(
        preds[t_idx:t_idx+1], segments=segments[t_idx:t_idx+1],
        cmap="fire", norm_percentile=99, vmin=.6,
        alpha_cmap=(0, .2), show_stimuli=True,
    )
    return fig

print("=== PICOS DE ATENCION (top 3) ===")
for rank, t in enumerate(peaks[:3], 1):
    print(f"\\nP{rank}: t={t}s | score={attn_score[t]:.3f} | texto: {segment_text(t)!r}")
    render_pick(t); plt.show()

print("\\n=== VALLES DE ATENCION (top 3) ===")
for rank, t in enumerate(valleys[:3], 1):
    print(f"\\nV{rank}: t={t}s | score={attn_score[t]:.3f} | texto: {segment_text(t)!r}")
    render_pick(t); plt.show()
""", "attn-render"))

cells.append(code("""# --- Generar informe en Markdown + PDF ---
from datetime import datetime

stem = video_path.stem
report_md_path  = CACHE_FOLDER / f"{stem}_informe.md"
report_pdf_path = CACHE_FOLDER / f"{stem}_informe.pdf"
report_txt_path = CACHE_FOLDER / f"{stem}_informe.txt"

# --- Markdown ---
def line_for(rank, t, kind):
    txt = segment_text(t) or "(silencio / sin habla)"
    return (f"- **{kind}{rank} - t={t}s** | score={attn_score[t]:.3f}"
            f" | audio={audio_energy[t]:.2f}\\n"
            f"    > {txt}")

md_lines = [
    f"# Informe RMN sintetica - {stem}",
    f"",
    f"_Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
    f"",
    f"## Resumen",
    f"- **Duracion:** {duration:.1f}s ({n_t} segundos analizados)",
    f"- **Palabras detectadas:** {len(words)}",
    f"- **Score medio de atencion:** {attn_score.mean():.3f}",
    f"- **Variabilidad (std):** {attn_score.std():.3f}",
    f"- **Asimetria hemisferica media:** LH={attn_lh.mean():.3f} | RH={attn_rh.mean():.3f}",
    f"",
    f"## Transcripcion",
    f"",
    transcript if transcript else "_(sin habla detectada)_",
    f"",
    f"## Picos de atencion (top {TOP_K})",
    f"Momentos donde el modelo predice mayor activacion cortical agregada.",
    f"",
]
md_lines += [line_for(i+1, t, "P") for i, t in enumerate(peaks)]
md_lines += [
    f"",
    f"## Valles de atencion (top {TOP_K})",
    f"Momentos de menor activacion cortical (posible aburrimiento / desenganche).",
    f"",
]
md_lines += [line_for(i+1, t, "V") for i, t in enumerate(valleys)]
md_lines += [
    f"",
    f"## Interpretacion",
    f"- TRIBE v2 estima respuesta BOLD en fsaverage5 a partir de features de LLaMA, V-JEPA2 y Wav2Vec.",
    f"- El score 0-1 es la norma L2 de la prediccion por segundo, no una medida psicologica de atencion.",
    f"- Picos suelen alinearse con eventos sensoriales o linguisticos densos; valles con silencios o monotonia.",
    f"",
    f"## Archivos",
    f"- Timeline: `{stem}_timeline.png`",
    f"- Resultados crudos: `{stem}_tribe_resultado.npz` (cargar en app local)",
]
report_md = "\\n".join(md_lines)
report_md_path.write_text(report_md, encoding="utf-8")
print("Markdown:", report_md_path)

# --- TXT plano ---
report_txt_path.write_text(report_md.replace("**", "").replace("`", ""), encoding="utf-8")
print("TXT:", report_txt_path)

# --- PDF con reportlab ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Image as RLImage, PageBreak)
    from reportlab.lib.units import cm

    doc = SimpleDocTemplate(str(report_pdf_path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]; h2 = styles["Heading2"]; body = styles["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=9, leading=11)

    story = []
    story.append(Paragraph(f"Informe RMN sintetica - {stem}", h1))
    story.append(Paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", small))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Resumen", h2))
    story.append(Paragraph(
        f"Duracion: {duration:.1f}s &middot; {len(words)} palabras &middot; "
        f"score medio {attn_score.mean():.3f} (std {attn_score.std():.3f})<br/>"
        f"Asimetria hemisferica: LH {attn_lh.mean():.3f} | RH {attn_rh.mean():.3f}",
        body))
    story.append(Spacer(1, 0.3*cm))

    if timeline_png.exists():
        story.append(RLImage(str(timeline_png), width=17*cm, height=7*cm))
        story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Transcripcion", h2))
    story.append(Paragraph((transcript or "(sin habla detectada)")[:4000]
                           .replace("&", "&amp;").replace("<", "&lt;"), body))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(f"Picos de atencion (top {TOP_K})", h2))
    for i, t in enumerate(peaks, 1):
        txt = (segment_text(t) or "(silencio)").replace("&", "&amp;").replace("<", "&lt;")
        story.append(Paragraph(
            f"<b>P{i} - t={t}s</b> &middot; score {attn_score[t]:.3f} "
            f"&middot; audio {audio_energy[t]:.2f}<br/>"
            f"<i>{txt}</i>", body))
        story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph(f"Valles de atencion (top {TOP_K})", h2))
    for i, t in enumerate(valleys, 1):
        txt = (segment_text(t) or "(silencio)").replace("&", "&amp;").replace("<", "&lt;")
        story.append(Paragraph(
            f"<b>V{i} - t={t}s</b> &middot; score {attn_score[t]:.3f} "
            f"&middot; audio {audio_energy[t]:.2f}<br/>"
            f"<i>{txt}</i>", body))
        story.append(Spacer(1, 0.15*cm))

    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Interpretacion", h2))
    story.append(Paragraph(
        "TRIBE v2 estima respuesta BOLD en fsaverage5 a partir de features de LLaMA, "
        "V-JEPA2 y Wav2Vec. El score 0-1 es la norma L2 de la prediccion por segundo, "
        "no una medida psicologica de atencion. Picos suelen alinearse con eventos "
        "sensoriales / linguisticos densos; valles con silencios o monotonia.",
        body))

    doc.build(story)
    print("PDF:", report_pdf_path, f"({report_pdf_path.stat().st_size/1e3:.0f} KB)")
except Exception as e:
    print("WARN: no se pudo generar PDF:", e)
    report_pdf_path = None

print("\\n--- Vista previa del informe ---\\n")
print(report_md[:1500])
""", "attn-report"))

# ============================================================
# 10. Export npz
# ============================================================
cells.append(md("""## 10. Exportar resultados a .npz

Archivo único para descargar y abrir en la app local.
""", "sec-export-md"))

cells.append(code("""import json as _json

out_path = CACHE_FOLDER / (video_path.stem + "_tribe_resultado.npz")

activation = np.asarray(preds, dtype=np.float32)
if activation.shape[0] < n_seconds:
    pad = np.zeros((n_seconds - activation.shape[0], activation.shape[1]), dtype=np.float32)
    activation = np.vstack([activation, pad])
else:
    activation = activation[:n_seconds]

# Resumen de atencion serializado
attention_report = {
    "score":   attn_score.tolist(),
    "lh":      attn_lh.tolist(),
    "rh":      attn_rh.tolist(),
    "peaks":   [{"t": int(t), "score": float(attn_score[t]),
                 "text": segment_text(t)} for t in peaks],
    "valleys": [{"t": int(t), "score": float(attn_score[t]),
                 "text": segment_text(t)} for t in valleys],
}

np.savez_compressed(
    out_path,
    activation=activation,
    audio_energy=audio_energy,
    transcript=np.array(transcript, dtype=object),
    words_json=np.array(_json.dumps(words, ensure_ascii=False), dtype=object),
    attention_json=np.array(_json.dumps(attention_report, ensure_ascii=False), dtype=object),
    duration=np.float32(duration),
    n_seconds=np.int32(n_seconds),
    mode=np.array("tribe", dtype=object),
)
print("Guardado en:", out_path, f"({out_path.stat().st_size/1e6:.1f} MB)")
""", "export-cell"))

# ============================================================
# 11. Download
# ============================================================
cells.append(md("""## 11. Descargar resultados

Bajan a tu PC: el `.npz` (para la app local) + el informe en PDF/MD.
""", "sec-download-md"))

cells.append(code("""from google.colab import files

for p in [out_path, report_pdf_path, report_md_path, timeline_png]:
    if p and p.exists():
        try:
            files.download(str(p))
        except Exception as e:
            print(f"WARN descargando {p.name}: {e}")
""", "download-cell"))

# ============================================================
# 12. Drive + email
# ============================================================
cells.append(md("""## 12. (Opcional) Video animado del cerebro + Drive + email

Genera un mp4 con el cerebro segundo a segundo, copia todo a Google Drive y opcionalmente envía un email con resumen + adjuntos chicos.

> Para el email necesitás un **App Password** de Gmail (16 chars, sin espacios). Generalo en https://myaccount.google.com/apppasswords
""", "sec-extra-md"))

cells.append(code("""# ============================================================
# Configuracion (editar antes de correr)
# ============================================================
DEST_EMAIL          = "lucianonunnez@gmail.com"
GMAIL_APP_PASSWORD  = ""                                 # 16 chars del App Password (vacio = no manda email)
GMAIL_SENDER        = "lucianonunnez@gmail.com"
DRIVE_FOLDER_NAME   = "RMN_videos_aburridos"
GENERATE_BRAIN_VIDEO = True                              # False si solo querés el informe / drive

import os, shutil, io, smtplib, traceback
from pathlib import Path
from email.message import EmailMessage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 1) Drive ---
drive_dir = None
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    drive_dir = Path(f"/content/drive/MyDrive/{DRIVE_FOLDER_NAME}")
    drive_dir.mkdir(parents=True, exist_ok=True)
    print("Drive montado en:", drive_dir)
except Exception as e:
    print("WARN: no se pudo montar Drive:", e)

# --- 2) Video animado del cerebro ---
brain_video_path = None
if GENERATE_BRAIN_VIDEO:
    print("\\nGenerando video animado del cerebro (puede tardar varios minutos)...")
    n_t = preds.shape[0]
    frames_np = []
    for t in range(n_t):
        fig = plotter.plot_timesteps(
            preds[t:t+1], segments=segments[t:t+1],
            cmap="fire", norm_percentile=99, vmin=.6,
            alpha_cmap=(0, .2), show_stimuli=True,
        )
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=85, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        buf.seek(0)
        frames_np.append(np.array(Image.open(buf).convert("RGB")))
        if (t+1) % 5 == 0 or t == n_t - 1:
            print(f"  frame {t+1}/{n_t}")

    max_h = max(f.shape[0] for f in frames_np)
    max_w = max(f.shape[1] for f in frames_np)
    padded = []
    for f in frames_np:
        pad = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        pad[:f.shape[0], :f.shape[1]] = f
        padded.append(pad)

    brain_video_path = CACHE_FOLDER / f"{video_path.stem}_brain_RMN.mp4"
    try:
        import imageio
        imageio.mimsave(str(brain_video_path), padded, fps=2, codec='libx264', quality=7)
    except Exception:
        import imageio.v2 as imageio
        imageio.mimsave(str(brain_video_path), padded, fps=2)
    print(f"\\nVideo del cerebro guardado: {brain_video_path} ({brain_video_path.stat().st_size/1e6:.1f} MB)")

# --- 3) Lista de archivos a compartir ---
files_to_share = [out_path, report_pdf_path, report_md_path, timeline_png]
if brain_video_path: files_to_share.append(brain_video_path)
files_to_share = [f for f in files_to_share if f and Path(f).exists()]

# --- 4) Copiar a Drive ---
if drive_dir:
    for f in files_to_share:
        try:
            shutil.copy(f, drive_dir / Path(f).name)
            print(f"  -> Drive: {Path(f).name}")
        except Exception as e:
            print(f"  WARN copia {f}: {e}")
    print(f"\\nTodo en Drive: https://drive.google.com/drive/u/0/my-drive (carpeta '{DRIVE_FOLDER_NAME}')")

# --- 5) Email (opcional) ---
if GMAIL_APP_PASSWORD:
    print("\\nEnviando email a", DEST_EMAIL, "...")
    try:
        msg = EmailMessage()
        msg["Subject"] = f"RMN procesada: {video_path.stem}"
        msg["From"] = GMAIL_SENDER
        msg["To"] = DEST_EMAIL
        body = f\"\"\"Hola!

Procesamiento TRIBE v2 terminado.

Resumen:
- Duracion: {duration:.1f}s
- Palabras detectadas: {len(words)}
- Score medio atencion: {attn_score.mean():.3f}

PICOS:
{chr(10).join(f'  P{i+1} t={t}s ({attn_score[t]:.2f}) -> {segment_text(t) or "(silencio)"}' for i, t in enumerate(peaks))}

VALLES:
{chr(10).join(f'  V{i+1} t={t}s ({attn_score[t]:.2f}) -> {segment_text(t) or "(silencio)"}' for i, t in enumerate(valleys))}

Transcripcion:
{transcript[:1000]}{'...' if len(transcript) > 1000 else ''}

Archivos adjuntos chicos + carpeta en Drive con todo:
https://drive.google.com/drive/u/0/my-drive (carpeta '{DRIVE_FOLDER_NAME}')

-- Notebook TRIBE v2 en Colab
\"\"\"
        msg.set_content(body)

        total_size = 0
        for f in files_to_share:
            sz = Path(f).stat().st_size
            if total_size + sz > 22 * 1024 * 1024:
                print(f"  skip adjunto {Path(f).name} ({sz/1e6:.1f} MB) - esta en Drive")
                continue
            with open(f, 'rb') as fh:
                msg.add_attachment(fh.read(), maintype='application',
                                   subtype='octet-stream', filename=Path(f).name)
            total_size += sz
            print(f"  adjuntado: {Path(f).name} ({sz/1e6:.1f} MB)")

        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login(GMAIL_SENDER, GMAIL_APP_PASSWORD.replace(" ", ""))
            s.send_message(msg)
        print(f"OK Email enviado a {DEST_EMAIL}")
    except Exception as e:
        print("ERROR al enviar email:", e)
        traceback.print_exc()
        print("\\nTip: GMAIL_APP_PASSWORD debe ser un App Password de Gmail (16 chars).")
else:
    print("\\n(Sin GMAIL_APP_PASSWORD - no se envia email. Los archivos estan en Drive.)")

print("\\n=== LISTO ===")
""", "extra-cell"))


# ============================================================
# Assemble
# ============================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "gpuType": "A100"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = Path("/home/user/dante/out/tribe_colab_v2_mejorado.ipynb")
out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Written: {out} ({out.stat().st_size/1024:.1f} KB) - {len(cells)} cells")
