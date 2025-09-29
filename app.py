# app.py
# Waste Classification (Organic vs Recycle) — Streamlit Inference
# UI: single-column (hasil di bawah input), kamera hanya aktif saat diminta,
# threshold kelas R, grafik Altair, tema gelap/terang, kompatibel versi Streamlit lama/baru.
# Perbaikan: Deteksi otomatis Rescaling(1./255) di dalam model untuk mencegah double-normalization.

import os, json, traceback, inspect
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import streamlit as st
import altair as alt
import tensorflow as tf
from tensorflow.keras import layers

# Page config
st.set_page_config(
    page_title="Waste Classifier (O vs R)",
    page_icon="♻️",
    layout="centered",
)

# Theme & Palette
ACCENT_O = "#10b981"   # Organic (green)
ACCENT_R = "#f59e0b"   # Recycle (amber)

# Tema in-app agar kontras aman di dark/light
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Dark Slate (disarankan)"

with st.sidebar:
    st.header("Tampilan")
    st.session_state.theme_choice = st.selectbox(
        "Mode tampilan", ["Dark Slate (disarankan)", "Eco Light"],
        index=0 if st.session_state.theme_choice.startswith("Dark") else 1,
        help="Pilih mode tampilan agar kontras teks optimal."
    )

if st.session_state.theme_choice.startswith("Dark"):
    COLORS = {
        "bg": "#0f172a",      # slate-900
        "text": "#e2e8f0",    # slate-200
        "head": "#f8fafc",    # slate-50
        "muted": "#cbd5e1",   # slate-300
        "card": "#111827",    # gray-900
        "border": "#1f2937",  # gray-800
        "axis": "#e2e8f0",
        "grid": "#334155",
    }
else:
    COLORS = {
        "bg": "#FAF8F1",
        "text": "#334155",
        "head": "#0f172a",
        "muted": "#475569",
        "card": "#ffffff",
        "border": "#e5e7eb",
        "axis": "#334155",
        "grid": "#94a3b8",
    }

st.markdown(
    f"""
    <style>
      :root{{
        --bg: {COLORS["bg"]};
        --text: {COLORS["text"]};
        --head: {COLORS["head"]};
        --muted: {COLORS["muted"]};
        --card: {COLORS["card"]};
        --border: {COLORS["border"]};
        --accentO: {ACCENT_O};
        --accentR: {ACCENT_R};
      }}
      html, body, .stApp {{ background: var(--bg); color: var(--text); }}
      h1,h2,h3,h4,h5,h6 {{ color: var(--head) !important; }}
      p, span, label, .stMarkdown, .stCaption, .stText {{ color: var(--text) !important; }}
      /* Sidebar */
      [data-testid="stSidebar"] {{ background: #0b1220; color: var(--text); }}
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: var(--head) !important;
      }}
      /* Cards */
      .wc-card {{
        background: var(--card); padding: 1rem 1.1rem; border-radius: 14px;
        box-shadow: 0 1px 8px rgba(0,0,0,0.12); border: 1px solid var(--border);
      }}
      .wc-badge {{
        display:inline-block; padding: .30rem .70rem; border-radius: 9999px; font-weight: 700; font-size: .95rem;
      }}
      .wc-badge-o {{ background: {ACCENT_O}22; color: {ACCENT_O}; border:1px solid {ACCENT_O}66; }}
      .wc-badge-r {{ background: {ACCENT_R}22; color: {ACCENT_R}; border:1px solid {ACCENT_R}66; }}
      .wc-note {{ color:var(--muted); font-size:0.88rem; }}
      .stButton>button {{ border-radius: 12px; padding: 0.5rem 1rem; }}
      .stSlider>div>div>div>div {{ background: linear-gradient(90deg, {ACCENT_O}, {ACCENT_R}); }}
      .section-title {{ font-weight:800; font-size:1.1rem; margin: .1rem 0 .6rem; }}
      .hr-soft {{ height:1px; background: var(--border); margin: 1rem 0; border:0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("♻️ Waste Classifier — Organic vs Recycle")
st.caption("Unggah gambar **atau** ambil foto dari **kamera** (mobile) → Sistem memprediksi **O**/**R** disertai probabilitas.")

# Paths & constants
MODEL_PATH = os.getenv("MODEL_PATH", "waste_classifier_model.keras")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.json")
IMG_SIZE: Tuple[int, int] = (224, 224)

# C. Custom Layer (ECALayer)
try:
    from keras.saving import register_keras_serializable  # Keras 3
except Exception:
    from tensorflow.keras.utils import register_keras_serializable  # TF/Keras 2

@register_keras_serializable(package="Custom", name="ECALayer")
class ECALayer(layers.Layer):
    """Efficient Channel Attention (Wang et al., CVPR 2020)."""
    def __init__(self, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b
        self.conv1d = None

    def build(self, input_shape):
        import numpy as _np
        channels = int(input_shape[-1])
        t = int(abs((_np.log2(max(1, channels)) + self.b) / self.gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        self.conv1d = layers.Conv1D(1, kernel_size=k, padding="same", use_bias=False)
        super().build(input_shape)

    def call(self, x):
        y = tf.reduce_mean(x, axis=[1, 2], keepdims=False)  # (B, C)
        y = tf.expand_dims(y, axis=-1)                      # (B, C, 1)
        y = self.conv1d(y)                                  # (B, C, 1)
        y = tf.nn.sigmoid(y)
        y = tf.squeeze(y, axis=-1)                          # (B, C)
        y = tf.reshape(y, (-1, 1, 1, tf.shape(y)[-1]))      # (B,1,1,C)
        return x * y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "b": self.b})
        return cfg

# Utilities
def _supports_kw(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False

def show_image_stretch(img, **kwargs):
    try:
        if _supports_kw(st.image, "width"):
            st.image(img, width="stretch", **kwargs)
        else:
            st.image(img, use_container_width=True, **kwargs)
    except TypeError:
        st.image(img, use_container_width=True, **kwargs)

def show_altair_chart_stretch(chart):
    try:
        if _supports_kw(st.altair_chart, "width"):
            st.altair_chart(chart, width="stretch")
        else:
            st.altair_chart(chart, use_container_width=True)
    except TypeError:
        st.altair_chart(chart, use_container_width=True)

@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Model tidak ditemukan di '{model_path}'. Pastikan file .keras sudah tersedia di repo/Cloud."
        )
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"ECALayer": ECALayer},
        compile=False
    )
    try:
        params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    except Exception:
        params = None
    return model, params

@st.cache_data(show_spinner=False)
def load_labels(label_path: str) -> Dict[str, Any]:
    default = {"idx_to_class": {0: "O", 1: "R"}, "class_to_idx": {"O": 0, "R": 1}}
    if not os.path.exists(label_path):
        return default
    with open(label_path, "r") as f:
        raw = json.load(f)
    idx_to_class = {int(k): v for k, v in raw.get("idx_to_class", {}).items()}
    class_to_idx = raw.get("class_to_idx") or {v: k for k, v in idx_to_class.items()}
    return {"idx_to_class": idx_to_class or default["idx_to_class"],
            "class_to_idx": class_to_idx or default["class_to_idx"]}

def _fix_exif(im: Image.Image) -> Image.Image:
    try:
        return ImageOps.exif_transpose(im)
    except Exception:
        return im

# --- DETEKSI RESCALING DI DALAM MODEL
def _detect_internal_rescaling(m: tf.keras.Model) -> bool:
    try:
        for lyr in m.layers:
            if isinstance(lyr, tf.keras.layers.InputLayer):
                continue
            # nested model
            if isinstance(lyr, tf.keras.Model):
                if _detect_internal_rescaling(lyr):
                    return True
            if isinstance(lyr, tf.keras.layers.Rescaling):
                return True
        return False
    except Exception:
        # jika gagal deteksi, default False
        return False

# Placeholder; akan di-set setelah model diload
HAS_INTERNAL_RESCALE = False

def preprocess(im: Image.Image, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize + (opsional) scaling /255"""
    im = _fix_exif(im).convert("RGB").resize(size)
    x = np.asarray(im, dtype=np.float32)
    if not HAS_INTERNAL_RESCALE:
        x = x / 255.0
    return x[None, ...]  # (1,H,W,3)

def predict_one(model, x: np.ndarray) -> np.ndarray:
    prob = model.predict(x, verbose=0)[0]  # softmax [p_O, p_R]
    return prob

def prob_chart(data, width=460, height=340):
    """Altair v5-safe layering"""
    domain = ["O", "R"]
    colors = [ACCENT_O, ACCENT_R]
    base_data = alt.Data(values=data)

    bars = (
        alt.Chart(base_data)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title="Kelas", sort=domain, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("prob:Q", title="Probabilitas", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=colors), legend=None),
            tooltip=[alt.Tooltip("label:N"), alt.Tooltip("prob:Q", format=".3f")],
        )
        .properties(width=width, height=height, title="Probabilitas Prediksi")
    )

    text = (
        alt.Chart(base_data)
        .mark_text(dy=-8, fontWeight="bold", color=COLORS["head"])
        .encode(x="label:N", y="prob:Q", text=alt.Text("prob:Q", format=".2f"))
    )

    layered = alt.layer(bars, text).configure_axis(
        labelColor=COLORS["axis"], titleColor=COLORS["axis"], gridColor=COLORS["grid"]
    ).configure_title(color=COLORS["head"])

    return layered

def show_error_box(err: Exception):
    with st.expander("Rincian error (untuk debugging)"):
        st.code("".join(traceback.format_exception(err)), language="python")

# Sidebar: Pengaturan model
with st.sidebar:
    st.header("Pengaturan")
    thr = st.slider("Decision threshold untuk **R** (Recycle)", 0.0, 1.0, 0.50, 0.01)
    st.caption("Jika Prob(R) ≥ threshold → Prediksi **R**, selain itu **O**.")
    show_probs = st.toggle("Tampilkan grafik probabilitas", value=True)
    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
    st.caption("Tips kamera: kamera hanya aktif saat kamu klik *Aktifkan kamera*.")

# Load resources
with st.spinner("Memuat model..."):
    try:
        (model, n_params) = load_model(MODEL_PATH)
    except Exception as e:
        st.error(str(e))
        show_error_box(e)
        st.stop()

# Deteksi apakah model sudah punya Rescaling internal
HAS_INTERNAL_RESCALE = _detect_internal_rescaling(model)

labels = load_labels(LABELS_PATH)
IDX2CLASS = labels["idx_to_class"]
CLASS2IDX = {k: int(v) for k, v in labels["class_to_idx"].items()}

if set(IDX2CLASS.values()) != {"O", "R"}:
    st.warning("Labels tidak persis {'O','R'}. Menggunakan mapping dari 'labels.json'. Pastikan urutan output model sesuai.")

# INPUT (single-column)
st.markdown('<div class="wc-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">1) Input Gambar</div>', unsafe_allow_html=True)

mode = st.radio(
    "Pilih sumber gambar:",
    ["Upload", "Kamera"],
    horizontal=True,
)

uploaded_file = None
camera_image = None

if mode == "Upload":
    uploaded_file = st.file_uploader("Upload gambar (*.jpg, *.jpeg, *.png)", type=["jpg", "jpeg", "png"])
else:
    # Kamera hanya aktif saat user menekan tombol
    if "cam_enabled" not in st.session_state:
        st.session_state.cam_enabled = False

    col1, col2 = st.columns([1,1])
    with col1:
        if not st.session_state.cam_enabled:
            if st.button("📷 Aktifkan kamera"):
                st.session_state.cam_enabled = True
                st.rerun()
        else:
            if st.button("✖️ Matikan kamera"):
                st.session_state.cam_enabled = False
                st.rerun()
    with col2:
        st.caption("Kamera membutuhkan izin browser.")

    if st.session_state.cam_enabled:
        camera_image = st.camera_input("Ambil foto dari kamera")
        st.caption("Jika kamera tidak tampil, cek izin browser & perangkat.")

st.markdown('</div>', unsafe_allow_html=True)

# HASIL PREDIKSI (di bawah input)
st.markdown('<div class="wc-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">2) Hasil Prediksi</div>', unsafe_allow_html=True)
st.caption("Prediksi & probabilitas akan muncul setelah gambar dipilih/dipotret.")

selected_image: Optional[Image.Image] = None
source = None

try:
    if mode == "Kamera" and camera_image is not None:
        selected_image = Image.open(camera_image)
        source = "camera"
    elif mode == "Upload" and uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        source = "upload"
except UnidentifiedImageError as e:
    st.error("Format gambar tidak dikenali. Coba file lain.")
    show_error_box(e)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if selected_image is not None:
    try:
        x = preprocess(selected_image, IMG_SIZE)
        prob = predict_one(model, x)  # [p_O, p_R]
        if len(prob) < 2:
            raise RuntimeError("Output model tidak berukuran 2 kelas. Pastikan model sesuai (O,R).")
        p_O, p_R = float(prob[0]), float(prob[1])
        pred = "R" if p_R >= thr else "O"
        conf = p_R if pred == "R" else p_O

        badge_html = f'<span class="wc-badge wc-badge-{"r" if pred=="R" else "o"}">{pred}</span>'

        # Tampilkan gambar
        show_image_stretch(_fix_exif(selected_image), caption=f"Sumber: {source}")

        # Tampilkan hasil
        st.markdown(f"### Prediksi: {badge_html}", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {conf:.2f}")

        if show_probs:
            show_altair_chart_stretch(
                prob_chart([{"label": "O", "prob": p_O}, {"label": "R", "prob": p_R}])
            )

        with st.expander("Detail angka"):
            st.write(
                {
                    "p(O)": round(p_O, 4),
                    "p(R)": round(p_R, 4),
                    "threshold_R": round(thr, 2),
                    "keputusan": pred,
                    "has_internal_rescaling": HAS_INTERNAL_RESCALE,
                }
            )
    except Exception as e:
        st.error("Terjadi kesalahan saat memproses gambar.")
        show_error_box(e)
else:
    st.info("Unggah gambar atau aktifkan kamera lalu ambil foto untuk memulai prediksi.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
with st.expander("ℹ️ Tentang & Catatan Teknis"):
    st.markdown(
        f"""
        - **Model**: Keras (`.keras`) kustom dengan blok Depthwise-Separable + ECA.
        - **Preprocessing**: Deteksi otomatis **Rescaling(1./255)** di dalam model → app menyesuaikan agar tidak double-normalization.
        - **Labels**: `labels.json` dengan `idx_to_class` & `class_to_idx`. Default fallback `{{0:'O', 1:'R'}}`.
        - **Parameter**: ~{('{:,}'.format(n_params)) if n_params else '—'} trainable params.
        - **Keputusan**: threshold pada kelas **R** (Recycle) agar mudah dikalibrasi.
        - **Perangkat**: Kamera hanya aktif saat tombol **Aktifkan kamera** ditekan.
        """
    )
