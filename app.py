import os, json, traceback, inspect
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import streamlit as st
import altair as alt
import tensorflow as tf
from tensorflow.keras import layers

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Waste Classifier (O vs R)",
    page_icon="♻️",
    layout="centered",
)

# ----------------------------
# Theme & palette
# ----------------------------
ACCENT_O = "#10b981"   # Organic (green)
ACCENT_R = "#f59e0b"   # Recycle (amber)

if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Dark Slate (recommended)"

with st.sidebar:
    st.header("Appearance")
    st.session_state.theme_choice = st.selectbox(
        "Theme mode", ["Dark Slate (recommended)", "Eco Light"],
        index=0 if st.session_state.theme_choice.startswith("Dark") else 1,
        help="Pick a theme with good text contrast."
    )

if st.session_state.theme_choice.startswith("Dark"):
    COLORS = {
        "bg": "#0f172a",
        "text": "#e2e8f0",
        "head": "#f8fafc",
        "muted": "#cbd5e1",
        "card": "#111827",
        "border": "#1f2937",
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
      .section-title {{ font-weight:800; font-size:1.1rem; margin: .1rem 0 .6rem; }}
      .hr-soft {{ height:1px; background: var(--border); margin: 1rem 0; border:0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("♻️ Waste Classifier — Organic vs Recycle")
st.caption("Upload an image **or** take a photo with the **camera** → the system predicts **O**/**R** with probabilities.")

# ----------------------------
# Paths & constants
# ----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "waste_classifier_model.keras")
LABELS_PATH = os.getenv("LABELS_PATH", "labels.json")
IMG_SIZE: Tuple[int, int] = (224, 224)

# ----------------------------
# Custom layer (ECA) for model load
# ----------------------------
try:
    from keras.saving import register_keras_serializable  # Keras 3
except Exception:
    from tensorflow.keras.utils import register_keras_serializable  # TF/Keras 2

@register_keras_serializable(package="Custom", name="ECALayer")
class ECALayer(layers.Layer):
    def __init__(self, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b

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
        y = self.conv1d(y)
        y = tf.nn.sigmoid(y)
        y = tf.squeeze(y, axis=-1)                          # (B, C)
        y = tf.reshape(y, (-1, 1, 1, tf.shape(y)[-1]))      # (B,1,1,C)
        return x * y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "b": self.b})
        return cfg

# ----------------------------
# Utilities
# ----------------------------
def _supports_kw(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False

def show_image_stretch(img, **kwargs):
    """Prefer width='stretch' when available; fallback to use_container_width=True for older Streamlit."""
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
            f"❌ Model file not found at '{model_path}'. Make sure the .keras file is present."
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

def _detect_internal_rescaling(m: tf.keras.Model) -> bool:
    """True if the model contains a tf.keras.layers.Rescaling layer (e.g., 1./255)."""
    try:
        for lyr in m.layers:
            if isinstance(lyr, tf.keras.layers.InputLayer):
                continue
            if isinstance(lyr, tf.keras.Model):
                if _detect_internal_rescaling(lyr):
                    return True
            if isinstance(lyr, tf.keras.layers.Rescaling):
                return True
        return False
    except Exception:
        return False

HAS_INTERNAL_RESCALE = False

def preprocess(im: Image.Image, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize + optional /255 scaling only if the model does NOT include Rescaling."""
    im = _fix_exif(im).convert("RGB").resize(size)
    x = np.asarray(im, dtype=np.float32)
    if not HAS_INTERNAL_RESCALE:
        x = x / 255.0
    return x[None, ...]  # (1,H,W,3)

def predict_one(model, x: np.ndarray) -> np.ndarray:
    return model.predict(x, verbose=0)[0]  # softmax [p_O, p_R]

def prob_chart(data, width=460, height=340):
    """Altair v5-safe layered bar + text; config applied on the layer object."""
    domain = ["O", "R"]
    colors = [ACCENT_O, ACCENT_R]
    base_data = alt.Data(values=data)

    bars = (
        alt.Chart(base_data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("label:N", title="Class", sort=domain, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("prob:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=colors), legend=None),
            tooltip=[alt.Tooltip("label:N"), alt.Tooltip("prob:Q", format=".3f")],
        )
        .properties(width=width, height=height, title="Prediction Probabilities")
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
    with st.expander("Error details (debug)"):
        st.code("".join(traceback.format_exception(err)), language="python")

# ----------------------------
# Sidebar: model settings
# ----------------------------
with st.sidebar:
    st.header("Settings")
    thr = st.slider("Decision threshold for **R** (Recycle)", 0.0, 1.0, 0.50, 0.01)
    st.caption("If Prob(R) ≥ threshold → predict **R**, otherwise **O**.")
    show_probs = st.toggle("Show probability chart", value=True)
    st.markdown('<div class="hr-soft"></div>', unsafe_allow_html=True)
    st.caption("Camera tip: it only activates after you press **Enable camera**.")

# ----------------------------
# Load resources
# ----------------------------
with st.spinner("Loading model..."):
    try:
        (model, n_params) = load_model(MODEL_PATH)
    except Exception as e:
        st.error(str(e))
        show_error_box(e)
        st.stop()

HAS_INTERNAL_RESCALE = _detect_internal_rescaling(model)

labels = load_labels(LABELS_PATH)
IDX2CLASS = labels["idx_to_class"]
CLASS2IDX = {k: int(v) for k, v in labels["class_to_idx"].items()}

if set(IDX2CLASS.values()) != {"O", "R"}:
    st.warning("Label set is not exactly {'O','R'}. Using mapping from labels.json. Ensure model output order matches.")

# ----------------------------
# INPUT (single-column)
# ----------------------------
st.markdown('<div class="wc-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">1) Image Input</div>', unsafe_allow_html=True)

mode = st.radio(
    "Pick image source:",
    ["Upload", "Camera"],
    horizontal=True,
)

uploaded_file = None
camera_image = None

if mode == "Upload":
    uploaded_file = st.file_uploader("Upload image (*.jpg, *.jpeg, *.png)", type=["jpg", "jpeg", "png"])
else:
    if "cam_enabled" not in st.session_state:
        st.session_state.cam_enabled = False

    col1, col2 = st.columns([1,1])
    with col1:
        if not st.session_state.cam_enabled:
            if st.button("📷 Enable camera"):
                st.session_state.cam_enabled = True
                st.rerun()
        else:
            if st.button("✖️ Disable camera"):
                st.session_state.cam_enabled = False
                st.rerun()
    with col2:
        st.caption("Browser permission is required for camera.")

    if st.session_state.cam_enabled:
        camera_image = st.camera_input("Take a photo")
        st.caption("If the camera is not shown, check browser/device permissions.")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# RESULTS (below input)
# ----------------------------
st.markdown('<div class="wc-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">2) Prediction</div>', unsafe_allow_html=True)
st.caption("Results & probabilities appear after you select or capture an image.")

selected_image: Optional[Image.Image] = None
source = None

try:
    if mode == "Camera" and camera_image is not None:
        selected_image = Image.open(camera_image)
        source = "camera"
    elif mode == "Upload" and uploaded_file is not None:
        selected_image = Image.open(uploaded_file)
        source = "upload"
except UnidentifiedImageError as e:
    st.error("Unsupported or corrupt image. Please try another file.")
    show_error_box(e)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if selected_image is not None:
    try:
        x = preprocess(selected_image, IMG_SIZE)
        prob = predict_one(model, x)  # [p_O, p_R]
        if len(prob) < 2:
            raise RuntimeError("Model output does not have 2 classes. Expected order (O, R).")
        p_O, p_R = float(prob[0]), float(prob[1])
        pred = "R" if p_R >= thr else "O"
        conf = p_R if pred == "R" else p_O

        badge_html = f'<span class="wc-badge wc-badge-{"r" if pred=="R" else "o"}">{pred}</span>'

        show_image_stretch(_fix_exif(selected_image), caption=f"Source: {source}")

        st.markdown(f"### Prediction: {badge_html}", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {conf:.2f}")

        if show_probs:
            show_altair_chart_stretch(
                prob_chart([{"label": "O", "prob": p_O}, {"label": "R", "prob": p_R}])
            )

        with st.expander("Numeric details"):
            st.write(
                {
                    "p(O)": round(p_O, 4),
                    "p(R)": round(p_R, 4),
                    "threshold_R": round(thr, 2),
                    "decision": pred,
                    "has_internal_rescaling": HAS_INTERNAL_RESCALE,
                }
            )
    except Exception as e:
        st.error("An error occurred while processing the image.")
        show_error_box(e)
else:
    st.info("Upload an image or enable the camera to start a prediction.")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
with st.expander("ℹ️ About & Technical Notes"):
    st.markdown(
        f"""
        - **Model**: Custom Keras (`.keras`) with Depthwise-Separable blocks + ECA.
        - **Preprocessing**: Auto-detect internal **Rescaling(1./255)** — app adapts to avoid double-normalization.
        - **Labels**: `labels.json` with `idx_to_class` & `class_to_idx` (default `{{0:'O', 1:'R'}}`).
        - **Parameters**: ~{('{:,}'.format(n_params)) if n_params else '—'} trainable params.
        - **Decision rule**: threshold on class **R** (Recycle) for easy field calibration.
        - **Camera**: Only activates after pressing **Enable camera**.
        """
    )
