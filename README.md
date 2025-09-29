# ♻️ Waste Classification — Organic vs Recycle (Streamlit + Keras)

A production‑ready web app to classify waste into **Organic (O)** or **Recycle (R)**.  
UI built with **Streamlit**; inference uses a **custom Keras CNN** with **Efficient Channel Attention (ECA)**.  
Camera capture is **permission‑based** (not auto‑on). Layout is **single column**: results appear **below** inputs.

---

## 📑 Project Scope (per agreement)

- ✅ **Original ML code** (no Kaggle Codes copy‑paste); rare/custom architecture (Depthwise‑Separable CNN + **ECALayer**).
- ✅ **Google Colab** training notebook and model export to `.keras`.
- ✅ **Streamlit web app** with:
  - Image **Upload** and **Camera** (activate manually; supports mobile).
  - **Decision threshold** slider for class **R** (Recycle).
  - **Probability chart** (Altair v5‑safe).
  - Responsive, high‑contrast **Dark/Light** theme.
- ✅ **Deployment** target: Streamlit Cloud (GitHub integration).
- ✅ **Documentation**: short Chapter 3 (BAB 3) + this **README**.
- ✅ **Support** until thesis completion (non‑feature changes).

> Bahasa Indonesia is used in the UI labels and docs where helpful.

---

## 🗂 Repository Structure

```
.
├─ app.py                          # Streamlit app (single-column; camera on-demand)
├─ labels.json                     # Class mapping: {"idx_to_class": {"0":"O","1":"R"}, "class_to_idx":{"O":0,"R":1}}
├─ waste-classfication-final.ipynb # Colab notebook for training & export (.keras)
├─ waste_classifier_model.keras    # Trained model (optional: store as release asset)
├─ README.md                       # This file
├─ requirements.txt                # (optional) see Dependencies
└─ assets/                         # (optional) screenshots, demo images, logo
```

> Large model files are better released as **GitHub Release assets** or stored externally; then point `MODEL_PATH` to the URL/path.

---

## 📊 Dataset

- **Source**: [Waste Classification Data — Kaggle (techsash)](https://www.kaggle.com/datasets/techsash/waste-classification-data)  
- **Task**: Binary classification — **O (Organic)** vs **R (Recycle)**.  
- **Preprocess**: Resize to **224×224**, RGB, normalize to `[0,1]`.

Please follow the dataset license and terms.

---

## 🧠 Model Overview

- **Backbone**: Lightweight **Depthwise‑Separable CNN**.
- **Attention**: **ECA (Efficient Channel Attention)** via custom Keras Layer (serializable).
- **Input**: `224×224×3` images; **/255** normalization.
- **Output**: Softmax `[p_O, p_R]`.
- **Export**: Saved as `.keras` for robust loading in Keras 3 / TF 2.x.

**Decision rule in app**: predict **R** if `p_R ≥ threshold_R` (default `0.50`), else **O**.

---

## 📈 Example Metrics (fill with your results)

Populate with the latest results from the Colab notebook:

| Split | Accuracy | F1 (macro) | ROC‑AUC | Notes |
|------:|---------:|-----------:|--------:|------|
| Val   |  _xx.x%_ |   _xx.x%_  | _0.xxx_ | |
| Test  |  _xx.x%_ |   _xx.x%_  | _0.xxx_ | |

Consider including **per‑class metrics** and a **confusion matrix** at the chosen threshold.

---

## 🛠 Dependencies

Minimal versions (CPU‑only):

```
python >= 3.10
streamlit >= 1.32
altair >= 5.0
numpy >= 1.24
pillow >= 10.0
tensorflow >= 2.13
```

Optional `requirements.txt`:
```txt
streamlit>=1.32
altair>=5.0
numpy>=1.24
pillow>=10.0
tensorflow>=2.13
```

> Note: If you see oneDNN logs or tiny numeric diffs, you can set `TF_ENABLE_ONEDNN_OPTS=0` for reproducibility.

---

## ▶️ Run Locally

1) **Create virtual env & install**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt  # or install the packages listed above
```

2) **Place model & labels**

```
waste_classifier_model.keras   # alongside app.py
labels.json                    # see example format above
```

(Or set environment variables `MODEL_PATH` and `LABELS_PATH`.)

3) **Run**

```bash
streamlit run app.py
```

Open the **Local URL** shown by Streamlit.

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to **GitHub**.  
2. On **Streamlit Community Cloud**, create new app → pick repo/branch → set **Main file path** to `app.py`.  
3. If the model is hosted elsewhere, set **Advanced settings → Environment variables**:
   - `MODEL_PATH` → URL/path to `.keras` model
   - `LABELS_PATH` → URL/path to `labels.json`

> Large models: store as **release asset** or object storage with a direct download link.

---

## 💻 How to Use

1. Choose input: **Upload** or **Kamera** (camera only activates after you press **Aktifkan kamera**).
2. Adjust **Threshold R** from the **Sidebar** (default `0.50`).
3. Results show: predicted **O/R**, **confidence**, and **probability chart** (toggle from sidebar).

**Preprocessing**: EXIF transpose → RGB → resize to `224×224` → `/255` normalization (handled in code).

---

## 🔁 Re‑training (Colab)

Use `waste-classfication-final.ipynb`:

- Prepare data; split train/val/test.
- Build the DS‑CNN + **ECALayer** model.
- Train & evaluate; log metrics.
- Export artifacts:
  - `model.save("waste_classifier_model.keras")`
  - `labels.json` with both mappings (`idx_to_class`, `class_to_idx`).

Upload exported files next to `app.py` or point the env vars accordingly.

---

## ✅ Validation Checklist

- [ ] Threshold slider updates decisions.
- [ ] Upload & camera flows work (camera only after **Activate**).
- [ ] Model/labels are discovered via env vars or local files.
- [ ] Altair chart renders (no v5 layering/config errors).
- [ ] Works in **Dark** and **Light** themes with good contrast.

---

## 🔒 Privacy & Notes

- Camera images are processed **in-session** for inference and not persisted by the UI code.
- This is a learning/POC tool; avoid critical decisions without broader validation and monitoring.

---

## 🙏 Acknowledgements

- Dataset: **Waste Classification Data** by techsash (Kaggle).  
- Attention: **Efficient Channel Attention** — Wang et al., CVPR 2020.

---

## 📄 License

Choose a suitable license (e.g., **MIT**) or keep private for academic submission.

```text
Copyright (c) 2025
All rights reserved.
```
