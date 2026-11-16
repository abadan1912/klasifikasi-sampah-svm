"""Demo Streamlit: Unggah & Scan Koleksi Sampah (Maks. 10 Foto).

Bisa dijalankan dengan:
    streamlit run app.py
atau cukup:
    python app.py
"""
import sys
from pathlib import Path

# Path direktori file app.py agar model selalu ditemukan dimanapun dijalankan
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_svm.joblib"

# Memungkinkan aplikasi dijalankan langsung via 'python app.py'
if __name__ == "__main__":
    try:
        from streamlit import runtime
        if not runtime.exists():
            from streamlit.web import cli as stcli
            sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
            sys.exit(stcli.main())
    except (ImportError, AttributeError):
        pass

import cv2
import joblib
import numpy as np
import streamlit as st

from fitur import ekstrak_fitur

# Konfigurasi Halaman
st.set_page_config(
    page_title="EcoScan — Klasifikasi Sampah",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Tema putih-biru dengan tipografi sans-serif dan aksen serif italic.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,600&family=Playfair+Display:ital,wght@1,500;1,600;1,700&display=swap');

    :root {
        color-scheme: light;
        --ink: #172033;
        --muted: #718096;
        --blue: #2878E8;
        --line: #E5EAF2;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--ink);
        background: #F7F9FC !important;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 252, 0.9) !important;
    }

    .main .block-container {
        padding: 2.5rem 2rem 4rem;
        max-width: 1120px;
    }

    .hero-container {
        text-align: center;
        padding: 3.5rem 2rem 3rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #F5F9FF 100%);
        border: 1px solid #E8EEF7;
        border-radius: 28px;
        box-shadow: 0 12px 40px rgba(32, 75, 135, 0.045);
        margin-bottom: 2.5rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.4rem 0.9rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #286CC8;
        background: #F0F6FF;
        border: 1px solid #DCEAFF;
        border-radius: 9999px;
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4rem);
        font-weight: 700;
        color: #172033;
        letter-spacing: -0.055em;
        margin-bottom: 0.75rem;
        line-height: 1.08;
    }

    .hero-title .italic-accent {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-weight: 500;
        color: var(--blue);
    }

    .hero-subtitle {
        font-size: 1rem;
        color: var(--muted);
        font-style: italic;
        font-weight: 400;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.7;
    }

    .content-card {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 1.35rem 1.5rem 1.1rem;
        box-shadow: 0 5px 20px rgba(20, 42, 74, 0.025);
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.3rem;
    }

    .card-subtitle {
        font-size: 0.84rem;
        color: var(--muted);
        margin-bottom: 0;
        line-height: 1.6;
    }

    [data-testid="stFileUploader"] {
        margin-top: 0.6rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 150px;
        background: #FFFFFF !important;
        border: 1.5px dashed #B8D0F1 !important;
        border-radius: 20px !important;
        transition: background 0.2s ease, border-color 0.2s ease;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background: #F8FBFF !important;
        border-color: #78A8EC !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        border: 1px solid #DCE5F1 !important;
        border-radius: 10px !important;
        color: #26364D !important;
        background: #FFFFFF !important;
    }

    div.stButton > button {
        min-height: 2.8rem;
        background: var(--blue);
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        border: 1px solid var(--blue);
        border-radius: 12px;
        padding: 0.55rem 1.35rem;
        box-shadow: 0 5px 14px rgba(40, 120, 232, 0.18);
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    }

    div.stButton > button:hover {
        background: #1768D4;
        border-color: #1768D4;
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(40, 120, 232, 0.24);
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-color: var(--line) !important;
        border-radius: 18px !important;
        box-shadow: 0 5px 20px rgba(20, 42, 74, 0.025);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-style: italic;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    .section-heading {
        margin: 2.2rem 0 1rem;
        color: var(--ink);
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.035em;
    }

    .badge-organik,
    .badge-anorganik {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .badge-organik {
        color: #167052;
        background: #EAF7F1;
        border: 1px solid #D4EFE2;
    }

    .badge-anorganik {
        color: #286CC8;
        background: #EEF5FF;
        border: 1px solid #DCEAFF;
    }

    .empty-state {
        text-align: center;
        padding: 2.6rem 1.5rem;
        color: var(--muted);
        background: #FFFFFF;
        border-radius: 20px;
        border: 1px solid var(--line);
        box-shadow: 0 5px 20px rgba(20, 42, 74, 0.025);
    }

    .empty-icon {
        display: grid;
        place-items: center;
        width: 52px;
        height: 52px;
        margin: 0 auto 0.9rem;
        border-radius: 16px;
        background: #EEF5FF;
        color: var(--blue);
        font-size: 1.55rem;
    }

    .empty-title {
        color: var(--ink);
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .empty-copy {
        color: var(--muted);
        font-size: 0.86rem;
        font-style: italic;
    }

    @media (max-width: 640px) {
        .main .block-container {
            padding: 1.5rem 1rem 3rem;
        }

        .hero-container {
            padding: 2.5rem 1.25rem;
            border-radius: 22px;
        }

        .hero-subtitle {
            font-size: 0.92rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def muat_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"File model tidak ditemukan di: {MODEL_PATH}")
    d = joblib.load(MODEL_PATH)
    return d["model"], d["kelas"]


# Muat model SVM
try:
    model, kelas = muat_model()
except FileNotFoundError as e:
    st.error(f"{e}. Silakan jalankan `python latih.py` terlebih dahulu.")
    st.stop()


# Header / Hero Section
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">Smart Waste Scanner System</div>
        <div class="hero-title">
            Eco<span class="italic-accent">Scan</span>
        </div>
        <div class="hero-subtitle">
            Kenali sampah organik dan anorganik dari foto, dengan cara yang sederhana.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Card Pengumpulan / Scanner
st.markdown(
    """
    <div class="content-card">
        <div class="card-title">Mulai pemindaian</div>
        <div class="card-subtitle">Pilih atau tarik hingga 10 foto ke area unggah. Mendukung JPG, JPEG, dan PNG.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Unggah foto sampah",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    # Batasi maksimal 10 foto
    if len(uploaded_files) > 10:
        st.warning("⚠️ Anda memilih lebih dari 10 foto. Hanya 10 foto pertama yang akan diproses ke dalam scanner.")
        file_list = uploaded_files[:10]
    else:
        file_list = uploaded_files

    st.markdown(f"**Terkumpul:** *{len(file_list)} foto siap dipindai.*")

    # Tombol Aksi Scanner
    col_btn, _ = st.columns([2, 5])
    with col_btn:
        mulai_scan = st.button("🔍 Pindai Semua Sampah", use_container_width=True)

    if mulai_scan:
        hasil_scan = []
        progress_bar = st.progress(0, text="Sedang memindai citra sampah...")

        for i, f in enumerate(file_list):
            data = np.frombuffer(f.read(), np.uint8)
            f.seek(0)  # Reset pointer file agar bisa dibaca ulang jika diperlukan
            citra = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if citra is None:
                hasil_scan.append({
                    "nama": f.name,
                    "error": True,
                    "citra_rgb": None
                })
            else:
                citra_rgb = cv2.cvtColor(citra, cv2.COLOR_BGR2RGB)
                fitur = ekstrak_fitur(citra).reshape(1, -1)
                prob = model.predict_proba(fitur)[0]
                idx = int(np.argmax(prob))

                hasil_scan.append({
                    "nama": f.name,
                    "error": False,
                    "citra_rgb": citra_rgb,
                    "prediksi": kelas[idx],
                    "keyakinan": prob[idx],
                })

            progress_bar.progress((i + 1) / len(file_list), text=f"Memproses {i + 1} dari {len(file_list)} foto...")

        progress_bar.empty()

        # Simpan ke session state agar hasil tetap tampil
        st.session_state["hasil_scan"] = hasil_scan

# Tampilkan Hasil Scan jika sudah ada
if "hasil_scan" in st.session_state and st.session_state["hasil_scan"]:
    hasil_list = st.session_state["hasil_scan"]

    total = len(hasil_list)
    total_organik = sum(1 for h in hasil_list if not h["error"] and h["prediksi"].lower() == "organik")
    total_anorganik = sum(1 for h in hasil_list if not h["error"] and h["prediksi"].lower() == "anorganik")

    st.markdown('<div class="section-heading">Ringkasan hasil</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for column, label, value in (
        (m1, "Foto dipindai", total),
        (m2, "Sampah organik", total_organik),
        (m3, "Sampah anorganik", total_anorganik),
    ):
        with column:
            with st.container(border=True):
                st.metric(label, value)

    st.markdown('<div class="section-heading">Hasil klasifikasi</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, item in enumerate(hasil_list):
        with cols[idx % 3]:
            with st.container(border=True):
                if item["error"]:
                    st.error(f"Gagal membaca file {item['nama']}")
                else:
                    st.image(item["citra_rgb"], use_container_width=True)
                    is_organik = item["prediksi"].lower() == "organik"
                    badge_class = "badge-organik" if is_organik else "badge-anorganik"
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; gap: 0.75rem;">
                            <span class="{badge_class}">{item['prediksi']}</span>
                            <span style="font-size: 0.85rem; font-weight: 600; color: #526176;">
                                {item['keyakinan'] * 100:.1f}%
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(item["nama"])
                    st.progress(float(item["keyakinan"]))
else:
    # State awal ketika belum ada file yang diunggah
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">↑</div>
            <div class="empty-title">Belum ada foto</div>
            <div class="empty-copy">Unggah 1–10 foto untuk mulai mengelompokkan sampah.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
