"""Demo Streamlit: unggah foto sampah, tampil prediksi Organik / Anorganik.

Jalankan: streamlit run app.py
"""
import cv2
import joblib
import numpy as np
import streamlit as st

from fitur import ekstrak_fitur

st.set_page_config(page_title="Klasifikasi Sampah", page_icon="♻️")


@st.cache_resource
def muat_model():
    d = joblib.load("model_svm.joblib")
    return d["model"], d["kelas"]


st.title("Klasifikasi Sampah Organik dan Anorganik")
st.caption("Fitur warna (HSV) + tekstur (GLCM) dengan klasifikasi SVM")

try:
    model, kelas = muat_model()
except FileNotFoundError:
    st.error("model_svm.joblib belum ada. Jalankan latih.py dulu.")
    st.stop()

file = st.file_uploader("Unggah foto sampah", type=["jpg", "jpeg", "png"])

if file is not None:
    data = np.frombuffer(file.read(), np.uint8)
    citra = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if citra is None:
        st.error("File gambar tidak bisa dibaca.")
        st.stop()

    st.image(cv2.cvtColor(citra, cv2.COLOR_BGR2RGB), caption="Citra input", use_container_width=True)

    fitur = ekstrak_fitur(citra).reshape(1, -1)
    prob = model.predict_proba(fitur)[0]
    idx = int(np.argmax(prob))

    st.subheader(f"Hasil: {kelas[idx]}")
    st.write(f"Keyakinan: **{prob[idx] * 100:.1f}%**")
    for nama, p in zip(kelas, prob):
        st.progress(float(p), text=f"{nama}: {p * 100:.1f}%")

    with st.expander("Lihat vektor fitur"):
        st.write(f"Jumlah fitur: {fitur.shape[1]}")
        st.dataframe(fitur)
