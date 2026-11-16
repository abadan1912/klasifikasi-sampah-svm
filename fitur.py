"""Ekstraksi fitur warna (HSV) dan tekstur (GLCM) untuk klasifikasi sampah."""
import os
import cv2
import numpy as np
from joblib import Parallel, delayed
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature.texture import graycomatrix, graycoprops

UKURAN = (128, 128)
# Struktur dataset Kaggle "Waste Classification Data": folder O (organik) dan R (anorganik/recyclable)
KELAS = {"O": 0, "R": 1}
NAMA_KELAS = ["Organik", "Anorganik"]

# 16 (H) + 8 (S) + 8 (V) bin histogram + 6 (mean & std HSV)
N_WARNA = 16 + 8 + 8 + 6
# 5 properti GLCM x 2 jarak
N_TEKSTUR = 5 * 2

PROPS_GLCM = ["contrast", "correlation", "energy", "homogeneity", "dissimilarity"]


def fitur_warna(hsv):
    h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
    hist = np.concatenate([h / h.sum(), s / s.sum(), v / v.sum()])
    mean, std = cv2.meanStdDev(hsv)
    return np.concatenate([hist, mean.flatten() / 255.0, std.flatten() / 255.0])


def fitur_tekstur(gray):
    # kuantisasi 256 -> 32 level agar GLCM cepat
    q = (gray // 8).astype(np.uint8)
    glcm = graycomatrix(
        q,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    hasil = []
    for p in PROPS_GLCM:
        nilai = graycoprops(glcm, p)  # bentuk (2 jarak, 4 sudut)
        hasil.extend(nilai.mean(axis=1))  # rata-rata antar sudut
    return np.array(hasil)


def ekstrak_fitur(citra_bgr):
    """Input: citra BGR (array OpenCV). Output: vektor fitur [warna | tekstur]."""
    citra = cv2.resize(citra_bgr, UKURAN)
    citra = cv2.GaussianBlur(citra, (3, 3), 0)
    hsv = cv2.cvtColor(citra, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(citra, cv2.COLOR_BGR2GRAY)
    return np.concatenate([fitur_warna(hsv), fitur_tekstur(gray)])


def _proses(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return ekstrak_fitur(img)


def muat_dataset(folder, maks_per_kelas=1000, n_jobs=-1, seed=42):
    """Baca folder/O dan folder/R. Return X, y, daftar path."""
    rng = np.random.RandomState(seed)
    paths, label = [], []
    for nama, kode in KELAS.items():
        sub = os.path.join(folder, nama)
        if not os.path.isdir(sub):
            raise FileNotFoundError(f"Folder tidak ditemukan: {sub}")
        files = sorted(
            f for f in os.listdir(sub) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        rng.shuffle(files)
        files = files[:maks_per_kelas]
        paths += [os.path.join(sub, f) for f in files]
        label += [kode] * len(files)

    hasil = Parallel(n_jobs=n_jobs)(delayed(_proses)(p) for p in paths)
    X, y, p_ok = [], [], []
    for fitur, lab, p in zip(hasil, label, paths):
        if fitur is not None:
            X.append(fitur)
            y.append(lab)
            p_ok.append(p)
    return np.array(X), np.array(y), np.array(p_ok)
