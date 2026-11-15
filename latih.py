"""Latih dan evaluasi klasifikasi sampah organik vs anorganik (SVM).

Contoh:
    python latih.py --data DATASET/TRAIN --maks 1000
"""
import argparse
import os

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from fitur import N_WARNA, NAMA_KELAS, muat_dataset


def buat_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def hitung(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return {
        "akurasi": accuracy_score(yte, pred),
        "presisi": precision_score(yte, pred, average="macro"),
        "recall": recall_score(yte, pred, average="macro"),
        "f1": f1_score(yte, pred, average="macro"),
    }


def simpan_salah(paths, y_asli, y_pred, file="salah_klasifikasi.png", maks=8):
    idx = np.where(y_asli != y_pred)[0][:maks]
    if len(idx) == 0:
        print("Tidak ada salah klasifikasi.")
        return
    kolom = 4
    baris = int(np.ceil(len(idx) / kolom))
    fig, ax = plt.subplots(baris, kolom, figsize=(3 * kolom, 3.2 * baris))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for a, i in zip(ax, idx):
        img = cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB)
        a.imshow(img)
        a.set_title(
            f"Asli: {NAMA_KELAS[y_asli[i]]}\nPrediksi: {NAMA_KELAS[y_pred[i]]}",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(file, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder berisi subfolder O dan R")
    ap.add_argument("--maks", type=int, default=1000, help="maks citra per kelas")
    ap.add_argument("--ulang", action="store_true", help="hitung ulang fitur")
    args = ap.parse_args()

    # 1. Fitur (disimpan ke cache agar tidak dihitung ulang)
    cache = "fitur_cache.npz"
    if os.path.exists(cache) and not args.ulang:
        d = np.load(cache, allow_pickle=True)
        X, y, paths = d["X"], d["y"], d["paths"]
        print(f"Fitur dimuat dari cache: {X.shape}")
    else:
        print("Ekstraksi fitur...")
        X, y, paths = muat_dataset(args.data, args.maks)
        np.savez(cache, X=X, y=y, paths=paths)
        print(f"Selesai: {X.shape}")

    # 2. Bagi data 80:20
    idx = np.arange(len(y))
    i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    Xtr, Xte, ytr, yte = X[i_tr], X[i_te], y[i_tr], y[i_te]
    paths_te = paths[i_te]
    print(f"Latih: {len(i_tr)} | Uji: {len(i_te)}")

    baris = []

    # 3. Skenario fitur (SVM RBF default)
    skenario = {
        "Warna saja": slice(0, N_WARNA),
        "Tekstur saja": slice(N_WARNA, None),
        "Warna + Tekstur": slice(0, None),
    }
    for nama, sl in skenario.items():
        m = buat_pipeline(SVC(kernel="rbf"))
        r = hitung(m, Xtr[:, sl], ytr, Xte[:, sl], yte)
        baris.append({"Percobaan": f"Fitur: {nama} (SVM RBF)", **r})

    # 4. Tuning SVM pada fitur gabungan: linear vs RBF
    grid = {
        "linear": {"model__kernel": ["linear"], "model__C": [0.1, 1, 10]},
        "rbf": {
            "model__kernel": ["rbf"],
            "model__C": [1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.001],
        },
    }
    terbaik, skor_terbaik = None, -1
    for kernel, pg in grid.items():
        gs = GridSearchCV(
            buat_pipeline(SVC()), pg, cv=5, scoring="accuracy", n_jobs=-1
        )
        gs.fit(Xtr, ytr)
        pred = gs.predict(Xte)
        akurasi = accuracy_score(yte, pred)
        print(f"SVM {kernel}: param terbaik {gs.best_params_} | uji {akurasi:.4f}")
        baris.append(
            {
                "Percobaan": f"SVM {kernel} + tuning",
                "akurasi": akurasi,
                "presisi": precision_score(yte, pred, average="macro"),
                "recall": recall_score(yte, pred, average="macro"),
                "f1": f1_score(yte, pred, average="macro"),
            }
        )
        if akurasi > skor_terbaik:
            skor_terbaik, terbaik = akurasi, gs.best_estimator_

    # 5. Pembanding: KNN dan Random Forest
    r = hitung(buat_pipeline(KNeighborsClassifier(n_neighbors=5)), Xtr, ytr, Xte, yte)
    baris.append({"Percobaan": "Pembanding: K-NN (k=5)", **r})
    r = hitung(
        buat_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)),
        Xtr, ytr, Xte, yte,
    )
    baris.append({"Percobaan": "Pembanding: Random Forest", **r})

    # 6. Model akhir: SVM terbaik + probabilitas (untuk demo)
    final = clone(terbaik)
    final.set_params(model__probability=True, model__random_state=42)
    final.fit(Xtr, ytr)
    pred = final.predict(Xte)

    print("\n=== Laporan klasifikasi (model akhir) ===")
    print(classification_report(yte, pred, target_names=NAMA_KELAS, digits=4))

    cm = confusion_matrix(yte, pred)
    ConfusionMatrixDisplay(cm, display_labels=NAMA_KELAS).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    simpan_salah(paths_te, yte, pred)

    # 7. Tabel perbandingan
    df = pd.DataFrame(baris).round(4)
    df.to_csv("hasil_perbandingan.csv", index=False)
    print("\n=== Tabel perbandingan ===")
    print(df.to_string(index=False))

    joblib.dump({"model": final, "kelas": NAMA_KELAS}, "model_svm.joblib")
    print("\nTersimpan: model_svm.joblib, confusion_matrix.png, salah_klasifikasi.png, hasil_perbandingan.csv")


if __name__ == "__main__":
    main()
