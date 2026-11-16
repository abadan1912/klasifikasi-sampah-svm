@echo off
title Klasifikasi Sampah Organik dan Anorganik
cd /d "%~dp0"

echo =======================================================
echo   Aplikasi Klasifikasi Sampah (SVM + Streamlit)
echo =======================================================
echo.

REM 1. Cek apakah Python terdeteksi
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [PERINGATAN] Perintah 'python' belum terbaca di sistem PATH.
    echo Mencoba perintah 'py'...
    py --version >nul 2>nul
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Python belum terpasang atau belum dimasukkan ke PATH Windows!
        echo Silakan pasang Python dari https://www.python.org/downloads/
        echo Jangan lupa centang "Add python.exe to PATH" saat instalasi.
        echo.
        pause
        exit /b 1
    )
    set PY_CMD=py
) else (
    set PY_CMD=python
)

echo Menggunakan: %PY_CMD%
echo.

REM 2. Cek dan instal dependensi dari requirements.txt
echo [1/2] Memeriksa dan menginstal library (joblib, streamlit, opencv, dll)...
%PY_CMD% -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Gagal menginstal library. Pastikan koneksi internet aktif.
    pause
    exit /b 1
)

echo.
echo [2/2] Menjalankan server aplikasi di localhost...
echo Browser akan otomatis terbuka ke http://localhost:8501
echo Tekan Ctrl + C di jendela ini jika ingin menghentikan aplikasi.
echo =======================================================
echo.

%PY_CMD% -m streamlit run app.py

pause
