@echo off
REM CRNN Model Eğitim Ortamı Kurulum Scripti (Windows)

setlocal enabledelayedexpansion

cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║     CRNN MODEL TRAINING ENVIRONMENT SETUP ^(WINDOWS^)            ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Python kontrolü
echo Adım 1: Python kontrolü...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python bulunamadı. Lütfen Python 3.8+ yükleyin.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo ✓ Python sürümü: %PY_VER%
echo.

REM Virtual environment oluştur
echo Adım 2: Virtual environment oluşturuluyor...
if not exist "venv" (
    python -m venv venv
    echo ✓ Virtual environment oluşturuldu
) else (
    echo ✓ Virtual environment zaten var
)
echo.

REM Virtual environment'ı etkinleştir
call venv\Scripts\activate.bat
echo ✓ Virtual environment etkinleştirildi
echo.

REM pip'i güncelle
echo Adım 3: pip ve setuptools güncelleniyor...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo ✓ pip güncellendi
echo.

REM Gerekli paketleri yükle
echo Adım 4: Bağımlılıklar yükleniyor...

echo   - PyTorch yükleniyor...
REM CPU veya CUDA GPU desteği ile PyTorch
pip install torch torchvision torchaudio >nul 2>&1

echo   - Diğer paketler yükleniyor...
pip install ^
    pandas ^
    numpy ^
    pyyaml ^
    scikit-learn ^
    matplotlib ^
    tqdm ^
    pillow ^
    jupyter ^
    ipython >nul 2>&1

echo ✓ Tüm paketler yüklendi
echo.

REM Klasör yapısını oluştur
echo Adım 5: Klasör yapısı oluşturuluyor...

if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\cropped_fields" mkdir data\cropped_fields
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "logs\evaluation" mkdir logs\evaluation
if not exist "configs" mkdir configs
if not exist "training" mkdir training
if not exist "notebooks" mkdir notebooks

echo ✓ Klasörler oluşturuldu
echo.

REM VPS veri indirme talimatları
echo Adım 6: Veri hazırlığı...
echo.
echo 📝 VPS'den label.csv dosyalarını indirmek için:
echo.
echo   1. PowerShell veya CMD ile şu komutu çalıştır:
echo      scp kullanici@vps_ip:/path/to/label.csv data\raw\
echo.
echo   2. Veya şu scripti çalıştır:
echo      python training/download_from_vps.py
echo.

REM Dataset hazırlama
echo Adım 7: Dataset hazırlığı...
echo.

if exist "data\raw\labels.csv" (
    echo ℹ CSV dosyaları bulundu. Dataset hazırlanıyor...
    python training\prepare_dataset.py
    echo ✓ Dataset hazırlandı
) else (
    dir /b data\raw\*.csv >nul 2>&1
    if !errorlevel! equ 0 (
        echo ℹ CSV dosyaları bulundu. Dataset hazırlanıyor...
        python training\prepare_dataset.py
        echo ✓ Dataset hazırlandı
    ) else (
        echo ⚠ CSV dosyaları data\raw\ klasöründe bulunamadı
        echo Lütfen önce VPS'den CSV dosyalarını indirin
    )
)
echo.

REM Özet
cls
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║     KURULUM TAMAMLANDI ✓                                       ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo Sonraki Adımlar:
echo.
echo 1️⃣  Dataset Hazırlama:
echo    python training\prepare_dataset.py
echo.
echo 2️⃣  Model Eğitme:
echo    python training\train_crnn.py
echo.
echo 3️⃣  Model Değerlendirme:
echo    python training\evaluate_model.py
echo.
echo 4️⃣  Jupyter Notebook açma:
echo    jupyter notebook
echo.
echo ℹ Virtual environment'ı deaktif etmek için:
echo    deactivate
echo.
echo ℹ Gelecekte virtual environment'ı etkinleştirmek için:
echo    venv\Scripts\activate.bat
echo.
pause
