@echo off
REM CRNN Model Egitim Ortami Kurulumu (Windows)
REM Kullanim: setup_environment.bat

echo.
echo ================================================================
echo      CRNN MODEL TRAINING ENVIRONMENT SETUP (WINDOWS)
echo ================================================================
echo.

REM Python kontrolu
echo Adim 1/4: Python kontrolu...
python --version >nul 2>&1
if errorlevel 1 (
    echo HATA: Python bulunamadi. Lutfen Python 3.9+ yukleyin:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do echo OK Python surumu: %%i

REM Virtual environment
echo.
echo Adim 2/4: Virtual environment...
if exist ".venv\Scripts\activate.bat" (
    set VENV_DIR=.venv
) else if exist "venv\Scripts\activate.bat" (
    set VENV_DIR=venv
) else (
    set VENV_DIR=venv
    python -m venv venv
)
call %VENV_DIR%\Scripts\activate.bat
echo OK Etkin: %VENV_DIR%

REM Paketler
echo.
echo Adim 3/4: Bagimliliklar yukleniyor (birkac dakika surebilir)...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo OK Tum paketler yuklendi

REM Veri kontrolu
echo.
echo Adim 4/4: Veri kontrolu...
if exist "data\processed\train.csv" (
    echo OK Hazir split'ler bulundu ^(data\processed\^)
) else (
    echo Split'ler bulunamadi, olusturuluyor...
    python training\prepare_dataset.py
)

REM Ozet
echo.
echo ================================================================
echo      KURULUM TAMAMLANDI
echo ================================================================
echo.
echo Egitimi baslatmak icin:
echo.
echo    python training\train_crnn.py
echo.
echo Egitim sonrasi degerlendirme:
echo.
echo    python training\evaluate_model.py
echo.
echo Yeni terminal oturumlarinda once venv'i etkinlestirin:
echo    %VENV_DIR%\Scripts\activate
echo.
pause
