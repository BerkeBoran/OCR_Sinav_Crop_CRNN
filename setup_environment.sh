#!/bin/bash

# CRNN Model Eğitim Ortamı Kurulum Scripti
# MacBook ve Linux uyumlu

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     CRNN MODEL TRAINING ENVIRONMENT SETUP                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Platform kontrolü
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    PY_VERSION="python3"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    PY_VERSION="python3"
else
    echo "❌ Desteklenmeyen işletim sistemi: $OSTYPE"
    exit 1
fi

echo "✓ Işletim Sistemi: $OS"
echo ""

# Python versiyonu kontrol et
echo "Adım 1: Python kontrolü..."
if ! command -v $PY_VERSION &> /dev/null; then
    echo "❌ Python3 bulunamadı. Lütfen Python 3.8+ yükleyin."
    exit 1
fi

PY_VER=$($PY_VERSION --version | cut -d' ' -f2)
echo "✓ Python sürümü: $PY_VER"
echo ""

# Virtual environment oluştur
echo "Adım 2: Virtual environment oluşturuluyor..."
if [ ! -d "venv" ]; then
    $PY_VERSION -m venv venv
    echo "✓ Virtual environment oluşturuldu"
else
    echo "✓ Virtual environment zaten var"
fi

# Virtual environment'ı etkinleştir
source venv/bin/activate
echo "✓ Virtual environment etkinleştirildi"
echo ""

# pip'i güncelle
echo "Adım 3: pip ve setuptools güncelleniyor..."
pip install --upgrade pip setuptools wheel --quiet
echo "✓ pip güncellendi"
echo ""

# Gerekli paketleri yükle
echo "Adım 4: Bağımlılıklar yükleniyor..."

# Base paketler
echo "  - PyTorch yükleniyor..."
if [[ "$OS" == "macOS" ]]; then
    # MacBook M1/M2/M3 için Metal Performance Shaders (MPS) desteği
    pip install torch torchvision torchaudio --quiet
else
    # Linux - CUDA desteği (GPU varsa)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
fi

echo "  - Diğer paketler yükleniyor..."
pip install -q \
    pandas \
    numpy \
    pyyaml \
    scikit-learn \
    matplotlib \
    tqdm \
    pillow \
    jupyter \
    ipython

echo "✓ Tüm paketler yüklendi"
echo ""

# Klasör yapısını oluştur
echo "Adım 5: Klasör yapısı oluşturuluyor..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/cropped_fields
mkdir -p models
mkdir -p logs
mkdir -p logs/evaluation
mkdir -p configs
mkdir -p training
mkdir -p notebooks

echo "✓ Klasörler oluşturuldu"
echo ""

# VPS veri indirme talimatları
echo "Adım 6: Veri hazırlığı..."
echo ""
echo "📝 VPS'den label.csv dosyalarını indirmek için:"
echo ""
echo "  1. SSH ile VPS'ye bağlan:"
echo "     ssh kullanici@vps_ip"
echo ""
echo "  2. CSV dosyalarını yerel bilgisayara kopyala:"
echo "     scp kullanici@vps_ip:/path/to/label.csv data/raw/"
echo ""
echo "  3. Veya şu komutu çalıştır:"
echo "     python3 training/download_from_vps.py"
echo ""

# Dataset hazırlama
echo "Adım 7: Dataset hazırlığı..."
echo ""

if [ -f "data/raw/labels.csv" ] || [ "$(ls -A data/raw/*.csv 2>/dev/null)" ]; then
    echo "ℹ CSV dosyaları bulundu. Dataset hazırlanıyor..."
    python3 training/prepare_dataset.py
    echo "✓ Dataset hazırlandı"
else
    echo "⚠ CSV dosyaları data/raw/ klasöründe bulunamadı"
    echo "Lütfen önce VPS'den CSV dosyalarını indirin"
fi
echo ""

# Eğitim scriptlerinin izni
chmod +x scripts/*.py
chmod +x training/*.py 2>/dev/null || true
echo ""

# Özet
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     KURULUM TAMAMLANDI ✓                                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Sonraki Adımlar:"
echo ""
echo "1️⃣  Dataset Hazırlama:"
echo "   python3 training/prepare_dataset.py"
echo ""
echo "2️⃣  Model Eğitme:"
echo "   python3 training/train_crnn.py"
echo ""
echo "3️⃣  Model Değerlendirme:"
echo "   python3 training/evaluate_model.py"
echo ""
echo "4️⃣  Jupyter Notebook açma:"
echo "   jupyter notebook"
echo ""
echo "ℹ Virtual environment'ı deaktif etmek için:"
echo "   deactivate"
echo ""
echo "ℹ Gelecekte virtual environment'ı etkinleştirmek için:"
echo "   source venv/bin/activate"
echo ""
