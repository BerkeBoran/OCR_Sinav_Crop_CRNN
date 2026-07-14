#!/bin/bash

# CRNN Model Eğitim Ortamı Kurulumu (macOS / Linux)
# Kullanım: bash setup_environment.sh

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     CRNN MODEL TRAINING ENVIRONMENT SETUP                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Platform kontrolü
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    echo "❌ Desteklenmeyen işletim sistemi: $OSTYPE"
    exit 1
fi
echo "✓ İşletim Sistemi: $OS"

# Python kontrolü
echo ""
echo "Adım 1/4: Python kontrolü..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 bulunamadı. Lütfen Python 3.9+ yükleyin."
    exit 1
fi
echo "✓ Python sürümü: $(python3 --version | cut -d' ' -f2)"

# Virtual environment (mevcut .venv veya venv kullanılır, yoksa oluşturulur)
echo ""
echo "Adım 2/4: Virtual environment..."
VENV_DIR=""
if [ -d ".venv" ]; then
    VENV_DIR=".venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
else
    VENV_DIR="venv"
    python3 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate
echo "✓ Etkin: $VENV_DIR"

# Paketler
echo ""
echo "Adım 3/4: Bağımlılıklar yükleniyor (birkaç dakika sürebilir)..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "✓ Tüm paketler yüklendi"

# Veri kontrolü
echo ""
echo "Adım 4/4: Veri kontrolü..."
if [ -f "data/processed/train.csv" ]; then
    echo "✓ Hazır split'ler bulundu (data/processed/)"
else
    echo "ℹ Split'ler bulunamadı, oluşturuluyor..."
    python3 training/prepare_dataset.py
fi

IMG_COUNT=$(find data/cropped_fields -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
echo "✓ Görüntü sayısı: $IMG_COUNT"

# Özet
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     KURULUM TAMAMLANDI ✓                                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Eğitimi başlatmak için:"
echo ""
echo "   python3 training/train_crnn.py"
echo ""
echo "Eğitim sonrası değerlendirme:"
echo ""
echo "   python3 training/evaluate_model.py"
echo ""
echo "ℹ Yeni terminal oturumlarında önce venv'i etkinleştirin:"
echo "   source $VENV_DIR/bin/activate"
echo ""
