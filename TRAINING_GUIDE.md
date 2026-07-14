# CRNN Model Eğitim Kılavuzu

Profesyonel bir CRNN (Convolutional Recurrent Neural Network) modeli eğitmek için adım adım kılavuz.

## 📋 İçindekiler

1. [Gereksinimler](#gereksinimler)
2. [Kurulum](#kurulum)
3. [Veri Hazırlığı](#veri-hazırlığı)
4. [Model Eğitimi](#model-eğitimi)
5. [Model Değerlendirmesi](#model-değerlendirmesi)
6. [Klasör Yapısı](#klasör-yapısı)
7. [Sorun Giderme](#sorun-giderme)

---

## 🔧 Gereksinimler

### Yazılım
- **Python 3.8+**
- **Git**
- **SSH** (VPS'den veri indirmek için)

### Donanım
- **MacBook** (Intel/M1/M2/M3)
- **Windows PC** (GPU opsiyonel)
- **Linux Sunucusu** (GPU opsiyonel)

### İnternet Bağlantısı
- PyPI paketleri indirebilmek için
- VPS'ye SSH bağlantısı (label.csv dosyaları için)

---

## 🚀 Kurulum

### 1. MacBook / Linux

```bash
# Proje klasörüne gir
cd /path/to/OCR_Sinav_Crop_CRNN

# Setup script'ini çalıştır
bash setup_environment.sh
```

### 2. Windows

```bash
# Proje klasörüne gir (PowerShell/CMD)
cd C:\path\to\OCR_Sinav_Crop_CRNN

# Setup script'ini çalıştır
setup_environment.bat
```

### 3. Manuel Kurulum (Tüm Platformlar)

```bash
# Virtual environment oluştur
python3 -m venv venv

# Virtual environment'ı etkinleştir
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Paketleri yükle
pip install -r requirements.txt

# Klasörleri oluştur
mkdir -p data/raw data/processed models logs configs
```

---

## 📂 Veri Hazırlığı

### Adım 1: VPS'den Veri İndir

VPS'deki label.csv dosyalarını indirmek için:

```bash
python3 training/download_from_vps.py
```

**İnternet İçeriği:**
- VPS Hostname/IP
- SSH Port (varsayılan 22)
- Kullanıcı Adı
- CSV dosyalarının yolu
- SSH Şifre

**Alternatif (Manuel SCP):**
```bash
# VPS'den yerel makineye kopyala
scp kullanici@vps_ip:/path/to/*.csv data/raw/
```

### Adım 2: Dataset Hazırlama

CSV dosyalarını birleştirip train/val/test split'leri oluştur:

```bash
python3 training/prepare_dataset.py
```

**Çıktı:**
- `data/processed/train.csv` (80% veri)
- `data/processed/val.csv` (10% veri)
- `data/processed/test.csv` (10% veri)

**Parametreler (değiştirmek için `prepare_dataset.py`'ı düzenle):**
- `test_size=0.2` - Test veri yüzdesi
- `val_size=0.1` - Validation veri yüzdesi

---

## 🧠 Model Eğitimi

### Seçenek 1: Interaktif Pipeline (Önerilen)

```bash
python3 training/run_full_pipeline.py
```

**Menü Seçenekleri:**
1. Tam pipeline (VPS → Eğitim → Değerlendirme)
2. Hızlı mod (sadece eğitim + değerlendirme)
3. Sadece dataset hazırla
4. Sadece model eğit
5. Sadece değerlendir

### Seçenek 2: Doğrudan Eğitim

```bash
python3 training/train_crnn.py
```

**Eğitim Parametreleri (`configs/training_config.yaml`):**

```yaml
num_epochs: 50              # Toplam epoch sayısı
batch_size: 32              # Batch boyutu
learning_rate: 0.001        # Öğrenme oranı
img_width: 128              # Resim genişliği
img_height: 32              # Resim yüksekliği
num_workers: 4              # Veri yükleme worker'ı sayısı
```

### Eğitim Sırasında

- **Log dosyaları:** `logs/training_*.log`
- **En iyi model:** `models/crnn_best_model.pth`
- **Eğitim geçmişi:** `logs/history_*.json`

### Eğitimi İptal Etme

- **Linux/macOS:** `Ctrl + C`
- **Windows:** `Ctrl + C`

---

## 📊 Model Değerlendirmesi

### Test Seti Üzerinde Değerlendir

```bash
python3 training/evaluate_model.py
```

**Çıktılar:**
- `logs/evaluation/test_results_*.csv` - Tahmin detayları
- `logs/evaluation/test_metrics_*.json` - Performans metrikleri

### Metriklerin Yorumlanması

| Metrik | İyi Değer | Yorumlar |
|--------|-----------|----------|
| Accuracy | > 0.90 | Genel doğruluk oranı |
| Precision | > 0.85 | Yanlış pozitif oranı |
| Recall | > 0.85 | Yanlış negatif oranı |
| F1-Score | > 0.87 | Precision ve Recall dengesi |

---

## 📁 Klasör Yapısı

```
OCR_Sinav_Crop_CRNN/
├── data/
│   ├── raw/                    # VPS'den indirilen CSV dosyaları
│   ├── processed/              # İşlenmiş veri (train/val/test.csv)
│   ├── cropped_fields/         # Kırpılmış resimler
│   ├── crnn_dataset/           # CRNN veri seti
│   └── metadata/               # Meta veri
├── training/
│   ├── train_crnn.py          # Ana eğitim scripti
│   ├── prepare_dataset.py      # Dataset hazırlama
│   ├── evaluate_model.py       # Model değerlendirmesi
│   ├── download_from_vps.py    # VPS veri indirme
│   ├── run_full_pipeline.py    # Pipeline yöneticisi
│   └── __init__.py
├── models/
│   └── crnn_best_model.pth    # Eğitilen model
├── configs/
│   └── training_config.yaml    # Eğitim konfigürasyonu
├── logs/
│   ├── training_*.log          # Eğitim logları
│   ├── history_*.json          # Eğitim geçmişi
│   └── evaluation/             # Değerlendirme sonuçları
├── utils/
│   ├── visualization.py        # Visualizasyon fonksiyonları
│   └── __init__.py
├── setup_environment.sh        # macOS/Linux kurulum
├── setup_environment.bat       # Windows kurulum
├── requirements.txt            # Python paketleri
└── TRAINING_GUIDE.md          # Bu dosya
```

---

## 🐛 Sorun Giderme

### Problem: "CUDA out of memory"
**Çözüm:**
```yaml
# configs/training_config.yaml
batch_size: 16  # 32'den 16'ya düşür
```

### Problem: "Module not found: torch"
**Çözüm:**
```bash
pip install --upgrade torch
```

### Problem: "CSV dosyası bulunamadı"
**Çözüm:**
1. VPS'den veri indir: `python3 training/download_from_vps.py`
2. `data/raw/` klasöründe CSV dosyaları olduğunu kontrol et

### Problem: "Resim dosyası bulunamadı"
**Çözüm:**
- CSV'deki dosya yollarının `data/cropped_fields/` altında olduğunu kontrol et
- Dosya isimlerinin tam olarak eşleştiğini kontrol et

### Problem: "Permission denied" (macOS/Linux)
**Çözüm:**
```bash
chmod +x setup_environment.sh
chmod +x training/*.py
```

### Problem: Virtual Environment Hatası

**macOS/Linux:**
```bash
# Eski virtual environment'ı sil
rm -rf venv

# Yenisini oluştur
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
# Eski virtual environment'ı sil
rmdir /s /q venv

# Yenisini oluştur
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📈 İleri Ayarlar

### Hiperparametre Tuning

`configs/training_config.yaml` dosyasını düzenle:

```yaml
# Model mimari
cnn_channels: [32, 64, 128]    # CNN kanal sayıları
rnn_hidden: 256                 # RNN gizli boyutu

# Eğitim
num_epochs: 100                 # Daha fazla epoch
learning_rate: 0.0005           # Daha düşük öğrenme oranı
weight_decay: 0.0001            # L2 regularization

# Scheduler
scheduler_patience: 10          # LR düşmek için patience
scheduler_factor: 0.3           # LR düşme faktörü
```

### GPU Kullanımı

- **NVIDIA GPU (CUDA):** Otomatik olarak kullanılır
- **Apple Metal (M1/M2/M3):** Otomatik olarak kullanılır
- **CPU:** CPU'da çalışır (daha yavaş)

GPU kullanımını kontrol etmek için:
```python
import torch
print(torch.cuda.is_available())  # NVIDIA
print(torch.backends.mps.is_available())  # Apple Metal
```

---

## 📞 Destek

Model eğitimi sırasında sorun yaşarsanız:

1. **Log dosyasını kontrol et:** `logs/training_*.log`
2. **Error mesajını dikkatle oku**
3. **Sorun Giderme bölümüne bak**
4. **Setup'ı yeniden çalıştır:**
   - macOS/Linux: `bash setup_environment.sh`
   - Windows: `setup_environment.bat`

---

## 📝 Notlar

- **Eğitim süresi:** GPU olmadan 1-2 saat, GPU ile 15-30 dakika
- **Model boyutu:** ~15-20 MB
- **Veri boyutu:** CSV dosya boyutuna bağlı (birkaç MB - birkaç GB)

---

**Başarılı eğitimler! 🚀**
