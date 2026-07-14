# ✅ Profesyonel CRNN Model Eğitim Setup'ı Tamamlandı

## 📋 Yapı Özeti

Tam otomatik ve profesyonel bir CRNN model eğitim pipeline'ı oluşturuldu.

### 🗂️ Yeni Klasörler

```
training/              # Eğitim scriptleri
├── train_crnn.py              # CRNN model eğitimi
├── prepare_dataset.py          # Dataset hazırlama
├── evaluate_model.py           # Model değerlendirmesi
├── download_from_vps.py        # VPS veri indirme
└── run_full_pipeline.py        # Tüm adımları koordine et

configs/               # Konfigürasyon dosyaları
├── training_config.yaml        # Eğitim parametreleri

models/                # Eğitilen modeller
└── (crnn_best_model.pth kaydedilecek)

logs/                  # Eğitim logları ve sonuçları
├── (training logları)
├── evaluation/        # Test sonuçları

utils/                 # Yardımcı fonksiyonlar
├── visualization.py           # Grafik ve görselleştirme

data/
├── raw/                # VPS'den indirilen CSV dosyaları
└── processed/          # İşlenmiş train/val/test splits
```

---

## 🚀 Başlamak için

### 1. MacBook/Linux
```bash
bash setup_environment.sh
```

### 2. Windows
```bash
setup_environment.bat
```

### 3. Manuel (Tüm Platformlar)
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

---

## 📊 Eğitim Pipeline'ı

```
┌─────────────────────┐
│  VPS Veri İndirme   │
│ (download_from_vps) │
└──────────┬──────────┘
           ↓
┌─────────────────────────┐
│  Dataset Hazırlama      │
│ (prepare_dataset.py)    │
│ • CSV Birleştir         │
│ • Train/Val/Test Split  │
└──────────┬──────────────┘
           ↓
┌──────────────────────┐
│  Model Eğitimi       │
│ (train_crnn.py)      │
│ • CNN + RNN Mimarisi │
│ • Otomatik Checkpoint│
└──────────┬───────────┘
           ↓
┌──────────────────────────┐
│  Model Değerlendirmesi   │
│ (evaluate_model.py)      │
│ • Metrikleri Hesapla     │
│ • Rapor Oluştur          │
└──────────────────────────┘
```

---

## 📁 Dosya Açıklaması

### Eğitim Scriptleri

| Dosya | Amaç | Giriş | Çıkış |
|-------|------|------|-------|
| `train_crnn.py` | CRNN modelini eğit | train.csv, val.csv | crnn_best_model.pth |
| `prepare_dataset.py` | CSV'leri hazırla | data/raw/*.csv | train/val/test.csv |
| `evaluate_model.py` | Modeli değerlendir | test.csv, model.pth | metrics.json, results.csv |
| `download_from_vps.py` | VPS'den indir | SSH bilgileri | data/raw/*.csv |
| `run_full_pipeline.py` | Tüm adımları yönet | - | Pipeline kontrolü |

### Konfigürasyon

**`configs/training_config.yaml`**
```yaml
num_epochs: 50              # Eğitim döngüsü sayısı
batch_size: 32              # Her adımda işlenen örnek sayısı
learning_rate: 0.001        # Ağırlıkları güncelleme hızı
img_width: 128              # Resim genişliği (pixel)
img_height: 32              # Resim yüksekliği (pixel)
```

### Yardımcı Fonksiyonlar

**`utils/visualization.py`**
- Eğitim geçmişi grafikleri
- Confusion matrix visualizasyonu
- Örnek tahmin gösterimi
- Özet rapor oluşturma

---

## ⚙️ CRNN Mimarisi

```
Giriş Resim (1 x 128 x 32)
        ↓
    CNN Kısmı
    ├─ Conv2d (3x3)
    ├─ BatchNorm
    ├─ ReLU
    └─ MaxPool
    [32 → 64 → 128 kanal]
        ↓
    Reshape + LSTM
    [Bidirectional RNN]
        ↓
    Global Average Pooling
        ↓
    Fully Connected Layers
    [256 → num_classes]
        ↓
    Sınıf Tahmini
```

**Model Parametreleri:**
- CNN Kanal: 32, 64, 128
- RNN Gizli Boyutu: 256 (bidirectional)
- Toplam Parametreler: ~2M

---

## 📈 Eğitim Örneği

```bash
# Terminal'de
python3 training/run_full_pipeline.py

# Menü seçin:
# 1. Tam pipeline
# 2. Hızlı mod
# 3-5. Bireysel adımlar
```

**Çıktı Örneği:**
```
[1/4] VPS Veri İndirme
    Label.csv dosyaları indiriliyor...
    ✓ 50 MB indirildi

[2/4] Dataset Hazırlama
    ✓ Train: 3500 örnek (70%)
    ✓ Val: 750 örnek (15%)
    ✓ Test: 750 örnek (15%)

[3/4] Model Eğitimi
    Epoch 1/50 - Train Loss: 2.341, Val Loss: 1.895, Val Acc: 0.654
    Epoch 2/50 - Train Loss: 1.523, Val Loss: 1.204, Val Acc: 0.782
    ...
    ✓ En iyi model kaydedildi

[4/4] Model Değerlendirmesi
    Accuracy:  0.8543
    Precision: 0.8421
    Recall:    0.8465
    F1-Score:  0.8443
```

---

## 🔧 Sistem Uyumluluğu

### MacBook
- ✅ Intel (x86_64)
- ✅ M1/M2/M3 (Apple Silicon)
- ✅ Metal Performance Shaders desteği
- ⚠️ GPU kullanımı M-serisi çipslerde otomatik

### Windows
- ✅ Windows 10/11 (64-bit)
- ✅ NVIDIA GPU (CUDA desteği)
- ✅ CPU (yavaş ama çalışır)
- ⚠️ PowerShell veya CMD gerekli

### Linux
- ✅ Ubuntu 20.04+
- ✅ NVIDIA GPU (CUDA)
- ✅ CPU/ROCm desteği
- ⚠️ Bash shell gerekli

---

## 💾 Bağımlılıklar

```
PyTorch >= 2.0           # Derin öğrenme framework'ü
pandas >= 2.0            # Veri işleme
scikit-learn >= 1.3      # ML metrikler
numpy >= 1.24            # Sayısal işlemler
Pillow >= 10.0           # Resim işleme
tqdm >= 4.66             # Progress bar
paramiko >= 3.3          # SSH (VPS bağlantısı)
matplotlib >= 3.7        # Visualizasyon
pyyaml >= 6.0            # Konfigürasyon
```

---

## 📊 Çıktı Dosyaları

### Modeller
- `models/crnn_best_model.pth` - Eğitilen PyTorch model
- Boyut: ~15-20 MB
- Checkpoint'ler: model ağırlıkları, optimizer state, label mapping

### Loglar
- `logs/training_*.log` - Eğitim logları
- `logs/history_*.json` - Epoch başına loss/accuracy
- `logs/evaluation/test_results_*.csv` - Tahmin detayları
- `logs/evaluation/test_metrics_*.json` - Performans metrikleri

### Grafikler (Otomatik oluşturulur)
- `logs/training_history.png` - Loss & Accuracy grafikleri
- `logs/confusion_matrix.png` - Confusion matrix
- `logs/sample_predictions.png` - Örnek tahminler

---

## ✨ Özellikler

✅ **Otomatik Pipeline** - Adım adım ortam kurulumu
✅ **Çoklu Platform** - macOS, Windows, Linux uyumlu
✅ **GPU Desteği** - NVIDIA CUDA, Apple Metal, CPU fallback
✅ **VPS Entegrasyonu** - SSH ile otomatik veri indirme
✅ **Profesyonel Mimarisi** - CNN + Bidirectional LSTM
✅ **Detaylı Loglar** - Her adımda ilerleme takibi
✅ **Otomatik Kontrol Noktası** - En iyi model kaydı
✅ **Kapsamlı Değerlendirme** - Metrikleri ve raporları otomatik hesapla
✅ **Visualizasyon** - Grafik ve tahmin gösterimi
✅ **Interaktif Pipeline** - Menü tabanlı adım seçimi

---

## 🚀 Sonraki Adımlar

1. **Kurulum:** `bash setup_environment.sh` (macOS/Linux) veya `setup_environment.bat` (Windows)

2. **Veri İndirme:** `python3 training/download_from_vps.py`

3. **Eğitim Başlatma:** `python3 training/run_full_pipeline.py`

4. **Sonuçları İnceleme:** `logs/` klasöründeki dosyaları kontrol et

---

## 📚 Dokümantasyon

- **[QUICK_START.md](QUICK_START.md)** - Hızlı başlangıç (3 adım)
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Detaylı kılavuz
- **Kod içi yorumlar** - Her scriptte açıklamalar

---

## 📞 Sorun Giderme

**Setup'ı yeniden çalıştır:**
```bash
# macOS/Linux
bash setup_environment.sh

# Windows
setup_environment.bat
```

**Virtual environment problemi:**
```bash
rm -rf venv          # Sil
python3 -m venv venv # Yeniden oluştur
source venv/bin/activate
pip install -r requirements.txt
```

**Detaylı yardım:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md#-sorun-giderme)

---

## 🎯 Hedefler

- ✅ Profesyonel klasör yapısı
- ✅ Tam otomatik setup
- ✅ Çoklu platform desteği
- ✅ VPS veri entegrasyonu
- ✅ İleri CRNN mimarisi
- ✅ Kapsamlı eğitim pipeline'ı
- ✅ Detaylı değerlendirme ve raporlama
- ✅ Yeterli dokümantasyon

---

**Başarılı bir eğitim sürecine hoşgeldiniz! 🎉**

*Sorularınız veya geri bildiriminiz varsa lütfen dokümantasyonu kontrol edin.*
