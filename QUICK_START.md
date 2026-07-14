# 🚀 Hızlı Başlangıç

Modeli eğitmek için 3 temel adım:

## 1️⃣ Ortamı Kur (İlk Seferinde)

### macOS / Linux
```bash
bash setup_environment.sh
```

### Windows
```bash
setup_environment.bat
```

## 2️⃣ VPS'den Veri İndir

```bash
python3 training/download_from_vps.py
```

**Gerekli bilgiler:**
- VPS IP/Hostname
- SSH Port (22)
- Kullanıcı Adı
- CSV dosyalarının klasörü
- Şifre

## 3️⃣ Model Eğit

### Tam Pipeline (Önerilen)
```bash
python3 training/run_full_pipeline.py
```

Menüden seçin:
- **1:** Tüm adımları çalıştır
- **2:** Hızlı mod (veri zaten hazırsa)
- **3-5:** Bireysel adımlar

### Doğrudan Eğitim (veri hazırsa)
```bash
# Dataset hazırla
python3 training/prepare_dataset.py

# Modeli eğit
python3 training/train_crnn.py

# Değerlendir
python3 training/evaluate_model.py
```

---

## 📊 Sonuçlar

Eğitim tamamlandıktan sonra:

- **Model:** `models/crnn_best_model.pth`
- **Loglar:** `logs/training_*.log`
- **Değerlendirme:** `logs/evaluation/test_*.csv`
- **Metrikleri:** `logs/evaluation/test_*.json`

---

## ⚙️ Ayarlar Değiştir

`configs/training_config.yaml` dosyasını düzenle:

```yaml
num_epochs: 50           # Epoch sayısı
batch_size: 32           # Batch boyutu
learning_rate: 0.001     # Öğrenme oranı
```

---

## 💡 İpuçları

- **GPU varsa** model çok daha hızlı eğitilir
- **Batch size'ı düşür** eğer bellek yetmiyorsa
- **Epoch sayısını artır** daha iyi sonuç için
- **Logları kontrol et** eğitim sırasında ilerlemeyi görmek için

---

**Detaylı kılavuz için:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
