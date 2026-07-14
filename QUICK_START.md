# 🚀 Hızlı Başlangıç

Veri seti repoya dahildir — clone'layıp doğrudan eğitebilirsiniz.

## 1️⃣ Kurulum (ilk seferde bir kez)

### macOS / Linux
```bash
bash setup_environment.sh
```

### Windows
```bash
setup_environment.bat
```

## 2️⃣ Model Eğit

```bash
python3 training/train_crnn.py
```

## 3️⃣ Değerlendir

```bash
python3 training/evaluate_model.py
```

---

## 📊 Sonuçlar

- **Model:** `models/crnn_best_model.pth`
- **Loglar + grafik:** `logs/`
- **Test sonuçları:** `logs/evaluation/`

## ⚙️ Ayarlar

`configs/training_config.yaml`:

```yaml
num_epochs: 50           # Epoch sayısı
batch_size: 32           # Bellek yetmezse 16 yapın
learning_rate: 0.001
```

## 💡 İpuçları

- GPU otomatik algılanır (NVIDIA CUDA / Apple Silicon MPS)
- Yeni terminalde önce venv'i etkinleştirin: `source venv/bin/activate`
- Yeni veri eklendiğinde: `python3 training/prepare_dataset.py`

**Detaylı bilgi:** [README.md](README.md) · [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
