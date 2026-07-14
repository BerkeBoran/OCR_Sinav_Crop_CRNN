# OCR Sınav Kağıdı Tanıma — CRNN

Sınav kağıtlarındaki **öğrenci numarası** ve **not** alanlarını okuyan CRNN (CNN + BiLSTM + CTC) modeli.

Veri seti repoya dahildir (1598 etiketli görüntü + hazır train/val/test split'leri) — clone'layıp doğrudan eğitime başlayabilirsiniz.

## 🚀 Hızlı Başlangıç (3 Adım)

### macOS / Linux

```bash
git clone <repo-url>
cd OCR_Sinav_Crop_CRNN
bash setup_environment.sh        # venv oluşturur + paketleri kurar
python3 training/train_crnn.py  # eğitimi başlat (ince ayar)
```

### Windows

```bash
git clone <repo-url>
cd OCR_Sinav_Crop_CRNN
setup_environment.bat
python training\train_crnn.py
```

> **Not:** Repoda sentetik verilerle ön-eğitilmiş bir model (`models/crnn_pretrained.pth`)
> hazır gelir; `train_crnn.py` bunu otomatik yükleyip gerçek verilerle ince ayar yapar.
> Ön-eğitimi kendiniz tekrarlamak isterseniz (MNIST ~11 MB indirir, ~20 dk):
> `python3 training/pretrain_synthetic.py`

> Sonraki oturumlarda sadece venv'i etkinleştirin:
> `source venv/bin/activate` (macOS/Linux) veya `venv\Scripts\activate` (Windows)

## 📊 Eğitim Sonrası

```bash
python3 training/evaluate_model.py
```

| Çıktı | Konum |
|-------|-------|
| En iyi model | `models/crnn_best_model.pth` |
| Eğitim logları + grafik | `logs/` |
| Test sonuçları (CSV + metrikler) | `logs/evaluation/` |

Değerlendirme, **sequence accuracy** (tam eşleşme) ve **CER** (karakter hata oranı) raporlar; öğrenci numarası / not alanları için ayrı kırılım verir.

## 🧠 Model

- **Giriş:** 32×128 gri tonlamalı alan görüntüsü
- **Mimari:** CNN (VGG-stili) → 2 katmanlı BiLSTM → CTC
- **Çıkış:** Rakam dizisi (0-9, değişken uzunluk) — eğitimde görülmemiş numaraları da okur
- **Donanım:** NVIDIA CUDA, Apple Silicon (MPS) ve CPU otomatik algılanır

## ⚙️ Ayarlar

Eğitim parametreleri: [configs/training_config.yaml](configs/training_config.yaml)

```yaml
num_epochs: 50
batch_size: 32        # bellek yetmezse 16'ya düşürün
learning_rate: 0.001
```

## 📁 Proje Yapısı

```
├── data/
│   ├── raw/labels.csv          # Ham etiketler
│   ├── processed/              # train/val/test split'leri (repoda hazır)
│   └── cropped_fields/         # Etiketli alan görüntüleri (repoda)
│       ├── ogrenci_numara/
│       └── not/
├── training/
│   ├── train_crnn.py           # Model eğitimi
│   ├── evaluate_model.py       # Test seti değerlendirmesi
│   ├── prepare_dataset.py      # Split'leri yeniden üretmek için (opsiyonel)
│   └── run_full_pipeline.py    # İnteraktif pipeline menüsü
├── configs/training_config.yaml
├── scripts/                    # Veri hazırlama araçları (crop, etiketleme)
├── models/                     # Eğitilen modeller (git'e girmez)
└── logs/                       # Eğitim çıktıları (git'e girmez)
```

## 🔄 Veri Güncelleme (Opsiyonel)

Yeni etiketli veri eklendiğinde split'leri yeniden üretin:

```bash
python3 training/prepare_dataset.py
```

Split'ler deterministiktir (seed=42) — herkes aynı split'lerle eğitir.

## 🐛 Sık Sorunlar

| Sorun | Çözüm |
|-------|-------|
| `No module named torch` | venv aktif mi? `source venv/bin/activate` |
| Bellek yetersiz | `configs/training_config.yaml` → `batch_size: 16` |
| Eğitim çok yavaş | GPU algılandı mı? Log başındaki `Cihaz:` satırına bakın |

Detaylı kılavuz: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
