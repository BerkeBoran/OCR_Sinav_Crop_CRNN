#!/usr/bin/env python3
"""
Alan tespit modeli (YOLOv8) eğitimi.

Sınav kağıdı fotoğrafında 'not' ve 'ogrenci_numara' alanlarını bulan
detektörü data/roboflow_export üzerindeki YOLO etiketleriyle eğitir.

Kullanım:
    python3 training/train_yolo.py                 # sadece gerçek veri
    python3 training/train_yolo.py --sentetik       # gerçek + sentetik şablonlar

--sentetik: data/synthetic_templates'i eğitime KATAR ama doğrulama/test hep
gerçek veride kalır — böylece sentetik verinin gerçek fayda sağlayıp
sağlamadığı dürüstçe ölçülür. Önce üretmeyi unutmayın:
    python3 scripts/generate_synthetic_templates.py --sayi 1500

Çıktılar:
    - En iyi model: models/yolo_fields.pt
    - Eğitim logları/grafikler: logs/yolo/
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = PROJECT_ROOT / "data" / "roboflow_export" / "data.yaml"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic_templates"
COMBINED_YAML = PROJECT_ROOT / "data" / "combined_data.yaml"
OUTPUT_MODEL = PROJECT_ROOT / "models" / "yolo_fields.pt"
LOG_DIR = PROJECT_ROOT / "logs" / "yolo"

BASE_MODEL = "yolov8s.pt"
EPOCHS = 300
# Tam sayfa fotoğrafta alanlar küçük kaldığı için yüksek çözünürlük kullanılır
IMG_SIZE = 960
BATCH_SIZE = 8


def fix_data_yaml():
    """Roboflow export'unun '../train/images' şeklindeki hatalı göreli
    yollarını klasör içine işaret edecek şekilde düzeltir."""
    with open(DATA_YAML, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    changed = False
    for split, folder in [("train", "train"), ("val", "valid"), ("test", "test")]:
        expected = f"{folder}/images"
        if data.get(split) != expected:
            data[split] = expected
            changed = True

    if changed:
        with open(DATA_YAML, "w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
        print(f"data.yaml yolları düzeltildi: {DATA_YAML}")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_combined_yaml():
    """Gerçek + sentetik eğitim setini birleştiren data.yaml yazar.
    Doğrulama ve test BİLEREK yalnızca gerçek veridir."""
    synth_train = SYNTHETIC_DIR / "train" / "images"
    if not synth_train.exists():
        raise FileNotFoundError(
            f"Sentetik veri bulunamadı: {synth_train}\n"
            "Önce üretin: python3 scripts/generate_synthetic_templates.py --sayi 1500"
        )

    real = DATA_YAML.parent
    data = {
        # Ultralytics birden çok train yolunu liste olarak kabul eder
        "train": [
            str(real / "train" / "images"),
            str(synth_train),
        ],
        "val": str(real / "valid" / "images"),
        "test": str(real / "test" / "images"),
        "nc": 2,
        "names": ["not", "ogrenci_numara"],
    }
    with open(COMBINED_YAML, "w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)

    synth_count = len(list(synth_train.glob("*.jpg")))
    print(f"Birleşik veri: gerçek train + {synth_count} sentetik sayfa "
          f"(doğrulama/test yalnızca gerçek)")
    return COMBINED_YAML


def main():
    parser = argparse.ArgumentParser(description="YOLO alan tespit modeli eğitimi")
    parser.add_argument("--sentetik", action="store_true",
                        help="Sentetik şablonları eğitime kat (data/synthetic_templates)")
    args = parser.parse_args()

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset bulunamadı: {DATA_YAML}\n"
            "data/roboflow_export klasörüne Roboflow YOLO export'unu koyun."
        )

    fix_data_yaml()

    if args.sentetik:
        data_yaml = build_combined_yaml()
    else:
        data_yaml = DATA_YAML

    device = pick_device()
    print(f"Cihaz: {device}")

    model = YOLO(BASE_MODEL)

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=str(LOG_DIR),
        name="fields",
        patience=25,
        # Şablon ezberini kırmaya yönelik augmentation:
        # alanların sayfadaki konumuna değil görünümüne odaklanmayı teşvik eder
        degrees=5.0,
        translate=0.15,
        scale=0.6,
        perspective=0.0005,
    )

    best_path = Path(model.trainer.best)
    if not best_path.exists():
        raise FileNotFoundError(f"Eğitim çıktısı bulunamadı: {best_path}")

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, OUTPUT_MODEL)
    print(f"\nEn iyi model kopyalandı: {OUTPUT_MODEL}")

    print("\nTest split üzerinde değerlendirme (yalnızca gerçek veri):")
    metrics = YOLO(str(OUTPUT_MODEL)).val(data=str(data_yaml), split="test", device=device)
    print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")

    print("\nEğitim tamamlandı.")
    print(f"Grafikler ve detaylar: {LOG_DIR}/fields/")


if __name__ == "__main__":
    main()
