#!/usr/bin/env python3
"""
Label Studio YOLO export'unu eğitim veri setine (data/roboflow_export) katar.

Label Studio'da: Project -> Export -> YOLO seçilip indirilen zip açılır,
sonra bu script o klasöre çalıştırılır.

Yaptıkları:
1. classes.txt okunur, Label Studio sınıf indeksleri projenin sırasına
   (not = 0, ogrenci_numara = 1) göre YENİDEN EŞLENİR — sıra farklıysa
   veri sessizce bozulacağı için bu adım kritiktir
2. Görsel + etiketler train/valid/test'e bölünüp kopyalanır
3. Zaten var olan dosya adları atlanır (tekrar import güvenli)

Kullanım:
    python3 scripts/labelstudio/merge_export.py ~/Downloads/project-1-at-2026-07-19-yolo
    python3 scripts/labelstudio/merge_export.py <klasor> --val-oran 0.1 --test-oran 0.1
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "roboflow_export"

# data.yaml ile aynı sıra
CLASS_ORDER = ["not", "ogrenci_numara"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def read_classes(export_dir: Path):
    """classes.txt -> {label_studio_index: sinif_adi}"""
    classes_file = export_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(
            f"classes.txt bulunamadı: {classes_file}\n"
            "Label Studio'dan 'YOLO' formatında export aldığınızdan emin olun."
        )

    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines()
             if line.strip()]

    bilinmeyen = [n for n in names if n not in CLASS_ORDER]
    if bilinmeyen:
        raise ValueError(
            f"Beklenmeyen sınıf(lar): {bilinmeyen}\n"
            f"Beklenen sınıflar: {CLASS_ORDER}\n"
            "Label Studio etiket yapılandırmasını kontrol edin (label_config.xml)."
        )

    return {index: name for index, name in enumerate(names)}


def choose_split(file_name: str, val_ratio: float, test_ratio: float):
    """Dosya adına göre deterministik bölme — aynı dosya hep aynı split'e gider."""
    digest = hashlib.md5(file_name.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF

    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "valid"
    return "train"


def remap_label_file(label_path: Path, index_to_name: dict):
    """Sınıf indekslerini projenin sırasına çevirir; satırları döndürür."""
    lines = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 5:
            print(f"  UYARI: bozuk satır atlandı ({label_path.name}): {raw}")
            continue

        ls_index = int(parts[0])
        name = index_to_name.get(ls_index)
        if name is None:
            print(f"  UYARI: bilinmeyen sınıf indeksi {ls_index} atlandı ({label_path.name})")
            continue

        parts[0] = str(CLASS_ORDER.index(name))
        lines.append(" ".join(parts))

    return lines


def main():
    parser = argparse.ArgumentParser(description="Label Studio YOLO export'unu veri setine kat")
    parser.add_argument("export", help="Label Studio YOLO export klasörü")
    parser.add_argument("--val-oran", type=float, default=0.1, help="Doğrulama oranı")
    parser.add_argument("--test-oran", type=float, default=0.1, help="Test oranı")
    parser.add_argument("--kuru", action="store_true",
                        help="Kuru çalıştırma: hiçbir dosya kopyalanmaz, sadece rapor")
    args = parser.parse_args()

    export_dir = Path(args.export).expanduser()
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"Beklenen klasörler yok: {images_dir} ve {labels_dir}")

    index_to_name = read_classes(export_dir)
    print(f"Label Studio sınıfları: {index_to_name}")
    print(f"Proje sırası          : {dict(enumerate(CLASS_ORDER))}")

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))
    if not images:
        raise SystemExit(f"Görsel bulunamadı: {images_dir}")

    sayac = {"train": 0, "valid": 0, "test": 0}
    atlanan = 0
    etiketsiz = 0

    for image_path in images:
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            etiketsiz += 1
            continue

        lines = remap_label_file(label_path, index_to_name)
        if not lines:
            etiketsiz += 1
            continue

        split = choose_split(image_path.name, args.val_oran, args.test_oran)
        target_image = DATASET_DIR / split / "images" / image_path.name
        target_label = DATASET_DIR / split / "labels" / (image_path.stem + ".txt")

        if target_image.exists():
            atlanan += 1
            continue

        if not args.kuru:
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)
            target_label.write_text("\n".join(lines) + "\n", encoding="utf-8")

        sayac[split] += 1

    print("\n" + "=" * 50)
    print("KURU ÇALIŞTIRMA (hiçbir şey kopyalanmadı)" if args.kuru else "BİRLEŞTİRME TAMAMLANDI")
    print("=" * 50)
    for split in ["train", "valid", "test"]:
        print(f"  {split}: +{sayac[split]}")
    print(f"  zaten mevcut (atlandı): {atlanan}")
    print(f"  etiketsiz görsel      : {etiketsiz}")

    if not args.kuru and sum(sayac.values()):
        print("\nSıradaki adım: YOLO'yu yeniden eğitin")
        print("  python3 training/train_yolo.py")


if __name__ == "__main__":
    main()
