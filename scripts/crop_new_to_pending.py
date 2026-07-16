#!/usr/bin/env python3
"""
Roboflow export içindeki YENİ görselleri kontrol klasörüne (pending) crop'lar.

Bu script:
1. data/cropped_fields/ ve data/cropped_fields_pending/ içindeki mevcut crop'lara bakar
2. Roboflow her export'ta dosya adına farklı hash eklediği için (.rf.XXXX),
   karşılaştırmayı hash'i atarak orijinal görsel adına göre yapar
3. Daha önce hiç crop'lanmamış görselleri data/cropped_fields_pending/ altına crop'lar

Sonraki adımlar:
- Etiketleme sitesinde "Yeni Veriler (Kontrol Bekleyen)" seçilerek etiketlenir
- Etiketleme bitince scripts/merge_pending_data.py ile mevcut verilere birleştirilir
"""

import csv
import re
from pathlib import Path

import cv2

from crop_fields import (
    load_class_names,
    find_image_file,
    read_yolo_label_file,
    select_one_box_per_class,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "roboflow_export"
CROPPED_DIR = PROJECT_ROOT / "data" / "cropped_fields"
PENDING_DIR = PROJECT_ROOT / "data" / "cropped_fields_pending"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

EXPECTED_CLASSES = ["not", "ogrenci_numara"]

SPLIT_PREFIX_PATTERN = re.compile(r"^(train|valid|test)_")
RF_HASH_PATTERN = re.compile(r"\.rf\..*$")


def normalize_original_name(file_name: str) -> str:
    """Split öneki ve Roboflow hash'ini atarak orijinal görsel adını döndürür."""
    name = SPLIT_PREFIX_PATTERN.sub("", file_name)
    return RF_HASH_PATTERN.sub("", name)


def collect_existing_originals() -> set[str]:
    """Mevcut crop'larda (main + pending) temsil edilen orijinal görsel adları."""
    originals = set()

    for base_dir in [CROPPED_DIR, PENDING_DIR]:
        for class_name in EXPECTED_CLASSES:
            class_dir = base_dir / class_name

            if not class_dir.exists():
                continue

            for image_path in class_dir.glob("*.jpg"):
                originals.add(normalize_original_name(image_path.name))

    return originals


def crop_and_save_to_pending(image, selected_boxes, image_stem, split_name):
    saved_files = []

    for class_name, detection in selected_boxes.items():
        x1, y1, x2, y2 = detection["box"]

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            saved_files.append({
                "class_name": class_name,
                "status": "empty_crop",
                "output_path": ""
            })
            continue

        output_file_name = f"{split_name}_{image_stem}_{class_name}.jpg"
        output_path = PENDING_DIR / class_name / output_file_name

        success = cv2.imwrite(str(output_path), crop)

        saved_files.append({
            "class_name": class_name,
            "status": "saved" if success else "write_failed",
            "output_path": str(output_path)
        })

    return saved_files


def process_split(split_name, class_names, existing_originals, report_rows):
    images_dir = DATASET_DIR / split_name / "images"
    labels_dir = DATASET_DIR / split_name / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"{split_name} klasörü bulunamadı, atlandı.")
        return 0

    label_files = sorted(labels_dir.glob("*.txt"))
    new_count = 0

    print(f"\n{split_name} işleniyor... (label dosyası: {len(label_files)})")

    for label_path in label_files:
        original_name = normalize_original_name(label_path.stem)

        if original_name in existing_originals:
            continue

        image_path = find_image_file(images_dir, label_path.stem)

        if image_path is None:
            report_rows.append({
                "split": split_name,
                "file_stem": label_path.stem,
                "status": "image_not_found",
                "warning": "Görsel bulunamadı."
            })
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            report_rows.append({
                "split": split_name,
                "file_stem": label_path.stem,
                "status": "image_read_failed",
                "warning": "Görsel okunamadı."
            })
            continue

        image_height, image_width = image.shape[:2]

        detections = read_yolo_label_file(
            label_path=label_path,
            class_names=class_names,
            image_width=image_width,
            image_height=image_height
        )

        selected_boxes, warnings = select_one_box_per_class(detections)

        saved_files = crop_and_save_to_pending(
            image=image,
            selected_boxes=selected_boxes,
            image_stem=label_path.stem,
            split_name=split_name
        )

        # Aynı orijinal görsel başka bir splitte de varsa tekrar crop'lanmasın
        existing_originals.add(original_name)
        new_count += 1

        status = "ok" if len(selected_boxes) == len(EXPECTED_CLASSES) else "missing_class"

        report_rows.append({
            "split": split_name,
            "file_stem": label_path.stem,
            "status": status,
            "warning": " | ".join(warnings),
            "saved_files": " | ".join([item["output_path"] for item in saved_files])
        })

    print(f"{split_name}: {new_count} yeni görsel crop'landı")
    return new_count


def save_report(report_rows):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOG_DIR / "crop_pending_report.csv"

    fieldnames = ["split", "file_stem", "status", "warning", "saved_files"]

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nRapor kaydedildi: {report_path}")


def count_pending():
    print("\nKontrol bekleyen crop sayıları:")

    for class_name in EXPECTED_CLASSES:
        class_dir = PENDING_DIR / class_name
        image_count = len(list(class_dir.glob("*.jpg"))) if class_dir.exists() else 0
        print(f"{class_name}: {image_count}")


def main():
    for class_name in EXPECTED_CLASSES:
        (PENDING_DIR / class_name).mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()

    existing_originals = collect_existing_originals()
    print(f"Mevcut crop'larda temsil edilen orijinal görsel: {len(existing_originals)}")

    report_rows = []
    total_new = 0

    for split_name in ["train", "valid", "test"]:
        total_new += process_split(split_name, class_names, existing_originals, report_rows)

    save_report(report_rows)
    count_pending()

    print(f"\nToplam {total_new} yeni görsel pending klasörüne crop'landı.")
    print("Sıradaki adım: etiketleme sitesinde 'Yeni Veriler (Kontrol Bekleyen)' seçip doğrulayın.")


if __name__ == "__main__":
    main()
