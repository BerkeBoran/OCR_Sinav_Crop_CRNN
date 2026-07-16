#!/usr/bin/env python3
"""
Uçtan uca pipeline değerlendirmesi: tam sayfa test fotoğraflarında
YOLO tespit + CRNN okuma zincirinin gerçek başarısını ölçer.

CRNN'in tek başına doğruluğu (evaluate_model.py) el ile kırpılmış
görsellerde ölçülür; bu script ise ürünün gerçek metriğini verir:
"fotoğraf ver → iki alan da doğru okunsun" yüzdesi.

Kullanım:
    python3 training/evaluate_pipeline.py
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "roboflow_export" / "test" / "images"
LABELS_PATH = PROJECT_ROOT / "data" / "crnn_dataset" / "labels.csv"
SKIPPED_PATH = PROJECT_ROOT / "data" / "crnn_dataset" / "skipped.csv"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "evaluation"

FIELD_CLASSES = ["not", "ogrenci_numara"]

SPLIT_PREFIX_PATTERN = re.compile(r"^(train|valid|test)_")
RF_HASH_PATTERN = re.compile(r"\.rf\..*$")


def normalize_original_name(name: str) -> str:
    """Split öneki ve Roboflow hash'ini atarak orijinal görsel adını döndürür."""
    name = SPLIT_PREFIX_PATTERN.sub("", name)
    return RF_HASH_PATTERN.sub("", name)


def crop_key(file_name: str):
    """Kırpım dosya adından (orijinal_ad, alan_tipi) anahtarı üretir."""
    stem = Path(file_name).stem

    for field_type in FIELD_CLASSES:
        suffix = f"_{field_type}"
        if stem.endswith(suffix):
            return normalize_original_name(stem[: -len(suffix)]), field_type

    return None


def load_ground_truth():
    """labels.csv'den (orijinal_ad, alan_tipi) -> etiket haritası kurar."""
    ground_truth = {}

    with open(LABELS_PATH, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = crop_key(row["file_name"])
            if key is not None:
                ground_truth[key] = row["label"].strip()

    unreadable = set()
    if SKIPPED_PATH.exists():
        with open(SKIPPED_PATH, encoding="utf-8") as file:
            for row in csv.DictReader(file):
                key = crop_key(row["file_name"])
                if key is not None:
                    unreadable.add(key)

    return ground_truth, unreadable


def main():
    from tqdm import tqdm
    from inference.pipeline import ExamReader

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Test görselleri bulunamadı: {TEST_IMAGES_DIR}")

    ground_truth, unreadable = load_ground_truth()
    reader = ExamReader()

    image_paths = sorted(
        path for path in TEST_IMAGES_DIR.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    )

    print(f"Test fotoğrafı: {len(image_paths)}")

    field_stats = {
        field_type: {"toplam": 0, "dogru": 0, "tespit_yok": 0, "yanlis_okuma": 0}
        for field_type in FIELD_CLASSES
    }
    page_total = 0
    page_correct = 0
    no_gt = 0
    detail_rows = []

    for image_path in tqdm(image_paths, unit="kağıt"):
        original_name = normalize_original_name(image_path.stem)

        expected = {
            field_type: ground_truth.get((original_name, field_type))
            for field_type in FIELD_CLASSES
        }
        expected = {k: v for k, v in expected.items() if v is not None}

        # Etiketi olmayan veya okunamaz işaretlenen sayfalar ölçüme girmez
        if not expected:
            no_gt += 1
            continue

        result = reader.read(image_path)

        page_fields_ok = True
        page_total += 1

        for field_type, gt_value in expected.items():
            if (original_name, field_type) in unreadable:
                continue

            stats = field_stats[field_type]
            stats["toplam"] += 1

            read = result[field_type]
            predicted = read["deger"]

            if read["durum"] == "alan_bulunamadi":
                stats["tespit_yok"] += 1
                page_fields_ok = False
            elif predicted == gt_value:
                stats["dogru"] += 1
            else:
                stats["yanlis_okuma"] += 1
                page_fields_ok = False

            detail_rows.append({
                "dosya": image_path.name,
                "alan": field_type,
                "gercek": gt_value,
                "okunan": predicted,
                "okuma_guveni": f"{read['guven']:.2f}",
                "kutu_guveni": f"{read['kutu_guven']:.2f}",
                "durum": read["durum"],
                "dogru_mu": "evet" if predicted == gt_value else "hayir",
            })

        if page_fields_ok:
            page_correct += 1

    # Rapor
    print("\n" + "=" * 60)
    print("UÇTAN UCA PİPELİNE SONUÇLARI")
    print("=" * 60)

    for field_type in FIELD_CLASSES:
        stats = field_stats[field_type]
        if stats["toplam"] == 0:
            continue
        accuracy = stats["dogru"] / stats["toplam"] * 100
        print(f"\n{field_type} ({stats['toplam']} alan):")
        print(f"  Doğru okunan  : {stats['dogru']} (%{accuracy:.1f})")
        print(f"  Tespit edilemedi: {stats['tespit_yok']}")
        print(f"  Yanlış okundu : {stats['yanlis_okuma']}")

    if page_total:
        print(f"\nSAYFA BAZINDA: {page_correct}/{page_total} kağıtta tüm alanlar doğru "
              f"(%{page_correct / page_total * 100:.1f})")

    if no_gt:
        print(f"\nNot: {no_gt} fotoğrafın etiketi olmadığı için ölçüme girmedi.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = OUTPUT_DIR / f"pipeline_results_{timestamp}.csv"

    with open(detail_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"Detaylı sonuçlar: {detail_path}")


if __name__ == "__main__":
    main()
