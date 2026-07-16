#!/usr/bin/env python3
"""
Toplu sınav kağıdı okuma: klasördeki tüm fotoğrafları okuyup CSV çıkarır.

Kullanım:
    python3 scripts/read_exams.py <klasör veya resim yolları...>
    python3 scripts/read_exams.py sinav_fotolari/ --cikti sonuclar.csv

Çıktı CSV sütunları:
    dosya, ogrenci_numara, ogrenci_numara_guven, not, not_guven, durum

durum "ok" olmayan satırlar elle kontrol edilmelidir.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def collect_images(paths):
    images = []

    for raw_path in paths:
        path = Path(raw_path)

        if path.is_dir():
            for extension in IMAGE_EXTENSIONS:
                images.extend(sorted(path.glob(f"*{extension}")))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            print(f"Uyarı: atlandı (resim değil veya bulunamadı): {path}")

    return images


def main():
    parser = argparse.ArgumentParser(description="Sınav kağıdı fotoğraflarını toplu oku")
    parser.add_argument("girdiler", nargs="+", help="Klasör veya resim dosyaları")
    parser.add_argument("--cikti", default="sonuclar.csv", help="Çıktı CSV yolu (varsayılan: sonuclar.csv)")
    args = parser.parse_args()

    images = collect_images(args.girdiler)

    if not images:
        print("Okunacak resim bulunamadı.")
        sys.exit(1)

    print(f"{len(images)} fotoğraf okunacak...")

    from tqdm import tqdm
    from inference.pipeline import ExamReader

    reader = ExamReader()

    rows = []
    kontrol_gereken = 0

    for image_path in tqdm(images, unit="kağıt"):
        try:
            sonuc = reader.read(image_path)
        except Exception as error:
            rows.append({
                "dosya": image_path.name,
                "ogrenci_numara": "",
                "ogrenci_numara_guven": "",
                "not": "",
                "not_guven": "",
                "durum": f"hata: {error}",
            })
            kontrol_gereken += 1
            continue

        numara = sonuc["ogrenci_numara"]
        not_alani = sonuc["not"]

        if sonuc["kontrol_gerekli"]:
            sorunlar = [
                f"{alan}: {sonuc[alan]['durum']}"
                for alan in ["ogrenci_numara", "not"]
                if sonuc[alan]["durum"] != "ok"
            ]
            durum = " | ".join(sorunlar)
            kontrol_gereken += 1
        else:
            durum = "ok"

        rows.append({
            "dosya": image_path.name,
            "ogrenci_numara": numara["deger"],
            "ogrenci_numara_guven": f"{numara['guven']:.2f}",
            "not": not_alani["deger"],
            "not_guven": f"{not_alani['guven']:.2f}",
            "durum": durum,
        })

    output_path = Path(args.cikti)
    fieldnames = ["dosya", "ogrenci_numara", "ogrenci_numara_guven", "not", "not_guven", "durum"]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSonuçlar kaydedildi: {output_path}")
    print(f"Toplam: {len(rows)} | Sorunsuz: {len(rows) - kontrol_gereken} | Elle kontrol gereken: {kontrol_gereken}")

    if kontrol_gereken:
        print("\nElle kontrol gerekenler:")
        for row in rows:
            if row["durum"] != "ok":
                print(f"  - {row['dosya']}: {row['durum']}")


if __name__ == "__main__":
    main()
