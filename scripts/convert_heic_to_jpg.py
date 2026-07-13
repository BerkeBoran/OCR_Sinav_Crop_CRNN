from pathlib import Path
import csv
from PIL import Image
from pillow_heif import register_heif_opener


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "roboflow_export"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

SPLITS = ["train", "valid", "test"]

HEIC_EXTENSIONS = [".heic", ".heif", ".HEIC", ".HEIF"]


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def convert_heic_to_jpg(heic_path):
    """
    HEIC/HEIF dosyasını aynı klasörde JPG olarak kaydeder.
    Orijinal HEIC dosyasını silmez.
    """

    output_path = heic_path.with_suffix(".jpg")

    if output_path.exists():
        return {
            "status": "skipped",
            "message": "JPG zaten mevcut.",
            "source_path": str(heic_path),
            "output_path": str(output_path)
        }

    try:
        image = Image.open(heic_path)

        # JPG formatı alpha kanalını sevmez, bu yüzden RGB'ye çeviriyoruz.
        image = image.convert("RGB")

        image.save(output_path, "JPEG", quality=95)

        return {
            "status": "converted",
            "message": "HEIC dosyası JPG'ye çevrildi.",
            "source_path": str(heic_path),
            "output_path": str(output_path)
        }

    except Exception as error:
        return {
            "status": "failed",
            "message": str(error),
            "source_path": str(heic_path),
            "output_path": ""
        }


def find_heic_files():
    heic_files = []

    for split in SPLITS:
        images_dir = DATASET_DIR / split / "images"

        if not images_dir.exists():
            print(f"{split}/images klasörü bulunamadı, atlandı.")
            continue

        for extension in HEIC_EXTENSIONS:
            heic_files.extend(images_dir.glob(f"*{extension}"))

    return sorted(set(heic_files))


def save_report(rows):
    report_path = LOG_DIR / "heic_conversion_report.csv"

    fieldnames = ["status", "message", "source_path", "output_path"]

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDönüştürme raporu kaydedildi:")
    print(report_path)


def main():
    register_heif_opener()
    ensure_dirs()

    heic_files = find_heic_files()

    print("HEIC/HEIF dosya taraması tamamlandı.")
    print(f"Bulunan HEIC/HEIF dosya sayısı: {len(heic_files)}")

    if len(heic_files) == 0:
        print("Dönüştürülecek HEIC/HEIF dosyası bulunamadı.")
        return

    rows = []

    converted_count = 0
    skipped_count = 0
    failed_count = 0

    for heic_path in heic_files:
        result = convert_heic_to_jpg(heic_path)
        rows.append(result)

        if result["status"] == "converted":
            converted_count += 1
        elif result["status"] == "skipped":
            skipped_count += 1
        elif result["status"] == "failed":
            failed_count += 1

        print(f"{result['status']}: {heic_path.name}")

    save_report(rows)

    print("\nÖzet:")
    print(f"Dönüştürülen: {converted_count}")
    print(f"Zaten mevcut olduğu için atlanan: {skipped_count}")
    print(f"Başarısız: {failed_count}")

    print("\nHEIC → JPG dönüştürme işlemi tamamlandı.")


if __name__ == "__main__":
    main()