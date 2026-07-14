from pathlib import Path
import cv2
import yaml
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "roboflow_export"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cropped_fields"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

CROP_PADDING = 8

EXPECTED_CLASSES = ["not", "ogrenci_numara"]


def ensure_dirs():
    for class_name in EXPECTED_CLASSES:
        (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_class_names():
    data_yaml_path = DATASET_DIR / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml bulunamadı: {data_yaml_path}")

    with open(data_yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names")

    if names is None:
        raise ValueError("data.yaml içinde 'names' alanı bulunamadı.")

    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(key): value for key, value in names.items()}

    raise ValueError("data.yaml içindeki names formatı desteklenmiyor.")


def find_image_file(images_dir, file_stem):
    for extension in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{file_stem}{extension}"

        if image_path.exists():
            return image_path

    return None


def yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height, padding=8):
    x_center_pixel = x_center * image_width
    y_center_pixel = y_center * image_height
    box_width_pixel = width * image_width
    box_height_pixel = height * image_height

    x1 = int(x_center_pixel - box_width_pixel / 2) - padding
    y1 = int(y_center_pixel - box_height_pixel / 2) - padding
    x2 = int(x_center_pixel + box_width_pixel / 2) + padding
    y2 = int(y_center_pixel + box_height_pixel / 2) + padding

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_width, x2)
    y2 = min(image_height, y2)

    return x1, y1, x2, y2


def read_yolo_label_file(label_path, class_names, image_width, image_height):
    detections = []

    with open(label_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):
        parts = line.strip().split()

        if len(parts) != 5:
            print(f"Uyarı: Hatalı label satırı atlandı: {label_path.name}, satır {line_number}")
            continue

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        class_name = class_names.get(class_id)

        if class_name not in EXPECTED_CLASSES:
            print(f"Uyarı: Beklenmeyen class atlandı: {class_name} - {label_path.name}")
            continue

        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center=x_center,
            y_center=y_center,
            width=width,
            height=height,
            image_width=image_width,
            image_height=image_height,
            padding=CROP_PADDING
        )

        area = (x2 - x1) * (y2 - y1)

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "box": [x1, y1, x2, y2],
            "area": area
        })

    return detections


def select_one_box_per_class(detections):
    selected = {}
    warnings = []

    for class_name in EXPECTED_CLASSES:
        class_detections = [
            detection for detection in detections
            if detection["class_name"] == class_name
        ]

        if len(class_detections) == 0:
            warnings.append(f"{class_name} bulunamadı.")
            continue

        if len(class_detections) > 1:
            warnings.append(f"{class_name} için {len(class_detections)} kutu bulundu. En büyük kutu seçildi.")

        best_detection = max(class_detections, key=lambda item: item["area"])
        selected[class_name] = best_detection

    return selected, warnings


def crop_and_save(image, selected_boxes, image_stem, split_name):
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
        output_path = OUTPUT_DIR / class_name / output_file_name

        success = cv2.imwrite(str(output_path), crop)

        saved_files.append({
            "class_name": class_name,
            "status": "saved" if success else "write_failed",
            "output_path": str(output_path)
        })

    return saved_files


def process_split(split_name, class_names, report_rows):
    images_dir = DATASET_DIR / split_name / "images"
    labels_dir = DATASET_DIR / split_name / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"{split_name} klasörü bulunamadı, atlandı.")
        return

    label_files = sorted(labels_dir.glob("*.txt"))

    print(f"\n{split_name} işleniyor...")
    print(f"Label dosyası sayısı: {len(label_files)}")

    for label_path in label_files:
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

        saved_files = crop_and_save(
            image=image,
            selected_boxes=selected_boxes,
            image_stem=label_path.stem,
            split_name=split_name
        )

        status = "ok" if len(selected_boxes) == len(EXPECTED_CLASSES) else "missing_class"

        report_rows.append({
            "split": split_name,
            "file_stem": label_path.stem,
            "status": status,
            "warning": " | ".join(warnings),
            "saved_files": " | ".join([item["output_path"] for item in saved_files])
        })


def save_report(report_rows):
    report_path = LOG_DIR / "crop_report.csv"

    fieldnames = ["split", "file_stem", "status", "warning", "saved_files"]

    with open(report_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nRapor kaydedildi: {report_path}")


def count_outputs():
    print("\nCrop çıktı sayıları:")

    for class_name in EXPECTED_CLASSES:
        class_dir = OUTPUT_DIR / class_name
        image_count = len(list(class_dir.glob("*.jpg")))

        print(f"{class_name}: {image_count}")


def main():
    ensure_dirs()

    class_names = load_class_names()

    print("Class eşleşmesi:")
    for class_id, class_name in class_names.items():
        print(f"{class_id} -> {class_name}")

    report_rows = []

    for split_name in ["train", "valid", "test"]:
        process_split(split_name, class_names, report_rows)

    save_report(report_rows)
    count_outputs()

    print("\nCrop işlemi tamamlandı.")


if __name__ == "__main__":
    main()