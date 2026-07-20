#!/usr/bin/env python3
"""
Label Studio için görselleri hazırlar ve mevcut YOLO modeliyle ÖN-ETİKETLER.

Ekip sıfırdan kutu çizmek yerine, modelin önerdiği kutuları düzeltir —
yeni şablonlarda bile bu çok daha hızlıdır.

Yaptıkları:
1. HEIC/JPG/PNG görselleri okur, EXIF yönünü UYGULAYIP JPG olarak kaydeder
   (yön bilgisi dosyaya gömülür; tarayıcı ile model aynı yönü görür)
2. models/yolo_fields.pt ile alanları tespit eder
3. Label Studio'ya aktarılacak tasks.json dosyasını üretir

Kullanım:
    python3 scripts/labelstudio/prepare_tasks.py "/path/to/Şablon_1" --parti sablon_1

Çıktılar:
    data/labeling_images/<parti>/*.jpg        (Label Studio'nun sunduğu görseller)
    data/labelstudio_tasks/<parti>.json       (Label Studio'ya import edilecek dosya)
"""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_DIR = PROJECT_ROOT / "data" / "labeling_images"
TASK_DIR = PROJECT_ROOT / "data" / "labelstudio_tasks"

INPUT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

# label_config.xml ve data.yaml ile aynı sıra
FIELD_CLASSES = ["not", "ogrenci_numara"]

register_heif_opener()


def collect_images(root: Path):
    if root.is_file():
        return [root] if root.suffix.lower() in INPUT_EXTENSIONS else []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in INPUT_EXTENSIONS and not p.name.startswith(".")
    )


def normalize_image(source: Path, target: Path, max_size: int):
    """EXIF yönünü uygulayıp (gömerek) JPG kaydeder. Zaten varsa atlar."""
    if target.exists():
        with Image.open(target) as existing:
            return existing.size

    image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")

    if max_size and max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "JPEG", quality=92)
    return image.size


def to_prediction(detections, width, height):
    """Tespit kutularını Label Studio 'prediction' formatına çevirir.
    Label Studio koordinatları YÜZDE cinsindendir (0-100)."""
    results = []
    scores = []

    for class_name in FIELD_CLASSES:
        if class_name not in detections:
            continue

        x1, y1, x2, y2 = detections[class_name]["box"]
        conf = detections[class_name]["conf"]
        scores.append(conf)

        results.append({
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": width,
            "original_height": height,
            "image_rotation": 0,
            "value": {
                "x": max(0.0, x1 / width * 100),
                "y": max(0.0, y1 / height * 100),
                "width": min(100.0, (x2 - x1) / width * 100),
                "height": min(100.0, (y2 - y1) / height * 100),
                "rotation": 0,
                "rectanglelabels": [class_name],
            },
        })

    if not results:
        return None

    return {
        "model_version": "yolo_fields",
        "score": sum(scores) / len(scores),
        "result": results,
    }


def load_detector():
    """Detektörü yükler; model yoksa None döner (ön-etiketsiz devam edilir)."""
    try:
        from inference.detector import FieldDetector
        return FieldDetector()
    except FileNotFoundError:
        print("UYARI: models/yolo_fields.pt bulunamadı — ön-etiketleme yapılmayacak.")
        return None
    except Exception as error:
        print(f"UYARI: Detektör yüklenemedi ({error}) — ön-etiketleme yapılmayacak.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Label Studio görev dosyası hazırla")
    parser.add_argument("girdi", help="Görsellerin bulunduğu klasör (veya tek dosya)")
    parser.add_argument("--parti", required=True,
                        help="Parti adı, ör. sablon_1 (klasör ve görev dosyası adı olur)")
    parser.add_argument("--max-boyut", type=int, default=1600,
                        help="Uzun kenar üst sınırı (0 = küçültme)")
    parser.add_argument("--on-etiket", dest="on_etiket", action="store_true", default=True,
                        help="Mevcut YOLO modeliyle ön-etiketle (varsayılan)")
    parser.add_argument("--on-etiket-yok", dest="on_etiket", action="store_false",
                        help="Ön-etiketleme yapma, boş görevler üret")
    args = parser.parse_args()

    root = Path(args.girdi).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {root}")

    sources = collect_images(root)
    if not sources:
        raise SystemExit(f"Görsel bulunamadı: {root}")

    print(f"{len(sources)} görsel bulundu -> parti '{args.parti}'")

    detector = load_detector() if args.on_etiket else None

    out_dir = IMAGE_DIR / args.parti
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    on_etiketli = 0

    for index, source in enumerate(sources, start=1):
        target = out_dir / (source.stem + ".jpg")
        width, height = normalize_image(source, target, args.max_boyut)

        # Label Studio yerel dosya sunumu: document root'a göre GÖRECELİ yol
        relative = f"{args.parti}/{target.name}"
        task = {"data": {"image": f"/data/local-files/?d={quote(relative)}"}}

        if detector is not None:
            prediction = to_prediction(detector.detect(target), width, height)
            if prediction is not None:
                task["predictions"] = [prediction]
                on_etiketli += 1

        tasks.append(task)

        if index % 25 == 0:
            print(f"  {index}/{len(sources)} işlendi")

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    task_file = TASK_DIR / f"{args.parti}.json"
    task_file.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGörseller : {out_dir}")
    print(f"Görev dosyası: {task_file}")
    if detector is not None:
        print(f"Ön-etiketlenen: {on_etiketli}/{len(tasks)} "
              f"(kalanına elle kutu çizilecek)")
    print("\nSıradaki adım: Label Studio'da projeye bu JSON'u import edin (LABELING.md).")


if __name__ == "__main__":
    main()
