"""
YOLOv8 ile sınav kağıdı fotoğrafında alan tespiti (not, ogrenci_numara).

Kullanım:
    from inference.detector import FieldDetector
    detector = FieldDetector()
    kirpimlar = detector.crop_fields(pil_image)   # {"not": (crop, conf), ...}
"""

import sys
from pathlib import Path

from PIL import Image

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'yolo_fields.pt'

# crop_fields.py ile aynı değer: CRNN bu kırpım stiliyle eğitildi,
# farklı padding kullanmak okuma doğruluğunu sessizce düşürür.
CROP_PADDING = 8

FIELD_CLASSES = ["not", "ogrenci_numara"]


class FieldDetector:
    """Tam sayfa görüntüde not ve öğrenci numarası alanlarını bulan detektör"""

    def __init__(self, model_path=None, conf_threshold=0.25, device=None):
        from ultralytics import YOLO

        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {model_path}\n"
                "Önce detektörü eğitin: python3 training/train_yolo.py"
            )

        self.model = YOLO(str(model_path))
        self.class_names = self.model.names
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, image):
        """
        Her alan sınıfı için en yüksek güvenli kutuyu döndürür.

        Args:
            image: PIL.Image veya dosya yolu

        Returns:
            {class_name: {"box": (x1, y1, x2, y2), "conf": float}}
            Bulunamayan sınıflar sonuçta yer almaz.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        best = {}
        for box in results[0].boxes:
            class_name = self.class_names[int(box.cls)]

            if class_name not in FIELD_CLASSES:
                continue

            conf = float(box.conf)
            if class_name not in best or conf > best[class_name]["conf"]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                best[class_name] = {"box": (x1, y1, x2, y2), "conf": conf}

        return best

    def crop_fields(self, image, detections=None):
        """
        Tespit edilen alanları eğitim kırpımlarıyla aynı padding'le keser.

        Returns:
            {class_name: (PIL.Image, conf)}
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        if detections is None:
            detections = self.detect(image)

        width, height = image.size
        crops = {}

        for class_name, detection in detections.items():
            x1, y1, x2, y2 = detection["box"]

            x1 = max(0, int(x1) - CROP_PADDING)
            y1 = max(0, int(y1) - CROP_PADDING)
            x2 = min(width, int(x2) + CROP_PADDING)
            y2 = min(height, int(y2) + CROP_PADDING)

            if x2 <= x1 or y2 <= y1:
                continue

            crops[class_name] = (image.crop((x1, y1, x2, y2)), detection["conf"])

        return crops


if __name__ == '__main__':
    # Hızlı komut satırı testi: python3 inference/detector.py <resim_yolu>
    if len(sys.argv) < 2:
        print("Kullanım: python3 inference/detector.py <resim_yolu>")
        sys.exit(1)

    detector = FieldDetector()
    detections = detector.detect(sys.argv[1])

    if not detections:
        print("Hiç alan bulunamadı.")
    for class_name, detection in detections.items():
        x1, y1, x2, y2 = detection["box"]
        print(f"{class_name}: güven %{detection['conf'] * 100:.1f} "
              f"kutu ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
