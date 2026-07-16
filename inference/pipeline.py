"""
Uçtan uca sınav kağıdı okuma: YOLO alan tespiti + CRNN rakam okuma.

Kullanım:
    from inference.pipeline import ExamReader
    reader = ExamReader()
    sonuc = reader.read("kagit.jpg")
    print(sonuc["ogrenci_numara"]["deger"], sonuc["not"]["deger"])
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.detector import FieldDetector, FIELD_CLASSES
from inference.predictor import CRNNPredictor

# Bu eşiklerin altındaki okumalar "kontrol gerekli" olarak işaretlenir
MIN_READ_CONFIDENCE = 0.80
MIN_BOX_CONFIDENCE = 0.50


def validate_value(field_type, value):
    """Etiketleme aracıyla aynı kurallar: alan tipine göre değer geçerli mi?"""
    if not value or not value.isdigit():
        return False

    if field_type == "ogrenci_numara":
        return 5 <= len(value) <= 15

    if field_type == "not":
        return 0 <= int(value) <= 100

    return True


class ExamReader:
    """Tam sayfa fotoğraftan öğrenci numarası ve notu okuyan pipeline"""

    def __init__(self, detector=None, predictor=None):
        self.detector = detector or FieldDetector()
        self.predictor = predictor or CRNNPredictor()

    def read(self, image):
        """
        Args:
            image: PIL.Image veya dosya yolu

        Returns:
            {
                "ogrenci_numara": {"deger", "guven", "kutu_guven", "kutu", "kirpim", "durum"},
                "not": {...},
                "kontrol_gerekli": bool,
            }
            kutu: (x1, y1, x2, y2) tespit kutusu veya None
            kirpim: kırpılmış PIL görsel veya None
            durum: "ok" | "dusuk_guven" | "gecersiz_deger" | "alan_bulunamadi"
        """
        from PIL import ImageOps

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        # Kutu koordinatları ile görselin aynı yönde olması için burada da
        # exif_transpose uygulanır (detector zaten uyguluyor).
        image = ImageOps.exif_transpose(image)

        detections = self.detector.detect(image)
        crops = self.detector.crop_fields(image, detections)

        result = {"kontrol_gerekli": False}

        for field_type in FIELD_CLASSES:
            if field_type not in crops:
                result[field_type] = {
                    "deger": "",
                    "guven": 0.0,
                    "kutu_guven": 0.0,
                    "kutu": None,
                    "kirpim": None,
                    "durum": "alan_bulunamadi",
                }
                result["kontrol_gerekli"] = True
                continue

            crop, box_conf = crops[field_type]
            text, read_conf = self.predictor.predict(crop)

            if not validate_value(field_type, text):
                durum = "gecersiz_deger"
            elif read_conf < MIN_READ_CONFIDENCE or box_conf < MIN_BOX_CONFIDENCE:
                durum = "dusuk_guven"
            else:
                durum = "ok"

            result[field_type] = {
                "deger": text,
                "guven": read_conf,
                "kutu_guven": box_conf,
                "kutu": detections[field_type]["box"],
                "kirpim": crop,
                "durum": durum,
            }

            if durum != "ok":
                result["kontrol_gerekli"] = True

        return result


if __name__ == '__main__':
    # Hızlı komut satırı testi: python3 inference/pipeline.py <resim_yolu>
    if len(sys.argv) < 2:
        print("Kullanım: python3 inference/pipeline.py <resim_yolu>")
        sys.exit(1)

    reader = ExamReader()
    sonuc = reader.read(sys.argv[1])

    for field_type in FIELD_CLASSES:
        alan = sonuc[field_type]
        print(f"{field_type}: '{alan['deger']}' "
              f"(okuma güveni %{alan['guven'] * 100:.1f}, "
              f"kutu güveni %{alan['kutu_guven'] * 100:.1f}, durum: {alan['durum']})")

    if sonuc["kontrol_gerekli"]:
        print("⚠ Bu kağıt elle kontrol edilmeli.")
