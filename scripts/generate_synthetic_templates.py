#!/usr/bin/env python3
"""
Sentetik sınav şablonu üreticisi.

Elimizdeki gerçek el yazısı kırpımlarını (data/cropped_fields/not ve
ogrenci_numara) programatik olarak üretilen ÇEŞİTLİ sahte sınav kağıdı
şablonlarına yapıştırır ve YOLO etiketini otomatik çıkarır.

Amaç: YOLO detektörü "alanın sayfadaki konumunu" ezberlemek yerine
"alanın nasıl göründüğünü" öğrensin — böylece hiç görmediği şablonlarda
da not ve öğrenci numarası alanlarını bulabilsin.

Kullanım:
    python3 scripts/generate_synthetic_templates.py --sayi 1000
    python3 scripts/generate_synthetic_templates.py --sayi 1000 --seed 42

Çıktı (YOLO formatı):
    data/synthetic_templates/
        train/images/*.jpg,  train/labels/*.txt
        valid/images/*.jpg,  valid/labels/*.txt
        data.yaml
"""

import argparse
import os
import random
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CROPPED_DIR = PROJECT_ROOT / "data" / "cropped_fields"
OUTPUT_DIR = PROJECT_ROOT / "data" / "synthetic_templates"

# data.yaml ile aynı sıra: not=0, ogrenci_numara=1
CLASS_IDS = {"not": 0, "ogrenci_numara": 1}

# Detektörün yok saymayı öğrenmesi için sahte (dikkat dağıtıcı) basılı başlıklar.
# Kırpımlar kendi 'Not:' / 'Öğrenci No:' etiketlerini zaten içerdiğinden burada
# alan etiketleri tekrar edilmez; bunlar yalnızca çevre metnidir.
DECOY_LABELS = ["Ad Soyad", "Adı Soyadı", "Sınıf", "Şube", "Ders", "Tarih",
                "İmza", "Okul", "Dönem", "Öğretmen", "Bölüm", "Süre", "Puan Dağılımı"]

FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Trebuchet MS.ttf",
    # Linux / Colab
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]


def discover_fonts():
    """Sistemde bulunan başlık fontlarını toplar (macOS + Linux)."""
    fonts = [path for path in FONT_CANDIDATES if os.path.exists(path)]

    if not fonts:
        # Son çare: matplotlib ile gelen DejaVu
        try:
            import matplotlib
            fonts = [str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")]
        except Exception:
            fonts = []

    if not fonts:
        raise RuntimeError(
            "Kullanılabilir TrueType font bulunamadı. "
            "Linux'ta: apt-get install fonts-dejavu"
        )

    return fonts


def load_crops():
    """Her sınıf için kırpım dosya yollarını toplar."""
    crops = {}
    for class_name in CLASS_IDS:
        files = sorted((CROPPED_DIR / class_name).glob("*.jpg"))
        if not files:
            raise FileNotFoundError(
                f"Kırpım bulunamadı: {CROPPED_DIR / class_name}\n"
                "Önce crop_fields.py çalıştırılmalı."
            )
        crops[class_name] = files
    return crops


def random_paper_color():
    """Hafif kirli beyaz / krem tonları."""
    base = random.randint(238, 255)
    tint = random.randint(-6, 4)
    return (
        max(230, min(255, base)),
        max(230, min(255, base + tint)),
        max(225, min(255, base + tint - random.randint(0, 6))),
    )


def random_ink_color():
    """Basılı metin için koyu tonlar (siyah, lacivert, koyu gri)."""
    return random.choice([
        (20, 20, 20), (10, 10, 10), (30, 30, 60), (40, 40, 40), (15, 25, 70),
    ])


def get_font(font_paths, size):
    for _ in range(5):
        try:
            return ImageFont.truetype(random.choice(font_paths), size)
        except Exception:
            continue
    return ImageFont.load_default()


def pick_usable_crop(files, max_tries=5):
    """Çok karanlık/gölgeli kırpımları eleyerek kullanılabilir bir kırpım seçer.
    Bu tür kırpımlar sayfada kara leke bırakır."""
    for _ in range(max_tries):
        path = random.choice(files)
        gray = Image.open(path).convert("L")
        # Ortalama parlaklık düşükse (gölgeli tarama) başka kırpım dene
        histogram = gray.histogram()
        total = sum(histogram) or 1
        mean = sum(i * c for i, c in enumerate(histogram)) / total
        if mean >= 135:
            return path
    return path  # hepsi karanlıksa sonuncuyu döndür


def white_point_value(gray_image, percentile=0.55):
    """Histogramdan beyaz-nokta değeri: bu değerin üstü beyaza haritalanır.
    Koyu pikselleri saf siyaha ezmeden zemini temizlemek için kullanılır."""
    histogram = gray_image.histogram()
    total = sum(histogram)
    if total == 0:
        return 255
    hedef = total * percentile
    biriken = 0
    for value, count in enumerate(histogram):
        biriken += count
        if biriken >= hedef:
            # En az orta-parlaklıkta bir beyaz-nokta garanti et
            return max(120, value)
    return 255


def composite_handwriting(page, crop_path, box):
    """
    Kırpımı verilen kutuya (x1,y1,x2,y2 - üst sınır) sığdırıp multiply ile
    harmanlar: beyaz zemin sayfayı karartmaz, sadece koyu yazı görünür.
    Gerçek yapıştırılan sıkı kutuyu (YOLO etiketi için) döndürür.
    """
    x1, y1, max_w, max_h = box
    crop = Image.open(crop_path).convert("RGB")

    # Zemini beyaza çek ama koyuları EZME: sadece yüksek uçtan gerdiriyoruz.
    # autocontrast koyu komşu-satır kalıntılarını saf siyaha çevirip sayfada
    # çirkin siyah bantlar bırakıyordu; beyaz-nokta germesi bunu önler.
    gray = crop.convert("L")
    white_point = white_point_value(gray, percentile=0.55)
    lut = [min(255, round(i * 255.0 / max(1, white_point))) for i in range(256)]
    gray = gray.point(lut)
    crop = Image.merge("RGB", (gray, gray, gray))

    cw, ch = crop.size
    scale = min(max_w / cw, max_h / ch) * random.uniform(0.82, 0.98)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    crop = crop.resize((nw, nh), Image.LANCZOS)

    # Kutu içinde hafif rastgele konum
    px = x1 + random.randint(0, max(0, max_w - nw))
    py = y1 + random.randint(0, max(0, max_h - nh))

    region = page.crop((px, py, px + nw, py + nh))
    blended = ImageChops.multiply(region, crop)
    page.paste(blended, (px, py))

    return (px, py, px + nw, py + nh)


def draw_label(draw, font, text, xy, color):
    draw.text(xy, text, font=font, fill=color)
    bbox = draw.textbbox(xy, text, font=font)
    return bbox  # (x1, y1, x2, y2)


def render_field(page, draw, crop_path, slot, ink):
    """
    Bir alanı (kırpımı) verilen bölgeye yerleştirir.

    Kırpımlar zaten kendi basılı etiketini ("Not:", "Öğrenci No:" vb.) ve
    kutusunu içerdiği için yanına ayrıca etiket ÇİZİLMEZ — bu tekrar olur.
    Çeşitlilik konumdan, zeminden, ölçekten ve kırpımların kendi doğal
    etiket çeşitliliğinden gelir. Ara sıra alana çerçeve/alt çizgi eklenir.

    slot = (x, y, genislik, yukseklik). Döndürür: sıkı kutu (x1,y1,x2,y2).
    """
    x, y, w, h = slot
    tight = composite_handwriting(page, crop_path, (x, y, w, h))
    x1, y1, x2, y2 = tight

    style = random.random()
    if style < 0.25:
        # Çerçeve
        pad = random.randint(4, 12)
        draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad],
                       outline=ink, width=random.randint(1, 2))
    elif style < 0.55:
        # Alt çizgi
        draw.line([(x1, y2 + random.randint(2, 6)), (x2, y2 + random.randint(2, 6))],
                  fill=ink, width=random.randint(1, 2))

    return tight


def render_page(fonts, crops, size):
    """Bir sentetik sınav kağıdı üretir. (image, [(cls, x1,y1,x2,y2), ...])"""
    W, H = size
    page = Image.new("RGB", (W, H), random_paper_color())
    draw = ImageDraw.Draw(page)
    ink = random_ink_color()

    boxes = []
    margin = random.randint(40, 90)
    title_font = get_font(fonts, random.randint(30, 46))
    body_font = get_font(fonts, random.randint(22, 34))

    # Üst başlık (ör. okul / sınav adı) — dikkat dağıtıcı basılı metin
    if random.random() < 0.8:
        title = random.choice([
            "SINAV KAĞIDI", "YAZILI SINAVI", "DEĞERLENDİRME FORMU",
            "1. DÖNEM 2. YAZILI", "ORTAK SINAV", "MATEMATİK SINAVI",
        ])
        tb = draw.textbbox((margin, margin // 2), title, font=title_font)
        draw.text(((W - (tb[2] - tb[0])) // 2, margin // 2), title, font=title_font, fill=ink)

    # Bazı sahte basılı satırlar (Ad Soyad, Sınıf, Tarih...)
    y_cursor = margin + random.randint(60, 110)
    for _ in range(random.randint(1, 4)):
        decoy = random.choice(DECOY_LABELS)
        draw.text((margin, y_cursor), f"{decoy}: " + "_" * random.randint(10, 24),
                  font=body_font, fill=ink)
        y_cursor += random.randint(40, 70)

    # Öğrenci numarası ve not alanlarını yerleştir.
    # Konumları BİLİNÇLİ olarak farklı düzenlerde dağıtıyoruz.
    field_specs = [
        ("ogrenci_numara", pick_usable_crop(crops["ogrenci_numara"])),
        ("not", pick_usable_crop(crops["not"])),
    ]
    random.shuffle(field_specs)

    placements = random.choice(["stacked", "corners", "spread"])

    for i, (class_name, crop_path) in enumerate(field_specs):
        if placements == "stacked":
            slot = (margin, y_cursor, W - 2 * margin, random.randint(70, 120))
            y_cursor += random.randint(90, 150)
        elif placements == "corners":
            # numara üst-sağ, not alt-sağ gibi köşelere
            if i == 0:
                slot = (W // 2, margin + random.randint(80, 140),
                        W // 2 - margin, random.randint(70, 110))
            else:
                slot = (W // 2, H - margin - random.randint(120, 220),
                        W // 2 - margin, random.randint(70, 110))
        else:  # spread
            slot = (random.randint(margin, W // 2),
                    random.randint(y_cursor, H - margin - 160),
                    random.randint(W // 3, W - 2 * margin),
                    random.randint(70, 120))

        tight = render_field(page, draw, crop_path, slot, ink)
        boxes.append((CLASS_IDS[class_name], *tight))

    # Hafif gerçekçilik: gürültü/bulanıklık/parlaklık
    if random.random() < 0.4:
        page = page.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    if random.random() < 0.3:
        page = ImageOps.autocontrast(page, cutoff=1)

    return page, boxes


def to_yolo_line(cls, x1, y1, x2, y2, W, H):
    xc = (x1 + x2) / 2 / W
    yc = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def write_data_yaml():
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "nc: 2\n"
        "names:\n- not\n- ogrenci_numara\n"
    )
    (OUTPUT_DIR / "data.yaml").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Sentetik sınav şablonu üret")
    parser.add_argument("--sayi", type=int, default=1000, help="Üretilecek sayfa sayısı")
    parser.add_argument("--val-oran", type=float, default=0.1, help="Doğrulama oranı")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu")
    args = parser.parse_args()

    random.seed(args.seed)

    fonts = discover_fonts()
    crops = load_crops()
    print(f"Font: {len(fonts)} adet | not: {len(crops['not'])} | "
          f"ogrenci_numara: {len(crops['ogrenci_numara'])} kırpım")

    for split in ["train", "valid"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    val_count = int(args.sayi * args.val_oran)

    for i in range(args.sayi):
        split = "valid" if i < val_count else "train"

        # A4'e yakın, çeşitli boyut ve yön
        if random.random() < 0.75:
            W, H = random.randint(1000, 1350), random.randint(1400, 1800)  # dikey
        else:
            W, H = random.randint(1400, 1800), random.randint(1000, 1350)  # yatay

        page, boxes = render_page(fonts, crops, (W, H))

        name = f"synth_{i:05d}"
        page.save(OUTPUT_DIR / split / "images" / f"{name}.jpg", quality=random.randint(80, 95))

        lines = [to_yolo_line(*b, W, H) for b in boxes]
        (OUTPUT_DIR / split / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{args.sayi} üretildi")

    write_data_yaml()

    print(f"\nTamamlandı: {args.sayi} sayfa -> {OUTPUT_DIR}")
    print(f"  train: {args.sayi - val_count} | valid: {val_count}")
    print("Sıradaki adım: bu veriyi gerçek veriyle birleştirip YOLO'yu yeniden eğit.")


if __name__ == "__main__":
    main()
