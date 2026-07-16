"""
Sınav Okuma sayfası — canlı sitede tam sayfa sınav fotoğrafı yükleyip
YOLO alan tespiti + CRNN rakam okuma hattını (uçtan uca) elle test etmek için.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Sınav Okuma", page_icon="📄", layout="wide")

st.title("📄 Sınav Okuma (Tam Sayfa)")
st.markdown(
    "Tam bir **sınav kağıdı fotoğrafı** yükleyin. Sistem önce YOLO ile "
    "öğrenci numarası ve not alanlarını bulur, sonra CRNN ile rakamları okur. "
    "Bu, alanları kendiniz kırpmadan uçtan uca okumayı test eder."
)


@st.cache_resource
def load_reader():
    from inference.pipeline import ExamReader
    return ExamReader()


try:
    reader = load_reader()
except FileNotFoundError as error:
    st.error(
        "⚠️ Gerekli model bulunamadı.\n\n"
        f"```\n{error}\n```\n\n"
        "Sunucuda `models/yolo_fields.pt` ve `models/crnn_best_model.pth` "
        "bulunmalı: repoya commit edilip `git pull` ile çekilmeli, ardından "
        "container yeniden başlatılmalı (rebuild)."
    )
    st.stop()

# Durum -> renk/etiket eşlemesi
DURUM_STILI = {
    "ok": ("#1a9850", "✅ Okundu"),
    "dusuk_guven": ("#f9a825", "⚠️ Düşük güven"),
    "gecersiz_deger": ("#e08000", "⚠️ Geçersiz değer"),
    "alan_bulunamadi": ("#d73027", "❌ Alan bulunamadı"),
}

ALAN_ADI = {"ogrenci_numara": "Öğrenci Numarası", "not": "Not"}


def annotate(image, result):
    """Tespit kutularını görselin üstüne çizer."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    # Görsel boyutuna göre çizgi kalınlığı ve font
    line_width = max(3, annotated.width // 300)
    font_size = max(16, annotated.width // 45)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for field_type in ["ogrenci_numara", "not"]:
        alan = result[field_type]
        if alan["kutu"] is None:
            continue

        renk = DURUM_STILI[alan["durum"]][0]
        x1, y1, x2, y2 = [int(v) for v in alan["kutu"]]
        draw.rectangle([x1, y1, x2, y2], outline=renk, width=line_width)

        etiket = f"{ALAN_ADI[field_type]}: {alan['deger'] or '?'}"
        tb = draw.textbbox((x1, y1), etiket, font=font)
        th = tb[3] - tb[1]
        draw.rectangle([x1, y1 - th - 8, tb[2] + 6, y1], fill=renk)
        draw.text((x1 + 3, y1 - th - 6), etiket, fill="white", font=font)

    return annotated


uploaded = st.file_uploader(
    "Sınav kağıdı fotoğrafı yükle (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Tüm sayfayı içeren bir fotoğraf yükleyin — alanları kırpmanıza gerek yok.",
)

if uploaded is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded))

    with st.spinner("Alanlar bulunuyor ve okunuyor..."):
        result = reader.read(image)

    annotated = annotate(image, result)

    col_img, col_sonuc = st.columns([3, 2])

    with col_img:
        st.subheader("Tespit Sonucu")
        st.image(annotated, use_container_width=True)

    with col_sonuc:
        st.subheader("Okunan Değerler")

        if result["kontrol_gerekli"]:
            st.warning("⚠️ Bu kağıt elle kontrol edilmeli — aşağıya bakın.")
        else:
            st.success("✅ Tüm alanlar güvenle okundu.")

        for field_type in ["ogrenci_numara", "not"]:
            alan = result[field_type]
            renk, etiket = DURUM_STILI[alan["durum"]]

            st.markdown(f"**{ALAN_ADI[field_type]}**")

            if alan["durum"] == "alan_bulunamadi":
                st.markdown(
                    f"<span style='color:{renk}'>{etiket}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"# `{alan['deger'] or '—'}`")
                st.markdown(
                    f"<span style='color:{renk}'>{etiket}</span> &nbsp;·&nbsp; "
                    f"okuma güveni %{alan['guven'] * 100:.0f} &nbsp;·&nbsp; "
                    f"kutu güveni %{alan['kutu_guven'] * 100:.0f}",
                    unsafe_allow_html=True,
                )
                if alan["kirpim"] is not None:
                    st.image(alan["kirpim"], caption="Kırpılan alan", width=260)

            st.divider()

    st.caption(
        "Renkler: yeşil = güvenle okundu, sarı = düşük güven, "
        "kırmızı = alan bulunamadı. Sarı/kırmızı alanları elle doğrulayın."
    )
