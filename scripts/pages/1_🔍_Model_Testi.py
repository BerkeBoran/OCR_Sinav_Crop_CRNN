"""
Model Testi sayfası — canlı sitede yeni bir görsel yükleyip
eğitilmiş CRNN modelinin okumasını test etmek için.
"""

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Model Testi", page_icon="🔍", layout="centered")

st.title("🔍 Model Testi")
st.markdown(
    "Eğitilmiş modelin **hiç görmediği** bir görüntüyü yükleyin — "
    "öğrenci numarası veya not alanının kırpılmış fotoğrafı. "
    "Model rakamları okuyup gösterecek."
)


@st.cache_resource
def load_predictor():
    from inference.predictor import CRNNPredictor
    return CRNNPredictor()


# Model yüklemesi (bir kez yapılır, sonraki isteklerde önbellekten gelir)
try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error(
        "⚠️ Eğitilmiş model bulunamadı (`models/crnn_best_model.pth`).\n\n"
        "Sunucuda modelin olduğundan emin olun: repoya commit edilip "
        "`git pull` ile çekilmeli, ardından container yeniden başlatılmalı."
    )
    st.stop()

info = predictor.checkpoint_info
if info.get('val_acc') is not None:
    st.caption(
        f"Yüklü model — epoch {info['epoch']}, "
        f"doğrulama başarısı: %{info['val_acc'] * 100:.1f}"
    )

uploaded = st.file_uploader(
    "Görüntü yükle (JPG / PNG / WEBP)",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="En iyi sonuç için sadece rakamların olduğu alanı kırpıp yükleyin."
)

if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Yüklenen Görüntü")
        st.image(image, use_container_width=True)

    with st.spinner("Model okuyor..."):
        text, confidence = predictor.predict(image)

    with col2:
        st.subheader("Modelin Okuduğu")
        if text:
            st.markdown(f"# `{text}`")

            if confidence >= 0.9:
                st.success(f"Güven: %{confidence * 100:.1f} — yüksek")
            elif confidence >= 0.7:
                st.warning(f"Güven: %{confidence * 100:.1f} — orta")
            else:
                st.error(f"Güven: %{confidence * 100:.1f} — düşük, sonuca temkinli yaklaşın")
        else:
            st.error("Model bu görüntüde rakam okuyamadı.")
            st.markdown(
                "**İpuçları:**\n"
                "- Sadece rakamların olduğu bölgeyi kırpın\n"
                "- Görüntünün net ve iyi aydınlatılmış olduğundan emin olun\n"
                "- Rakamlar yatay bir satır halinde olmalı"
            )

    # Doğrulama bölümü: gerçek değeri girip karşılaştır
    st.divider()
    st.subheader("Sonucu Doğrula (opsiyonel)")
    gercek = st.text_input(
        "Görüntüdeki gerçek değer neydi?",
        placeholder="örn. 231601045",
        max_chars=15
    )

    if gercek:
        gercek = gercek.strip()
        if gercek == text:
            st.success(f"✅ Model DOĞRU okudu: {text}")
        else:
            # Kaç rakam farklı?
            from training.train_crnn import levenshtein
            fark = levenshtein(text, gercek)
            st.error(
                f"❌ Model YANLIŞ okudu\n\n"
                f"- Gerçek: `{gercek}`\n"
                f"- Model: `{text if text else '(boş)'}`\n"
                f"- Fark: {fark} rakam"
            )

        # Doğrulanan görseli veri setine ekle: görsel pending klasörüne,
        # etiketi pending_labels.csv'ye yazılır; sıradaki birleştirmede
        # (merge_pending_data.py) eğitim verisine katılır.
        st.divider()
        st.subheader("Veri Setine Ekle")
        st.caption(
            "Doğru değeriyle birlikte bu görsel eğitim verisine eklenir — "
            "modelin yanlış okuduğu örnekler özellikle değerlidir."
        )

        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from labeling_tool import (
            ANNOTATORS,
            CROPPED_PENDING_DIR,
            save_label,
            to_relative_path,
            validate_label,
        )

        tahmini_tur = "not" if gercek.isdigit() and len(gercek) <= 3 else "ogrenci_numara"

        col_tur, col_kisi = st.columns(2)
        with col_tur:
            field_type = st.selectbox(
                "Görsel türü",
                ["ogrenci_numara", "not"],
                index=["ogrenci_numara", "not"].index(tahmini_tur),
            )
        with col_kisi:
            annotator = st.selectbox("Ekleyen kişi", ANNOTATORS)

        if st.button("📥 Veri setine ekle", type="primary"):
            is_valid, message = validate_label(field_type, gercek)

            if not is_valid:
                st.error(message)
            else:
                import hashlib
                import io

                buffer = io.BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=95)
                content = buffer.getvalue()

                # İçerik hash'iyle adlandırma: aynı görsel iki kez eklenirse
                # dosya ve etiket üzerine yazılır, çift kayıt oluşmaz.
                digest = hashlib.md5(content).hexdigest()[:10]
                file_name = f"upload_{digest}_{field_type}.jpg"

                output_dir = CROPPED_PENDING_DIR / field_type
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / file_name
                output_path.write_bytes(content)

                row = {
                    "image_path": to_relative_path(output_path),
                    "type": field_type,
                    "file_name": file_name,
                    "split": "upload",
                }
                save_label(row, gercek, annotator, dataset="pending")

                st.success(
                    f"✅ Eklendi: `{file_name}` — etiket: `{gercek}`. "
                    "Sıradaki veri birleştirmesinde eğitim setine katılacak."
                )
