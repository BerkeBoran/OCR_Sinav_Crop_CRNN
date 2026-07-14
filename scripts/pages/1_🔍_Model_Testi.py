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
