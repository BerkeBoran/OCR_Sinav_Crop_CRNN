import hashlib
import threading
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CROPPED_DIR = PROJECT_ROOT / "data" / "cropped_fields"
CRNN_DIR = PROJECT_ROOT / "data" / "crnn_dataset"

LABELS_PATH = CRNN_DIR / "labels.csv"
SKIPPED_PATH = CRNN_DIR / "skipped.csv"

# Streamlit çalıştırır her oturumu ayrı bir thread'de; birden fazla kişi
# aynı anda kaydettiğinde labels.csv/skipped.csv'nin oku-değiştir-yaz
# döngüsü çakışmasın diye bu kilit kullanılır.
_CSV_LOCK = threading.Lock()

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]

FIELD_TYPES = ["ogrenci_numara", "not"]

# Her görsel bu listedeki sıraya göre sabit bir kişiye atanır;
# sıralamayı değiştirmek atamaları da değiştirir.
ANNOTATORS = ["Mehmet", "Berke", "Aslı", "Sevde"]


def assign_annotator(image_path: str) -> str:
    digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(ANNOTATORS)
    return ANNOTATORS[index]


def ensure_dirs():
    CRNN_DIR.mkdir(parents=True, exist_ok=True)


def to_relative_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def infer_split_from_filename(file_name: str) -> str:
    if file_name.startswith("train_"):
        return "train"
    if file_name.startswith("valid_"):
        return "valid"
    if file_name.startswith("test_"):
        return "test"
    return "unknown"


def collect_crop_images() -> pd.DataFrame:
    rows = []

    for field_type in FIELD_TYPES:
        field_dir = CROPPED_DIR / field_type

        if not field_dir.exists():
            continue

        for extension in IMAGE_EXTENSIONS:
            for image_path in sorted(field_dir.glob(f"*{extension}")):
                relative_path = to_relative_path(image_path)
                rows.append({
                    "image_path": relative_path,
                    "file_name": image_path.name,
                    "type": field_type,
                    "split": infer_split_from_filename(image_path.name),
                    "assigned_to": assign_annotator(relative_path),
                })

    return pd.DataFrame(rows)


def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str).fillna("")

    return pd.DataFrame(columns=columns)


def save_csv_safely(df: pd.DataFrame, path: Path):
    temp_path = path.with_suffix(".tmp")
    df.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(path)


def load_labels() -> pd.DataFrame:
    columns = [
        "image_path",
        "label",
        "type",
        "file_name",
        "split",
        "annotator",
        "annotated_at",
    ]
    return load_csv(LABELS_PATH, columns)


def load_skipped() -> pd.DataFrame:
    columns = [
        "image_path",
        "type",
        "file_name",
        "split",
        "reason",
        "annotator",
        "skipped_at",
    ]
    return load_csv(SKIPPED_PATH, columns)


def validate_label(field_type: str, label: str):
    label = label.strip()

    if label == "":
        return False, "Boş değer kaydedilemez."

    if not label.isdigit():
        return False, "Sadece rakam girilmelidir."

    if field_type == "ogrenci_numara":
        if len(label) < 5 or len(label) > 15:
            return False, "Öğrenci numarası 5 ile 15 hane arasında olmalıdır."

    if field_type == "not":
        value = int(label)

        if value < 0 or value > 100:
            return False, "Not değeri 0 ile 100 arasında olmalıdır."

    return True, ""


def save_label(row: dict, label: str, annotator: str):
    with _CSV_LOCK:
        labels_df = load_labels()

        new_row = {
            "image_path": row["image_path"],
            "label": label.strip(),
            "type": row["type"],
            "file_name": row["file_name"],
            "split": row["split"],
            "annotator": annotator,
            "annotated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if not labels_df.empty and row["image_path"] in labels_df["image_path"].values:
            labels_df.loc[labels_df["image_path"] == row["image_path"], list(new_row.keys())] = list(new_row.values())
        else:
            labels_df = pd.concat([labels_df, pd.DataFrame([new_row])], ignore_index=True)

        save_csv_safely(labels_df, LABELS_PATH)


def save_skip(row: dict, reason: str, annotator: str):
    with _CSV_LOCK:
        skipped_df = load_skipped()

        new_row = {
            "image_path": row["image_path"],
            "type": row["type"],
            "file_name": row["file_name"],
            "split": row["split"],
            "reason": reason.strip(),
            "annotator": annotator,
            "skipped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if not skipped_df.empty and row["image_path"] in skipped_df["image_path"].values:
            skipped_df.loc[skipped_df["image_path"] == row["image_path"], list(new_row.keys())] = list(new_row.values())
        else:
            skipped_df = pd.concat([skipped_df, pd.DataFrame([new_row])], ignore_index=True)

        save_csv_safely(skipped_df, SKIPPED_PATH)


def rerun_app():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def main():
    ensure_dirs()

    st.set_page_config(
        page_title="OCR Crop Labeling Tool",
        layout="wide"
    )

    st.title("OCR Crop Labeling Tool")
    st.caption("Kırpılmış öğrenci numarası ve not görsellerini etiketleme aracı")

    all_images_df = collect_crop_images()
    labels_df = load_labels()
    skipped_df = load_skipped()

    if all_images_df.empty:
        st.error("Hiç crop görseli bulunamadı. data/cropped_fields klasörünü kontrol et.")
        return

    with st.sidebar:
        st.header("Ayarlar")

        annotator = st.selectbox("Etiketleyen kişi", ANNOTATORS)

        selected_type = st.selectbox(
            "Görsel türü",
            ["Tümü", "ogrenci_numara", "not"]
        )

        include_skipped = st.checkbox("Atlananları tekrar göster", value=False)

        st.divider()

        st.write("Beklenen klasörler:")
        st.code("data/cropped_fields/ogrenci_numara")
        st.code("data/cropped_fields/not")

    type_filtered_df = all_images_df.copy()

    if selected_type != "Tümü":
        type_filtered_df = type_filtered_df[type_filtered_df["type"] == selected_type]

    labeled_paths = set(labels_df["image_path"].tolist()) if not labels_df.empty else set()
    skipped_paths = set(skipped_df["image_path"].tolist()) if not skipped_df.empty else set()

    def remaining_of(df):
        if include_skipped:
            return df[~df["image_path"].isin(labeled_paths)]

        return df[
            ~df["image_path"].isin(labeled_paths)
            & ~df["image_path"].isin(skipped_paths)
        ]

    own_df = type_filtered_df[type_filtered_df["assigned_to"] == annotator]
    own_remaining_df = remaining_of(own_df)

    helping = None

    if not own_remaining_df.empty:
        filtered_df = own_df
        remaining_df = own_remaining_df
    else:
        donor_counts = {
            other: len(remaining_of(type_filtered_df[type_filtered_df["assigned_to"] == other]))
            for other in ANNOTATORS
            if other != annotator
        }
        donor = max(donor_counts, key=donor_counts.get) if donor_counts else None

        if donor and donor_counts[donor] > 0:
            helping = donor
            filtered_df = type_filtered_df[type_filtered_df["assigned_to"] == donor]
            remaining_df = remaining_of(filtered_df)
        else:
            filtered_df = own_df
            remaining_df = own_remaining_df

    total_count = len(filtered_df)
    labeled_count = len(filtered_df[filtered_df["image_path"].isin(labeled_paths)])
    skipped_count = len(filtered_df[filtered_df["image_path"].isin(skipped_paths)])
    remaining_count = len(remaining_df)

    if helping:
        st.info(f"Kendi görsellerin bitti — {helping} kişisinin kalanlarına yardım ediyorsun.")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Toplam", total_count)
    col2.metric("Etiketlenen", labeled_count)
    col3.metric("Atlanan", skipped_count)
    col4.metric("Kalan", remaining_count)

    if total_count > 0:
        progress_value = labeled_count / total_count
        st.progress(progress_value)

    st.divider()

    if remaining_df.empty:
        st.success("Bu filtre için etiketlenecek görsel kalmadı.")
        st.write(f"labels.csv konumu: `{LABELS_PATH}`")
        return

    # Gösterilen görsel, bu tarayıcı oturumuna sabitlenir; böylece başka
    # biri bir yerlerde kayıt yapıp kalan listesini değiştirse bile görsel
    # bu kullanıcının altından değişmez. Sabitlenen görsel etiketlenip/
    # atlanıp kalan listeden düşünce (bu kullanıcı ya da yardım modunda
    # aynı kuyruğa bakan başka biri tarafından) yeni bir görsel seçilir.
    remaining_paths = set(remaining_df["image_path"].tolist())
    pinned_path = st.session_state.get("current_image_path")

    if pinned_path not in remaining_paths:
        pinned_path = remaining_df.sample(n=1).iloc[0]["image_path"]
        st.session_state["current_image_path"] = pinned_path

    current_row = remaining_df[remaining_df["image_path"] == pinned_path].iloc[0].to_dict()
    image_abs_path = PROJECT_ROOT / current_row["image_path"]

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Görsel")

        if image_abs_path.exists():
            image = Image.open(image_abs_path)
            st.image(image, caption=current_row["file_name"], use_container_width=True)
        else:
            st.error(f"Görsel bulunamadı: {image_abs_path}")
            return

    with right_col:
        st.subheader("Etiket Bilgisi")

        st.write("Tür:")
        st.code(current_row["type"])

        st.write("Dosya:")
        st.code(current_row["file_name"])

        st.write("Split:")
        st.code(current_row["split"])

        with st.form("label_form", clear_on_submit=True):
            label = st.text_input("Görselde yazan değer")

            save_button = st.form_submit_button("Kaydet ve sıradakine geç")

            if save_button:
                is_valid, message = validate_label(current_row["type"], label)

                if not is_valid:
                    st.error(message)
                else:
                    save_label(current_row, label, annotator)
                    st.success("Kaydedildi.")
                    rerun_app()

        with st.form("skip_form", clear_on_submit=True):
            skip_reason = st.text_input("Atlama sebebi", value="okunamıyor veya hatalı crop")
            skip_button = st.form_submit_button("Bu görseli atla")

            if skip_button:
                save_skip(current_row, skip_reason, annotator)
                st.warning("Görsel atlandı.")
                rerun_app()

    st.divider()

    st.subheader("Son Kaydedilenler")

    if labels_df.empty:
        st.info("Henüz kayıt yok.")
    else:
        st.dataframe(labels_df.tail(10), use_container_width=True)

    st.divider()

    st.subheader("Girilen Değerleri Kontrol Et")

    own_labeled_df = labels_df[labels_df["annotator"] == annotator]

    if selected_type != "Tümü":
        own_labeled_df = own_labeled_df[own_labeled_df["type"] == selected_type]

    own_labeled_df = own_labeled_df.reset_index(drop=True)

    if own_labeled_df.empty:
        st.info("Kontrol edilecek etiketli görsel yok.")
    else:
        review_index = st.session_state.get("review_index", 0) % len(own_labeled_df)
        review_row = own_labeled_df.iloc[review_index].to_dict()
        review_abs_path = PROJECT_ROOT / review_row["image_path"]

        st.caption(f"{review_index + 1} / {len(own_labeled_df)}")

        review_col1, review_col2 = st.columns([2, 1])

        with review_col1:
            if review_abs_path.exists():
                st.image(Image.open(review_abs_path), caption=review_row["file_name"], use_container_width=True)
            else:
                st.error(f"Görsel bulunamadı: {review_abs_path}")

        with review_col2:
            with st.form("review_form"):
                reviewed_label = st.text_input("Değer", value=review_row["label"])
                save_and_next = st.form_submit_button("Kaydet ve sıradakine geç")

                if save_and_next:
                    is_valid, message = validate_label(review_row["type"], reviewed_label)

                    if not is_valid:
                        st.error(message)
                    else:
                        save_label(review_row, reviewed_label, annotator)
                        st.session_state["review_index"] = (review_index + 1) % len(own_labeled_df)
                        rerun_app()

            nav_prev, nav_next = st.columns(2)

            if nav_prev.button("Önceki"):
                st.session_state["review_index"] = (review_index - 1) % len(own_labeled_df)
                rerun_app()

            if nav_next.button("Değiştirmeden sıradakine geç"):
                st.session_state["review_index"] = (review_index + 1) % len(own_labeled_df)
                rerun_app()

            jump_input, jump_go = st.columns(2)

            jump_target = jump_input.number_input(
                "Fotoğraf no",
                min_value=1,
                max_value=len(own_labeled_df),
                value=review_index + 1,
                step=1,
            )

            if jump_go.button("Git"):
                st.session_state["review_index"] = int(jump_target) - 1
                rerun_app()

    st.subheader("Dosya Konumları")

    st.code(f"labels.csv: {LABELS_PATH}")
    st.code(f"skipped.csv: {SKIPPED_PATH}")


if __name__ == "__main__":
    main()