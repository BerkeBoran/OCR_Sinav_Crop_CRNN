#!/usr/bin/env python3
"""
Yeni kontrol edilen görselleri (pending) mevcut verilere birleştirir.

Bu script:
1. data/cropped_fields_pending/ klasöründeki görselleri mevcut klasörlere taşır
2. pending_labels.csv'yi labels.csv'ye birleştirir
3. pending_skipped.csv'yi skipped.csv'ye birleştirir

Güvenlik: Mevcut veriler korunur, yalnızca yeni veriler eklenir.
"""

import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CROPPED_DIR = PROJECT_ROOT / "data" / "cropped_fields"
CROPPED_PENDING_DIR = PROJECT_ROOT / "data" / "cropped_fields_pending"
CRNN_DIR = PROJECT_ROOT / "data" / "crnn_dataset"

LABELS_PATH = CRNN_DIR / "labels.csv"
SKIPPED_PATH = CRNN_DIR / "skipped.csv"
PENDING_LABELS_PATH = CRNN_DIR / "pending_labels.csv"
PENDING_SKIPPED_PATH = CRNN_DIR / "pending_skipped.csv"


def create_backup(path: Path) -> Path:
    """Orijinal dosyadan backup oluştur."""
    if not path.exists():
        return None
    backup_path = path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    shutil.copy2(path, backup_path)
    print(f"✅ Backup oluşturuldu: {backup_path}")
    return backup_path


def merge_csv(main_path: Path, pending_path: Path):
    """pending_path'den main_path'e veri birleştir."""
    if not pending_path.exists():
        print(f"⚠️  Pending dosya bulunamadı: {pending_path}")
        return

    main_df = pd.read_csv(main_path, dtype=str) if main_path.exists() else pd.DataFrame()
    pending_df = pd.read_csv(pending_path, dtype=str)

    if pending_df.empty:
        print(f"⚠️  {pending_path.name} boş, birleştirme yapılmadı")
        return

    print(f"📊 Birleştiriliyor: {main_path.name}")
    print(f"   Mevcut: {len(main_df)} satır, Yeni: {len(pending_df)} satır")

    merged_df = pd.concat([main_df, pending_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["image_path"], keep="last")

    merged_df.to_csv(main_path, index=False, encoding="utf-8")
    print(f"✅ Birleştirildi: {len(merged_df)} satır")


def move_images(src_dir: Path, dst_dir: Path, field_type: str):
    """Görselleri pending klasöründen mevcut klasöre taşı."""
    src_field = src_dir / field_type
    dst_field = dst_dir / field_type

    if not src_field.exists():
        print(f"⚠️  Kaynak klasör bulunamadı: {src_field}")
        return

    dst_field.mkdir(parents=True, exist_ok=True)

    image_count = 0
    for image_file in src_field.iterdir():
        if image_file.is_file():
            shutil.move(str(image_file), str(dst_field / image_file.name))
            image_count += 1

    print(f"✅ {field_type}: {image_count} görsel taşındı")


def main():
    print("\n" + "="*60)
    print("🔄 YENİ VERİLERİ MEVCUT VERILERE BİRLEŞTİRME")
    print("="*60)

    if not CROPPED_PENDING_DIR.exists() or not PENDING_LABELS_PATH.exists():
        print("\n❌ Hata: Pending veriler bulunamadı!")
        print(f"   Kontrol et: {CROPPED_PENDING_DIR}")
        print(f"   Kontrol et: {PENDING_LABELS_PATH}")
        return

    print("\n1️⃣  BACKUP OLUŞTURULUYOR...")
    create_backup(LABELS_PATH)
    create_backup(SKIPPED_PATH)

    print("\n2️⃣  GÖRSELLER TAŞINIYORU...")
    move_images(CROPPED_PENDING_DIR, CROPPED_DIR, "not")
    move_images(CROPPED_PENDING_DIR, CROPPED_DIR, "ogrenci_numara")

    print("\n3️⃣  CSV DOSYALARI BİRLEŞTİRİLİYOR...")
    merge_csv(LABELS_PATH, PENDING_LABELS_PATH)
    merge_csv(SKIPPED_PATH, PENDING_SKIPPED_PATH)

    print("\n4️⃣  TEMIZLEME...")
    if PENDING_LABELS_PATH.exists():
        PENDING_LABELS_PATH.unlink()
        print(f"✅ Silindi: {PENDING_LABELS_PATH.name}")

    if PENDING_SKIPPED_PATH.exists():
        PENDING_SKIPPED_PATH.unlink()
        print(f"✅ Silindi: {PENDING_SKIPPED_PATH.name}")

    print("\n" + "="*60)
    print("✅ BİRLEŞTİRME BAŞARILI!")
    print("="*60)
    print("\nVeriler şu konumlarda:")
    print(f"  📁 Görseller: {CROPPED_DIR}")
    print(f"  📊 Labels: {LABELS_PATH}")
    print(f"  📊 Skipped: {SKIPPED_PATH}")
    print()


if __name__ == "__main__":
    main()
