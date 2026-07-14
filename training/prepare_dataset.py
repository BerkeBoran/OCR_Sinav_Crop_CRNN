"""
VPS'den indirilen label.csv dosyalarını birleştirip eğitim için hazırla
"""

import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import sys
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_csv_files(raw_data_dir):
    """Tüm CSV dosyalarını birleştir"""
    raw_dir = Path(raw_data_dir)

    if not raw_dir.exists():
        logger.error(f"Klasör bulunamadı: {raw_dir}")
        return None

    csv_files = sorted(set(raw_dir.rglob('*.csv')))

    if not csv_files:
        logger.error(f"{raw_dir} içinde CSV dosyası bulunamadı")
        return None

    logger.info(f"Bulundu: {len(csv_files)} CSV dosyası")

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Yüklendi: {csv_file.name} ({len(df)} satır)")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Hata: {csv_file.name} - {e}")

    if not dfs:
        logger.error("Hiçbir CSV dosyası başarıyla yüklenemedi")
        return None

    # Tüm dataframe'leri birleştir
    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Toplam birleştirilmiş satır: {len(merged_df)}")

    return merged_df


def prepare_splits(df, test_size=0.2, val_size=0.1, random_state=42):
    """Train/Val/Test split'leri oluştur"""

    # Label'lar çoğunlukla benzersiz olduğundan stratify için 'type' kullan
    stratify_col = df['type'] if 'type' in df.columns else None

    # Test split
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Train/Val split
    val_percentage = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_percentage,
        random_state=random_state,
        stratify=train_val_df['type'] if 'type' in train_val_df.columns else None
    )

    logger.info(f"Train set: {len(train_df)} satır ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Val set: {len(val_df)} satır ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"Test set: {len(test_df)} satır ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def validate_images(df, img_dir):
    """Resimlerin var olup olmadığını kontrol et ve çözülen yolu kaydet"""
    img_path = Path(img_dir)
    resolved_paths = []
    invalid_indices = []

    for idx, row in df.iterrows():
        candidates = []

        # 1. CSV'deki image_path (proje köküne göre relative)
        if 'image_path' in row and pd.notna(row['image_path']):
            candidates.append(Path(row['image_path']))

        # 2. img_dir/type/file_name (alt klasör yapısı)
        file_name = row.get('file_name')
        field_type = row.get('type')
        if pd.notna(file_name):
            if pd.notna(field_type):
                candidates.append(img_path / str(field_type) / str(file_name))
            candidates.append(img_path / str(file_name))

        resolved = next((c for c in candidates if c.exists()), None)

        if resolved:
            resolved_paths.append((idx, str(resolved)))
        else:
            invalid_indices.append(idx)

    logger.info(f"Geçerli resim: {len(resolved_paths)}/{len(df)}")

    if invalid_indices:
        logger.warning(f"Bulunmayan resimler: {len(invalid_indices)} dosya")
        df = df.drop(invalid_indices)
        logger.info(f"Geçersiz satırlar çıkarıldı. Yeni toplam: {len(df)}")

    # Çözülen tam yolları image_path sütununa yaz
    for idx, path in resolved_paths:
        if idx in df.index:
            df.loc[idx, 'image_path'] = path

    return df


def process_dataset(raw_data_dir, output_dir, img_dir, test_size=0.2, val_size=0.1):
    """Tüm dataset processing pipeline'ı"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # CSV dosyalarını birleştir
    logger.info("Step 1: CSV dosyaları birleştiriliyor...")
    df = merge_csv_files(raw_data_dir)

    if df is None:
        return False

    # Gerekli sütunları kontrol et
    logger.info("Step 2: Veri yapısı kontrol ediliyor...")
    required_cols = ['label']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Gerekli sütunlar: {required_cols}. Mevcut: {df.columns.tolist()}")
        return False

    # Dosya adını normalize et
    if 'file_name' not in df.columns and 'image_path' in df.columns:
        df['file_name'] = df['image_path'].apply(lambda x: Path(x).name)

    # Resimleri valide et
    logger.info("Step 3: Resimlerin varlığı kontrol ediliyor...")
    df = validate_images(df, img_dir)

    if len(df) == 0:
        logger.error("Hiçbir geçerli resim bulunamadı")
        return False

    # Geçersiz label'ları filtrele (sadece rakamlardan oluşanlar geçerli)
    logger.info("Step 4: Label'lar temizleniyor...")
    original_len = len(df)
    df['label'] = df['label'].astype(str).str.strip()
    df = df[df['label'].str.fullmatch(r'\d+')]
    if original_len - len(df) > 0:
        logger.warning(f"Rakam olmayan label'lı satırlar çıkarıldı: {original_len - len(df)}")

    # Duplikatları kaldır
    logger.info("Step 5: Duplikatlar kaldırılıyor...")
    original_len = len(df)
    df = df.drop_duplicates(subset=['file_name'], keep='first')
    logger.info(f"Duplikat: {original_len - len(df)}")

    # Train/Val/Test split'leri oluştur
    logger.info("Step 5: Train/Val/Test split'leri oluşturuluyor...")
    train_df, val_df, test_df = prepare_splits(df, test_size=test_size, val_size=val_size)

    # CSV dosyalarını kaydet
    logger.info("Step 6: Dosyalar kaydediliyor...")
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)

    # İstatistikler
    logger.info("Step 7: İstatistikler hesaplanıyor...")
    if 'label' in df.columns:
        logger.info("\nLabel dağılımı:")
        logger.info(f"Train set: {len(df[df['label'].isin(train_df['label'])])}")
        logger.info(f"\nUnique label'lar: {df['label'].nunique()}")
        logger.info(f"Toplam örnek: {len(df)}")

    logger.info(f"\n✓ Dataset hazırlama tamamlandı!")
    logger.info(f"Output: {output_path}")

    return True


def main():
    """Ana fonksiyon"""

    # Konfigürasyon
    raw_data_dir = 'data/raw'
    output_dir = 'data/processed'
    img_dir = 'data/cropped_fields'

    # İşlem
    success = process_dataset(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        img_dir=img_dir,
        test_size=0.2,
        val_size=0.1
    )

    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
