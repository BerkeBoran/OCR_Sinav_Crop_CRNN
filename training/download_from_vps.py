"""
VPS'den label.csv dosyalarını indir
"""

import os
import paramiko
import logging
from pathlib import Path
import sys
from getpass import getpass
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_vps_config():
    """VPS konfigürasyonunu yükle veya oluştur"""
    config_path = Path('configs/vps_config.json')

    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)

    # Kullanıcıdan bilgi al
    logger.info("VPS Konfigürasyonu Gerekli")
    logger.info("-" * 50)

    config = {}
    config['host'] = input("VPS Hostname/IP: ").strip()
    config['port'] = int(input("SSH Port (varsayılan 22): ").strip() or "22")
    config['username'] = input("Kullanıcı Adı: ").strip()
    config['remote_path'] = input("VPS'deki CSV dosyalarının yolu (/root/labels/ gibi): ").strip()

    # Konfigürasyonu kaydet
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"✓ Konfigürasyon kaydedildi: {config_path}")
    return config


def connect_vps(host, port, username, password):
    """VPS'ye SSH bağlantısı aç"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        logger.info(f"VPS'ye bağlanılıyor: {username}@{host}:{port}")
        ssh.connect(host, port=port, username=username, password=password, timeout=10)

        logger.info("✓ Bağlantı başarılı")
        return ssh

    except paramiko.AuthenticationException:
        logger.error("❌ Kimlik doğrulama başarısız. Kullanıcı adı/şifre hatalı.")
        return None
    except paramiko.SSHException as e:
        logger.error(f"❌ SSH hatası: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Bağlantı hatası: {e}")
        return None


def list_remote_files(ssh, remote_path):
    """VPS'deki CSV dosyalarını listele"""
    try:
        sftp = ssh.open_sftp()

        # Klasörün varlığını kontrol et
        try:
            sftp.listdir(remote_path)
        except IOError:
            logger.error(f"❌ Klasör bulunamadı: {remote_path}")
            sftp.close()
            return []

        # CSV dosyalarını bul
        files = []
        for item in sftp.listdir(remote_path):
            if item.endswith('.csv'):
                files.append(item)

        sftp.close()
        return files

    except Exception as e:
        logger.error(f"❌ Dosya listeme hatası: {e}")
        return []


def download_files(ssh, remote_path, local_path, files):
    """Dosyaları indir"""
    local_dir = Path(local_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    sftp = ssh.open_sftp()

    downloaded = 0
    failed = 0

    for filename in files:
        remote_file = f"{remote_path}/{filename}".replace('\\', '/')
        local_file = local_dir / filename

        try:
            logger.info(f"İndiriliyor: {filename}...", end=' ')
            sftp.get(remote_file, str(local_file))
            logger.info(f"✓ ({local_file.stat().st_size / 1024 / 1024:.2f} MB)")
            downloaded += 1

        except Exception as e:
            logger.error(f"❌ {e}")
            failed += 1

    sftp.close()

    return downloaded, failed


def merge_csvs(input_dir):
    """İndirilen CSV dosyalarını birleştir"""
    import pandas as pd

    input_path = Path(input_dir)
    csv_files = list(input_path.glob('*.csv'))

    if not csv_files:
        logger.warning("Birleştirilecek CSV dosyası bulunamadı")
        return None

    logger.info(f"\n{len(csv_files)} CSV dosyası birleştiriliyor...")

    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            logger.info(f"  ✓ {csv_file.name}: {len(df)} satır")
        except Exception as e:
            logger.error(f"  ❌ {csv_file.name}: {e}")

    if not dfs:
        return None

    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"\nToplam satır: {len(merged_df)}")

    return merged_df


def main():
    """Ana fonksiyon"""

    logger.info("=" * 60)
    logger.info("VPS'DEN VERİ İNDİRME")
    logger.info("=" * 60)
    logger.info("")

    # Konfigürasyon
    config = load_vps_config()
    logger.info("")

    # Şifre al
    password = getpass("SSH Şifre: ")

    # VPS'ye bağlan
    ssh = connect_vps(config['host'], config['port'], config['username'], password)
    if not ssh:
        sys.exit(1)

    # Dosyaları listele
    logger.info("")
    logger.info(f"VPS'de aranıyor: {config['remote_path']}")
    files = list_remote_files(ssh, config['remote_path'])

    if not files:
        logger.error("CSV dosyası bulunamadı")
        ssh.close()
        sys.exit(1)

    logger.info(f"✓ {len(files)} CSV dosyası bulundu:")
    for f in files:
        logger.info(f"  - {f}")

    logger.info("")

    # Kullanıcıdan onay al
    response = input("Bu dosyaları indir? (e/h): ").strip().lower()
    if response != 'e':
        logger.info("İşlem iptal edildi")
        ssh.close()
        sys.exit(0)

    logger.info("")

    # Dosyaları indir
    local_path = 'data/raw'
    downloaded, failed = download_files(ssh, config['remote_path'], local_path, files)

    logger.info("")
    logger.info(f"İndirme tamamlandı: {downloaded} başarılı, {failed} başarısız")

    ssh.close()

    # CSV dosyalarını birleştir
    logger.info("")
    response = input("\nCSV dosyalarını birleştir? (e/h): ").strip().lower()

    if response == 'e':
        logger.info("")
        merged_df = merge_csvs(local_path)

        if merged_df is not None:
            merged_path = Path(local_path) / 'merged_labels.csv'
            merged_df.to_csv(merged_path, index=False)
            logger.info(f"\n✓ Birleştirilmiş dosya kaydedildi: {merged_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ TAMAMLANDI")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Sonraki Adım: Dataset hazırla")
    logger.info("  python3 training/prepare_dataset.py")
    logger.info("")


if __name__ == '__main__':
    main()
