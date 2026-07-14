#!/usr/bin/env python3
"""
Tam eğitim pipeline'ı - VPS'den veri indirmeden model değerlendirilmesine kadar
"""

import subprocess
import sys
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Eğitim pipeline yöneticisi"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.steps = []
        self.current_step = 0

    def add_step(self, name, description, command):
        """Pipeline'a adım ekle"""
        self.steps.append({
            'name': name,
            'description': description,
            'command': command
        })

    def print_header(self, text):
        """Başlık yazdır"""
        print("\n" + "=" * 70)
        print(f"║ {text.center(66)} ║")
        print("=" * 70 + "\n")

    def print_step_info(self, step_num, total, name, description):
        """Adım bilgisini yazdır"""
        print(f"\n[{step_num}/{total}] {name}")
        print(f"    {description}")
        print("-" * 70)

    def run_step(self, step):
        """Bir adımı çalıştır"""
        try:
            logger.info(f"Çalıştırılıyor: {step['command']}")

            # Komutu çalıştır
            result = subprocess.run(
                step['command'],
                shell=True,
                cwd=str(self.project_root),
                capture_output=False
            )

            if result.returncode != 0:
                logger.error(f"❌ Adım başarısız: {step['name']}")
                return False

            logger.info(f"✓ Adım tamamlandı: {step['name']}")
            return True

        except Exception as e:
            logger.error(f"❌ Hata: {e}")
            return False

    def configure_pipeline(self):
        """Pipeline'ı yapılandır"""

        # Step 1: VPS'den indir
        self.add_step(
            'VPS Veri İndirme',
            'VPS sunucusundan label.csv dosyalarını indir',
            'python3 training/download_from_vps.py'
        )

        # Step 2: Dataset hazırlama
        self.add_step(
            'Dataset Hazırlama',
            'CSV dosyalarını birleştir ve train/val/test split\'leri oluştur',
            'python3 training/prepare_dataset.py'
        )

        # Step 3: Model eğitimi
        self.add_step(
            'Model Eğitimi',
            'CRNN modelini eğit',
            'python3 training/train_crnn.py'
        )

        # Step 4: Değerlendirme
        self.add_step(
            'Model Değerlendirmesi',
            'Test seti üzerinde model performansını değerlendir',
            'python3 training/evaluate_model.py'
        )

    def run(self, start_step=0, skip_steps=None):
        """Pipeline'ı çalıştır"""
        if skip_steps is None:
            skip_steps = []

        self.print_header("CRNN MODEL TRAINING PIPELINE")

        logger.info(f"Toplam adım: {len(self.steps)}")
        logger.info(f"Başlangıç adımı: {start_step + 1}")
        if skip_steps:
            logger.info(f"Atlanacak adımlar: {[s+1 for s in skip_steps]}")

        total_steps = len(self.steps)
        success_count = 0
        failed_steps = []

        for i, step in enumerate(self.steps[start_step:], start=start_step + 1):
            if i - 1 in skip_steps:
                logger.info(f"\n[{i}/{total_steps}] {step['name']} (ATLANDI)")
                continue

            self.print_step_info(i, total_steps, step['name'], step['description'])

            if self.run_step(step):
                success_count += 1
            else:
                failed_steps.append(step['name'])
                logger.error(f"Pipeline durdu. Adım başarısız: {step['name']}")
                break

        # Özet
        self.print_header("PIPELINE OZETI")

        logger.info(f"Başarılı adımlar: {success_count}/{total_steps}")

        if failed_steps:
            logger.error(f"Başarısız adımlar:")
            for step in failed_steps:
                logger.error(f"  - {step}")
            return False
        else:
            logger.info("✓ Tüm adımlar başarıyla tamamlandı!")
            return True

    def run_quick_mode(self):
        """Hızlı mode - sadece eğitim ve değerlendirme (veri zaten hazır)"""
        self.print_header("CRNN MODEL TRAINING - HIZLI MOD")

        logger.info("⚠ Hızlı mod: Dataset hazırlama adımları atlanacak")
        logger.info("ℹ Data/processed/ klasöründe train.csv, val.csv ve test.csv olması gerekiyor")
        logger.info("")

        # Veri kontrol et
        processed_dir = self.project_root / 'data' / 'processed'
        required_files = ['train.csv', 'val.csv', 'test.csv']

        missing_files = [f for f in required_files if not (processed_dir / f).exists()]

        if missing_files:
            logger.error(f"❌ Eksik dosyalar: {missing_files}")
            logger.info("Lütfen önce dataset'i hazırla: python3 training/prepare_dataset.py")
            return False

        # Pipeline'ı sadece eğitim ve değerlendirmeyle çalıştır
        self.run(start_step=2, skip_steps=[])
        return True


def print_menu():
    """Ana menüyü yazdır"""
    print("\n" + "=" * 70)
    print("║ CRNN MODEL TRAINING PIPELINE".ljust(67) + "║")
    print("=" * 70)
    print("\nSeçin:")
    print("  1. Tam pipeline (VPS'den indir → Değerlendir)")
    print("  2. Hızlı mod (sadece eğit + değerlendir)")
    print("  3. Sadece dataset hazırla")
    print("  4. Sadece model eğit")
    print("  5. Sadece değerlendir")
    print("  6. Çıkış")
    print()


def main():
    """Ana fonksiyon"""

    while True:
        print_menu()
        choice = input("Seçim: ").strip()

        pipeline = TrainingPipeline()
        pipeline.configure_pipeline()

        if choice == '1':
            success = pipeline.run()
            sys.exit(0 if success else 1)

        elif choice == '2':
            success = pipeline.run_quick_mode()
            sys.exit(0 if success else 1)

        elif choice == '3':
            logger.info("Dataset hazırlama başlatılıyor...")
            subprocess.run(['python3', 'training/prepare_dataset.py'], cwd=str(pipeline.project_root))

        elif choice == '4':
            logger.info("Model eğitimi başlatılıyor...")
            subprocess.run(['python3', 'training/train_crnn.py'], cwd=str(pipeline.project_root))

        elif choice == '5':
            logger.info("Model değerlendirmesi başlatılıyor...")
            subprocess.run(['python3', 'training/evaluate_model.py'], cwd=str(pipeline.project_root))

        elif choice == '6':
            logger.info("Çıkılıyor...")
            sys.exit(0)

        else:
            logger.warning("Geçersiz seçim")


if __name__ == '__main__':
    main()
