"""
Eğitilen CRNN modelini test seti üzerinde değerlendir.

Metrikler: exact-match accuracy, karakter hata oranı (CER), alan tipine göre kırılım.
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train_crnn import (
    CRNN, CRNNDataset, decode_greedy, get_device, levenshtein
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model değerlendirme sınıfı"""

    def __init__(self, model_path):
        self.device = get_device()
        logger.info(f"Cihaz: {self.device}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']

        self.model = CRNN(rnn_hidden=self.config.get('rnn_hidden', 256)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        logger.info(f"Model yüklendi: {model_path}")
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch')}, Val Acc: {checkpoint.get('val_acc', 0):.4f}")

    def setup_dataloader(self, csv_path, img_dir):
        """Test dataloader'ını oluştur"""
        transform = transforms.Compose([
            transforms.Resize((self.config['img_height'], self.config['img_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        dataset = CRNNDataset(csv_path, img_dir, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        return loader

    def evaluate(self, dataloader):
        """Tahminleri topla"""
        all_preds = []
        all_labels = []
        all_files = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Değerlendirme'):
                images = batch['image'].to(self.device)

                logits = self.model(images)
                preds = decode_greedy(logits)

                all_preds.extend(preds)
                all_labels.extend(batch['label'])
                all_files.extend(batch['file_name'])

        return all_preds, all_labels, all_files

    def calculate_metrics(self, preds, labels):
        """Sekans ve karakter seviyesi metrikleri hesapla"""
        total = len(labels)
        correct = sum(p == t for p, t in zip(preds, labels))

        char_errors = sum(levenshtein(p, t) for p, t in zip(preds, labels))
        char_total = sum(len(t) for t in labels)

        return {
            'sequence_accuracy': correct / max(total, 1),
            'character_error_rate': char_errors / max(char_total, 1),
            'total_samples': total,
            'correct_samples': correct,
        }

    def save_results(self, preds, labels, files, output_dir, set_name='test'):
        """Sonuçları kaydet"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        metrics = self.calculate_metrics(preds, labels)

        results_df = pd.DataFrame({
            'file_name': files,
            'true_label': labels,
            'predicted_label': preds,
            'correct': [p == t for p, t in zip(preds, labels)],
            'edit_distance': [levenshtein(p, t) for p, t in zip(preds, labels)],
        })

        # Alan tipine göre kırılım (dosya adından çıkar)
        results_df['field_type'] = results_df['file_name'].apply(
            lambda f: 'ogrenci_numara' if 'ogrenci_numara' in f else ('not' if '_not' in f else 'diger')
        )

        by_type = {}
        for field_type, group in results_df.groupby('field_type'):
            by_type[field_type] = {
                'accuracy': float(group['correct'].mean()),
                'samples': len(group),
            }
        metrics['by_field_type'] = by_type

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_csv = output_path / f'{set_name}_results_{timestamp}.csv'
        results_df.to_csv(results_csv, index=False)

        metrics_json = output_path / f'{set_name}_metrics_{timestamp}.json'
        with open(metrics_json, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"\nSonuçlar kaydedildi:")
        logger.info(f"  - CSV: {results_csv}")
        logger.info(f"  - JSON: {metrics_json}")

        return metrics, results_df

    def print_metrics(self, metrics):
        """Metrikleri yazdır"""
        logger.info("\n" + "=" * 50)
        logger.info("DEĞERLENDİRME SONUÇLARI")
        logger.info("=" * 50)
        logger.info(f"Sequence Accuracy:    {metrics['sequence_accuracy']:.4f}")
        logger.info(f"Character Error Rate: {metrics['character_error_rate']:.4f}")
        logger.info(f"Doğru / Toplam:       {metrics['correct_samples']} / {metrics['total_samples']}")

        for field_type, stats in metrics.get('by_field_type', {}).items():
            logger.info(f"  {field_type}: acc={stats['accuracy']:.4f} ({stats['samples']} örnek)")

        logger.info("=" * 50 + "\n")

    def print_errors(self, results_df, max_show=20):
        """Yanlış tahminleri göster"""
        wrong = results_df[~results_df['correct']]

        if len(wrong) == 0:
            logger.info("✓ Tüm tahminler doğru!")
            return

        logger.info(f"\nYanlış Tahminler ({len(wrong)}):")
        logger.info("-" * 80)

        for _, row in wrong.head(max_show).iterrows():
            logger.info(
                f"{row['file_name']}: "
                f"Gerçek={row['true_label']}, Tahmin={row['predicted_label']} "
                f"(edit={row['edit_distance']})"
            )

        if len(wrong) > max_show:
            logger.info(f"... ve {len(wrong) - max_show} daha")


def main():
    model_path = 'models/crnn_best_model.pth'
    test_csv = 'data/processed/test.csv'
    img_dir = 'data/cropped_fields'
    output_dir = 'logs/evaluation'

    if not Path(model_path).exists():
        logger.error(f"Model bulunamadı: {model_path}")
        logger.info("Önce modeli eğitin: python3 training/train_crnn.py")
        sys.exit(1)

    if not Path(test_csv).exists():
        logger.error(f"Test CSV bulunamadı: {test_csv}")
        logger.info("Önce dataset'i hazırlayın: python3 training/prepare_dataset.py")
        sys.exit(1)

    evaluator = ModelEvaluator(model_path)

    logger.info("Dataloader oluşturuluyor...")
    dataloader = evaluator.setup_dataloader(test_csv, img_dir)

    logger.info("Tahminler yapılıyor...")
    preds, labels, files = evaluator.evaluate(dataloader)

    metrics, results_df = evaluator.save_results(preds, labels, files, output_dir)

    evaluator.print_metrics(metrics)
    evaluator.print_errors(results_df)

    logger.info("✓ Değerlendirme tamamlandı!")


if __name__ == '__main__':
    main()
