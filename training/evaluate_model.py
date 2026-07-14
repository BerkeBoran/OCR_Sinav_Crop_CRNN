"""
Eğitilen CRNN modelini değerlendir
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train_crnn import CRNN, CRNNDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Model değerlendirme sınıfı"""

    def __init__(self, model_path, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        # Model yükle
        self.load_model(model_path)
        logger.info(f"Cihaz: {self.device}")

    def load_model(self, model_path):
        """Model ve checkpoint'i yükle"""
        checkpoint = torch.load(model_path, map_location=self.device)

        num_classes = len(checkpoint['unique_labels'])
        self.model = CRNN(num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.label_to_idx = checkpoint['label_to_idx']
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.unique_labels = checkpoint['unique_labels']

        logger.info(f"Model yüklendi: {model_path}")
        logger.info(f"Sınıf sayısı: {num_classes}")

    def setup_dataloader(self, csv_path, img_dir):
        """Dataloader'ı oluştur"""
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

        return loader, dataset

    def evaluate(self, dataloader):
        """Model'i değerlendir"""
        all_preds = []
        all_labels = []
        all_files = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Değerlendirme')
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label']
                files = batch['file_name']

                # Tahmin yap
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()

                # Label'ları indekse dönüştür
                label_indices = np.array([self.label_to_idx[label] for label in labels])

                all_preds.extend(preds)
                all_labels.extend(label_indices)
                all_files.extend(files)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return all_preds, all_labels, all_files

    def calculate_metrics(self, preds, labels):
        """Metrik'leri hesapla"""
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def generate_report(self, preds, labels):
        """Detaylı rapport oluştur"""
        report = classification_report(
            labels, preds,
            target_names=self.unique_labels,
            zero_division=0,
            output_dict=True
        )

        return report

    def save_results(self, preds, labels, files, output_dir, set_name='test'):
        """Sonuçları kaydet"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Metrik'ler
        metrics = self.calculate_metrics(preds, labels)

        # Detaylı rapport
        report = self.generate_report(preds, labels)

        # Sonuçları CSV'ye kaydet
        results_df = pd.DataFrame({
            'file_name': files,
            'true_label': [self.idx_to_label[int(label)] for label in labels],
            'predicted_label': [self.idx_to_label[int(pred)] for pred in preds],
            'correct': preds == labels
        })

        # Dosyaları kaydet
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_csv = output_path / f'{set_name}_results_{timestamp}.csv'
        results_df.to_csv(results_csv, index=False)

        metrics_json = output_path / f'{set_name}_metrics_{timestamp}.json'
        with open(metrics_json, 'w') as f:
            json.dump({
                'metrics': metrics,
                'report': report
            }, f, indent=2)

        logger.info(f"\nSonuçlar kaydedildi:")
        logger.info(f"  - CSV: {results_csv}")
        logger.info(f"  - JSON: {metrics_json}")

        return metrics, results_df

    def print_metrics(self, metrics):
        """Metrik'leri yazdır"""
        logger.info("\n" + "="*50)
        logger.info("DEĞERLENDİRME SONUÇLARI")
        logger.info("="*50)
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1']:.4f}")
        logger.info("="*50 + "\n")

    def print_confusion_details(self, preds, labels, results_df):
        """Yanlış tahminleri göster"""
        wrong_predictions = results_df[~results_df['correct']]

        if len(wrong_predictions) == 0:
            logger.info("✓ Tüm tahminler doğru!")
            return

        logger.info(f"\nYanlış Tahminler ({len(wrong_predictions)}):")
        logger.info("-" * 80)

        for idx, row in wrong_predictions.head(20).iterrows():
            logger.info(
                f"{row['file_name']}: "
                f"Gerçek={row['true_label']}, Tahmin={row['predicted_label']}"
            )

        if len(wrong_predictions) > 20:
            logger.info(f"... ve {len(wrong_predictions) - 20} daha")


def main():
    """Ana fonksiyon"""

    # Konfigürasyon
    config = {
        'img_width': 128,
        'img_height': 32,
        'batch_size': 32
    }

    model_path = 'models/crnn_best_model.pth'
    test_csv = 'data/processed/test.csv'
    img_dir = 'data/cropped_fields'
    output_dir = 'logs/evaluation'

    # Model kontrol et
    if not Path(model_path).exists():
        logger.error(f"Model bulunamadı: {model_path}")
        sys.exit(1)

    if not Path(test_csv).exists():
        logger.error(f"Test CSV bulunamadı: {test_csv}")
        sys.exit(1)

    # Değerlendir
    evaluator = ModelEvaluator(model_path, config)

    logger.info("Dataloader oluşturuluyor...")
    dataloader, dataset = evaluator.setup_dataloader(test_csv, img_dir)

    logger.info("Tahminler yapılıyor...")
    preds, labels, files = evaluator.evaluate(dataloader)

    logger.info("Sonuçlar kaydediliyor...")
    metrics, results_df = evaluator.save_results(preds, labels, files, output_dir)

    evaluator.print_metrics(metrics)
    evaluator.print_confusion_details(preds, labels, results_df)

    logger.info("✓ Değerlendirme tamamlandı!")


if __name__ == '__main__':
    main()
