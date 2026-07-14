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
            lambda f: 'Öğrenci Numarası' if 'ogrenci_numara' in f else ('Not' if '_not' in f else 'Diğer')
        )

        by_type = {}
        for field_type, group in results_df.groupby('field_type'):
            by_type[field_type] = {
                'accuracy': float(group['correct'].mean()),
                'correct': int(group['correct'].sum()),
                'samples': len(group),
            }
        metrics['by_field_type'] = by_type

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_csv = output_path / f'{set_name}_results_{timestamp}.csv'
        results_df.to_csv(results_csv, index=False)

        metrics_json = output_path / f'{set_name}_metrics_{timestamp}.json'
        with open(metrics_json, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Herkesin okuyabileceği düz metin rapor
        rapor_txt = output_path / f'{set_name}_rapor_{timestamp}.txt'
        self.write_simple_report(rapor_txt, metrics, results_df)

        logger.info(f"\nSonuçlar kaydedildi:")
        logger.info(f"  - Okunabilir rapor: {rapor_txt}")
        logger.info(f"  - Detay CSV:        {results_csv}")
        logger.info(f"  - Metrikler JSON:   {metrics_json}")

        return metrics, results_df, rapor_txt

    def build_simple_report(self, metrics, results_df):
        """Basit, anlaşılır karşılaştırma raporunu metin olarak oluştur"""
        lines = []
        add = lines.append

        add("=" * 70)
        add("MODEL KARNESİ — Test Sonuçları")
        add("=" * 70)
        add("")
        add("Model, daha önce HİÇ GÖRMEDİĞİ sınav kağıdı alanlarını okudu.")
        add("Aşağıda gerçek değer ile modelin okuduğu değer karşılaştırılıyor.")
        add("")

        # Genel özet
        total = metrics['total_samples']
        correct = metrics['correct_samples']
        add("-" * 70)
        add("GENEL SONUÇ")
        add("-" * 70)
        add(f"  Toplam okunan alan : {total}")
        add(f"  Doğru okunan       : {correct}  (%{correct / max(total, 1) * 100:.1f})")
        add(f"  Yanlış okunan      : {total - correct}  (%{(total - correct) / max(total, 1) * 100:.1f})")
        add("")

        # Alan tipine göre
        add("-" * 70)
        add("ALAN TİPİNE GÖRE BAŞARI")
        add("-" * 70)
        for field_type, stats in metrics['by_field_type'].items():
            bar_len = int(stats['accuracy'] * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            add(f"  {field_type:<18}: {bar} %{stats['accuracy'] * 100:5.1f}  ({stats['correct']}/{stats['samples']} doğru)")
        add("")

        # Yanlış okunanlar — tam liste, tip bazında
        wrong = results_df[~results_df['correct']]

        add("-" * 70)
        if len(wrong) == 0:
            add("✓ TÜM ALANLAR DOĞRU OKUNDU — hiç hata yok!")
        else:
            add(f"YANLIŞ OKUNANLAR — {len(wrong)} alan")
            add("-" * 70)
            add("  (Gerçek değer → Modelin okuduğu. 'fark' = kaç rakam hatalı)")
            add("")
            for field_type, group in wrong.groupby('field_type'):
                add(f"  ▼ {field_type} ({len(group)} hata):")
                for _, row in group.iterrows():
                    pred_shown = row['predicted_label'] if row['predicted_label'] else '(boş okudu)'
                    add(f"    ✗ {row['true_label']:>12} → {pred_shown:<12} (fark: {row['edit_distance']} rakam)   [{row['file_name']}]")
                add("")

        # Doğru okunanlardan örnekler
        right = results_df[results_df['correct']]
        if len(right) > 0:
            add("-" * 70)
            add(f"DOĞRU OKUNANLARDAN ÖRNEKLER (toplam {len(right)} doğru)")
            add("-" * 70)
            for field_type, group in right.groupby('field_type'):
                add(f"  ▼ {field_type}:")
                for _, row in group.head(5).iterrows():
                    add(f"    ✓ {row['true_label']:>12} → {row['predicted_label']}")
                add("")

        add("=" * 70)
        add("NASIL YORUMLANIR?")
        add("  • %90 üzeri doğruluk  : Model iyi durumda")
        add("  • %70-90 arası        : Daha fazla veri veya epoch gerekebilir")
        add("  • %70 altı            : Veri kalitesini ve etiketleri kontrol edin")
        add("  • 'fark: 1 rakam'     : Model sadece 1 rakamı karıştırmış (yakın hata)")
        add("=" * 70)

        return '\n'.join(lines)

    def write_simple_report(self, path, metrics, results_df):
        """Raporu dosyaya yaz"""
        report = self.build_simple_report(metrics, results_df)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report + '\n')

    def print_simple_report(self, metrics, results_df):
        """Raporu ekrana yazdır"""
        print('\n' + self.build_simple_report(metrics, results_df) + '\n')


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

    metrics, results_df, rapor_txt = evaluator.save_results(preds, labels, files, output_dir)

    evaluator.print_simple_report(metrics, results_df)

    logger.info(f"✓ Değerlendirme tamamlandı! Rapor: {rapor_txt}")


if __name__ == '__main__':
    main()
