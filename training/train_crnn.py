"""
CRNN (CNN + BiLSTM + CTC) modeli ile rakam dizisi tanıma eğitimi.

Öğrenci numarası ve not alanlarındaki rakam dizilerini karakter karakter okur.
CTC loss sayesinde model eğitimde görmediği numaraları da okuyabilir.
MacBook (MPS), NVIDIA GPU (CUDA) ve CPU destekler.
"""

import os

# MPS'de desteklenmeyen operasyonlar için CPU fallback (torch import'undan önce)
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import sys
import yaml
import json
import logging
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import multiprocessing

Path('logs').mkdir(exist_ok=True)

# Dosya log'u sadece ana süreçte aç — DataLoader worker'ları boş log oluşturmasın
_handlers = [logging.StreamHandler()]
if multiprocessing.parent_process() is None:
    _handlers.append(
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=_handlers
)
logger = logging.getLogger(__name__)

# Karakter seti: rakamlar. Index 0 CTC blank için ayrılmıştır.
CHARSET = '0123456789'
BLANK_IDX = 0
NUM_CLASSES = len(CHARSET) + 1  # +1 blank

CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}


def encode_label(label):
    """Label string'ini indeks dizisine çevir"""
    return [CHAR_TO_IDX[c] for c in str(label)]


def decode_greedy(logits):
    """
    CTC greedy decode: (T, N, C) logits → tahmin string listesi.
    Ardışık tekrarları birleştir, blank'leri at.
    """
    preds = logits.argmax(dim=2).permute(1, 0)  # (N, T)
    results = []
    for seq in preds:
        chars = []
        prev = BLANK_IDX
        for idx in seq.tolist():
            if idx != BLANK_IDX and idx != prev:
                chars.append(IDX_TO_CHAR[idx])
            prev = idx
        results.append(''.join(chars))
    return results


def get_device():
    """Platforma göre en iyi cihazı seç"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class CRNNDataset(Dataset):
    """Rakam dizisi tanıma veri seti"""

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Geçerli dosyaları ve label'ları filtrele
        self.samples = []
        skipped_label = 0
        skipped_image = 0

        for _, row in self.df.iterrows():
            label = str(row['label']).strip()

            # Sadece charset'teki karakterlerden oluşan label'lar
            if not label or not all(c in CHAR_TO_IDX for c in label):
                skipped_label += 1
                continue

            img_path = self._resolve_image_path(row)
            if img_path is None:
                skipped_image += 1
                continue

            self.samples.append((img_path, label))

        logger.info(
            f"Dataset: {len(self.samples)} örnek "
            f"(atlanan: {skipped_label} geçersiz label, {skipped_image} eksik resim)"
        )

    def _resolve_image_path(self, row):
        """Resim yolunu çöz: önce image_path, sonra img_dir/type/file_name"""
        candidates = []

        img_path_col = row.get('image_path')
        if pd.notna(img_path_col):
            candidates.append(Path(img_path_col))

        file_name = row.get('file_name')
        field_type = row.get('type')
        if pd.notna(file_name):
            if pd.notna(field_type):
                candidates.append(self.img_dir / str(field_type) / str(file_name))
            candidates.append(self.img_dir / str(file_name))

        return next((c for c in candidates if c.exists()), None)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'file_name': img_path.name
        }


class CRNN(nn.Module):
    """CNN + BiLSTM + CTC tabanlı sekans tanıma modeli"""

    def __init__(self, num_classes=NUM_CLASSES, rnn_hidden=256):
        super().__init__()

        # CNN: (1, 32, W) → (512, 1, W/4)
        # Yükseklik agresif, genişlik hafif düşürülür ki zaman adımları korunur
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 64 x 16 x W/2

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 128 x 8 x W/4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256 x 4 x W/4

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512 x 2 x W/4

            nn.Conv2d(512, 512, 2),        # 512 x 1 x (W/4 - 1)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(
            512, rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=False,
            dropout=0.2
        )

        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)                     # (N, 512, 1, T)
        conv = conv.squeeze(2)                 # (N, 512, T)
        conv = conv.permute(2, 0, 1)           # (T, N, 512)

        rnn_out, _ = self.rnn(conv)            # (T, N, 2*hidden)
        logits = self.fc(rnn_out)              # (T, N, num_classes)

        return logits


class Trainer:
    """Model eğitim sınıfı"""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = get_device()
        logger.info(f"Cihaz: {self.device}")
        logger.info(f"İşletim Sistemi: {platform.system()} ({platform.machine()})")

        torch.manual_seed(self.config.get('seed', 42))

        self.model = CRNN(rnn_hidden=self.config.get('rnn_hidden', 256)).to(self.device)
        logger.info(f"Model parametreleri: {sum(p.numel() for p in self.model.parameters()):,}")

        self.setup_dataloaders()

        self.criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0)
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.get('scheduler_factor', 0.5),
            patience=self.config.get('scheduler_patience', 5)
        )

        self.best_val_acc = -1.0  # ilk epoch her zaman kaydedilsin
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_cer': []}

    def setup_dataloaders(self):
        """Eğitim ve validation dataloader'larını oluştur"""
        data_dir = Path(self.config['data_dir'])
        processed_dir = data_dir / 'processed'

        transform = transforms.Compose([
            transforms.Resize((self.config['img_height'], self.config['img_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train_dataset = CRNNDataset(
            processed_dir / 'train.csv',
            data_dir / 'cropped_fields',
            transform=transform
        )
        val_dataset = CRNNDataset(
            processed_dir / 'val.csv',
            data_dir / 'cropped_fields',
            transform=transform
        )

        num_workers = self.config.get('num_workers', 2)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=num_workers
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers
        )

        logger.info(f"Eğitim veri seti: {len(train_dataset)} örnek")
        logger.info(f"Validation veri seti: {len(val_dataset)} örnek")

    def compute_ctc_loss(self, logits, labels):
        """CTC loss hesapla"""
        T, N, _ = logits.shape

        targets = torch.cat([
            torch.tensor(encode_label(label), dtype=torch.long)
            for label in labels
        ])
        target_lengths = torch.tensor(
            [len(label) for label in labels], dtype=torch.long
        )
        input_lengths = torch.full((N,), T, dtype=torch.long)

        log_probs = logits.log_softmax(2)

        # CTC loss CPU'da daha stabil (özellikle MPS'de)
        return self.criterion(
            log_probs.cpu(), targets, input_lengths, target_lengths
        )

    def train_epoch(self):
        """Bir epoch eğitim"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc='Eğitim')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label']

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.compute_ctc_loss(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validation: loss + exact-match accuracy + karakter hata oranı (CER)"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        char_errors = 0
        char_total = 0
        examples = []  # epoch sonunda gösterilecek örnek tahminler

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label']

                logits = self.model(images)
                loss = self.compute_ctc_loss(logits, labels)
                total_loss += loss.item()

                preds = decode_greedy(logits)
                for pred, true in zip(preds, labels):
                    if pred == true:
                        correct += 1
                    total += 1
                    char_errors += levenshtein(pred, true)
                    char_total += len(true)

                    if len(examples) < 6:
                        examples.append((true, pred))

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss = total_loss / len(self.val_loader)
        accuracy = correct / max(total, 1)
        cer = char_errors / max(char_total, 1)

        return val_loss, accuracy, cer, examples

    def _trend(self, current, previous, lower_is_better):
        """Önceki epoch'a göre iyileşme/kötüleşme işareti üret"""
        if previous is None:
            return ''
        if current == previous:
            return '→ değişmedi'
        improved = (current < previous) if lower_is_better else (current > previous)
        arrow = '↓' if current < previous else '↑'
        return f"{arrow} {'iyileşiyor ✓' if improved else 'kötüleşti ⚠'}"

    def print_epoch_summary(self, epoch, num_epochs, train_loss, val_loss, val_acc, val_cer, examples):
        """Epoch sonucunu herkesin anlayacağı şekilde yazdır"""
        h = self.history
        prev = lambda key: h[key][-1] if h[key] else None

        logger.info("")
        logger.info("─" * 68)
        logger.info(f"EPOCH {epoch + 1}/{num_epochs} SONUCU")
        logger.info("─" * 68)
        logger.info(f"  Hata puanı (train loss) : {train_loss:8.4f}  {self._trend(train_loss, prev('train_loss'), True):<20} [AZALMALI]")
        logger.info(f"  Hata puanı (val loss)   : {val_loss:8.4f}  {self._trend(val_loss, prev('val_loss'), True):<20} [AZALMALI]")
        logger.info(f"  Doğru okuma oranı       : %{val_acc * 100:6.2f}  {self._trend(val_acc, prev('val_acc'), False):<20} [ARTMALI]")
        logger.info(f"  Rakam hata oranı (CER)  : %{val_cer * 100:6.2f}  {self._trend(val_cer, prev('val_cer'), True):<20} [AZALMALI]")
        logger.info("")
        logger.info("  Örnek okumalar (gerçek → modelin okuduğu):")
        for true, pred in examples:
            mark = '✓' if true == pred else '✗'
            logger.info(f"    {mark} {true} → {pred if pred else '(boş)'}")
        logger.info("─" * 68)

    def train(self):
        """Tam eğitim döngüsü"""
        num_epochs = self.config['num_epochs']
        logger.info(f"Eğitim başladı - Epoch: {num_epochs}")
        logger.info("")
        logger.info("NASIL OKUNUR?")
        logger.info("  • Hata puanı (loss)     : Modelin hata ölçüsü — her epoch AZALMALI")
        logger.info("  • Doğru okuma oranı     : Tamamen doğru okunan alan yüzdesi — ARTMALI")
        logger.info("  • Rakam hata oranı (CER): Yanlış okunan rakam yüzdesi — AZALMALI")
        logger.info("  İlk epoch'larda sonuçlar kötüdür, zamanla düzelir. Bu normaldir.")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc, val_cer, examples = self.validate()

            self.print_epoch_summary(
                epoch, num_epochs, train_loss, val_loss, val_acc, val_cer, examples
            )

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_cer'].append(val_cer)

            self.scheduler.step(val_loss)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(epoch, val_acc)
                logger.info(f"✓ Şu ana kadarki EN İYİ model kaydedildi (doğruluk: %{val_acc * 100:.2f})")

        self.save_history()
        logger.info("")
        logger.info(f"EĞİTİM BİTTİ — En iyi doğru okuma oranı: %{self.best_val_acc * 100:.2f}")
        logger.info("Sıradaki adım: python3 training/evaluate_model.py")

    def save_model(self, epoch, val_acc):
        """Model checkpoint'ini kaydet"""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'val_acc': val_acc,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'charset': CHARSET,
        }

        torch.save(checkpoint, models_dir / 'crnn_best_model.pth')

    def save_history(self):
        """Eğitim geçmişini kaydet"""
        history_path = Path('logs') / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Eğitim geçmişi kaydedildi: {history_path}")

        # Grafik oluştur
        try:
            from utils.visualization import plot_training_history
            plot_path = plot_training_history(self.history)
            logger.info(f"Eğitim grafiği kaydedildi: {plot_path}")
        except Exception as e:
            logger.warning(f"Grafik oluşturulamadı: {e}")


def levenshtein(s1, s2):
    """İki string arasındaki edit distance"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            current.append(min(
                previous[j + 1] + 1,      # silme
                current[j] + 1,           # ekleme
                previous[j] + (c1 != c2)  # değiştirme
            ))
        previous = current

    return previous[-1]


def main():
    config_path = 'configs/training_config.yaml'

    if not Path(config_path).exists():
        logger.error(f"Konfigürasyon dosyası bulunamadı: {config_path}")
        sys.exit(1)

    trainer = Trainer(config_path)
    trainer.train()
    logger.info("✓ Eğitim tamamlandı!")


if __name__ == '__main__':
    main()
