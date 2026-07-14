import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import logging
from datetime import datetime
import json
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import platform

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CRNNDataset(Dataset):
    """CRNN eğitim veri seti"""

    def __init__(self, csv_path, img_dir, transform=None, max_width=128, max_height=32):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.max_width = max_width
        self.max_height = max_height

        # Geçerli dosyaları filtrele
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            img_path = self.img_dir / row['file_name']
            if img_path.exists():
                self.valid_indices.append(idx)

        logger.info(f"Dataset oluşturuldu: {len(self.valid_indices)} geçerli örnek")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]

        # Resim yükle
        img_path = self.img_dir / row['file_name']
        image = Image.open(img_path).convert('L')

        # Transform uygula
        if self.transform:
            image = self.transform(image)

        # Label'ı sayıya dönüştür
        label = str(row['label'])

        return {
            'image': image,
            'label': label,
            'file_name': row['file_name']
        }


class CRNN(nn.Module):
    """CNN-RNN tabanlı CRNN modeli"""

    def __init__(self, num_classes, cnn_channels=[32, 64, 128], rnn_hidden=256):
        super(CRNN, self).__init__()

        # CNN kısmı
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
        )

        # RNN kısmı
        self.rnn = nn.Sequential(
            nn.LSTM(cnn_channels[2] * 4, rnn_hidden, batch_first=True, bidirectional=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # CNN çıktısı: (batch, channels, height, width)
        cnn_out = self.cnn(x)

        # RNN için reshape: (batch, width, channels * height)
        batch, channels, height, width = cnn_out.size()
        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous()
        cnn_out = cnn_out.view(batch, width, channels * height)

        # RNN çıktısı
        rnn_out, _ = self.rnn(cnn_out)

        # Global average pooling
        rnn_out = rnn_out.mean(dim=1)

        # Sınıf tahmini
        out = self.fc(rnn_out)

        return out


class Trainer:
    """Model eğitim sınıfı"""

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Cihaz: {self.device}")
        logger.info(f"İşletim Sistemi: {platform.system()}")

        # Model kurulumu
        self.setup_model()
        self.setup_dataloaders()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def setup_model(self):
        """Model ve unique label'ları setup et"""
        processed_dir = Path(self.config['data_dir']) / 'processed'

        # Unique label'ları bul
        train_csv = processed_dir / 'train.csv'
        df = pd.read_csv(train_csv)
        self.unique_labels = sorted(df['label'].astype(str).unique())
        self.num_classes = len(self.unique_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.unique_labels)}

        logger.info(f"Sınıf sayısı: {self.num_classes}")

        self.model = CRNN(self.num_classes).to(self.device)
        logger.info(f"Model parametreleri: {sum(p.numel() for p in self.model.parameters()):,}")

    def setup_dataloaders(self):
        """Eğitim ve validation dataloaderlarını oluştur"""
        data_dir = Path(self.config['data_dir'])
        processed_dir = data_dir / 'processed'

        # Transform
        transform = transforms.Compose([
            transforms.Resize((self.config['img_height'], self.config['img_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Veri setleri
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

        # Dataloaderlar
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        logger.info(f"Eğitim veri seti: {len(train_dataset)} örnek")
        logger.info(f"Validation veri seti: {len(val_dataset)} örnek")

    def train_epoch(self):
        """Bir epoch eğitimi"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc='Eğitim')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label']

            # Label'ları indekse dönüştür
            label_indices = torch.tensor(
                [self.label_to_idx[label] for label in labels],
                device=self.device
            )

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, label_indices)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item():.4f})

        return total_loss / len(self.train_loader)

    def validate(self):
        """Validation döngüsü"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label']

                # Label'ları indekse dönüştür
                label_indices = torch.tensor(
                    [self.label_to_idx[label] for label in labels],
                    device=self.device
                )

                outputs = self.model(images)
                loss = self.criterion(outputs, label_indices)
                total_loss += loss.item()

                # Tahminler
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label_indices.cpu().numpy())

                pbar.set_postfix({'loss': loss.item():.4f})

        val_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return val_loss, accuracy

    def train(self):
        """Tam eğitim döngüsü"""
        logger.info(f"Eğitim başladı - Epoch: {self.config['num_epochs']}")

        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Scheduler adımı
            self.scheduler.step(val_loss)

            # En iyi model'i kaydet
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch)
                logger.info(f"✓ En iyi model kaydedildi (Val Loss: {val_loss:.4f})")

        self.save_history()

    def save_model(self, epoch):
        """Model'i kaydet"""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'label_to_idx': self.label_to_idx,
            'unique_labels': self.unique_labels
        }

        model_path = models_dir / f"crnn_best_model.pth"
        torch.save(checkpoint, model_path)
        logger.info(f"Model kaydedildi: {model_path}")

    def save_history(self):
        """Eğitim geçmişini kaydet"""
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)

        history_path = logs_dir / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Eğitim geçmişi kaydedildi: {history_path}")


def main():
    """Ana fonksiyon"""
    config_path = 'configs/training_config.yaml'

    if not Path(config_path).exists():
        logger.error(f"Konfigürasyon dosyası bulunamadı: {config_path}")
        sys.exit(1)

    trainer = Trainer(config_path)
    trainer.train()
    logger.info("✓ Eğitim tamamlandı!")


if __name__ == '__main__':
    main()
