"""
Sentetik MNIST şeritleriyle CRNN ön-eğitimi.

Model önce yapay rakam dizileriyle rakam okumayı öğrenir; ardından
train_crnn.py gerçek verilerle ince ayar yapar (models/crnn_pretrained.pth
varsa otomatik yüklenir).
"""

import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.train_crnn import (
    CRNN, CRNNDataset, BLANK_IDX, encode_label, decode_greedy,
    get_device, levenshtein
)
from training.synthetic_dataset import SyntheticDigitStrips

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

IMG_H, IMG_W = 32, 256
BATCH_SIZE = 64
NUM_EPOCHS = 15
SAMPLES_PER_EPOCH = 20000
LR = 0.001


def compute_ctc_loss(criterion, logits, labels):
    T, N, _ = logits.shape
    targets = torch.cat([
        torch.tensor(encode_label(label), dtype=torch.long) for label in labels
    ])
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    input_lengths = torch.full((N,), T, dtype=torch.long)
    return criterion(logits.log_softmax(2).cpu(), targets, input_lengths, target_lengths)


def evaluate_on_real(model, loader, device):
    """Gerçek validation seti üzerinde doğruluk ve CER ölç"""
    model.eval()
    correct, total, char_err, char_total = 0, 0, 0, 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            preds = decode_greedy(model(images))
            for pred, true in zip(preds, batch['label']):
                correct += (pred == true)
                total += 1
                char_err += levenshtein(pred, true)
                char_total += len(true)

    return correct / max(total, 1), char_err / max(char_total, 1)


def main():
    device = get_device()
    logger.info(f"Cihaz: {device}")

    transform_train = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.RandomAffine(degrees=2, translate=(0.03, 0.06),
                                scale=(0.92, 1.08), shear=2, fill=255),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    transform_val = transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    logger.info("MNIST indiriliyor / yükleniyor (~11 MB, ilk seferde)...")
    synth = SyntheticDigitStrips('data/mnist', num_samples=SAMPLES_PER_EPOCH,
                                 transform=transform_train, seed=42)
    synth_loader = DataLoader(synth, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    real_val = CRNNDataset('data/processed/val.csv', 'data/cropped_fields',
                           transform=transform_val)
    real_val_loader = DataLoader(real_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_cer = float('inf')
    out_path = Path('models/crnn_pretrained.pth')
    out_path.parent.mkdir(exist_ok=True)

    logger.info(f"Ön-eğitim başladı: {NUM_EPOCHS} epoch × {SAMPLES_PER_EPOCH} sentetik örnek")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        pbar = tqdm(synth_loader, desc=f'Ön-eğitim {epoch + 1}/{NUM_EPOCHS}')
        for batch in pbar:
            images = batch['image'].to(device)
            optimizer.zero_grad()
            loss = compute_ctc_loss(criterion, model(images), batch['label'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

        scheduler.step()

        real_acc, real_cer = evaluate_on_real(model, real_val_loader, device)
        logger.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
            f"Sentetik loss: {total_loss / len(synth_loader):.4f} | "
            f"GERÇEK veri: doğruluk %{real_acc * 100:.2f}, rakam hatası %{real_cer * 100:.2f}"
        )

        if real_cer < best_cer:
            best_cer = real_cer
            torch.save({'model_state': model.state_dict(), 'real_cer': real_cer}, out_path)
            logger.info(f"✓ Ön-eğitilmiş model kaydedildi (gerçek CER: %{real_cer * 100:.2f})")

    logger.info("")
    logger.info(f"✓ Ön-eğitim bitti. En iyi gerçek-veri CER: %{best_cer * 100:.2f}")
    logger.info(f"Model: {out_path}")
    logger.info("Sıradaki adım (ince ayar): python3 training/train_crnn.py")


if __name__ == '__main__':
    main()
