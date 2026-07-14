"""
MNIST rakamlarından sentetik "numara şeridi" üreten dataset.

Gerçek sınav kağıdı alanlarını taklit eder:
- Not alanları: 1-3 haneli sayılar
- Öğrenci numaraları: 9-12 haneli sayılar
- Soldaki matbaa yazısı ("NUMARASI:" gibi) gri metinle taklit edilir
"""

import random
import string

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import datasets


# Gerçek verideki uzunluk dağılımını taklit et (not: 1-3, numara: 9-12 hane)
LENGTH_CHOICES = [1, 2, 2, 2, 3, 9, 9, 10, 10, 10, 11, 12]


class SyntheticDigitStrips(Dataset):
    """MNIST rakamlarını yan yana dizerek sentetik numara görüntüleri üretir"""

    def __init__(self, mnist_root, num_samples=20000, transform=None, seed=None):
        self.num_samples = num_samples
        self.transform = transform
        self.rng = random.Random(seed)

        mnist = datasets.MNIST(root=mnist_root, train=True, download=True)

        # Rakamlara göre gruplandır (numpy, siyah rakam / beyaz zemin olacak şekilde ters çevrilmiş)
        self.digit_images = {d: [] for d in range(10)}
        data = mnist.data.numpy()      # (60000, 28, 28), beyaz rakam / siyah zemin
        targets = mnist.targets.numpy()

        for img, label in zip(data, targets):
            self.digit_images[int(label)].append(255 - img)  # ters çevir

    def __len__(self):
        return self.num_samples

    def _make_strip(self):
        rng = self.rng
        length = rng.choice(LENGTH_CHOICES)
        digits = [rng.randrange(10) for _ in range(length)]
        label = ''.join(str(d) for d in digits)

        canvas_h = 48
        bg = rng.randint(215, 255)  # hafif değişken zemin tonu

        # Rakam görüntülerini seç ve rastgele ölçekle
        tiles = []
        for d in digits:
            arr = rng.choice(self.digit_images[d])
            scale = rng.uniform(0.8, 1.3)
            size = max(16, int(28 * scale))
            tile = np.array(
                Image.fromarray(arr).resize((size, size), Image.BILINEAR)
            )
            tiles.append(tile)

        spacing = [rng.randint(-2, 8) for _ in tiles]
        left_margin = rng.randint(4, 60)
        right_margin = rng.randint(4, 40)

        width = left_margin + sum(t.shape[1] for t in tiles) + sum(spacing) + right_margin
        canvas = np.full((canvas_h, max(width, 32)), bg, dtype=np.uint8)

        # Rakamları yerleştir (koyu piksel kazanır)
        x = left_margin
        for tile, gap in zip(tiles, spacing):
            th, tw = tile.shape
            y = rng.randint(0, max(0, canvas_h - th))
            # Zemine göre normalize: MNIST'in beyazı zeminle aynı olsun
            tile_adj = np.minimum(tile.astype(np.int16) + (bg - 255), 255).clip(0, 255).astype(np.uint8)
            region = canvas[y:y + th, x:x + tw]
            canvas[y:y + th, x:x + tw] = np.minimum(region, tile_adj)
            x += tw + gap

        img = Image.fromarray(canvas, mode='L')

        # Matbaa yazısı taklidi: sola gri büyük harf + ':' (gerçek kırpımlardaki
        # "NUMARASI:", "NOT:" yazıları gibi) — model bunu yok saymayı öğrenir
        if rng.random() < 0.5 and left_margin >= 30:
            draw = ImageDraw.Draw(img)
            text = ''.join(rng.choices(string.ascii_uppercase, k=rng.randint(2, 6))) + ':'
            gray = rng.randint(40, 130)
            draw.text((2, rng.randint(2, 30)), text, fill=gray)

        return img, label

    def __getitem__(self, idx):
        img, label = self._make_strip()

        if self.transform:
            img = self.transform(img)

        return {
            'image': img,
            'label': label,
            'file_name': f'synthetic_{idx}.png'
        }
