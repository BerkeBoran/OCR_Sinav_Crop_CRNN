"""
Eğitilmiş CRNN modeliyle tek görüntüden rakam dizisi okuma.

Kullanım:
    from inference.predictor import CRNNPredictor
    predictor = CRNNPredictor()
    text, confidence = predictor.predict(pil_image)
"""

import sys
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train_crnn import CRNN, BLANK_IDX, IDX_TO_CHAR

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'crnn_best_model.pth'


class CRNNPredictor:
    """Tek görüntüden rakam dizisi okuyan tahmin sınıfı (CPU uyumlu)"""

    def __init__(self, model_path=None, device=None):
        self.device = torch.device(device) if device else torch.device('cpu')

        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {model_path}\n"
                "Önce modeli eğitin: python3 training/train_crnn.py"
            )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})

        self.model = CRNN(rnn_hidden=config.get('rnn_hidden', 256)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((config.get('img_height', 32), config.get('img_width', 256))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.checkpoint_info = {
            'epoch': checkpoint.get('epoch'),
            'val_acc': checkpoint.get('val_acc'),
        }

    @torch.no_grad()
    def predict(self, image):
        """
        Görüntüden rakam dizisi oku.

        Args:
            image: PIL.Image veya dosya yolu

        Returns:
            (text, confidence): okunan rakam dizisi ve 0-1 arası güven skoru
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image)

        tensor = self.transform(image.convert('L')).unsqueeze(0).to(self.device)

        logits = self.model(tensor)            # (T, 1, C)
        probs = logits.softmax(dim=2)[:, 0, :]  # (T, C)

        # CTC greedy decode + güven skoru (seçilen karakterlerin ortalama olasılığı)
        best_probs, best_indices = probs.max(dim=1)

        chars = []
        char_probs = []
        prev = BLANK_IDX
        for idx, p in zip(best_indices.tolist(), best_probs.tolist()):
            if idx != BLANK_IDX and idx != prev:
                chars.append(IDX_TO_CHAR[idx])
                char_probs.append(p)
            prev = idx

        text = ''.join(chars)
        confidence = sum(char_probs) / len(char_probs) if char_probs else 0.0

        return text, confidence


if __name__ == '__main__':
    # Hızlı komut satırı testi: python3 inference/predictor.py <resim_yolu>
    if len(sys.argv) < 2:
        print("Kullanım: python3 inference/predictor.py <resim_yolu>")
        sys.exit(1)

    predictor = CRNNPredictor()
    text, conf = predictor.predict(sys.argv[1])
    print(f"Okunan: {text} (güven: %{conf * 100:.1f})")
