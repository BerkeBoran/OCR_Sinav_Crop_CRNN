"""Visualizasyon Fonksiyonları"""

import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import numpy as np

# Matplotlib'i non-interactive mod'da çalıştır
matplotlib.use('Agg')


def plot_training_history(history, save_path='logs/training_history.png'):
    """Eğitim geçmişini grafiğe dök"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_confusion_matrix(cm, labels, save_path='logs/confusion_matrix.png'):
    """Confusion matrix'i visualize et"""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        ylabel='True label',
        xlabel='Predicted label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Heatmap'a değerleri yazdır
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.title('Confusion Matrix')
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def plot_sample_predictions(images, true_labels, pred_labels, save_path='logs/sample_predictions.png'):
    """Örnek tahminleri visualize et"""
    num_samples = min(12, len(images))

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(num_samples):
        ax = axes[i]

        # Resmi normalize et
        img = images[i].numpy()
        if len(img.shape) == 3:
            img = img[0]  # Kanal

        ax.imshow(img, cmap='gray')
        ax.set_title(
            f"True: {true_labels[i]}\nPred: {pred_labels[i]}",
            fontsize=8
        )
        ax.axis('off')

    for i in range(num_samples, 12):
        axes[i].axis('off')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def create_summary_report(metrics, report, save_path='logs/summary_report.txt'):
    """Özet rapport oluştur"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n\n")

        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n")
        f.write(str(report))

    return save_path
