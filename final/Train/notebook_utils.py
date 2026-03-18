import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

# Local imports - ensuring paths work from Train/ directory or project root
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.multimodal_dataset import MultimodalDataset
from dataset.collate import multimodal_collate

def load_val_data(feature_root="features", batch_size=32, num_workers=0):
    """
    Loads the validation dataset and returns a DataLoader.
    """
    val_dataset = MultimodalDataset(f"{feature_root}/validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=multimodal_collate,
    )
    print(f"Loaded {len(val_dataset)} validation samples.")
    return val_loader

def evaluate_model(model, loader, device="cuda"):
    """
    Evaluates the model on the provided loader and plots a confusion matrix.
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    all_preds = []
    all_labels = []

    class_names = ["happy", "sad", "neutral", "fear", "disgust"]

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            texts, audios, visuals, _, _, labels = batch
            
            texts = texts.to(device)
            audios = audios.to(device)
            visuals = visuals.to(device)
            labels = labels.squeeze(1).to(device)

            _, _, _, _, logits = model(texts, audios, visuals)
            
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            functional.reset_net(model)

    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*30)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print("="*30)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Validation Data')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return {
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "predictions": all_preds,
        "labels": all_labels
    }
