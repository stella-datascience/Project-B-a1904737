import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import seaborn as sns

def plot_sample_predictions(model, test_dataset, class_names, num_samples=20, save_path=None):
    """Plot sample predictions with true and predicted labels"""
    model.ema_model.eval()
    
    # Create a data loader with batch size = num_samples
    sample_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_samples, shuffle=True)
    data_iter = iter(sample_loader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        images = images.to(model.device)
        outputs = model.ema_model(images)
        _, predictions = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, 1)
        confidences = probabilities.max(1)[0]
    
    images = images.cpu()
    predictions = predictions.cpu()
    labels = labels.cpu()
    confidences = confidences.cpu()
    
    # Plot
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx in range(num_samples):
        image = images[idx].permute(1, 2, 0)
        # Denormalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        pred_class = class_names[predictions[idx]]
        true_class = class_names[labels[idx]]
        confidence = confidences[idx]
        
        correct = predictions[idx] == labels[idx]
        color = 'green' if correct else 'red'
        
        axes[idx].imshow(image)
        axes[idx].set_title(f'True: {true_class}\nPred: {pred_class}\nConf: {confidence:.3f}', 
                          color=color, fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_distribution(labels, class_names, title="Class Distribution", save_path=None):
    """Plot distribution of classes"""
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar([class_names[i] for i in unique], counts, color='lightblue', alpha=0.7)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_space_comparison(embeddings1, labels1, embeddings2, labels2, 
                                title1="Source", title2="Target", save_path=None):
    """Compare feature spaces using PCA"""
    from sklearn.decomposition import PCA
    
    # Combine embeddings
    all_embeddings = np.vstack([embeddings1, embeddings2])
    all_labels = np.concatenate([labels1, labels2])
    dataset_indicator = np.concatenate([np.zeros(len(embeddings1)), np.ones(len(embeddings2))])
    
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    all_embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Split back
    embeddings1_2d = all_embeddings_2d[:len(embeddings1)]
    embeddings2_2d = all_embeddings_2d[len(embeddings1):]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Source domain
    scatter1 = ax1.scatter(embeddings1_2d[:, 0], embeddings1_2d[:, 1], 
                          c=labels1, cmap='tab10', alpha=0.6, s=10)
    ax1.set_title(f'{title1} Domain (CIFAR-100)', fontsize=14)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1)
    
    # Target domain
    scatter2 = ax2.scatter(embeddings2_2d[:, 0], embeddings2_2d[:, 1], 
                          c=labels2, cmap='tab10', alpha=0.6, s=10)
    ax2.set_title(f'{title2} Domain (CIFAR-10)', fontsize=14)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confidence_calibration(predictions, targets, probs, save_path=None):
    """Plot reliability diagram for confidence calibration"""
    from sklearn.calibration import calibration_curve
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(targets, probs.max(axis=1), n_bins=10)
    
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    
    # Model calibration
    plt.plot(prob_pred, prob_true, 's-', label='Model')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add histogram of predicted probabilities
    plt.twinx()
    plt.hist(probs.max(axis=1), bins=20, alpha=0.3, color='gray')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()