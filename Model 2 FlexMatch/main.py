import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import sys

from flexmatch import FlexMatch
from flexmatch_utils import get_cifar100_dataset, get_cifar10_datasets

def pretrain_on_cifar100():
    """Pre-train model on CIFAR-100"""
    print("Pre-training on CIFAR-100...")
    
    # Load CIFAR-100 datasets
    train_dataset, test_dataset = get_cifar100_dataset()
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create model for CIFAR-100 (100 classes)
    model = FlexMatch(num_classes=100, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training loop
    model.model.train()
    optimizer = torch.optim.SGD(model.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    best_acc = 0
    for epoch in range(2):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(model.device), target.to(model.device)
            
            optimizer.zero_grad()
            output = model.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        scheduler.step()
        
        # Evaluate on test set
        test_acc = evaluate_supervised(model.model, test_loader, 100)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.model.state_dict(), 'checkpoints/cifar100_pretrained.pth')
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, '
                  f'Train Acc: {100.*correct/total:.2f}%, Test Acc: {test_acc:.2f}%')
    
    print(f"Pre-training completed! Best test accuracy: {best_acc:.2f}%")
    return model

def evaluate_supervised(model, test_loader, num_classes):
    """Evaluate supervised model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total

def transfer_to_cifar10(pretrained_model_path=None):
    """Transfer learning from CIFAR-100 to CIFAR-10 using FlexMatch"""
    print("Starting transfer learning to CIFAR-10...")
    
    # Load CIFAR-10 datasets with 5% labeled data
    labeled_dataset, unlabeled_dataset, unlabeled_dataset_strong, test_dataset = get_cifar10_datasets(labeled_ratio=0.05)
    
    print(f"Labeled samples: {len(labeled_dataset)}")
    print(f"Unlabeled samples: {len(unlabeled_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=320, shuffle=True, num_workers=2, drop_last=True)
    unlabeled_loader_strong = DataLoader(unlabeled_dataset_strong, batch_size=320, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create FlexMatch model for CIFAR-10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FlexMatch(num_classes=10, device=device, lambda_u=1.0, T=0.5, threshold=0.95)
    
    # Load pre-trained weights (except the final layer)
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print("Loading pre-trained weights...")
        pretrained_dict = torch.load(pretrained_model_path, map_location=device)
        
        # Get current model state dict
        model_dict = model.model.state_dict()
        
        # Filter out final layer weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and 'fc' not in k and v.size() == model_dict[k].size()}
        
        # Update model state dict
        model_dict.update(pretrained_dict)
        model.model.load_state_dict(model_dict)
        
        # Also update EMA model
        ema_dict = model.ema_model.state_dict()
        ema_dict.update(pretrained_dict)
        model.ema_model.load_state_dict(ema_dict)
    
    # Training loop with FlexMatch
    num_epochs = 2
    best_acc = 0
    
    for epoch in range(num_epochs):
        model.model.train()
        
        # Create iterators for the loaders
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        unlabeled_strong_iter = iter(unlabeled_loader_strong)
        
        total_loss_l = 0
        total_loss_u = 0
        total_mask_ratio = 0
        num_batches = min(len(labeled_loader), len(unlabeled_loader))
        
        for batch_idx in range(num_batches):
            try:
                labeled_batch = next(labeled_iter)
                unlabeled_batch_weak = next(unlabeled_iter)
                unlabeled_batch_strong = next(unlabeled_strong_iter)
            except StopIteration:
                break
            
            # FlexMatch training step
            metrics = model.train_step(labeled_batch, unlabeled_batch_weak, unlabeled_batch_strong)
            
            total_loss_l += metrics['loss_l']
            total_loss_u += metrics['loss_u']
            total_mask_ratio += metrics['mask_ratio']
        
        # Update learning rate
        model.scheduler.step()
        
        # Evaluate on test set
        if (epoch + 1) % 50 == 0 or epoch == num_epochs - 1:
            test_acc = evaluate_flexmatch(model, test_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}] - '
                  f'Loss_l: {total_loss_l/num_batches:.4f}, '
                  f'Loss_u: {total_loss_u/num_batches:.4f}, '
                  f'Mask_ratio: {total_mask_ratio/num_batches:.4f}, '
                  f'Test Acc: {test_acc:.2f}%')
            
            if test_acc > best_acc:
                best_acc = test_acc
                model.save_model('checkpoints/flexmatch_best.pth')
    
    print(f"Transfer learning completed! Best test accuracy: {best_acc:.2f}%")
    return model

def evaluate_flexmatch(model, test_loader):
    """Evaluate FlexMatch model on test set"""
    model.ema_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(model.device), target.to(model.device)
            outputs = model.ema_model(data)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total

class ModelEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def get_embeddings_and_predictions(self, dataloader):
        """Extract embeddings and predictions from the model"""
        self.model.ema_model.eval()
        
        all_embeddings = []
        all_predictions = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, targets in tqdm(dataloader, desc='Extracting embeddings'):
                data = data.to(self.device)
                
                # Get embeddings (output before final layer)
                features = self.model.ema_model.conv1(data)
                features = self.model.ema_model.block1(features)
                features = self.model.ema_model.block2(features)
                features = self.model.ema_model.block3(features)
                features = self.model.ema_model.relu(self.model.ema_model.bn1(features))
                features = F.avg_pool2d(features, 8)
                embeddings = features.view(features.size(0), -1)
                
                # Get predictions
                logits = self.model.ema_model.fc(embeddings)
                probs = torch.softmax(logits, dim=1)
                _, predictions = torch.max(logits, 1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
                all_probs.append(probs.cpu().numpy())
        
        return (np.concatenate(all_embeddings),
                np.concatenate(all_predictions),
                np.concatenate(all_targets),
                np.concatenate(all_probs))
    
    def plot_tsne(self, embeddings, targets, title="t-SNE Visualization", save_path=None):
        """Create t-SNE visualization of embeddings"""
        print("Computing t-SNE...")
        # Sample a subset for faster t-SNE computation
        if len(embeddings) > 5000:
            indices = np.random.choice(len(embeddings), 5000, replace=False)
            embeddings = embeddings[indices]
            targets = targets[indices]
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=targets, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter)
        plt.title(title, fontsize=16)
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return embeddings_2d
    
    def plot_pca(self, embeddings, targets, title="PCA Visualization", save_path=None):
        """Create PCA visualization of embeddings"""
        print("Computing PCA...")
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=targets, cmap='tab10', alpha=0.6, s=10)
        plt.colorbar(scatter)
        plt.title(f'{title}\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}', fontsize=16)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return embeddings_2d
    
    def plot_confusion_matrix(self, predictions, targets, class_names=None, 
                            normalize=True, save_path=None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(10)]
        
        cm = confusion_matrix(targets, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_confidence_distribution(self, probs, targets, predictions, save_path=None):
        """Plot confidence distribution for correct and incorrect predictions"""
        correct_mask = predictions == targets
        correct_confidences = probs[correct_mask].max(axis=1)
        incorrect_confidences = probs[~correct_mask].max(axis=1)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidences, bins=50, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        box_data = [correct_confidences, incorrect_confidences]
        plt.boxplot(box_data, labels=['Correct', 'Incorrect'])
        plt.title('Confidence Box Plot')
        plt.ylabel('Confidence')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return correct_confidences, incorrect_confidences
    
    def generate_classification_report(self, predictions, targets, class_names=None):
        """Generate detailed classification report"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(10)]
        
        report = classification_report(targets, predictions, 
                                     target_names=class_names, output_dict=True)
        
        # Print detailed report
        print("\n" + "="*60)
        print("Detailed Classification Report")
        print("="*60)
        print(classification_report(targets, predictions, target_names=class_names))
        
        # Create accuracy per class plot
        class_accuracy = []
        for i in range(len(class_names)):
            class_mask = targets == i
            if class_mask.sum() > 0:
                acc = (predictions[class_mask] == i).mean()
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(class_names)), class_accuracy, color='skyblue', alpha=0.7)
        plt.axhline(y=np.mean(class_accuracy), color='red', linestyle='--', 
                   label=f'Average: {np.mean(class_accuracy):.3f}')
        plt.xlabel('Classes')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Class')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('evaluation_results/class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report

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
        image = images[idx]
        
        # Denormalize for visualization
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        # Convert to HWC for matplotlib
        image = image.permute(1, 2, 0).numpy()
        
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

def create_evaluation_report(model, test_loader, class_names, save_dir='evaluation_results'):
    """Create comprehensive evaluation report"""
    os.makedirs(save_dir, exist_ok=True)
    
    evaluator = ModelEvaluator(model)
    
    print("Starting comprehensive evaluation...")
    
    # Get embeddings and predictions
    embeddings, predictions, targets, probs = evaluator.get_embeddings_and_predictions(test_loader)
    
    # Calculate overall accuracy
    accuracy = (predictions == targets).mean()
    print(f"\nOverall Test Accuracy: {accuracy:.4f}")
    
    # 1. t-SNE visualization
    print("\n1. Creating t-SNE visualization...")
    tsne_embeddings = evaluator.plot_tsne(embeddings, targets, 
                                        "t-SNE Visualization of CIFAR-10 Test Set",
                                        f"{save_dir}/tsne_visualization.png")
    
    # 2. PCA visualization
    print("\n2. Creating PCA visualization...")
    pca_embeddings = evaluator.plot_pca(embeddings, targets,
                                      "PCA Visualization of CIFAR-10 Test Set",
                                      f"{save_dir}/pca_visualization.png")
    
    # 3. Confusion matrix
    print("\n3. Creating confusion matrix...")
    cm = evaluator.plot_confusion_matrix(predictions, targets, class_names,
                                       save_path=f"{save_dir}/confusion_matrix.png")
    
    # 4. Confidence distribution
    print("\n4. Analyzing confidence distribution...")
    correct_conf, incorrect_conf = evaluator.plot_confidence_distribution(
        probs, targets, predictions, 
        save_path=f"{save_dir}/confidence_distribution.png")
    
    # 5. Detailed classification report
    print("\n5. Generating classification report...")
    report = evaluator.generate_classification_report(predictions, targets, class_names)
    
    # 6. Save numerical results
    save_numerical_results(accuracy, cm, report, correct_conf, incorrect_conf, save_dir)
    
    print(f"\nEvaluation complete! Results saved to '{save_dir}' directory.")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'embeddings': embeddings,
        'predictions': predictions,
        'targets': targets
    }

def save_numerical_results(accuracy, cm, report, correct_conf, incorrect_conf, save_dir):
    """Save numerical results to files"""
    # Save accuracy and basic metrics
    with open(f'{save_dir}/results_summary.txt', 'w') as f:
        f.write("FlexMatch Transfer Learning Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        f.write("Per-class Accuracy:\n")
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            f.write(f"{class_name}: {metrics['precision']:.3f} precision, "
                   f"{metrics['recall']:.3f} recall, {metrics['f1-score']:.3f} f1-score\n")
        
        f.write(f"\nMacro Average F1: {report['macro avg']['f1-score']:.3f}\n")
        f.write(f"Weighted Average F1: {report['weighted avg']['f1-score']:.3f}\n")
        
        f.write(f"\nConfidence Analysis:\n")
        f.write(f"Correct predictions mean confidence: {correct_conf.mean():.3f}\n")
        f.write(f"Incorrect predictions mean confidence: {incorrect_conf.mean():.3f}\n")
        f.write(f"Confidence gap: {correct_conf.mean() - incorrect_conf.mean():.3f}\n")
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm)
    cm_df.columns = [f'Pred_{i}' for i in range(cm.shape[1])]
    cm_df.index = [f'True_{i}' for i in range(cm.shape[0])]
    cm_df.to_csv(f'{save_dir}/confusion_matrix.csv')
    
    # Save detailed classification report as CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{save_dir}/classification_report.csv')

def comprehensive_evaluation(model_path='checkpoints/flexmatch_best.pth'):
    """Comprehensive evaluation with visualizations"""
    print("Starting comprehensive evaluation...")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FlexMatch(num_classes=10, device=device)
    model.load_model(model_path)
    
    # Load test dataset
    _, _, _, test_dataset = get_cifar10_datasets(labeled_ratio=0.05)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create evaluation report
    results = create_evaluation_report(model, test_loader, class_names)
    
    # Additional visualizations
    print("\nGenerating sample predictions...")
    plot_sample_predictions(model, test_dataset, class_names, 
                          save_path='evaluation_results/sample_predictions.png')
    
    print("\nComprehensive evaluation completed!")
    return results

def run_complete_experiment():
    """Run the complete experiment from start to finish"""
    print("FlexMatch Transfer Learning Experiment")
    print("=" * 60)
    print("Source: CIFAR-100 → Target: CIFAR-10 (5% labeled)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    
    try:
        # Step 1: Pre-train on CIFAR-100
        print("\n🚀 Step 1: Pre-training on CIFAR-100...")
        pretrain_on_cifar100()
        
        # Step 2: Transfer learning to CIFAR-10
        print("\n🚀 Step 2: Transfer learning to CIFAR-10 with FlexMatch...")
        model = transfer_to_cifar10('checkpoints/cifar100_pretrained.pth')
        
        # Step 3: Comprehensive evaluation
        print("\n🚀 Step 3: Comprehensive Evaluation and Visualization...")
        results = comprehensive_evaluation('checkpoints/flexmatch_best.pth')
        
        # Generate final summary
        end_time = time.time()
        duration = end_time - start_time
        
        summary = f"""
🎉 EXPERIMENT COMPLETED SUCCESSFULLY! 🎉

Summary:
- Source Dataset: CIFAR-100 (100 classes)
- Target Dataset: CIFAR-10 (10 classes) 
- Labeled Data: 5% of CIFAR-10 training set
- Final Accuracy: {results['accuracy']:.4f}
- Total Time: {duration:.2f} seconds ({duration/60:.2f} minutes)

Results saved in:
- checkpoints/ : Model weights
- evaluation_results/ : Visualizations and analysis

Check evaluation_results/ for:
✓ t-SNE and PCA visualizations
✓ Confusion matrix
✓ Confidence distributions  
✓ Sample predictions
✓ Detailed classification reports
"""
        print(summary)
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed with error: {e}")
        return False

def main():
    # Check if we're in a Jupyter environment
    if 'ipykernel' in sys.modules:
        print("Running in Jupyter notebook mode...")
        print("Available functions:")
        print("1. run_complete_experiment() - Run full pipeline")
        print("2. pretrain_on_cifar100() - Pre-train on CIFAR-100")
        print("3. transfer_to_cifar10() - Transfer learning")
        print("4. comprehensive_evaluation() - Evaluation with visualizations")
        return
    
    # Command line mode
    parser = argparse.ArgumentParser(description='FlexMatch Transfer Learning')
    parser.add_argument('--pretrain', action='store_true', help='Pre-train on CIFAR-100')
    parser.add_argument('--transfer', action='store_true', help='Transfer to CIFAR-10')
    parser.add_argument('--evaluate', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--complete', action='store_true', help='Run complete experiment (all steps)')
    parser.add_argument('--pretrained_path', type=str, default='checkpoints/cifar100_pretrained.pth', 
                       help='Path to pre-trained model')
    
    args = parser.parse_args()
    
    if args.complete:
        success = run_complete_experiment()
        exit(0 if success else 1)
    
    if args.pretrain:
        pretrain_on_cifar100()
    
    if args.transfer:
        if not os.path.exists(args.pretrained_path) and args.pretrained_path == 'checkpoints/cifar100_pretrained.pth':
            print("No pre-trained model found. Pre-training first...")
            pretrain_on_cifar100()
        
        transfer_to_cifar10(args.pretrained_path)
    
    if args.evaluate:
        comprehensive_evaluation()

if __name__ == '__main__':
    main()