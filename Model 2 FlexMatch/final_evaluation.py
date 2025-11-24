#!/usr/bin/env python3
"""
Final evaluation and visualization script
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from main import comprehensive_evaluation
from flexmatch import FlexMatch
from flexmatch_utils import get_cifar10_datasets, get_cifar100_dataset
from evaluation_utils import ModelEvaluator
from visualization import plot_feature_space_comparison, plot_confidence_calibration
import os

def compare_with_baseline():
    """Compare FlexMatch with supervised baseline"""
    print("Comparing with supervised baseline...")
    
    # Load FlexMatch results
    flexmatch_model = FlexMatch(num_classes=10, device='cuda')
    flexmatch_model.load_model('checkpoints/flexmatch_best.pth')
    
    # Load test dataset
    _, _, _, test_dataset = get_cifar10_datasets(labeled_ratio=0.05)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Evaluate FlexMatch
    flexmatch_evaluator = ModelEvaluator(flexmatch_model)
    flexmatch_embeddings, flexmatch_predictions, flexmatch_targets, flexmatch_probs = \
        flexmatch_evaluator.get_embeddings_and_predictions(test_loader)
    flexmatch_accuracy = (flexmatch_predictions == flexmatch_targets).mean()
    
    print(f"FlexMatch Accuracy: {flexmatch_accuracy:.4f}")
    
    # You could add a supervised baseline model here for comparison
    # For now, we'll just show the FlexMatch results
    
    return {
        'flexmatch': {
            'accuracy': flexmatch_accuracy,
            'embeddings': flexmatch_embeddings,
            'predictions': flexmatch_predictions,
            'targets': flexmatch_targets,
            'probs': flexmatch_probs
        }
    }

def plot_final_summary(results):
    """Create a final summary plot"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Accuracy comparison (placeholder for future baseline comparisons)
    methods = ['FlexMatch\n(5% labeled)']
    accuracies = [results['flexmatch']['accuracy']]
    
    bars = ax1.bar(methods, accuracies, color=['skyblue'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12)
    
    # Confidence distribution
    correct_mask = results['flexmatch']['predictions'] == results['flexmatch']['targets']
    correct_conf = results['flexmatch']['probs'][correct_mask].max(axis=1)
    incorrect_conf = results['flexmatch']['probs'][~correct_mask].max(axis=1)
    
    ax2.hist(correct_conf, bins=30, alpha=0.7, label='Correct', color='green')
    ax2.hist(incorrect_conf, bins=30, alpha=0.7, label='Incorrect', color='red')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Class-wise accuracy
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
    class_accuracy = []
    for i in range(10):
        class_mask = results['flexmatch']['targets'] == i
        if class_mask.sum() > 0:
            acc = (results['flexmatch']['predictions'][class_mask] == i).mean()
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    bars = ax3.bar(range(10), class_accuracy, color='lightcoral', alpha=0.7)
    ax3.axhline(y=np.mean(class_accuracy), color='blue', linestyle='--', 
               label=f'Average: {np.mean(class_accuracy):.3f}')
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy per Class')
    ax3.set_xticks(range(10))
    ax3.set_xticklabels(class_names, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracy):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Confidence calibration
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(
        results['flexmatch']['targets'], 
        results['flexmatch']['probs'].max(axis=1), 
        n_bins=10
    )
    
    ax4.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax4.plot(prob_pred, prob_true, 's-', label='FlexMatch')
    ax4.set_xlabel('Mean predicted probability')
    ax4.set_ylabel('Fraction of positives')
    ax4.set_title('Reliability Diagram')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/final_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run final evaluation and visualization"""
    print("Final Evaluation and Visualization")
    print("=" * 50)
    
    # Create directories
    os.makedirs('evaluation_results', exist_ok=True)
    
    # Run comprehensive evaluation
    print("\n1. Running comprehensive evaluation...")
    results = comprehensive_evaluation()
    
    # Compare with baseline (if available)
    print("\n2. Comparing with baseline...")
    comparison_results = compare_with_baseline()
    
    # Create final summary
    print("\n3. Creating final summary...")
    plot_final_summary(comparison_results)
    
    print("\n" + "="*50)
    print("Final evaluation completed!")
    print("All results saved to 'evaluation_results' directory")
    print("="*50)

if __name__ == '__main__':
    main()