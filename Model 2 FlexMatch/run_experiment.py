#!/usr/bin/env python3
"""
Script to run the complete FlexMatch transfer learning experiment
"""

import subprocess
import sys
import os
import time

def run_experiment():
    print("Starting FlexMatch Transfer Learning Experiment")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('evaluation_results', exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Step 1: Pre-train on CIFAR-100
        print("\n" + "="*50)
        print("Step 1: Pre-training on CIFAR-100...")
        print("="*50)
        pretrain_result = subprocess.run([
            sys.executable, 'main.py', '--pretrain'
        ], capture_output=True, text=True)
        
        if pretrain_result.returncode != 0:
            print(f"Pre-training failed: {pretrain_result.stderr}")
            return False
        print("✓ Pre-training completed successfully!")
        
        # Step 2: Transfer learning to CIFAR-10
        print("\n" + "="*50)
        print("Step 2: Transfer learning to CIFAR-10 with 5% labeled data...")
        print("="*50)
        transfer_result = subprocess.run([
            sys.executable, 'main.py', '--transfer'
        ], capture_output=True, text=True)
        
        if transfer_result.returncode != 0:
            print(f"Transfer learning failed: {transfer_result.stderr}")
            return False
        print("✓ Transfer learning completed successfully!")
        
        # Step 3: Comprehensive evaluation and visualization
        print("\n" + "="*50)
        print("Step 3: Comprehensive Evaluation and Visualization...")
        print("="*50)
        eval_result = subprocess.run([
            sys.executable, 'final_evaluation.py'
        ], capture_output=True, text=True)
        
        if eval_result.returncode != 0:
            print(f"Evaluation failed: {eval_result.stderr}")
            return False
        print("✓ Evaluation and visualization completed successfully!")
        
        # Step 4: Generate summary report
        print("\n" + "="*50)
        print("Step 4: Generating Final Summary...")
        print("="*50)
        generate_summary(start_time)
        
        print("\n" + "="*60)
        print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        print("Check the following directories for results:")
        print("  - checkpoints/ : Saved model weights")
        print("  - evaluation_results/ : Visualizations and analysis")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        return False

def generate_summary(start_time):
    """Generate a final summary of the experiment"""
    end_time = time.time()
    duration = end_time - start_time
    
    summary = f"""
FLEXMATCH TRANSFER LEARNING EXPERIMENT SUMMARY
{'=' * 50}

Experiment Configuration:
- Source Dataset: CIFAR-100 (100 classes)
- Target Dataset: CIFAR-10 (10 classes)
- Labeled Data: 5% of CIFAR-10 training set
- Unlabeled Data: 95% of CIFAR-10 training set
- Method: FlexMatch Semi-Supervised Learning

Training Pipeline:
1. Pre-training: Supervised learning on CIFAR-100
2. Transfer: Semi-supervised learning on CIFAR-10 with FlexMatch
3. Evaluation: Comprehensive analysis and visualization

Results Directory Structure:
checkpoints/
├── cifar100_pretrained.pth
└── flexmatch_best.pth

evaluation_results/
├── tsne_visualization.png
├── pca_visualization.png
├── confusion_matrix.png
├── confidence_distribution.png
├── sample_predictions.png
├── class_distribution.png
├── final_summary.png
├── results_summary.txt
├── confusion_matrix.csv
└── classification_report.csv

Total Experiment Time: {duration:.2f} seconds ({duration/60:.2f} minutes)

Next Steps:
1. Check evaluation_results/ for detailed analysis
2. Review confusion_matrix.png for error patterns
3. Examine tsne_visualization.png for feature space structure
4. Read results_summary.txt for numerical results

Experiment completed at: {time.ctime()}
"""
    
    print(summary)
    
    # Save summary to file
    with open('evaluation_results/experiment_summary.txt', 'w') as f:
        f.write(summary)

def run_individual_step(step):
    """Run individual experiment steps"""
    if step == 'pretrain':
        print("Running pre-training only...")
        subprocess.run([sys.executable, 'main.py', '--pretrain'])
    
    elif step == 'transfer':
        print("Running transfer learning only...")
        subprocess.run([sys.executable, 'main.py', '--transfer'])
    
    elif step == 'evaluate':
        print("Running evaluation only...")
        subprocess.run([sys.executable, 'final_evaluation.py'])
    
    else:
        print(f"Unknown step: {step}")
        print("Available steps: pretrain, transfer, evaluate")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FlexMatch Transfer Learning Experiment')
    parser.add_argument('--step', type=str, choices=['pretrain', 'transfer', 'evaluate', 'all'],
                       default='all', help='Which step to run')
    
    args = parser.parse_args()
    
    if args.step == 'all':
        success = run_experiment()
        sys.exit(0 if success else 1)
    else:
        run_individual_step(args.step)