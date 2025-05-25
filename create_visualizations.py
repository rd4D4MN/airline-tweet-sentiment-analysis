"""
Create Visualizations for Assignment Documentation

This script generates appealing plots and diagrams for README.md, 
ASSIGNMENT_SUMMARY.md, and reflection.md based on experimental results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for professional-looking plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_results():
    """Load all experimental results."""
    results_dir = Path("experiments/results")
    
    # Load experiment comparison
    with open(results_dir / "experiment_comparison.json", 'r') as f:
        experiment_data = json.load(f)
    
    # Load data augmentation results
    with open(results_dir / "data_augmentation_results.json", 'r') as f:
        augmentation_data = json.load(f)
    
    # Load enhanced features results
    with open(results_dir / "enhanced_features_results.json", 'r') as f:
        features_data = json.load(f)
    
    return experiment_data, augmentation_data, features_data

def create_experiment_comparison_plot(experiment_data):
    """Create a comprehensive experiment comparison plot."""
    df = pd.DataFrame(experiment_data['summary'])
    df = df.sort_values('test_f1', ascending=True)  # Sort for horizontal bar chart
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))  # Made wider for better fit
    
    # Plot 1: F1-Score Comparison
    colors = ['#ff7f7f' if x < 0.72 else '#90EE90' if x < 0.74 else '#32CD32' for x in df['test_f1']]
    bars1 = ax1.barh(df['experiment_id'], df['test_f1'], color=colors, alpha=0.8)
    ax1.set_xlabel('Weighted F1-Score')
    ax1.set_title('Model Performance Comparison\n(Higher is Better)', fontweight='bold')
    ax1.set_xlim(0.64, 0.76)  # Extended range to fit all labels
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, df['test_f1'])):
        ax1.text(value + 0.003, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontweight='bold')
    
    # Highlight best performer
    best_idx = df['test_f1'].idxmax()
    bars1[best_idx].set_color('#FFD700')
    bars1[best_idx].set_edgecolor('red')
    bars1[best_idx].set_linewidth(2)
    
    # Plot 2: Per-Class Performance of Best Model
    best_model = experiment_data['best_experiment']
    classes = ['Negative', 'Neutral', 'Positive']
    f1_scores = [best_model['negative_f1'], best_model['neutral_f1'], best_model['positive_f1']]
    
    colors2 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars2 = ax2.bar(classes, f1_scores, color=colors2, alpha=0.8)
    ax2.set_ylabel('F1-Score')
    ax2.set_title(f'Best Model Per-Class Performance\n{best_model["experiment_id"].upper()}', fontweight='bold')
    ax2.set_ylim(0, 0.9)
    
    # Add value labels on bars
    for bar, value in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 0.01, 
                f'{value:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/methodology/experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def create_augmentation_comparison_plot(augmentation_data):
    """Create data augmentation comparison plot."""
    baseline = augmentation_data['baseline_results']
    augmented = augmentation_data['augmented_results']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Overall Performance Comparison
    metrics = ['Accuracy', 'Weighted F1', 'Macro F1']
    baseline_values = [
        baseline['accuracy'],
        baseline['weighted_avg']['f1-score'],
        baseline['macro_avg']['f1-score']
    ]
    augmented_values = [
        augmented['accuracy'],
        augmented['weighted_avg']['f1-score'],
        augmented['macro_avg']['f1-score']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_values, width, label='Baseline', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, augmented_values, width, label='With Augmentation', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Data Augmentation Impact\n(Synonym Replacement)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0.65, 0.75)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Per-Class Impact
    classes = ['Negative', 'Neutral', 'Positive']
    baseline_class = [baseline['per_class'][cls]['f1-score'] for cls in ['negative', 'neutral', 'positive']]
    augmented_class = [augmented['per_class'][cls]['f1-score'] for cls in ['negative', 'neutral', 'positive']]
    
    x2 = np.arange(len(classes))
    bars3 = ax2.bar(x2 - width/2, baseline_class, width, label='Baseline', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, augmented_class, width, label='With Augmentation', color='#e74c3c', alpha=0.8)
    
    ax2.set_ylabel('F1-Score')
    ax2.set_title('Per-Class Augmentation Impact', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.set_ylim(0.5, 0.85)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('docs/methodology/augmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_engineering_plot(features_data):
    """Create enhanced features comparison plot."""
    baseline_f1 = features_data['analysis']['baseline_svm_f1']
    enhanced_svm_f1 = features_data['analysis']['enhanced_svm_f1']
    enhanced_lr_f1 = features_data['analysis']['enhanced_lr_f1']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Feature Engineering Impact
    models = ['SVM\n(Baseline)', 'SVM\n(Enhanced)', 'LR\n(Enhanced)']
    f1_scores = [baseline_f1, enhanced_svm_f1, enhanced_lr_f1]
    colors = ['#2ecc71', '#e67e22', '#9b59b6']
    
    bars = ax1.bar(models, f1_scores, color=colors, alpha=0.8)
    ax1.set_ylabel('Weighted F1-Score')
    ax1.set_title('Feature Engineering Impact\n(GloVe + Handcrafted Features)', fontweight='bold')
    ax1.set_ylim(0.68, 0.75)
    
    # Add value labels
    for bar, value in zip(bars, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, value + 0.002, 
                f'{value:.3f}', ha='center', fontweight='bold')
    
    # Highlight that baseline is still best
    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(2)
    
    # Plot 2: Feature Dimensions Comparison
    dimensions = ['Original\nGloVe', 'Enhanced\nFeatures']
    dim_values = [100, features_data['experiment_info']['enhanced_dimensions']]
    
    bars2 = ax2.bar(dimensions, dim_values, color=['#3498db', '#e74c3c'], alpha=0.8)
    ax2.set_ylabel('Feature Dimensions')
    ax2.set_title('Feature Space Expansion\n(100D -> 105D)', fontweight='bold')
    
    # Add value labels
    for bar, value in zip(bars2, dim_values):
        ax2.text(bar.get_x() + bar.get_width()/2, value + 1, 
                f'{value}D', ha='center', fontweight='bold')
    
    # Add annotation about added features
    ax2.text(0.5, 50, '+5 Features:\n• Text length\n• Word count\n• Sentiment words\n• Balance score', 
             ha='center', va='center', transform=ax2.transData, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('docs/methodology/feature_engineering.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_summary_plot(experiment_data):
    """Create a simplified performance summary with two focused plots."""
    best = experiment_data['best_experiment']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance Category (Industry Benchmarks)
    performance = best['test_f1']
    
    # Create horizontal bar chart showing performance ranges
    categories = ['Poor\n(<65%)', 'Fair\n(65-70%)', 'Good\n(70-75%)', 'Excellent\n(>75%)']
    ranges = [0.65, 0.70, 0.75, 0.80]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    
    y_pos = np.arange(len(categories))
    bars = ax1.barh(y_pos, ranges, color=colors, alpha=0.3, height=0.6)
    
    # Add category labels
    for i, (cat, r) in enumerate(zip(categories, ranges)):
        ax1.text(r/2, i, cat, ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Mark current performance
    ax1.axvline(x=performance, color='red', linewidth=4, label=f'Our Model: {performance:.1%}')
    ax1.set_xlim(0.6, 0.8)
    ax1.set_xlabel('F1-Score')
    ax1.set_title('Performance Category\n(Industry Benchmarks)', fontweight='bold')
    ax1.legend()
    ax1.set_yticks([])
    
    # Plot 2: Dataset Class Distribution
    class_support = [1835, 620, 473]  # From confusion matrix
    class_names = ['Negative\n(62.7%)', 'Neutral\n(21.2%)', 'Positive\n(16.1%)']
    colors2 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    wedges, texts, autotexts = ax2.pie(class_support, labels=class_names, colors=colors2, 
                                      autopct='%1.0f', startangle=90)
    ax2.set_title('Dataset Class Distribution\n(Test Set)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/model_evaluation/performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_methodology_flowchart():
    """Create an enhanced methodology flowchart with detailed explanations."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define boxes with more detailed information
    boxes = [
        {'text': 'Data Loading\n14.6k airline tweets\n(Train: 11.7k, Test: 2.9k)', 'pos': (2, 10.5), 'color': '#3498db', 'size': (2.2, 1.0)},
        {'text': 'GloVe Embeddings\n400k vocabulary\n100-dimensional vectors\nPre-trained on Twitter', 'pos': (2, 8.5), 'color': '#e74c3c', 'size': (2.2, 1.2)},
        {'text': 'Text Preprocessing\n• Lowercase conversion\n• URL/mention removal\n• Tokenization\n• Stop word handling', 'pos': (2, 6.2), 'color': '#f39c12', 'size': (2.2, 1.4)},
        {'text': 'Vectorization\nMean aggregation\nof word embeddings\n100D feature vectors', 'pos': (2, 4), 'color': '#9b59b6', 'size': (2.2, 1.0)},
        {'text': 'Systematic Experiments\n• 10+ model configurations\n• Cross-validation\n• Hyperparameter tuning\n• Performance comparison', 'pos': (6, 7), 'color': '#2ecc71', 'size': (2.4, 1.4)},
        {'text': 'Best Model Selection\nSVM with RBF kernel\nC=10, gamma=0.1\nClass weight balanced', 'pos': (10, 7), 'color': '#e67e22', 'size': (2.2, 1.2)},
        {'text': 'Model Evaluation\n74.38% weighted F1-score\n73.57% accuracy\nConfusion matrix analysis', 'pos': (10, 4.5), 'color': '#1abc9c', 'size': (2.2, 1.2)},
        {'text': 'Error Analysis\nMisclassification patterns\nSarcasm detection issues\nClass imbalance effects', 'pos': (10, 2), 'color': '#34495e', 'size': (2.2, 1.2)},
        {'text': 'Optional Experiments\n• Data augmentation\n• Feature engineering\n• Performance comparison', 'pos': (6, 2), 'color': '#8e44ad', 'size': (2.4, 1.2)},
    ]
    
    # Draw boxes with detailed information
    for box in boxes:
        width, height = box['size']
        rect = plt.Rectangle((box['pos'][0]-width/2, box['pos'][1]-height/2), width, height, 
                           facecolor=box['color'], alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box['pos'][0], box['pos'][1], box['text'], 
               ha='center', va='center', fontweight='bold', fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    # Define consistent arrows with better positioning
    arrows = [
        # Main pipeline flow (vertical on left)
        {'start': (2, 10), 'end': (2, 9.1), 'style': 'main'},
        {'start': (2, 7.9), 'end': (2, 6.9), 'style': 'main'},
        {'start': (2, 5.5), 'end': (2, 4.5), 'style': 'main'},
        
        # From vectorization to experiments (diagonal)
        {'start': (3.1, 4), 'end': (4.8, 7), 'style': 'flow', 'label': 'Feature vectors'},
        
        # Experiments to best model (horizontal)
        {'start': (7.2, 7), 'end': (8.8, 7), 'style': 'flow', 'label': 'Best configuration'},
        
        # Best model to evaluation (vertical on right)
        {'start': (10, 6.4), 'end': (10, 5.1), 'style': 'main'},
        
        # Evaluation to error analysis (vertical on right)
        {'start': (10, 3.9), 'end': (10, 2.6), 'style': 'main'},
        
        # Optional experiments branch
        {'start': (6, 6.3), 'end': (6, 2.6), 'style': 'optional', 'label': 'Additional\nexperiments'},
    ]
    
    # Draw arrows with consistent styling
    for arrow in arrows:
        if arrow['style'] == 'main':
            # Main pipeline arrows - thick and dark
            ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                       arrowprops=dict(arrowstyle='->', lw=3, color='#2c3e50'))
        elif arrow['style'] == 'flow':
            # Flow arrows - medium thickness
            ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='#34495e'))
        elif arrow['style'] == 'optional':
            # Optional arrows - dashed
            ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                       arrowprops=dict(arrowstyle='->', lw=2, color='#7f8c8d', linestyle='dashed'))
        
        # Add labels with better positioning
        if 'label' in arrow:
            start_x, start_y = arrow['start']
            end_x, end_y = arrow['end']
            
            # Calculate label position based on arrow direction
            if abs(end_x - start_x) > abs(end_y - start_y):  # More horizontal
                label_x = (start_x + end_x) / 2
                label_y = max(start_y, end_y) + 0.3
            else:  # More vertical
                label_x = max(start_x, end_x) + 0.5
                label_y = (start_y + end_y) / 2
            
            ax.text(label_x, label_y, arrow['label'], fontsize=9, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8, edgecolor='black'))
    
    # Add title and methodology description
    ax.text(6, 11.5, 'Systematic ML Methodology for Sentiment Analysis', 
           ha='center', va='center', fontsize=18, fontweight='bold')
    
    # Add legend for arrow types
    legend_elements = [
        plt.Line2D([0], [0], color='#2c3e50', lw=3, label='Main Pipeline'),
        plt.Line2D([0], [0], color='#34495e', lw=2.5, label='Data Flow'),
        plt.Line2D([0], [0], color='#7f8c8d', lw=2, linestyle='dashed', label='Optional Branch')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
    ax.text(6, 0.8, 'Key Principles: Reproducibility • Systematic Comparison • Honest Reporting • Error Analysis', 
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.savefig('docs/methodology/methodology_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all visualizations."""
    print("Creating visualizations for documentation...")
    
    # Create docs directory structure
    Path("docs").mkdir(exist_ok=True)
    Path("docs/methodology").mkdir(exist_ok=True)
    Path("docs/model_evaluation").mkdir(exist_ok=True)
    Path("docs/data_analysis").mkdir(exist_ok=True)
    
    # Load all results
    experiment_data, augmentation_data, features_data = load_results()
    
    # Generate plots
    print("Creating experiment comparison plot...")
    create_experiment_comparison_plot(experiment_data)
    
    print("Creating data augmentation plot...")
    create_augmentation_comparison_plot(augmentation_data)
    
    print("Creating feature engineering plot...")
    create_feature_engineering_plot(features_data)
    
    print("Creating performance summary plot...")
    create_performance_summary_plot(experiment_data)
    
    print("Creating methodology flowchart...")
    create_methodology_flowchart()
    
    print("All visualizations created in organized 'docs/' structure!")
    print("\nGenerated files:")
    print("- docs/methodology/experiment_comparison.png")
    print("- docs/methodology/augmentation_comparison.png") 
    print("- docs/methodology/feature_engineering.png")
    print("- docs/methodology/methodology_flowchart.png")
    print("- docs/model_evaluation/performance_summary.png")

if __name__ == "__main__":
    main() 