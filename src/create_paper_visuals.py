import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def create_visualizations(csv_path, output_dir):
    """
    Generates and saves a series of publication-quality visualizations for model performance analysis.

    Args:
        csv_path (str): The path to the comprehensive summary CSV file.
        output_dir (str): The directory where the generated plots will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(csv_path)

    # Clean up model names for better presentation
    df['Model'] = df['Model'].replace({
        'efficientnet_b0': 'EfficientNet-B0',
        'resnet50': 'ResNet-50',
        'clip': 'CLIP'
    })

    # --- 1. Grouped Bar Charts: Direct Performance Comparison ---
    classification_types = df['Classification'].unique()
    for classification_type in classification_types:
        plt.figure(figsize=(14, 8))
        subset = df[df['Classification'] == classification_type]
        
        sns.barplot(
            data=subset,
            x='Plant',
            y='Accuracy',
            hue='Model',
            palette='viridis'
        )
        
        plt.title(f'Model Performance Comparison ({classification_type.capitalize()} Classification)', fontsize=16)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.xlabel('Plant Type', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0.5, 1.0) # Start y-axis at 50% for better visual distinction
        plt.legend(title='Model', fontsize=10)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'grouped_bar_chart_{classification_type}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Generated: {output_path}")

    # --- 2. Heatmaps: Performance Landscape ---
    for classification_type in classification_types:
        plt.figure(figsize=(10, 6))
        subset = df[df['Classification'] == classification_type]
        
        heatmap_data = subset.pivot_table(
            index='Model',
            columns='Plant',
            values='Accuracy'
        )
        
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
        
        plt.title(f'Performance Heatmap ({classification_type.capitalize()} Classification)', fontsize=16)
        plt.ylabel('Model', fontsize=12)
        plt.xlabel('Plant Type', fontsize=12)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'performance_heatmap_{classification_type}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Generated: {output_path}")

    # --- 3. Box Plots: Robustness and Consistency Analysis ---
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=df,
        x='Model',
        y='Accuracy',
        hue='Classification',
        palette='pastel'
    )
        
    plt.title('Model Robustness and Consistency Across All Experiments', fontsize=16)
    plt.ylabel('Accuracy Distribution', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.legend(title='Classification', fontsize=10)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'model_robustness_boxplot.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated: {output_path}")

    # --- 4. Violin Plots: Detailed Distribution Analysis ---
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        data=df,
        x='Model',
        y='Accuracy',
        hue='Classification',
        palette='muted',
        split=True,
        inner='quartile'
    )
        
    plt.title('Detailed Performance Distribution Analysis', fontsize=16)
    plt.ylabel('Accuracy Distribution', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.legend(title='Classification', fontsize=10)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'model_distribution_violinplot.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated: {output_path}")

if __name__ == '__main__':
    CSV_FILE_PATH = 'outputs/comprehensive_plant_classification_20250710_174532/comprehensive_summary.csv'
    OUTPUT_VISUALS_DIR = 'outputs/paper_visuals/'
    
    create_visualizations(CSV_FILE_PATH, OUTPUT_VISUALS_DIR)