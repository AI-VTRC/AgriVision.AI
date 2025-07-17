import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import re

def parse_evaluation_summary(file_path):
    """
    Parses an evaluation_summary.txt file to extract key metrics.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    def get_metric(metric_name):
        match = re.search(f"{metric_name}:\s*(\d+\.\d+)", content)
        return float(match.group(1)) if match else None

    accuracy = get_metric("Accuracy")
    f1_score = get_metric("F1-Score")
    precision = get_metric("Precision")
    recall = get_metric("Recall")

    return accuracy, f1_score, precision, recall

def main():
    """
    Main function to analyze experiment results.
    """
    base_dir = 'outputs/comprehensive_plant_classification_20250710_174532'
    data = []

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            summary_file = os.path.join(subdir_path, 'evaluation_summary.txt')
            if os.path.exists(summary_file):
                parts = subdir.split('_')
                plant_name = parts[0]
                classification_type = parts[1]
                model_name = '_'.join(parts[2:-1])

                accuracy, f1_score, precision, recall = parse_evaluation_summary(summary_file)
                
                if f1_score is not None:
                    data.append({
                        'Plant': plant_name,
                        'Classification': classification_type,
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'F1-Score': f1_score,
                        'Precision': precision,
                        'Recall': recall
                    })

    df = pd.DataFrame(data)
    csv_path = os.path.join(base_dir, 'comprehensive_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Comprehensive summary saved to {csv_path}")

    figures_dir = os.path.join(base_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    chart_files = []
    for (plant, classification), group in df.groupby(['Plant', 'Classification']):
        plt.figure(figsize=(10, 6))
        group.set_index('Model')['F1-Score'].plot(kind='bar', rot=45)
        plt.title(f'F1-Scores for {plant} - {classification} Classification')
        plt.ylabel('F1-Score')
        plt.tight_layout()
        chart_path = os.path.join(figures_dir, f'{plant}_{classification}_f1_scores.png')
        plt.savefig(chart_path)
        plt.close()
        chart_files.append(chart_path)
        print(f"Chart saved to {chart_path}")
        
    if chart_files:
        images = [Image.open(f) for f in chart_files]
        widths, heights = zip(*(i.size for i in images))

        # Create a collage of images
        # For simplicity, lets just stack them vertically
        total_height = sum(heights)
        max_width = max(widths)
        
        composite_image = Image.new('RGB', (max_width, total_height), color='white')
        
        y_offset = 0
        for img in images:
            composite_image.paste(img, (0, y_offset))
            y_offset += img.size[1]
        
        overview_path = os.path.join(figures_dir, 'performance_overview.png')
        composite_image.save(overview_path)
        print(f"Consolidated performance overview saved to {overview_path}")


if __name__ == '__main__':
    main()