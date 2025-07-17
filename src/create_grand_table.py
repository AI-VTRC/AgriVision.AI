import os
import pandas as pd
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
    Main function to analyze all experiment results and create a grand table.
    """
    base_dir = 'outputs'
    data = []

    # Recursively find all evaluation_summary.txt files
    search_pattern = os.path.join(base_dir, '**', 'evaluation_summary.txt')
    summary_files = glob.glob(search_pattern, recursive=True)

    for summary_file in summary_files:
        try:
            # The evaluation directory is the parent of the summary file
            eval_dir_name = os.path.basename(os.path.dirname(summary_file))
            
            # Extract info from directory name like "Apple_detailed_clip_evaluation"
            parts = eval_dir_name.split('_')
            if len(parts) >= 4 and parts[-1] == 'evaluation':
                plant_name = parts[0]
                classification_type = parts[1]
                model_name = '_'.join(parts[2:-1])

                accuracy, f1_score, precision, recall = parse_evaluation_summary(summary_file)
                
                if f1_score is not None:
                    # Get the timestamped run directory name for traceability
                    run_dir = os.path.dirname(os.path.dirname(summary_file))
                    run_name = os.path.basename(run_dir)

                    data.append({
                        'Run': run_name,
                        'Plant': plant_name,
                        'Classification': classification_type,
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'F1-Score': f1_score,
                        'Precision': precision,
                        'Recall': recall
                    })
        except Exception as e:
            print(f"Could not process file {summary_file}: {e}")

    if not data:
        print("No evaluation summaries found. Exiting.")
        return

    df = pd.DataFrame(data)
    
    # Remove duplicate rows if any experiment was run multiple times with same result
    df.drop_duplicates(inplace=True)
    
    # Sort the table for consistency
    df.sort_values(by=['Plant', 'Classification', 'Model', 'Run'], inplace=True)
    
    output_path = os.path.join(base_dir, 'grand_results_table.csv')
    df.to_csv(output_path, index=False)
    print(f"Grand results table saved to {output_path}")

if __name__ == '__main__':
    main()