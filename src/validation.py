import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
from tqdm import tqdm


def test_model(model, test_loader, device, criterion=None):
    """
    Test the model on the test dataset.
    
    Args:
        model: The model to test
        test_loader: DataLoader for the test set
        device: The device to test on (CPU or GPU)
        criterion: Loss function (optional)
        
    Returns:
        tuple: (test_loss, test_acc, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        
        for inputs, targets in test_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            if criterion:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
                
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store predictions and targets for further analysis
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            test_acc = 100. * correct / total
            test_pbar.set_postfix({
                'acc': test_acc
            })
    
    test_loss = running_loss / total if criterion else 0
    test_acc = 100. * correct / total
    
    return test_loss, test_acc, np.array(all_preds), np.array(all_targets)


def plot_confusion_matrix(predictions, targets, class_names, save_path=None):
    """
    Plot the confusion matrix for the model predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: Names of the classes
        save_path: Path to save the plot (optional)
    """
    unique_classes = sorted(np.unique(np.concatenate([predictions, targets])))
    class_indices = unique_classes
    
    # Filter class names to include only those present in the data
    used_class_names = [class_names[i] for i in class_indices if i < len(class_names)]
    
    cm = confusion_matrix(targets, predictions, labels=class_indices)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=used_class_names, yticklabels=used_class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def compute_metrics(predictions, targets, class_names):
    """
    Compute various classification metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: Names of the classes
        
    Returns:
        dict: Dictionary containing the metrics
    """
    unique_classes = sorted(np.unique(np.concatenate([predictions, targets])))
    class_indices = unique_classes
    
    # Filter class names to include only those present in the data
    used_class_names = [class_names[i] for i in class_indices if i < len(class_names)]
    
    report = classification_report(targets, predictions, 
                                  target_names=used_class_names, 
                                  output_dict=True)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average='weighted'
    )
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'per_class': report
    }
    
    return metrics


def visualize_predictions(model, dataloader, device, class_names, num_samples=5, save_dir=None):
    """
    Visualize model predictions on random samples from the dataset.
    
    Args:
        model: The model to use for predictions
        dataloader: DataLoader containing the samples
        device: The device to run the model on
        class_names: Names of the classes
        num_samples: Number of samples to visualize
        save_dir: Directory to save the visualizations (optional)
    """
    model.eval()
    
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    indices = torch.randperm(len(images))[:num_samples]
    images = images[indices]
    labels = labels[indices]
    
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
    
    # Convert images for display
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        title_color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}", 
                    color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'predictions.png')
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def evaluate_model(model, test_loader, device, class_names, criterion=None, save_dir=None):
    """
    Comprehensive evaluation of the model.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for the test set
        device: The device to evaluate on
        class_names: Names of the classes
        criterion: Loss function (optional)
        save_dir: Directory to save the evaluation results (optional)
        
    Returns:
        tuple: (test_loss, test_acc, metrics)
    """
    test_loss, test_acc, predictions, targets = test_model(
        model, test_loader, device, criterion
    )
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    unique_classes = sorted(np.unique(np.concatenate([predictions, targets])))
    class_indices = unique_classes
    
    # Filter class names to include only those present in the data
    used_class_names = [class_names[i] for i in class_indices if i < len(class_names)]
    
    metrics = compute_metrics(predictions, targets, class_names)
    
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=used_class_names))
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    cm_save_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
    plot_confusion_matrix(predictions, targets, class_names, cm_save_path)
    
    visualize_predictions(
        model, test_loader, device, class_names, 
        num_samples=5, save_dir=save_dir
    )
    
    return test_loss, test_acc, metrics 