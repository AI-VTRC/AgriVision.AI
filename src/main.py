import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np

from model import get_model
from training import create_dataloaders, Trainer
from validation import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Leaf Classification Training')
    
    parser.add_argument('--plant', type=str, default='Maize', help='Plant name (e.g., Apple)')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--model_name', type=str, default='clip', 
                        help='Base model name (efficientnet_b0, resnet50, clip)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model instead of training')
    parser.add_argument('--pretrained', type=int, default=1, help='Use pretrained weights (1=True, 0=False)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    
    return parser.parse_args()


def plot_training_metrics(metrics, save_path=None):
    """
    Plot training and validation metrics.
    
    Args:
        metrics (dict): Dictionary containing training and validation metrics
        save_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, metrics['train_loss'], label='Train Loss')
    ax1.plot(epochs, metrics['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, metrics['train_acc'], label='Train Accuracy')
    ax2.plot(epochs, metrics['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    """Main function."""
    args = parse_args()
    
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    class_names = [
        f"{args.plant} Healthy Real",
        f"{args.plant} Unhealthy Real",
        f"{args.plant} Healthy SDXL",
        f"{args.plant} Unhealthy SDXL",
        f"{args.plant} Healthy SPx2Px",
        f"{args.plant} Unhealthy SPx2Px",
        f"{args.plant} Healthy DCGAN",
        f"{args.plant} Unhealthy DCGAN",
        f"{args.plant} Healthy StyleGAN2",
        f"{args.plant} Unhealthy StyleGAN2"
    ]
    
    model_type = 'clip' if args.model_name == 'clip' or args.model_name.startswith('clip-') else None
    
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        plant_name=args.plant,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_seed,
        model_type=model_type
    )
    
    model = get_model(
        model_name=args.model_name,
        num_classes=len(class_names),
        pretrained=bool(args.pretrained)
    )
    model = model.to(device)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.CrossEntropyLoss()
    
    if args.eval:
        print("Evaluating model...")
        eval_dir = os.path.join(args.output_dir, f"{args.plant}_evaluation")
        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            criterion=criterion,
            save_dir=eval_dir
        )
    else:
        # For CLIP models, we only need to train the classifier head, which requires less learning rate
        if model_type == 'clip':
            lr = args.lr * 0.1  # Use smaller learning rate for fine-tuning on top of CLIP
        else:
            lr = args.lr

        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        print("Training model...")
        metrics = trainer.train(
            num_epochs=args.epochs,
            save_path=os.path.join(args.output_dir, f"{args.plant}_best_model.pth")
        )
        
        plot_training_metrics(
            metrics=metrics,
            save_path=os.path.join(args.output_dir, f"{args.plant}_training_metrics.png")
        )
        
        print("\nEvaluating model after training...")
        eval_dir = os.path.join(args.output_dir, f"{args.plant}_evaluation")
        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            criterion=criterion,
            save_dir=eval_dir
        )


if __name__ == '__main__':
    main() 