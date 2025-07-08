import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime

from model import get_model
from training import create_dataloaders, Trainer
from validation import evaluate_model

def setup_logging(output_dir, plant_name):
    """Set up logging configuration with file and console output."""
    log_dir = os.path.join(output_dir, f"{plant_name}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'training.log')
    
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_dir

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Leaf Classification Training')
    
    parser.add_argument('--plant', type=str, default='Maize', help='Plant name (e.g., Apple)')
    parser.add_argument('--data_dir', type=str, default='./Dataset', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--model_name', type=str, default='clip',
                        help='Base model name (efficientnet_b0, resnet50, clip)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--progress_interval', type=int, default=10,
                        help='Number of batches between detailed progress reports')
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
    try:
        args = parse_args()
        
        output_base_dir = os.path.join(args.output_dir, args.plant)
        log_dir = setup_logging(output_base_dir, args.plant)
        
        logger.info("="*80)
        logger.info(f"Starting training for {args.plant}")
        logger.info(f"Configuration: {vars(args)}")
        logger.info("="*80)
        
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        model_dir = os.path.join(log_dir, 'models')
        figures_dir = os.path.join(log_dir, 'figures')
        results_dir = os.path.join(log_dir, 'results')
        
        for dir_path in [model_dir, figures_dir, results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"Output directory: {log_dir}")
        
        class_names = [
            f"{args.plant}-Healthy-Diffusion-DS8",
            f"{args.plant}-Healthy-Diffusion-SPx2Px",
            f"{args.plant}-Healthy-GAN-StyleGAN2",
            f"{args.plant}-Healthy-GAN-StyleGAN3",
            f"{args.plant}-Healthy-Real-Real",
            f"{args.plant}-Unhealthy-Diffusion-DS8",
            f"{args.plant}-Unhealthy-Diffusion-SPx2Px",
            f"{args.plant}-Unhealthy-GAN-StyleGAN2",
            f"{args.plant}-Unhealthy-GAN-StyleGAN3",
            f"{args.plant}-Unhealthy-Real-Real"
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
        
        model_num_classes = None
        
        if args.eval and args.checkpoint:
            checkpoint_path = args.checkpoint
            if not os.path.exists(checkpoint_path):
                alt_path = os.path.join('../outputs', f'{args.plant}_best_model.pth')
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                else:
                    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {alt_path}")
            
            temp_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # Find the output layer size
            num_classes_in_checkpoint = None
            for key in temp_checkpoint['model_state_dict'].keys():
                if 'classifier.4.weight' in key:  # This is the final layer for CLIP model
                    num_classes_in_checkpoint = temp_checkpoint['model_state_dict'][key].shape[0]
                    break
            
            model_num_classes = num_classes_in_checkpoint
            
            if num_classes_in_checkpoint:
                logger.info(f"Checkpoint was trained with {num_classes_in_checkpoint} classes")
                model = get_model(
                    model_name=args.model_name,
                    num_classes=num_classes_in_checkpoint, 
                    pretrained=bool(args.pretrained)
                )
            else:
                model = get_model(
                    model_name=args.model_name,
                    num_classes=len(class_names),
                    pretrained=bool(args.pretrained)
                )
            del temp_checkpoint 
        else:
            model = get_model(
                model_name=args.model_name,
                num_classes=len(class_names),
                pretrained=bool(args.pretrained)
            )
        
        model = model.to(device)
        
        if args.checkpoint:
            checkpoint_path = args.checkpoint
            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
                alt_path = os.path.join('../outputs', f'{args.plant}_best_model.pth')
                if os.path.exists(alt_path):
                    logger.info(f"Found checkpoint at alternative path: {alt_path}")
                    checkpoint_path = alt_path
                else:
                    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path} or {alt_path}")
            
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            logger.info(f"Checkpoint file size: {os.path.getsize(checkpoint_path) / 1024**2:.2f} MB")
            
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
                
                if 'class_names' in checkpoint:
                    logger.info(f"Checkpoint class names: {checkpoint['class_names']}")
                    logger.info(f"Number of classes in checkpoint: {len(checkpoint['class_names'])}")
                
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                    for key in model_state.keys():
                        if 'fc' in key or 'classifier' in key or 'head' in key:
                            if 'weight' in key:
                                logger.info(f"Final layer {key} shape: {model_state[key].shape}")
                                expected_classes = model_state[key].shape[0]
                                if expected_classes != len(class_names):
                                    logger.warning(f"Model expects {expected_classes} classes but got {len(class_names)}")
                
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Checkpoint loaded successfully")
                
                if model_num_classes and model_num_classes > len(class_names):
                    logger.warning(f"Model has {model_num_classes} classes but only {len(class_names)} class names provided")
                    for i in range(len(class_names), model_num_classes):
                        class_names.append(f"Unknown_Class_{i}")
                    logger.info(f"Extended class names to: {class_names}")
                
            except Exception as e:
                logger.error(f"Error loading checkpoint: {str(e)}")
                raise
        
        criterion = nn.CrossEntropyLoss()
        
        if args.eval:
            print("Evaluating model...")
            logger.info("Starting evaluation mode")
            logger.info(f"Class names: {class_names}")
            logger.info(f"Number of classes: {len(class_names)}")
            
            eval_dir = os.path.join(log_dir, 'evaluation') 
            logger.info(f"Evaluation results will be saved to: {eval_dir}")
            
            logger.info(f"Test dataset size: {len(test_loader.dataset)}")
            logger.info(f"Test batch size: {test_loader.batch_size}")
            logger.info(f"Number of test batches: {len(test_loader)}")
            
            try:
                sample_batch = next(iter(test_loader))
                logger.info(f"Sample batch shape: images={sample_batch[0].shape}, labels={sample_batch[1].shape}")
                logger.info(f"Unique labels in sample batch: {torch.unique(sample_batch[1]).tolist()}")
            except Exception as e:
                logger.error(f"Error checking sample batch: {str(e)}")
            
            try:
                evaluate_model(
                    model=model,
                    test_loader=test_loader,
                    device=device,
                    class_names=class_names,
                    criterion=criterion,
                    save_dir=eval_dir
                )
                logger.info("Evaluation completed successfully")
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                raise
        else:
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
                scheduler=scheduler,
                progress_interval=args.progress_interval,
                plant_name=args.plant
            )
            
            logger.info("Starting model training...")
            metrics = trainer.train(
                num_epochs=args.epochs,
                save_path=os.path.join(model_dir, f"{args.plant}_best_model.pth")
            )
            
            metrics_plot_path = os.path.join(figures_dir, f"{args.plant}_training_metrics.png")
            plot_training_metrics(
                metrics=metrics,
                save_path=metrics_plot_path
            )
            logger.info(f"Training metrics plot saved to: {metrics_plot_path}")
            
            import json
            metrics_data_path = os.path.join(results_dir, 'training_metrics.json')
            with open(metrics_data_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Training metrics data saved to: {metrics_data_path}")
            
            logger.info("\nEvaluating model after training...")
            eval_dir = os.path.join(results_dir, 'evaluation')
            evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device,
                class_names=class_names,
                criterion=criterion,
                save_dir=eval_dir
            )
            
            logger.info("="*80)
            logger.info("Training completed successfully!")
            logger.info(f"All results saved to: {log_dir}")
            logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main() 