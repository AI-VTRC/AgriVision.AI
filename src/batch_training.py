#!/usr/bin/env python
"""
Batch Training Script for Multiple Plants
This script trains models for Apple, Maize, and Tomato plants sequentially.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
import time
import json

# Configure logging
def setup_batch_logging():
    """Set up logging for batch training."""
    log_dir = f"./outputs/batch_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'batch_training.log')
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_dir

# Initialize logging
logger, batch_log_dir = setup_batch_logging()

def run_plant_training(plant_name, epochs=5, batch_size=32):
    """
    Run training for a specific plant.
    
    Args:
        plant_name (str): Name of the plant (Apple, Maize, Tomato)
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (success, duration, output_dir)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting training for {plant_name}")
    logger.info(f"{'='*80}")
    
    start_time = time.time()
    
    # Construct the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "main.py",
        "--plant", plant_name,
        "--data_dir", "../Dataset",
        "--output_dir", "./outputs",
        "--model_name", "clip",
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", "0.001",
        "--weight_decay", "0.0001",
        "--num_workers", "4",
        "--train_ratio", "0.7",
        "--val_ratio", "0.15",
        "--random_seed", "42"
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the training command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        output_lines = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line.strip())  # Print to console
                    output_lines.append(line.strip())
                    logger.debug(line.strip())  # Log to file
        
        # Wait for process to complete
        return_code = process.wait()
        
        duration = time.time() - start_time
        
        if return_code == 0:
            logger.info(f"✓ {plant_name} training completed successfully in {duration:.2f} seconds")
            
            # Find the output directory from the logs
            output_dir = None
            for line in output_lines:
                if "Output directory:" in line:
                    output_dir = line.split("Output directory:")[-1].strip()
                    break
            
            return True, duration, output_dir
        else:
            logger.error(f"✗ {plant_name} training failed with return code {return_code}")
            return False, duration, None
            
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"✗ Error training {plant_name}: {str(e)}")
        return False, duration, None

def main():
    """Main batch training function."""
    logger.info("="*80)
    logger.info("BATCH TRAINING STARTED")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Batch log directory: {batch_log_dir}")
    logger.info("="*80)
    
    # Define plants to train
    plants = ["Apple", "Maize", "Tomato"]
    
    # Training parameters
    epochs = 5
    batch_size = 32
    
    logger.info(f"Plants to train: {', '.join(plants)}")
    logger.info(f"Training parameters: epochs={epochs}, batch_size={batch_size}")
    
    # Results tracking
    results = {
        "start_time": datetime.now().isoformat(),
        "parameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.001,
            "weight_decay": 0.0001
        },
        "plants": {}
    }
    
    total_start_time = time.time()
    successful_plants = []
    failed_plants = []
    
    # Train each plant
    for plant in plants:
        success, duration, output_dir = run_plant_training(plant, epochs, batch_size)
        
        results["plants"][plant] = {
            "success": success,
            "duration_seconds": duration,
            "output_directory": output_dir
        }
        
        if success:
            successful_plants.append(plant)
        else:
            failed_plants.append(plant)
            logger.warning(f"Continuing with remaining plants despite {plant} failure...")
    
    # Calculate total duration
    total_duration = time.time() - total_start_time
    results["total_duration_seconds"] = total_duration
    results["end_time"] = datetime.now().isoformat()
    
    # Save results summary
    results_file = os.path.join(batch_log_dir, "batch_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BATCH TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logger.info(f"Successful plants ({len(successful_plants)}): {', '.join(successful_plants)}")
    
    if failed_plants:
        logger.error(f"Failed plants ({len(failed_plants)}): {', '.join(failed_plants)}")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Batch log saved to: {os.path.join(batch_log_dir, 'batch_training.log')}")
    
    # Exit with appropriate code
    if failed_plants:
        logger.error("\n⚠️  Some plants failed to train successfully!")
        sys.exit(1)
    else:
        logger.info("\n✅ All plants trained successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()