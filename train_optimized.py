#!/usr/bin/env python3
"""
Optimized training script for Informer Option Pricing Model.

This script includes modern PyTorch optimizations, comprehensive logging,
and professional MLOps practices.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from utils.utils import set_seed, get_logger, calculate_metrics
from data.dataset import get_dataloaders
from models.informer import Informer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)


class OptimizedTrainer:
    """
    Professional trainer with modern optimizations and best practices.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        scaler_data: Optional[object] = None,
    ):
        """
        Initialize the optimized trainer.
        
        Args:
            model: The neural network model
            config: Training configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader (optional)
            scaler_data: Data scaler for inverse transform
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler_data = scaler_data
        
        # Initialize device and distributed training
        self.device = config["device"]
        self.use_ddp = config.get("use_ddp", False)
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize optimization components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        self._setup_amp()
        
        # Initialize monitoring
        self._setup_monitoring()
        
        # Initialize model compilation if available
        self._setup_model_compilation()
        
        # Best metrics tracking
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        self.patience_counter = 0
        
        # Timing
        self.epoch_times = []
        self.step_times = []

    def _setup_optimizer(self) -> None:
        """Setup optimizer with advanced configurations."""
        optimizer_name = self.config.get("optimizer", "adamw").lower()
        lr = self.config["learning_rate"]
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        logger.info(f"Using optimizer: {optimizer_name} with lr={lr}")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_name = self.config.get("scheduler", "cosine").lower()
        
        if scheduler_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"],
                eta_min=self.config["learning_rate"] * 0.01,
            )
        elif scheduler_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config["epochs"] // 3,
                gamma=0.5,
            )
        elif scheduler_name == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
            )
        elif scheduler_name == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        logger.info(f"Using scheduler: {scheduler_name}")

    def _setup_criterion(self) -> None:
        """Setup loss function."""
        loss_name = self.config.get("loss", "mse").lower()
        
        if loss_name == "mse":
            self.criterion = nn.MSELoss()
        elif loss_name == "mae":
            self.criterion = nn.L1Loss()
        elif loss_name == "huber":
            self.criterion = nn.HuberLoss()
        elif loss_name == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        
        logger.info(f"Using loss function: {loss_name}")

    def _setup_amp(self) -> None:
        """Setup Automatic Mixed Precision."""
        self.use_amp = self.config.get("use_amp", False) and self.device.type == "cuda"
        if self.use_amp:
            self.scaler_amp = GradScaler()
            logger.info("Using Automatic Mixed Precision (AMP)")
        else:
            self.scaler_amp = None

    def _setup_monitoring(self) -> None:
        """Setup monitoring and logging."""
        # TensorBoard
        log_dir = Path(self.config.get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir / "tensorboard")
        
        # MLflow
        mlflow.set_experiment(self.config.get("experiment_name", "informer-option-pricing"))
        
        logger.info(f"Logging to: {log_dir}")

    def _setup_model_compilation(self) -> None:
        """Setup model compilation for PyTorch 2.0+."""
        use_compile = self.config.get("use_compile", False)
        
        if use_compile and hasattr(torch, "compile"):
            try:
                compile_mode = self.config.get("compile_mode", "default")
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    fullgraph=self.config.get("compile_fullgraph", False),
                )
                logger.info(f"Model compiled with mode: {compile_mode}")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
                logger.info("Continuing without compilation")

    def run_epoch(
        self,
        dataloader: DataLoader,
        is_train: bool = True,
        epoch: int = 0,
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run one epoch of training or validation.
        
        Args:
            dataloader: Data loader for the epoch
            is_train: Whether this is a training epoch
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, predictions, true_values)
        """
        self.model.train() if is_train else self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_trues = []
        
        # Initialize progress bar
        desc = "Training" if is_train else "Validation"
        pbar = tqdm(dataloader, desc=f"{desc} Epoch {epoch+1}", leave=False)
        
        step_times = []
        
        for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
            step_start = time.time()
            
            # Move to device with non-blocking transfer
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            batch_x_mark = batch_x_mark.to(self.device, non_blocking=True)
            batch_y_mark = batch_y_mark.to(self.device, non_blocking=True)
            
            # Forward pass with optional mixed precision
            with autocast(enabled=self.use_amp):
                # Prepare decoder input
                decoder_input = torch.cat([
                    batch_y[:, :self.config['label_len'], :],
                    torch.zeros_like(batch_y[:, -self.config['pred_len']:, :])
                ], dim=1)
                
                # Model forward pass
                outputs = self.model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
                true_values = batch_y[:, -self.config['pred_len']:, :]
                loss = self.criterion(outputs, true_values)
            
            # Backward pass and optimization
            if is_train:
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler_amp.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.get("gradient_clip_val", 0) > 0:
                        self.scaler_amp.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["gradient_clip_val"]
                        )
                    
                    self.scaler_amp.step(self.optimizer)
                    self.scaler_amp.update()
                else:
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.get("gradient_clip_val", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["gradient_clip_val"]
                        )
                    
                    self.optimizer.step()
                
                # Log training metrics
                if batch_idx % self.config.get("log_every_n_steps", 100) == 0:
                    self.writer.add_scalar("Loss/Train_Step", loss.item(), 
                                         epoch * len(dataloader) + batch_idx)
                    self.writer.add_scalar("Learning_Rate", 
                                         self.optimizer.param_groups[0]['lr'],
                                         epoch * len(dataloader) + batch_idx)
                    
                    # Memory usage
                    if self.device.type == "cuda":
                        memory_used = torch.cuda.max_memory_allocated() / 1024**3
                        self.writer.add_scalar("Memory/GPU_GB", memory_used,
                                             epoch * len(dataloader) + batch_idx)
            
            else:
                # Validation - collect predictions
                all_preds.append(outputs.detach().cpu().numpy())
                all_trues.append(true_values.detach().cpu().numpy())
            
            total_loss += loss.item()
            
            # Timing
            step_time = time.time() - step_start
            step_times.append(step_time)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg_Loss': f'{total_loss/(batch_idx+1):.6f}',
                'Step_Time': f'{step_time:.3f}s'
            })
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        
        # Average step time
        avg_step_time = np.mean(step_times) if step_times else 0
        self.step_times.append(avg_step_time)
        
        # Process validation predictions
        if not is_train and all_preds:
            all_preds = np.concatenate(all_preds, axis=0)
            all_trues = np.concatenate(all_trues, axis=0)
            
            # Reshape for metric calculation
            all_preds = all_preds.reshape(-1, self.config['c_out'])
            all_trues = all_trues.reshape(-1, self.config['c_out'])
            
            return avg_loss, all_preds, all_trues
        
        return avg_loss, None, None

    def train(self) -> Dict:
        """
        Main training loop with comprehensive monitoring.
        
        Returns:
            Dictionary with training results and metrics
        """
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config}")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params({
                k: v for k, v in self.config.items() 
                if k not in ['attention_factory', 'device']
            })
            
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            
            training_start = time.time()
            
            for epoch in range(self.config['epochs']):
                epoch_start = time.time()
                
                # Training phase
                train_loss, _, _ = self.run_epoch(
                    self.train_loader, is_train=True, epoch=epoch
                )
                
                # Validation phase
                val_loss, val_preds, val_trues = self.run_epoch(
                    self.val_loader, is_train=False, epoch=epoch
                )
                
                # Calculate metrics
                if val_preds is not None and self.scaler_data is not None:
                    mae, da = calculate_metrics(val_preds, val_trues, self.scaler_data)
                else:
                    mae, da = 0.0, 0.0
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Timing
                epoch_time = time.time() - epoch_start
                self.epoch_times.append(epoch_time)
                
                # Logging
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1:03d}/{self.config['epochs']:03d} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Val MAE: ${mae:.4f} | "
                    f"Val DA: {da:.2f}% | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
                
                # TensorBoard logging
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Metrics/MAE", mae, epoch)
                self.writer.add_scalar("Metrics/DA", da, epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)
                self.writer.add_scalar("Time/Epoch", epoch_time, epoch)
                
                # MLflow logging
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mae": mae,
                    "val_da": da,
                    "learning_rate": current_lr,
                    "epoch_time": epoch_time,
                }, step=epoch)
                
                # Model checkpointing
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_metrics = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "mae": mae,
                        "da": da,
                    }
                    self.patience_counter = 0
                    
                    # Save model
                    checkpoint_path = Path(self.config['save_model_path'])
                    checkpoint_path.parent.mkdir(exist_ok=True)
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'best_val_loss': self.best_val_loss,
                        'config': self.config,
                    }, checkpoint_path)
                    
                    # MLflow model logging
                    mlflow.pytorch.log_model(
                        self.model,
                        "model",
                        registered_model_name="InformerOptionPricing"
                    )
                    
                    logger.info(f"✓ Model saved with val_loss: {self.best_val_loss:.6f}")
                
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config['patience']:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Memory cleanup
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # Training summary
            training_time = time.time() - training_start
            avg_epoch_time = np.mean(self.epoch_times)
            avg_step_time = np.mean(self.step_times)
            
            logger.info("Training completed!")
            logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
            logger.info(f"Total training time: {training_time:.1f}s")
            logger.info(f"Average epoch time: {avg_epoch_time:.1f}s")
            logger.info(f"Average step time: {avg_step_time:.3f}s")
            
            # Log final metrics
            mlflow.log_metrics({
                "best_val_loss": self.best_val_loss,
                "total_training_time": training_time,
                "avg_epoch_time": avg_epoch_time,
                "avg_step_time": avg_step_time,
            })
            
            # Close TensorBoard writer
            self.writer.close()
            
            return {
                "best_val_loss": self.best_val_loss,
                "best_metrics": self.best_metrics,
                "training_time": training_time,
                "avg_epoch_time": avg_epoch_time,
                "avg_step_time": avg_step_time,
            }


def main():
    """Main training function."""
    # Set up configuration
    config = get_config()
    logger.info(f"Using device: {config['device']}")
    
    # Set random seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(config, logger)
    
    # Create model
    logger.info("Creating model...")
    model = Informer(config).to(config['device'])
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = OptimizedTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler_data=scaler,
    )
    
    # Train model
    try:
        results = trainer.train()
        logger.info("Training completed successfully!")
        return results
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()