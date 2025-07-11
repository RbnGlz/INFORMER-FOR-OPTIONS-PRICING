# /train.py
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import os
import numpy as np
import mlflow
import joblib

from config import get_config
from utils.utils import set_seed, get_logger, calculate_metrics
from data.dataset import get_dataloaders
from models.informer import Informer

def _run_epoch(model, dataloader, optimizer, criterion, scaler_amp, device, config, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, all_preds, all_trues = 0.0, [], []

    for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = [d.to(device, non_blocking=True) for d in [batch_x, batch_y, batch_x_mark, batch_y_mark]]
        
        with autocast(enabled=config["use_amp"]):
            decoder_input = torch.cat([batch_y[:,:config['label_len'],:], torch.zeros_like(batch_y[:, -config['pred_len']:, :])], dim=1)
            outputs = model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
            true_values = batch_y[:, -config['pred_len']:, :]
            loss = criterion(outputs, true_values)
        
        if is_train:
            optimizer.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            all_preds.append(outputs.detach().cpu().numpy())
            all_trues.append(true_values.detach().cpu().numpy())
            
        total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    if not is_train:
        all_preds = np.concatenate(all_preds, axis=0).reshape(-1, config['c_out'])
        all_trues = np.concatenate(all_trues, axis=0).reshape(-1, config['c_out'])
        return avg_loss, all_preds, all_trues
    return avg_loss, None, None

def main():
    config = get_config()
    logger = get_logger()
    set_seed()
    
    mlflow.set_experiment("Informer Option Pricing")
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params({k: v for k, v in config.items() if k != 'attention_factory'})

        train_loader, val_loader, _, scaler = get_dataloaders(config, logger)
        joblib.dump(scaler, "scaler.gz")
        mlflow.log_artifact("scaler.gz")

        model = Informer(config).to(config['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.MSELoss()
        scaler_amp = GradScaler(enabled=config["use_amp"])
        
        best_val_loss = float('inf')
        for epoch in range(config['epochs']):
            train_loss, _, _ = _run_epoch(model, train_loader, optimizer, criterion, scaler_amp, config['device'], config, is_train=True)
            val_loss, val_preds, val_trues = _run_epoch(model, val_loader, None, criterion, None, config['device'], config, is_train=False)
            mae, da = calculate_metrics(val_preds, val_trues, scaler)
            
            logger.info(f"Epoch {epoch+1:02} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: ${mae:.4f} | Val DA: {da:.2f}%")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "val_mae": mae, "val_da": da}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), config['save_model_path'])
                mlflow.log_artifact(config['save_model_path'], artifact_path="model")
                logger.info(f"  -> Model saved. Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
