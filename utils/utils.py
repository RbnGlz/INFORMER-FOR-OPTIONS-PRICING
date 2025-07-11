# /utils/utils.py
import torch
import numpy as np
import random
import logging

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_logger(name="informer_logger"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])
    return logging.getLogger(name)

def calculate_metrics(preds, trues, scaler):
    # Asumimos que la última columna es el precio de la opción
    preds_unscaled = scaler.inverse_transform(np.pad(preds[..., -1:], ((0,0),(0,scaler.n_features_in_-1)), 'constant'))[:,-1]
    trues_unscaled = scaler.inverse_transform(np.pad(trues[..., -1:], ((0,0),(0,scaler.n_features_in_-1)), 'constant'))[:,-1]
    mae = np.mean(np.abs(preds_unscaled - trues_unscaled))
    preds_diff, trues_diff = np.diff(preds_unscaled), np.diff(trues_unscaled)
    correct_direction = (np.sign(preds_diff) == np.sign(trues_diff)).sum()
    da = (correct_direction / len(trues_diff)) * 100 if len(trues_diff) > 0 else 0.0
    return mae, da
