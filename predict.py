# /predict.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import joblib
import argparse
import os

from config import get_config
from models.informer import Informer
from data.dataset import get_dataloaders
from utils.utils import get_logger, set_seed

def predict(run_id, config):
    logger = get_logger()
    set_seed()
    device = config['device']
    
    logger.info(f"Cargando modelo y scaler del Run de MLflow ID: {run_id}")
    client = mlflow.tracking.MlflowClient()
    local_dir = "temp_artifacts"
    if not os.path.exists(local_dir): os.makedirs(local_dir)
    
    model_path = client.download_artifacts(run_id, "model/" + config['save_model_path'], local_dir)
    scaler_path = client.download_artifacts(run_id, "scaler.gz", local_dir)

    model = Informer(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)
    
    _, _, test_loader, _ = get_dataloaders(config, logger)
    
    logger.info("Realizando predicciones en el conjunto de prueba...")
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [d.to(device) for d in [batch_x, batch_y, batch_x_mark, batch_y_mark]]
            decoder_input = torch.cat([batch_y[:,:config['label_len'],:], torch.zeros_like(batch_y[:, -config['pred_len']:, :])], dim=1)
            outputs = model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
            all_preds.append(outputs.detach().cpu().numpy())
            all_trues.append(batch_y[:, -config['pred_len']:, :].detach().cpu().numpy())
            
    preds = np.concatenate(all_preds, axis=0).reshape(-1, config['c_out'])
    trues = np.concatenate(all_trues, axis=0).reshape(-1, config['c_out'])
    
    preds_unscaled = scaler.inverse_transform(np.pad(preds[..., -1:], ((0,0),(0,scaler.n_features_in_-1)), 'constant'))[:,-1]
    trues_unscaled = scaler.inverse_transform(np.pad(trues[..., -1:], ((0,0),(0,scaler.n_features_in_-1)), 'constant'))[:,-1]
    
    return preds_unscaled, trues_unscaled

def plot_predictions(preds, trues, logger):
    plt.figure(figsize=(15, 7))
    plt.plot(trues, label='Valores Reales', color='blue', alpha=0.7)
    plt.plot(preds, label='Predicciones del Modelo', color='red', linestyle='--')
    plt.title('Comparación de Predicciones del Precio de la Opción')
    plt.xlabel('Horizonte de Tiempo (Días)')
    plt.ylabel('Precio de la Opción ($)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = "prediction_vs_actual.png"
    plt.savefig(plot_filename)
    logger.info(f"Gráfica de predicciones guardada en '{plot_filename}'")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realizar predicciones con un modelo entrenado de MLflow.")
    parser.add_argument("--run_id", type=str, required=True, help="El ID del Run de MLflow del cual cargar el modelo.")
    args = parser.parse_args()
    
    config = get_config()
    preds, trues = predict(args.run_id, config)
    plot_predictions(preds, trues, get_logger())
