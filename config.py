# /config.py
import torch
from utils.utils import get_device
from models.attention import ProbAttention

try:
    from models.attention import XFormersAttention
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False
    
def get_attention_factory(name: str):
    """Fábrica que retorna la clase de atención solicitada."""
    factories = {'prob': ProbAttention}
    if XFORMERS_AVAILABLE:
        factories['xformers'] = XFormersAttention
    
    if name not in factories:
        raise ValueError(f"Tipo de atención desconocido: {name}. Opciones: {list(factories.keys())}")
    return factories[name]

def get_config():
    """Retorna un diccionario con toda la configuración del proyecto."""
    device = get_device()
    attention_type = 'xformers' if XFORMERS_AVAILABLE and device.type == 'cuda' else 'prob'

    return {
        # Flags de Optimización
        "use_amp": True and device.type == 'cuda',
        "use_checkpointing": False,
        # Parámetros de datos y rutas
        "csv_path": "sample_option_data.csv",
        "save_model_path": "best_informer_model.pth",
        "freq": 'd',
        # Parámetros de secuencias
        "seq_len": 30, "label_len": 5, "pred_len": 30,
        # Parámetros del modelo
        "enc_in": 6, "dec_in": 6, "c_out": 6,
        "d_model": 32, "n_heads": 3, "e_layers": 1, "d_layers": 2, "d_ff": 8,
        "dropout": 0.06, "factor": 3, "distil": True, "activation": 'gelu',
        # Parámetros de entrenamiento
        "batch_size": 64, "learning_rate": 0.0001, "epochs": 50, "patience": 5,
        "loss": "mse", "device": device,
        # Fábrica de Atención
        "attention_factory": get_attention_factory(attention_type)
    }
