# /data/dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from pydantic import ValidationError
from .validation import OptionDataRow

def _validate_dataframe(df, logger):
    # ... (implementación idéntica a la anterior)
    errors = []
    for i, row in df.iterrows():
        try: _ = OptionDataRow(**row.to_dict())
        except ValidationError as e: errors.append(f"Fila {i+2}: {e}\n")
    if errors:
        raise ValueError(f"Errores de validación en los datos: {len(errors)}.\n" + "".join(errors[:5]))
    logger.info("Validación de datos de entrada completada.")

def _create_time_features(df):
    # ... (implementación idéntica a la anterior)
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['month'] = df.fecha.dt.month
    df['day'] = df.fecha.dt.day
    df['weekday'] = df.fecha.dt.weekday
    return df[['month', 'day', 'weekday']].values

class OptionsDataset(Dataset):
    # ... (implementación idéntica a la anterior)
    def __init__(self, data, data_stamps, seq_len, label_len, pred_len, **_):
        self.data, self.data_stamps = data, data_stamps
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.total_len = seq_len + pred_len
    def __len__(self): return len(self.data) - self.total_len + 1
    def __getitem__(self, idx):
        s_begin, s_end = idx, idx + self.seq_len
        l_begin, l_end = s_end - self.label_len, s_end + self.pred_len
        return (torch.FloatTensor(self.data[s_begin:s_end]), torch.FloatTensor(self.data[l_begin:l_end]),
                torch.FloatTensor(self.data_stamps[s_begin:s_end]), torch.FloatTensor(self.data_stamps[l_begin:l_end]))

def get_dataloaders(config, logger):
    # ... (implementación idéntica a la anterior)
    df = pd.read_csv(config["csv_path"])
    _validate_dataframe(df, logger)
    time_features = _create_time_features(df.copy())
    data_features = ['precio_subyacente', 'volatilidad_implicita', 'tiempo_hasta_vencimiento', 'precio_ejercicio', 'tipo_opcion', 'precio_opcion']
    train_size, val_size = int(len(df) * 0.7), int(len(df) * 0.15)
    train_df, val_df, test_df = df[:train_size], df[train_size:train_size+val_size], df[train_size+val_size:]
    scaler = MinMaxScaler()
    train_data, val_data, test_data = scaler.fit_transform(train_df[data_features]), scaler.transform(val_df[data_features]), scaler.transform(test_df[data_features])
    train_stamps, val_stamps, test_stamps = time_features[:train_size], time_features[train_size:train_size+val_size], time_features[train_size+val_size:]
    pin_mem = config["device"].type == 'cuda'
    train_ds, val_ds, test_ds = OptionsDataset(train_data, train_stamps, **config), OptionsDataset(val_data, val_stamps, **config), OptionsDataset(test_data, test_stamps, **config)
    return (DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=pin_mem),
            DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=pin_mem),
            DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=pin_mem),
            scaler)
