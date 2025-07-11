# /profile.py
import torch
import time
import cProfile
import pstats
import argparse
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity

from config import get_config
from utils.utils import set_seed, get_logger
from data.dataset import get_dataloaders
from models.informer import Informer

def profile_cpu_data_loading(config, logger):
    logger.info("--- Iniciando Profiling de CPU para Carga de Datos ---")
    profiler = cProfile.Profile()
    profiler.enable()
    get_dataloaders(config, logger)
    profiler.disable()
    logger.info("--- Resultados del Profiling de CPU ---")
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)

def profile_model_inference(config, logger):
    logger.info("--- Iniciando Profiling de Modelo (CPU+GPU) ---")
    device = config['device']
    if device.type == 'cpu': logger.warning("Profiling de GPU no disponible, se ejecutará solo en CPU.")

    model = Informer(config).to(device).eval()
    train_loader, _, _, _ = get_dataloaders(config, logger)
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
    batch_x, batch_y, batch_x_mark, batch_y_mark = [d.to(device) for d in [batch_x, batch_y, batch_x_mark, batch_y_mark]]
    decoder_input = torch.cat([batch_y[:,:config['label_len'],:], torch.zeros_like(batch_y[:, -config['pred_len']:, :])], dim=1)

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda': activities.append(ProfilerActivity.CUDA)
    
    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        with record_function("model_inference"):
            _ = model(batch_x, batch_x_mark, decoder_input, batch_y_mark)

    logger.info("--- Resultados del Profiling del Modelo ---")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=15))

def benchmark_model(config, logger):
    logger.info("--- Iniciando Benchmark de Inferencia y Memoria ---")
    device = config['device']
    model = Informer(config).to(device).eval()
    train_loader, _, _, _ = get_dataloaders(config, logger)
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
    batch_x, batch_y, batch_x_mark, batch_y_mark = [d.to(device) for d in [batch_x, batch_y, batch_x_mark, batch_y_mark]]
    decoder_input = torch.cat([batch_y[:,:config['label_len'],:], torch.zeros_like(batch_y[:, -config['pred_len']:, :])], dim=1)

    if device.type == 'cuda':
        logger.info("Realizando pasadas de calentamiento (warm-up)...")
        for _ in range(10): _ = model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        torch.cuda.synchronize()

    timings = []
    logger.info("Midiendo latencia de inferencia...")
    for _ in range(100):
        start_time = time.perf_counter()
        _ = model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.perf_counter()
        timings.append((end_time - start_time) * 1000)

    logger.info("--- Resultados del Benchmark de Latencia ---")
    logger.info(f"Latencia promedio: {np.mean(timings):.3f} ms | Desviación estándar: {np.std(timings):.3f} ms | Latencia P95: {np.percentile(timings, 95):.3f} ms")

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        _ = model(batch_x, batch_x_mark, decoder_input, batch_y_mark)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        logger.info(f"--- Pico de memoria GPU utilizada: {peak_memory_mb:.2f} MB ---")

def main():
    parser = argparse.ArgumentParser(description="Herramientas de Profiling y Benchmarking para el Modelo Informer.")
    parser.add_argument("task", choices=['cpu', 'gpu', 'benchmark'], help="La tarea a ejecutar.")
    args = parser.parse_args()
    config, logger = get_config(), get_logger()
    set_seed()
    if args.task == 'cpu': profile_cpu_data_loading(config, logger)
    elif args.task == 'gpu': profile_model_inference(config, logger)
    elif args.task == 'benchmark': benchmark_model(config, logger)

if __name__ == "__main__":
    main()
