"""
Optimized configuration module with type safety and performance improvements.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, validator

from utils.utils import get_device

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Model configuration with validation."""
    
    # Model architecture parameters
    d_model: int = Field(default=512, ge=32, le=2048, description="Model dimension")
    n_heads: int = Field(default=8, ge=1, le=32, description="Number of attention heads")
    e_layers: int = Field(default=2, ge=1, le=8, description="Number of encoder layers")
    d_layers: int = Field(default=1, ge=1, le=8, description="Number of decoder layers")
    d_ff: int = Field(default=2048, ge=128, le=8192, description="Feed-forward dimension")
    dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="Dropout rate")
    factor: int = Field(default=5, ge=1, le=10, description="Attention factor")
    distil: bool = Field(default=True, description="Enable distillation")
    activation: str = Field(default="gelu", regex="^(relu|gelu|swish)$")
    
    # Input/output dimensions
    enc_in: int = Field(default=7, ge=1, le=50, description="Encoder input features")
    dec_in: int = Field(default=7, ge=1, le=50, description="Decoder input features")
    c_out: int = Field(default=7, ge=1, le=50, description="Output channels")
    
    # Sequence parameters
    seq_len: int = Field(default=96, ge=12, le=512, description="Input sequence length")
    label_len: int = Field(default=48, ge=6, le=256, description="Label sequence length")
    pred_len: int = Field(default=96, ge=12, le=256, description="Prediction length")
    
    @validator("d_model")
    def validate_d_model_divisible_by_heads(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure d_model is divisible by n_heads."""
        if "n_heads" in values and v % values["n_heads"] != 0:
            raise ValueError(f"d_model ({v}) must be divisible by n_heads ({values['n_heads']})")
        return v
    
    @validator("label_len")
    def validate_label_len(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure label_len is less than seq_len."""
        if "seq_len" in values and v >= values["seq_len"]:
            raise ValueError(f"label_len ({v}) must be less than seq_len ({values['seq_len']})")
        return v


class TrainingConfig(BaseModel):
    """Training configuration with validation."""
    
    batch_size: int = Field(default=32, ge=1, le=1024, description="Batch size")
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-1, description="Learning rate")
    epochs: int = Field(default=100, ge=1, le=1000, description="Training epochs")
    patience: int = Field(default=7, ge=1, le=50, description="Early stopping patience")
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1e-1, description="Weight decay")
    warmup_steps: int = Field(default=4000, ge=0, le=50000, description="Warmup steps")
    
    # Loss and optimization
    loss_fn: str = Field(default="mse", regex="^(mse|mae|huber|quantile)$")
    optimizer: str = Field(default="adamw", regex="^(adam|adamw|sgd|rmsprop)$")
    scheduler: str = Field(default="cosine", regex="^(step|cosine|plateau|none)$")
    
    # Regularization
    gradient_clip_val: float = Field(default=1.0, ge=0.0, le=10.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.3)
    
    # Performance optimizations
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")
    use_compile: bool = Field(default=True, description="Use torch.compile")
    use_checkpointing: bool = Field(default=False, description="Use gradient checkpointing")
    num_workers: int = Field(default=4, ge=0, le=16, description="DataLoader workers")
    pin_memory: bool = Field(default=True, description="Pin memory for faster transfer")


class DataConfig(BaseModel):
    """Data configuration with validation."""
    
    csv_path: str = Field(default="data/sample_option_data.csv", description="Data file path")
    freq: str = Field(default="h", regex="^(h|d|m|s)$", description="Data frequency")
    target_col: str = Field(default="precio_opcion", description="Target column name")
    
    # Data splits
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9, description="Training split ratio")
    val_ratio: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation split ratio")
    test_ratio: float = Field(default=0.1, ge=0.1, le=0.5, description="Test split ratio")
    
    # Data preprocessing
    normalize: bool = Field(default=True, description="Normalize data")
    scale_method: str = Field(default="standard", regex="^(standard|minmax|robust)$")
    
    @validator("train_ratio", "val_ratio", "test_ratio")
    def validate_ratios_sum(cls, v: float, values: Dict[str, Any]) -> float:
        """Ensure ratios sum to 1.0."""
        if len(values) == 2:  # All three ratios are present
            total = sum(values.values()) + v
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Train, validation, and test ratios must sum to 1.0, got {total}")
        return v


class OptimizationConfig(BaseModel):
    """Performance optimization configuration."""
    
    # Attention optimization
    attention_type: str = Field(default="flash", regex="^(flash|xformers|vanilla)$")
    use_flash_attn: bool = Field(default=True, description="Use FlashAttention if available")
    
    # Model compilation
    compile_mode: str = Field(default="default", regex="^(default|reduce-overhead|max-autotune)$")
    compile_fullgraph: bool = Field(default=False, description="Compile full graph")
    
    # Memory optimization
    memory_efficient: bool = Field(default=True, description="Enable memory efficient attention")
    offload_to_cpu: bool = Field(default=False, description="Offload to CPU when possible")
    
    # Distributed training
    use_ddp: bool = Field(default=False, description="Use DistributedDataParallel")
    find_unused_parameters: bool = Field(default=False, description="Find unused parameters in DDP")


class ExperimentConfig(BaseModel):
    """Experiment tracking configuration."""
    
    experiment_name: str = Field(default="informer-option-pricing", description="Experiment name")
    run_name: Optional[str] = Field(default=None, description="Run name")
    tags: list[str] = Field(default_factory=list, description="Experiment tags")
    
    # Logging
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_every_n_steps: int = Field(default=100, ge=1, le=1000, description="Log frequency")
    save_every_n_epochs: int = Field(default=10, ge=1, le=100, description="Save frequency")
    
    # Paths
    checkpoint_dir: str = Field(default="checkpoints", description="Checkpoint directory")
    log_dir: str = Field(default="logs", description="Log directory")
    model_save_path: str = Field(default="best_model.pth", description="Model save path")


class Config(BaseModel):
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    
    # Device configuration
    device: str = Field(default="auto", description="Device to use")
    seed: int = Field(default=42, ge=0, le=2**32-1, description="Random seed")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
        
    def __post_init__(self) -> None:
        """Post-initialization setup."""
        if self.device == "auto":
            self.device = str(get_device())
        
        # Set up logging
        logging.basicConfig(
            level=getattr(logging, self.experiment.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def get_attention_factory(attention_type: str) -> Type[nn.Module]:
    """
    Factory function to get attention mechanism.
    
    Args:
        attention_type: Type of attention mechanism
        
    Returns:
        Attention class
        
    Raises:
        ValueError: If attention type is not supported
    """
    from models.attention import ProbAttention
    
    factories = {"prob": ProbAttention, "vanilla": ProbAttention}
    
    # Try to import optimized attention mechanisms
    try:
        from models.attention import XFormersAttention
        factories["xformers"] = XFormersAttention
    except ImportError:
        logger.warning("XFormers not available, falling back to ProbAttention")
    
    try:
        from models.attention import FlashAttention
        factories["flash"] = FlashAttention
    except ImportError:
        logger.warning("FlashAttention not available, falling back to ProbAttention")
    
    if attention_type not in factories:
        available = list(factories.keys())
        raise ValueError(f"Unknown attention type: {attention_type}. Available: {available}")
    
    return factories[attention_type]


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    if config_path:
        try:
            cfg = OmegaConf.load(config_path)
            return Config(**cfg)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    
    return Config()


def get_optimized_config() -> Dict[str, Any]:
    """
    Get optimized configuration with performance improvements.
    
    Returns:
        Optimized configuration dictionary
    """
    config = load_config()
    device = get_device()
    
    # Enable optimizations based on device capabilities
    if device.type == "cuda":
        config.training.use_amp = True
        config.training.pin_memory = True
        config.optimization.use_flash_attn = True
        config.optimization.memory_efficient = True
        
        # Enable compilation for better performance
        if hasattr(torch, "compile"):
            config.optimization.use_compile = True
            config.training.use_compile = True
    else:
        # CPU optimizations
        config.training.use_amp = False
        config.training.pin_memory = False
        config.optimization.use_flash_attn = False
        config.training.num_workers = min(4, torch.get_num_threads())
    
    # Get attention factory
    attention_factory = get_attention_factory(config.optimization.attention_type)
    
    # Convert to dictionary for backward compatibility
    config_dict = config.dict()
    config_dict["device"] = device
    config_dict["attention_factory"] = attention_factory
    
    return config_dict


# Backward compatibility
def get_config() -> Dict[str, Any]:
    """Get configuration dictionary (backward compatibility)."""
    return get_optimized_config()