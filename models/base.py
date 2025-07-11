# /models/base.py
from torch.nn import Module
from abc import ABC, abstractmethod

class BaseAttention(Module, ABC):
    """
    Clase base abstracta para todas las implementaciones de atención.
    Define la interfaz que todas las capas de atención deben seguir.
    """
    @abstractmethod
    def forward(self, queries, keys, values, attn_mask):
        """
        El método forward debe ser implementado por todas las subclases.
        """
        raise NotImplementedError
