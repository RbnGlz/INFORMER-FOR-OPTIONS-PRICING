# /data/validation.py
from pydantic import BaseModel, Field, conint
from datetime import date

class OptionDataRow(BaseModel):
    fecha: date
    precio_subyacente: float = Field(..., gt=0)
    volatilidad_implicita: float = Field(..., ge=0)
    tiempo_hasta_vencimiento: float = Field(..., ge=0)
    precio_ejercicio: float = Field(..., gt=0)
    tipo_opcion: conint(ge=0, le=1)
    precio_opcion: float = Field(..., ge=0)
    class Config: coerce_numbers_to_str = True
