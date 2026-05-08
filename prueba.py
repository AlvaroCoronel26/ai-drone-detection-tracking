# pruebagpu.py
import torch

print("¿CUDA disponible?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Número de GPUs:", torch.cuda.device_count())
    print("Nombre de GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA no detectada correctamente.")
