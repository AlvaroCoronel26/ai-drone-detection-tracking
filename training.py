from ultralytics import YOLO

def main():
    model = YOLO(r"Direccion de modelo")  # o la ruta de tu modelo base
    results = model.train(
        data="config.yaml",       # tu archivo de configuración
        epochs=100,               # número de épocas
        batch=8,                  # tamaño de lote
        device=0,                 # usa tu GPU (0)
        resume=True                  
    )

if __name__ == "__main__":
    main()
