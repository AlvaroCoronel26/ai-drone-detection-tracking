from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"Direccion de modelo")
    

    results = model.train(
        data="C:/Users/Alvaro/Desktop/ENTRENAMIENTO/config.yaml",
        epochs=100,
        batch=8,
        device=0,
        resume=True
    )
