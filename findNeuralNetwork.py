from ultralytics import YOLO

model = YOLO(r"Direccion de modelo") 

print(model.info(verbose=True))