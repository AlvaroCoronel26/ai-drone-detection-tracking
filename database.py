import cv2
import time
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO

# ---------------------------------------------------------
# CONFIGURACIÓN DE FIREBASE
# ---------------------------------------------------------
credentials_path = "YOUR_CREDENTIALS.json"

# ---------------------------------------------------------
# CARGA DEL MODELO YOLO
# ---------------------------------------------------------
model = YOLO(r"Direccion de modelo")   # ← CAMBIAR por tu mejor modelo
                            # ej: "runs/detect/train_yolo11n/weights/best.pt"

# ---------------------------------------------------------
# CAPTURA DE VIDEO
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)  # webcam. Para archivo coloca: "video.mp4"

# ---------------------------------------------------------
# LOOP PRINCIPAL
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar inferencia
    results = model(frame, device=0)  # usa GPU

    drone_detected = False
    coords = {"x": None, "y": None, "w": None, "h": None}

    for r in results:
        boxes = r.boxes

        if len(boxes) > 0:  # dron encontrado
            drone_detected = True
            box = boxes[0]  # tomamos el más confiable

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # Coordenadas del centro del bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            w = int(x2 - x1)
            h = int(y2 - y1)

            coords = {"x": cx, "y": cy, "w": w, "h": h}

            # Dibujar la caja
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
            cv2.putText(frame, "DRON", (cx-30, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ---------------------------------------------------------
    # ENVIAR DATOS A FIREBASE
    # ---------------------------------------------------------
    data_to_send = {
        "detectado": drone_detected,
        "coordenadas": coords,
        "timestamp": time.time()
    }

    ref.set(data_to_send)

    # Mostrar video
    cv2.imshow("Deteccion de Dron", frame)

    if cv2.waitKey(1) == 27:  # tecla ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
