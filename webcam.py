import cv2
import time
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Cargar modelo YOLO (usa aquí tu modelo entrenado)
# ---------------------------------------------------------
model = YOLO(r"Direccion de modelo")  

# ---------------------------------------------------------
# 2. Inicializar cámara (0 = webcam principal)
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
    exit()

# Para mejorar la captura
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()

# ---------------------------------------------------------
# 3. Bucle de detección en tiempo real
# ---------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tiempo inicio para FPS
    start_time = time.time()

    # Realizar detección
    results = model(frame, verbose=False)

    # Extraer coordenadas
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]

                # Coordenadas del centro
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)

                # Dibujar bounding box
                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 255, 0), 2)

                # Dibujar centro
                cv2.circle(frame, (x_center, y_center), 4, (0, 0, 255), -1)

                # Mostrar coordenadas en pantalla
                cv2.putText(frame,
                            f"Centro: ({x_center}, {y_center})",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

    # ---------------------------------------------------------
    # 4. Calcular FPS
    # ---------------------------------------------------------
    end_time = time.time()
    fps = 1 / (end_time - start_time + 1e-8)

    # Mostrar FPS en pantalla
    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2)

    # Mostrar la imagen
    cv2.imshow("Deteccion en tiempo real", frame)

    # Salir con tecla Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------------------------
# 5. Liberar recursos
# ---------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
