import cv2
import time
import threading
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, db

# ------------------------
# FIREBASE INIT
# ------------------------
credentials_path = "YOUR_CREDENTIALS.json"

# ------------------------
# YOLO INIT
# ------------------------
model = YOLO(r"Direccion de modelo")

# ------------------------
# VARIABLES GLOBALES
# ------------------------
latest_coords = None
send_interval = 0.2  # 5 Hz para el PID
running = True

# Detección persistente
dron_detectado = False
tiempo_inicio_deteccion = None
UMBRAL_ALERTA = 30  # segundos

def enviar_sms_simulado(mensaje):
    print("\n======================")
    print("SIMULACIÓN SMS SIM800L")
    print("======================")
    print("→ ALERTA ENVIADA:")
    print(mensaje)
    print("======================\n")

# ------------------------
# THREAD PARA FIREBASE
# ------------------------
def firebase_sender():
    global latest_coords, running
    last_sent_time = 0
    
    while running:
        if latest_coords is not None:
            now = time.time()
            if now - last_sent_time >= send_interval:
                try:
                    ref.update(latest_coords)
                    last_sent_time = now
                except Exception as e:
                    print("Firebase error:", e)

        time.sleep(0.01)

threading.Thread(target=firebase_sender, daemon=True).start()

# ------------------------
# LOOP WEBCAM
# ------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    results = model(frame, verbose=False)
    detections = results[0].boxes

    if len(detections) > 0:
        # Se detectó dron
        box = detections[0].xyxy[0]
        x1, y1, x2, y2 = box
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)

        # Dibujar
        cv2.circle(frame, (x_center, y_center), 5, (0,255,0), -1)

        # Actualizar coordenadas globales
        latest_coords = {
            "x": x_center,
            "y": y_center,
            "timestamp": int(time.time() * 1000)
        }

        # Tiempo persistente
        if not dron_detectado:
            dron_detectado = True
            tiempo_inicio_deteccion = time.time()

        else:
            tiempo_presente = time.time() - tiempo_inicio_deteccion

            cv2.putText(frame, f"Dron detectado por {tiempo_presente:.1f} s",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # ALERTA
            if tiempo_presente >= UMBRAL_ALERTA:
                enviar_sms_simulado(
                    f"ALERTA: Dron detectado por {UMBRAL_ALERTA} segundos continuos."
                )
                tiempo_inicio_deteccion = time.time()  # Reinicia para evitar spam

    else:
        # No detectado → reset
        dron_detectado = False
        tiempo_inicio_deteccion = None

    # FPS
    fps = 1 / (time.time() - start + 1e-8)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Detección y Alerta", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running = False
cap.release()
cv2.destroyAllWindows()
