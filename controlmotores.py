import time
import matplotlib.pyplot as plt

# --- Parámetros del motor ---
STEP_ANGLE = 1.8              # grados por paso
MICROSTEPPING = 16
RESOLUTION = STEP_ANGLE / MICROSTEPPING  # grados por microstep

# --- Cámara (centro de imagen) ---
CAM_W, CAM_H = 640, 480
CX, CY = CAM_W // 2, CAM_H // 2

# --- PID ---
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
    
    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp*error + self.ki*self.integral + self.kd*derivative

pid_pan = PID(0.4, 0.01, 0.05)
pid_tilt = PID(0.4, 0.01, 0.05)

angle_pan = 0
angle_tilt = 0

# --- Simulación de coordenadas del dron ---
coords_dron = [
    (350, 200), (360, 220), (370, 230), (380, 240),
    (400, 260), (430, 280), (460, 300), (500, 320)
]

hist_angles_pan = []
hist_angles_tilt = []

print("\n--- Simulacion PAN/TILT ---")

for (x,y) in coords_dron:
    start = time.time()

    # Error en pixeles
    error_x = CX - x
    error_y = CY - y

    # PID
    dt = 0.1
    control_pan = pid_pan.update(error_x, dt)
    control_tilt = pid_tilt.update(error_y, dt)

    # Convertir control -> ángulo
    angle_pan += control_pan * 0.05     # factor de velocidad
    angle_tilt += control_tilt * 0.05

    # Convertir ángulo → pasos del motor
    steps_pan = angle_pan / RESOLUTION
    steps_tilt = angle_tilt / RESOLUTION

    print(f"\nCoordenadas dron: {x},{y}")
    print(f"Error → X:{error_x:.1f} px  Y:{error_y:.1f} px")
    print(f"Ángulos → PAN:{angle_pan:.2f}°  TILT:{angle_tilt:.2f}°")
    print(f"Pasos → PAN:{steps_pan:.1f}  TILT:{steps_tilt:.1f}")

    hist_angles_pan.append(angle_pan)
    hist_angles_tilt.append(angle_tilt)

    time.sleep(0.1)

# --- Graficar para tu tesis ---
plt.plot(hist_angles_pan, label="Ángulo TILT (°)")
plt.plot(hist_angles_tilt, label="Ángulo PAN (°)")
plt.xlabel("Tiempo (iteraciones)")
plt.ylabel("Ángulo (°)")    
plt.title("Simulación del Control PAN/TILT usando PID")
plt.legend()
plt.grid()
plt.show()
