import numpy as np
import matplotlib.pyplot as plt

# Puntos detectados

#derecha hacia abajo 
#P1 = np.array([287, 245])
#P2 = np.array([324, 231])

#derecha hacia arriba

#P1 = np.array([250, 260])
#P2 = np.array([330, 220])

#izquierda hacia abajo
P1 = np.array([350, 200])
P2 = np.array([290, 260])


# Centro de la cámara
C = np.array([320, 240])

# Errores
e1 = P1 - C
e2 = P2 - C

# Control proporcional
Kp = 0.1  # ganancia asumida
theta1 = Kp * e1
theta2 = Kp * e2

# Simulación temporal de corrección (10 pasos)
t = np.linspace(0,1,30)
x_traj = np.linspace(P1[0], P2[0], len(t))
y_traj = np.linspace(P1[1], P2[1], len(t))

# Cálculo del error en el tiempo
ex = x_traj - C[0]
ey = y_traj - C[1]
theta_x = Kp * ex
theta_y = Kp * ey

# Graficar errores
plt.figure()
plt.plot(t, ex, label='Error en X (PAN)')
plt.plot(t, ey, label='Error en Y (TILT)')
plt.xlabel("Tiempo (s)")
plt.ylabel("Error (pixeles)")
plt.title("Evolución del error entre Punto 1 y Punto 2")
plt.legend()
plt.grid(True)
plt.show()

# Graficar posiciones angulares
plt.figure()
plt.plot(t, theta_x, label='Ángulo PAN (°)')
plt.plot(t, theta_y, label='Ángulo TILT (°)')
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo (°)")
plt.title("Ángulos requeridos para posicionar la cámara")
plt.legend()
plt.grid(True)
plt.show()