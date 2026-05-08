import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

class MotorController:
    def __init__(self, motor_type="Nema23_SL57STH56"):
        self.motor_type = motor_type
        # Parámetros del motor Nema 23 SL57STH56-1.5A
        self.steps_per_revolution = 200  # Pasos por revolución
        self.max_speed = 1500  # pasos/segundo (aprox)
        self.current_position_steps = 0  # Posición actual en pasos
        
    def move_to_position(self, target_steps, current_time):
        """Simula el movimiento del motor a una posición específica"""
        # Limitar la velocidad máxima
        max_steps_per_cycle = int(self.max_speed * 0.1)  # Para simulación de 10Hz
        
        if abs(target_steps - self.current_position_steps) > max_steps_per_cycle:
            if target_steps > self.current_position_steps:
                self.current_position_steps += max_steps_per_cycle
            else:
                self.current_position_steps -= max_steps_per_cycle
        else:
            self.current_position_steps = target_steps
            
        return self.current_position_steps
    
    def steps_to_angle(self, steps):
        """Convierte pasos a ángulo en grados"""
        return (steps % self.steps_per_revolution) * (360.0 / self.steps_per_revolution)

class PanTiltController:
    def __init__(self):
        # Centro de la imagen (resolución típica 640x480)
        self.image_center_x = 320
        self.image_center_y = 240
        self.image_width = 640
        self.image_height = 480
        
        # Motores
        self.pan_motor = MotorController()
        self.tilt_motor = MotorController()
        
        # Control PID simplificado
        self.kp_pan = 0.8   # Ganancia proporcional PAN
        self.kp_tilt = 0.8  # Ganancia proporcional TILT
        
        # Historial para gráficas
        self.history = {
            'time': [],
            'drone_x': [],
            'drone_y': [],
            'pan_position': [],
            'tilt_position': [],
            'error_x': [],
            'error_y': []
        }
        
        self.start_time = time.time()
    
    def calculate_error(self, drone_x, drone_y):
        """Calcula el error entre la posición del dron y el centro"""
        error_x = drone_x - self.image_center_x
        error_y = drone_y - self.image_center_y
        return error_x, error_y
    
    def control_law(self, error_x, error_y):
        """Ley de control para calcular movimiento de motores"""
        # Convertir error de píxeles a pasos de motor
        pan_steps = -int(error_x * self.kp_pan)  # Negativo porque si el dron está a la derecha, movemos a la izquierda
        tilt_steps = -int(error_y * self.kp_tilt)  # Negativo porque si el dron está abajo, movemos arriba
        
        return pan_steps, tilt_steps
    
    def update_position(self, drone_x, drone_y):
        """Actualiza la posición basada en las coordenadas del dron"""
        current_time = time.time() - self.start_time
        
        # Calcular error
        error_x, error_y = self.calculate_error(drone_x, drone_y)
        
        # Calcular movimiento necesario
        pan_steps, tilt_steps = self.control_law(error_x, error_y)
        
        # Mover motores
        pan_position = self.pan_motor.move_to_position(
            self.pan_motor.current_position_steps + pan_steps, current_time)
        tilt_position = self.tilt_motor.move_to_position(
            self.tilt_motor.current_position_steps + tilt_steps, current_time)
        
        # Guardar historial
        self.history['time'].append(current_time)
        self.history['drone_x'].append(drone_x)
        self.history['drone_y'].append(drone_y)
        self.history['pan_position'].append(pan_position)
        self.history['tilt_position'].append(tilt_position)
        self.history['error_x'].append(error_x)
        self.history['error_y'].append(error_y)
        
        return pan_position, tilt_position, error_x, error_y
    
    def plot_results(self):
        """Genera gráficas del comportamiento del control"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfica 1: Posición del dron vs tiempo
        ax1.plot(self.history['time'], self.history['drone_x'], 'r-', label='Dron X', linewidth=2)
        ax1.plot(self.history['time'], self.history['drone_y'], 'b-', label='Dron Y', linewidth=2)
        ax1.axhline(y=self.image_center_x, color='r', linestyle='--', alpha=0.7, label='Centro X')
        ax1.axhline(y=self.image_center_y, color='b', linestyle='--', alpha=0.7, label='Centro Y')
        ax1.set_title('Posición del Dron en la Imagen')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Posición (píxeles)')
        ax1.legend()
        ax1.grid(True)
        
        # Gráfica 2: Error de seguimiento
        ax2.plot(self.history['time'], self.history['error_x'], 'r-', label='Error X', linewidth=2)
        ax2.plot(self.history['time'], self.history['error_y'], 'b-', label='Error Y', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Error de Seguimiento')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Error (píxeles)')
        ax2.legend()
        ax2.grid(True)
        
        # Gráfica 3: Posición de motores PAN
        ax3.plot(self.history['time'], self.history['pan_position'], 'g-', linewidth=2)
        ax3.set_title('Posición del Motor PAN')
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Pasos del Motor')
        ax3.grid(True)
        
        # Gráfica 4: Posición de motores TILT
        ax4.plot(self.history['time'], self.history['tilt_position'], 'purple', linewidth=2)
        ax4.set_title('Posición del Motor TILT')
        ax4.set_xlabel('Tiempo (s)')
        ax4.set_ylabel('Pasos del Motor')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def print_status(self, iteration, drone_x, drone_y, pan_pos, tilt_pos, error_x, error_y):
        """Imprime el estado actual del sistema"""
        print(f"Iteración {iteration}:")
        print(f"  Dron: ({drone_x}, {drone_y})")
        print(f"  Error: X={error_x:.1f}, Y={error_y:.1f}")
        print(f"  Motor PAN: {pan_pos} pasos")
        print(f"  Motor TILT: {tilt_pos} pasos")
        print(f"  Ángulo PAN: {self.pan_motor.steps_to_angle(pan_pos):.1f}°")
        print(f"  Ángulo TILT: {self.tilt_motor.steps_to_angle(tilt_pos):.1f}°")
        print("-" * 50)

# SIMULACIÓN COMPLETA
def simulate_tracking():
    print("=== SIMULACIÓN SEGUIMIENTO DE DRON CON MOTORES NEMA 23 ===")
    print("Configuración:")
    print("- Motores: Nema 23 SL57STH56-1.5A")
    print("- Control: Proporcional (P)")
    print("- Resolución imagen: 640x480")
    print("- Centro objetivo: (320, 240)")
    print("=" * 60)
    
    # Crear controlador
    controller = PanTiltController()
    
    # Coordenadas del dron (simulando movimiento)
    drone_positions = [
        (287, 245),   # Posición inicial
        (324, 231),   # Movimiento diagonal
        (350, 200),   # Hacia esquina superior derecha
        (300, 280),   # Hacia abajo-izquierda
        (320, 240),   # Cerca del centro
        (280, 220),   # Esquina superior izquierda
        (360, 260),   # Esquina inferior derecha
        (320, 240)    # Regreso al centro
    ]
    
    print("Iniciando simulación de seguimiento...")
    
    # Ejecutar simulación
    for i, (drone_x, drone_y) in enumerate(drone_positions):
        # Simular procesamiento en tiempo real
        time.sleep(0.5)
        
        # Actualizar control
        pan_pos, tilt_pos, error_x, error_y = controller.update_position(drone_x, drone_y)
        
        # Mostrar estado
        controller.print_status(i + 1, drone_x, drone_y, pan_pos, tilt_pos, error_x, error_y)
    
    # Generar gráficas
    print("Generando gráficas de resultados...")
    controller.plot_results()
    
    # Resumen final
    print("\n=== RESUMEN FINAL ===")
    final_error_x, final_error_y = controller.calculate_error(
        drone_positions[-1][0], drone_positions[-1][1])
    print(f"Error final: X={final_error_x}, Y={final_error_y}")
    print(f"Posición final PAN: {controller.pan_motor.current_position_steps} pasos")
    print(f"Posición final TILT: {controller.tilt_motor.current_position_steps} pasos")
    
    if abs(final_error_x) < 10 and abs(final_error_y) < 10:
        print("✅ SEGUIMIENTO EXITOSO - Dron en posición central")
    else:
        print("⚠️  SEGUIMIENTO PARCIAL - Ajustar ganancias del controlador")

# Ejecutar simulación
if __name__ == "__main__":
    simulate_tracking()