import pandas as pd
import matplotlib.pyplot as plt

# Cargar archivos CSV
y8s = pd.read_csv("results_yolov8s4000.csv")
y8 = pd.read_csv("results_yolov8s.csv")
y9 = pd.read_csv("results_yolov9t.csv")
y10 = pd.read_csv("results_yolov10n.csv")
y11 = pd.read_csv("results_yolov11n.csv")



# Crear una lista de métricas a comparar
metricas = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 
            'metrics/mAP50-95(B)', 'train/box_loss', 'train/cls_loss']

# Crear subplots
plt.figure(figsize=(12, 10))

for i, m in enumerate(metricas):
    plt.subplot(3, 2, i+1)
    plt.plot(y8s[m], label='YOLOv8s-4k', color='brown')
    plt.plot(y8[m], label='YOLOv8s', color='royalblue')
    plt.plot(y9[m], label='YOLOv9t', color='deepskyblue')
    plt.plot(y10[m], label='YOLOv10n', color='pink')
    plt.plot(y11[m], label='YOLOv11n', color='orange')
    

    plt.title(m)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
