import numpy as np
from matplotlib import pyplot as plt
import cv2


def detectar(id, base_path, corte_sagital, filo_superior_izq, filo_superior_der, punto_centro):
    corte_sagital_copia = np.copy(corte_sagital)
    x, y = punto_centro
    x1, y1 = filo_superior_izq[0]
    x2, y2 = filo_superior_der[0]
    x_medio_sacral = (x1 + x2) // 2
    y_medio_sacral = (y1 + y2) // 2

    # Dibuja los puntos y las líneas
    cv2.circle(corte_sagital_copia, (x, y), 1, (0, 255, 0), 2)  # Punto verde en el centro de ambos femurs.
    cv2.circle(corte_sagital_copia, (x_medio_sacral, y_medio_sacral), 1, (0, 255, 0), 2)  # Punto verde en el medio sacral.

    cv2.line(corte_sagital_copia, (x, y), (x, y - 80), (0, 255, 0), 1)  # Línea verde paralela al piso (vertical).
    cv2.line(corte_sagital_copia, (x, y), (x_medio_sacral, y_medio_sacral), (0, 255, 0),1)  # Línea verde al centro sacral.



    # Calcula las diferencias de coordenadas para los vectores
    dx_oblicuo = x_medio_sacral - x
    dy_oblicuo = y_medio_sacral - y

    dx_vertical = 0  # La línea vertical tiene diferencia 0 en x
    dy_vertical = -80  # Diferencia en y

    # Calcula los ángulos de las dos líneas respecto al eje horizontal
    angulo_vertical = np.arctan2(dy_vertical, dx_vertical)  # Angulo de la recta vertical
    angulo_oblicuo = np.arctan2(dy_oblicuo, dx_oblicuo)  # Angulo de la recta oblicua

    # Calcula la diferencia de ángulos
    angulo_entre_rectas = np.abs(angulo_oblicuo - angulo_vertical)

    # Convierte el ángulo a grados
    PelvicTiltAngle = round(np.degrees(angulo_entre_rectas))

    #guarda imagen
    plt.figure(figsize=(10, 7))
    plt.imshow(corte_sagital_copia, cmap="gray", aspect='auto')
    plt.axis('off')  # Desactiva los ejes
    output_path = f"{base_path}/PelvicTilt.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    return PelvicTiltAngle,output_path
