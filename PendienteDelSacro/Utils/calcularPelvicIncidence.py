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
    cv2.circle(corte_sagital_copia, (x, y), 1, (0, 255, 0), 2)  # Punto verde en el centro de ambos femurs.
    cv2.circle(corte_sagital_copia, (x_medio_sacral, y_medio_sacral), 1, (0, 255, 0), 2)  # Punto verde en el medio sacral.



    x_medio_sacral = (x1 + x2) // 2
    y_medio_sacral = (y1 + y2) // 2

    # Calcular la pendiente de la línea original
    if x2 - x1 != 0:  # Evita la división por cero
        pendiente_original = (y2 - y1) / (x2 - x1)
        pendiente_perpendicular = -1 / pendiente_original  # Pendiente de la línea perpendicular
    else:
        pendiente_perpendicular = 0  # La perpendicular a una línea vertical es horizontal

    # Definir la longitud de la línea perpendicular
    longitud_perpendicular = 100

    # Calcular los puntos de la línea perpendicular desde el punto medio
    dx = longitud_perpendicular / np.sqrt(1 + pendiente_perpendicular**2)
    dy = pendiente_perpendicular * dx

    # Puntos extremos de la línea perpendicular
    x_perpendicular_1 = int(x_medio_sacral + dx)
    y_perpendicular_1 = int(y_medio_sacral + dy)


    cv2.line(corte_sagital_copia,(x_medio_sacral, y_medio_sacral), (x, y), (0, 255, 0),1)  # Línea verde al centro sacral.
    cv2.line(corte_sagital_copia, (x_medio_sacral, y_medio_sacral), (x_perpendicular_1, y_perpendicular_1), (0, 255, 0), 1)


    # Definir los vectores
    vector_1 = (x - x_medio_sacral, y - y_medio_sacral)
    vector_2 = (x_perpendicular_1 - x_medio_sacral, y_perpendicular_1 - y_medio_sacral)

    # Producto escalar y magnitudes
    dot_product = np.dot(vector_1, vector_2)
    magnitude_1 = np.sqrt(vector_1[0]**2 + vector_1[1]**2)
    magnitude_2 = np.sqrt(vector_2[0]**2 + vector_2[1]**2)

    # Calcular el ángulo entre los dos vectores utilizando arctan2
    cos_theta = dot_product / (magnitude_1 * magnitude_2)
    theta_rad = np.arccos(cos_theta)  # Ángulo en radianes

    # Convertir a grados
    PelvicIncidenceAngle = round(np.degrees(theta_rad))

    #guarda imagen
    plt.figure(figsize=(10, 7))
    plt.imshow(corte_sagital_copia, cmap="gray", aspect='auto')
    plt.axis('off')  # Desactiva los ejes
    output_path = f"{base_path}/PelvicIncidence.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    return PelvicIncidenceAngle,output_path
