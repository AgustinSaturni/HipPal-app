import numpy as np
from matplotlib import pyplot as plt
import cv2


def detectar(id,base_path,corte_sagital, filo_superior_izq, filo_superior_der):
    x1, y1 = filo_superior_izq[0]
    x2, y2 = filo_superior_der[0]
    
    corte_sagital_copia = np.copy(corte_sagital)
    cv2.circle(corte_sagital_copia, (x1, y1), 2, (0, 255, 0), -1)  # Dibuja un punto rojo en la posición del vértice con menor x
    cv2.circle(corte_sagital_copia, (x2, y2), 2, (0, 255, 0), -1)  # Dibuja un punto azul en la posición del segundo punto

    # Dibujar una línea entre los dos puntos hallados
    cv2.line(corte_sagital_copia, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Línea verde que une ambos puntos
    # Dibujar la línea vertical desde el punto derecho
    cv2.line(corte_sagital_copia, (x2, y2), (x1-20,y2) , (0, 255, 0), 1) #recta  verde paralela al piso

    # Calcular el ángulo del Sacral Slope
    # Calcular la diferencia en las coordenadas
    delta_y = y2 - y1
    delta_x = x2 - x1
    
    # Calcular el ángulo en radianes y convertir a grados
    SacralSlopeAngle = round(np.degrees(np.arctan2(delta_y, delta_x)))

    plt.figure(figsize=(10, 7))
    plt.imshow(corte_sagital_copia, cmap="gray", aspect='auto')
    plt.axis('off')  # Desactiva los ejes
    output_path = f"{base_path}/SacralSlope.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    return abs(SacralSlopeAngle),output_path
