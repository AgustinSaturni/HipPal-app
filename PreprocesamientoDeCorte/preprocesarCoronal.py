import numpy as np
import cv2




#Aplica rotaciones y normalizaciones a los cortes ESTO ES PARA CUALQUIER MEDICION EN EL AXIAL
def procesarCorte(corte):
    
    corte = np.rot90(corte,k=1)

    # Normalizar la imagen CT para asegurar que esté en el rango [0, 255]
    corte_normalizado = cv2.normalize(corte, None, 0, 255, cv2.NORM_MINMAX)
    corte_normalizado = np.uint8(corte_normalizado)

    # Crear una imagen en color a partir de la imagen en escala de grises
    corte_color = cv2.cvtColor(corte_normalizado, cv2.COLOR_GRAY2RGB)

    # Rotar la imagen 90° a la derecha
    #corte_rotado = np.rot90(corte_color)  # k=-1 rota 90° a la derecha
    #corte_espejado = cv2.flip(corte_color, 1)

    return corte_color