import numpy as np
import cv2
import math


#Detecto PASA
def detectar(imagen_HU, x1_centroide, y1_centroide, x2_centroide_opuesto, y2_centroide_opuesto, radio, lado='derecho'):
    punto_blanco_encontrado = None  # Para almacenar el punto donde se encuentra el píxel blanco
    if lado == 'izquierdo':
        angulo_inicial = -180  # Ángulo inicial en grados para el lado derecho
        angulo_incremento = 1  # Decremento del ángulo para el lado derecho
    else:
        angulo_inicial = 180  # Ángulo inicial en grados para el lado izquierdo
        angulo_incremento = -1  # Incremento del ángulo para el lado izquierdo

    while abs(angulo_inicial) >= 0:
        # Calcular el ángulo de la línea original
        angulo_original = math.atan2(y2_centroide_opuesto - y1_centroide, x2_centroide_opuesto - x1_centroide)

        # Convertir el ángulo a radianes
        angulo_nuevo = math.radians(angulo_inicial)

        # Calcular el nuevo ángulo sumando el ángulo dado
        angulo_final = angulo_original + angulo_nuevo

        # Definir la longitud de la nueva línea (puedes ajustar esta longitud)
        longitud_linea = 100

        # Calcular las coordenadas del punto final de la nueva línea
        x_final = int(x1_centroide + longitud_linea * math.cos(angulo_final))
        y_final = int(y1_centroide + longitud_linea * math.sin(angulo_final))

        # Asegurarse de que la imagen es en escala de grises
        if len(imagen_HU.shape) > 2:
            imagen2 = cv2.cvtColor(imagen_HU, cv2.COLOR_BGR2GRAY)
        else:
            imagen2 = imagen_HU

        # Detección de píxel blanco
        hay_pixel_blanco = False
        for t in np.linspace((radio) / longitud_linea, 1, max(abs(x_final - x1_centroide), abs(y_final - y1_centroide))):
            x = int(x1_centroide + t * (x_final - x1_centroide))
            y = int(y1_centroide + t * (y_final - y1_centroide))
            if imagen2[y, x] != 0:
                hay_pixel_blanco = True
                punto_blanco_encontrado = (x, y)
                break
        
        #print(f"Ángulo: {angulo_inicial}, Hay pixel blanco: {hay_pixel_blanco}")

        if hay_pixel_blanco:
            break
        
        # Ajustar el ángulo
        angulo_inicial += angulo_incremento
    
    return x_final, y_final, angulo_inicial,punto_blanco_encontrado
