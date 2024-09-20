import numpy as np
import cv2




#Detecta contornos circulares para acercarnos a la cabeza del femur
def detectar(corte_segmentado,min_value):
    #Flag de que hallo contorno
    existe_contorno_circular=False;

    # Detectar contornos
    contours, _ = cv2.findContours(corte_segmentado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

    #Cuando se comienza a detectar el reborde del cuello frena para no pasarse.
    if len(contours) > 1:
            # Ordenar los contornos por área
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            segundo_contorno_mas_grande = sorted_contours[1]  # Segundo contorno más grande
            if(cv2.contourArea(segundo_contorno_mas_grande) > 60):
                 return False,0
    else:
            segundo_contorno_mas_grande = None  # No hay segundo contorno  

    if contours:
        # Encontrar el contorno con el área más grande
        contorno_mas_grande = max(contours, key=cv2.contourArea)

        # Calcular el área y el perímetro del contorno más grande
        area = cv2.contourArea(contorno_mas_grande)
        perimetro = cv2.arcLength(contorno_mas_grande, True)
        
        #Evitar división por cero
        if perimetro > 0:
            # Calcular la circularidad
            circularidad = (4 * np.pi * area) / (perimetro ** 2)
           
            # Verificar si el contorno es suficientemente circular
            if min_value <= circularidad <= 1:  # Puedes ajustar los umbrales según sea necesario
                existe_contorno_circular = True
            else:
                existe_contorno_circular = False
                
                 # Encontrar el segundo contorno más grande si hay más de uno
     
        return existe_contorno_circular,contorno_mas_grande
    else:
        return existe_contorno_circular,0
