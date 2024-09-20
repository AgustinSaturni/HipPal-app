import numpy as np
import cv2


#Detecta circulo para identificar donde esta la cabeza del femur
def detectar_2(corte_segmentado, corte_segmentado_gris,contorno_mas_grande,tomografia_original):
    flag_cabeza_femur = False

    # Crear una máscara en blanco del mismo tamaño que la imagen
    mask = np.zeros_like(corte_segmentado_gris)

    # Dibujar el contorno más grande en la máscara
    cv2.drawContours(mask, [contorno_mas_grande], -1, (255), thickness=cv2.FILLED)

    # Aplicar la máscara a la imagen rotada
    result_image = cv2.bitwise_and(corte_segmentado, corte_segmentado, mask=mask)

    # Aplicar HoughCircles para detectar círculos
    gray_result = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)  # Convertir a escala de grises

    imagen_hueso_desenfocada = cv2.GaussianBlur(gray_result, (5,5),0,0)

    # Parámetros de HoughCircles (ajustar según sea necesario)
    circles = cv2.HoughCircles(
        imagen_hueso_desenfocada,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=50,
        param2=10,
        minRadius=12,
        maxRadius=100
    )

    if circles is not None:
        
        # Convertir las coordenadas de los círculos a enteros
        circles = np.round(circles[0, :]).astype("int")

        # Dibujar los círculos detectados en la imagen
        for (x, y, r) in circles:
            flag_cabeza_femur = True
            circulo=(x,y,r)
            #print(r)

        return flag_cabeza_femur,circulo
    else:
        return flag_cabeza_femur,None





# Detecta círculo para identificar dónde está la cabeza del fémur
def detectar(corte_segmentado, corte_segmentado_gris, contorno_mas_grande,tomografia_original):
    flag_cabeza_femur = False
    
    # Crear una máscara en blanco del mismo tamaño que la imagen
    mask = np.zeros_like(corte_segmentado_gris)

    # Dibujar el contorno más grande en la máscara
    cv2.drawContours(mask, [contorno_mas_grande], -1, (255), thickness=cv2.FILLED)

    # Aplicar la máscara a la imagen
    result_image = cv2.bitwise_and(corte_segmentado, corte_segmentado, mask=mask)

    # Calcular el centroide del contorno más grande
    M = cv2.moments(contorno_mas_grande)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    
    # Inicializar las distancias mínima y máxima
    min_dist = float('inf')
    max_dist = 0

    # Encontrar las distancias mínima y máxima del centroide a los puntos del contorno
    for point in contorno_mas_grande:
        dist = np.sqrt((cX - point[0][0])**2 + (cY - point[0][1])**2)
        if dist > max_dist:
            max_dist = dist
        if dist < min_dist:
            min_dist = dist
    
    # Calcular el radio como el promedio de la distancia máxima y mínima
    radius = int((max_dist+min_dist)/2)

    if  radius > 10:
        flag_cabeza_femur = True
        circulo = (cX, cY, radius)

        return flag_cabeza_femur, circulo
    else:
        return flag_cabeza_femur, None
    



