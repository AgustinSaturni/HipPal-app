
import numpy as np
from CentroBordeLateral.Utils import esCorteEcuatorialCoronal
from PreprocesamientoDeCorte import preprocesarCoronal, preprocesarCoronalSegmentado, preprocesarSagital, preprocesarSagitalSegmentado
from matplotlib import pyplot as plt
import cv2
from SectorAcetabular.Angulos import aasa



def calcularSacralSlope(corte_sagital, filo_superior_izq, filo_superior_der):
    x1, y1 = filo_superior_izq[0]
    x2, y2 = filo_superior_der[0]
    
    # Dibujar una línea entre los dos puntos hallados
    cv2.line(corte_sagital, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Línea amarilla que une ambos puntos
    # Dibujar la línea vertical desde el punto derecho
    cv2.line(corte_sagital, (x2, y2), (x1,y2) , (255, 255, 0), 1) #recta paralela al piso

    # Calcular el ángulo del Sacral Slope
    # Calcular la diferencia en las coordenadas
    delta_y = y2 - y1
    delta_x = x2 - x1
    
    # Calcular el ángulo en radianes y convertir a grados
    SacralSlopeAngle = np.degrees(np.arctan2(delta_y, delta_x))  # Ángulo en grados

    return corte_sagital, abs(SacralSlopeAngle)






def detectar(id, base_path, cabezas_femur_axiales, tomografia_original, tomografia_segmentada):
    _, _, eje_ordenadas = tomografia_original.shape

    x_izq, y_izq, _ = cabezas_femur_axiales["ecuatorial"]["izquierdo"]["coordenadas"]
    numero_corte_izq = cabezas_femur_axiales["ecuatorial"]["izquierdo"]["numero_corte"]
    x_der, y_der, _ = cabezas_femur_axiales["ecuatorial"]["derecho"]["coordenadas"]
    numero_corte_der = cabezas_femur_axiales["ecuatorial"]["derecho"]["numero_corte"]

    # Cálculo del punto medio de la recta entre los dos puntos verdes
    z_medio = (x_izq + x_der) // 2
    x_medio = (y_izq + y_der) // 2
    y_medio = eje_ordenadas - ((numero_corte_izq + numero_corte_der) // 2)

    corte_ecuatorial_sagital_izq = tomografia_original[z_medio, :, :]
    corte_ecuatorial_sagital_izq = preprocesarSagital.procesarCorte(corte_ecuatorial_sagital_izq)

    sagital_slice = tomografia_segmentada[z_medio, :, :, 3]
    sagital_slice = preprocesarSagitalSegmentado.procesarCorte(sagital_slice)

    sagital_slice_gray = cv2.cvtColor(sagital_slice, cv2.COLOR_BGR2GRAY)
    # Detección de contornos
    contours, _ = cv2.findContours(sagital_slice_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar el contorno con mayor área
        sacro = max(contours, key=cv2.contourArea)

        # Dibujar solo el contorno más grande en verde
        cv2.drawContours(sagital_slice, [sacro], -1, (0, 255, 0), 2)

        # Aproximación de contorno para encontrar los vértices del contorno más grande
        epsilon = 0.02 * cv2.arcLength(sacro, True)
        approx = cv2.approxPolyDP(sacro, epsilon, True)

        # Buscar el punto con la menor coordenada x
        filo_superior_izq = min(approx, key=lambda point: point[0][0])

        # Dibujar solo ese punto en rojo
        x, y = filo_superior_izq[0]
        cv2.circle(corte_ecuatorial_sagital_izq, (x, y), 2, (255, 0, 0), -1)  # Dibuja un punto rojo en la posición del vértice con menor x

        # Encontrar el siguiente punto con la menor distancia en x al punto de menor x
        remaining_points = [point for point in approx if not (point == filo_superior_izq).all()]
        filo_superior_der = min(remaining_points, key=lambda point: abs(point[0][0] - filo_superior_izq[0][0]))

        # Dibujar este segundo punto en azul
        x2, y2 = filo_superior_der[0]
        cv2.circle(corte_ecuatorial_sagital_izq, (x2, y2), 2, (255, 0, 0), -1)  # Dibuja un punto azul en la posición del segundo punto

        corte_ecuatorial_sagital_izq,SacralSlopeAngle = calcularSacralSlope(corte_ecuatorial_sagital_izq,filo_superior_izq,filo_superior_der)
        print(SacralSlopeAngle)


        # Dibujar una línea entre los dos puntos hallados
        #cv2.line(corte_ecuatorial_sagital_izq, (x, y), (x2, y2), (255, 255, 0), 1)  # Línea amarilla que une ambos puntos

        # Dibuj angulo pendiente sacro
        #cv2.line(corte_ecuatorial_sagital_izq, (x2, y2), (x,y2) , (255, 255, 0), 1)  # Línea amarilla que une ambos puntos

        # Calcular el punto medio
       # x_medio_sacral = (x + x2) // 2
        #y_medio_sacral = (y + y2) // 2

        # Dibuj angulo pendiente sacro
        #cv2.line(corte_ecuatorial_sagital_izq, (x2, y2), (x,y2) , (255, 255, 0), 1)  # Línea amarilla que une ambos puntos

        cv2.circle(corte_ecuatorial_sagital_izq, (x_medio, y_medio), 1, (0, 255, 0), 2)  # Punto verde en (x_opuesto, y_opuesto)

        # Dibuj angulo pendiente sacro
        #cv2.line(corte_ecuatorial_sagital_izq, (x_medio, y_medio), (x_medio_sacral,y_medio_sacral) , (0, 255, 0), 1)  # Línea amarilla que une ambos puntos


    # Mostrar las imágenes procesadas
    plt.figure(figsize=(10, 7))
    plt.imshow(corte_ecuatorial_sagital_izq, cmap="gray", aspect='auto')
    plt.axis('off')  # Desactiva los ejes
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.imshow(sagital_slice, cmap="gray", aspect='auto')
    plt.axis('off')  # Desactiva los ejes
    plt.show()

    return True
