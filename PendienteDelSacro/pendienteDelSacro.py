from PendienteDelSacro.Utils import calcularPelvicIncidence, calcularPelvicTilt, calcularSacralSlope
from PreprocesamientoDeCorte import  preprocesarSagital, preprocesarSagitalSegmentado
import cv2



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

    sagital_slice_segmentada = tomografia_segmentada[z_medio, :, :, 3]
    sagital_slice_segmentada = preprocesarSagitalSegmentado.procesarCorte(sagital_slice_segmentada)

    sagital_slice_segmentada_gris = cv2.cvtColor(sagital_slice_segmentada, cv2.COLOR_BGR2GRAY)
    # Detección de contornos
    contours, _ = cv2.findContours(sagital_slice_segmentada_gris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Encontrar el contorno con mayor área
        sacro = max(contours, key=cv2.contourArea)

        # Dibujar solo el contorno más grande en verde
        cv2.drawContours(sagital_slice_segmentada, [sacro], -1, (0, 255, 0), 2)

        # Aproximación de contorno para encontrar los vértices del contorno más grande
        epsilon = 0.02 * cv2.arcLength(sacro, True)
        approx = cv2.approxPolyDP(sacro, epsilon, True)

        # Buscar el punto con la menor coordenada x
        filo_superior_izq = min(approx, key=lambda point: point[0][0])

        # Encontrar el siguiente punto con la menor distancia en x al punto de menor x
        remaining_points = [point for point in approx if not (point == filo_superior_izq).all()]
        filo_superior_der = min(remaining_points, key=lambda point: abs(point[0][0] - filo_superior_izq[0][0]))

        SacralSlopeAngle,output_path_ss = calcularSacralSlope.detectar(id,base_path,corte_ecuatorial_sagital_izq,filo_superior_izq,filo_superior_der)

        PelvicTiltAngle,output_path_pt=calcularPelvicTilt.detectar(id,base_path,corte_ecuatorial_sagital_izq, filo_superior_izq, filo_superior_der,(x_medio,y_medio))

        PelvicIncidenceAngle,output_path_pi=calcularPelvicIncidence.detectar(id,base_path,corte_ecuatorial_sagital_izq, filo_superior_izq, filo_superior_der,(x_medio,y_medio))


    angulo_SS=[{
                "name": "Sacral_Slope",
                "path":output_path_ss,
                "izquierdo":[{
                    "name":"sacralSlope",
                    "value":SacralSlopeAngle
                }]
        }]
 
 
    angulo_PT=[{
                "name": "Pelvic_Tilt",
                "path":output_path_pt,
                "izquierdo":[{
                    "name":"pelvicTilt",
                    "value":PelvicTiltAngle
                }]
        }]
 
     
    angulo_PI=[{
                "name": "Pelvic_Incidence",
                "path":output_path_pi,
                "izquierdo":[{
                    "name":"pelvicIncidence",
                    "value":PelvicIncidenceAngle
                }]
        }]
    


    return angulo_SS,angulo_PT,angulo_PI
