import numpy as np
from PreprocesamientoDeCorte import  preprocesarAxialOriginal, preprocesarAxialSegmentado
from matplotlib import pyplot as plt
import cv2
from SectorAcetabular.Angulos import aasa, pasa


def graficar(id, base_path, corte_original_acetabulo_derecho, corte_original_acetabulo_izquierdo,
             filo_sup, filo_inf, filo_inf_opuesto,lado):
    
    orientacion = 1
    if(lado == "Izquierdo"):
        orientacion=1
    else:
        orientacion=-1
    # Cortar cada imagen por la mitad
    mitad_derecha_izq = corte_original_acetabulo_izquierdo[:, corte_original_acetabulo_izquierdo.shape[1] // 2:]
    mitad_izquierda_der = corte_original_acetabulo_derecho[:, :corte_original_acetabulo_derecho.shape[1] // 2]

    # Combinar las mitades
    composicion = np.hstack((mitad_izquierda_der, mitad_derecha_izq))

   
    
    # Calcular el vector de la línea original
    dx = filo_inf_opuesto[0] - filo_inf[0]
    dy = filo_inf_opuesto[1] - filo_inf[1]
    
    # Obtener el vector perpendicular (intercambiar dx y dy y negar uno)
    perpendicular_vector = (-dy, dx)

    # Definir el punto final de la línea perpendicular (escala el vector para darle una longitud)
    escala = 0.3  # Puedes ajustar la longitud de la línea
    #Achicar ESTO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    punto_final_perpendicular = (int(filo_inf[0] + perpendicular_vector[0] * escala ),
                                 int(filo_inf[1] + perpendicular_vector[1] * escala * orientacion ))

    # Dibujar la primera línea entre filo_inf_izq y filo_inf_der
    cv2.circle(composicion, filo_inf, 1, (0, 255, 0), 2)  # Punto verde en filo_inf
    cv2.circle(composicion, filo_sup, 1, (0, 255, 0), 2)  # Punto verde en filo_sup
    cv2.circle(composicion, filo_inf_opuesto, 1, (0, 255, 0), 2)  # Punto verde en filo_inf_opuesto

    #Plomada
    cv2.line(composicion, filo_inf, filo_inf_opuesto, (0, 255, 0), 1)  # Línea verde entre los puntos

    # Dibujar la línea perpendicular
    cv2.line(composicion, filo_inf, punto_final_perpendicular, (0, 255, 0), 1)  # Línea verde perpendicular

    #De filo inf a sup
    cv2.line(composicion, filo_inf, filo_sup, (0, 255, 0), 1)  # Línea verde entre los filos acetabulares

    # Calcular los vectores
    vector1 = np.array([filo_inf_opuesto[0] - filo_inf[0], filo_inf_opuesto[1] - filo_inf[1]])
    vector2 = np.array([filo_sup[0] - filo_inf[0], filo_sup[1] - filo_inf[1]])

    # Calcular el producto punto de los vectores
    producto_punto = np.dot(vector1, vector2)

    # Calcular las magnitudes (longitud) de los vectores
    magnitud_vector1 = np.linalg.norm(vector1)
    magnitud_vector2 = np.linalg.norm(vector2)

    # Calcular el coseno del ángulo
    cos_theta = producto_punto / (magnitud_vector1 * magnitud_vector2)

    # Asegurarse de que el valor de cos_theta esté entre -1 y 1
    cos_theta = np.clip(cos_theta, -1, 1)

    # Calcular el ángulo en radianes y luego convertir a grados
    angulo_radianes = np.arccos(cos_theta)
    angulo_grados = np.degrees(angulo_radianes)


    # Mostrar y guardar la imagen
    plt.figure(figsize=(18, 10))
    plt.subplot(1, 1, 1)
    plt.imshow(composicion, cmap='gray')
    plt.axis('off')
    output_path = f"{base_path}/VersionAcetabular{lado}.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    return round(angulo_grados),output_path


def detectar(id,base_path,cabezas_femur_axiales,tomografia_original,tomografia_segmentada):




    numero_corte_izquierdo=cabezas_femur_axiales["ecuatorial"]["izquierdo"]["numero_corte"]
    numero_corte_derecho=cabezas_femur_axiales["ecuatorial"]["derecho"]["numero_corte"]
    cabeza_der=cabezas_femur_axiales["ecuatorial"]["derecho"]["coordenadas"]
    cabeza_izq=cabezas_femur_axiales["ecuatorial"]["izquierdo"]["coordenadas"]



    #Segmentado
    corte_segmentado_acetabulo_izquierdo = tomografia_segmentada[:,:,numero_corte_izquierdo,5]
    corte_segmentado_acetabulo_izquierdo= preprocesarAxialSegmentado.procesarCorte(corte_segmentado_acetabulo_izquierdo) 

    corte_segmentado_acetabulo_derecho = tomografia_segmentada[:,:,numero_corte_derecho,4]
    corte_segmentado_acetabulo_derecho= preprocesarAxialSegmentado.procesarCorte(corte_segmentado_acetabulo_derecho)   

    x_der,y_der,r_der = cabeza_der
    x_izq,y_izq,r_izq =cabeza_izq

    #Detecto PASA   
    _, _, _,filo_inf_der  = pasa.detectar(corte_segmentado_acetabulo_derecho,x_der,y_der,x_izq,y_izq,r_der,"derecho")
    _, _, _,filo_inf_izq  = pasa.detectar(corte_segmentado_acetabulo_izquierdo,x_izq,y_izq,x_der,y_der,r_izq,"izquierdo")

    #Detecto AASA
    _, _, _,filo_sup_der  = aasa.detectar(corte_segmentado_acetabulo_derecho,x_der,y_der,x_izq,y_izq,r_der,"derecho")
    _, _, _,filo_sup_izq  = aasa.detectar(corte_segmentado_acetabulo_izquierdo,x_izq,y_izq,x_der,y_der,r_izq,"izquierdo")

    #Original
            #Original
    corte_original_acetabulo_izquierdo = tomografia_original[:,:,numero_corte_izquierdo]
    corte_original_acetabulo_izquierdo= preprocesarAxialOriginal.procesarCorte(corte_original_acetabulo_izquierdo) 
    
    corte_original_acetabulo_derecho = tomografia_original[:,:,numero_corte_derecho]
    corte_original_acetabulo_derecho= preprocesarAxialOriginal.procesarCorte(corte_original_acetabulo_derecho) 


    angulo_VA_izq,output_path_izq = graficar(id, base_path,corte_original_acetabulo_derecho,corte_original_acetabulo_izquierdo,filo_sup_izq,filo_inf_izq,filo_inf_der,"Izquierdo")
    angulo_VA_der,output_path_der= graficar(id, base_path,corte_original_acetabulo_derecho,corte_original_acetabulo_izquierdo,filo_sup_der,filo_inf_der,filo_inf_izq,"Derecho")

    angulo_VA_izq=[{
                "name": "Izquierdo",
                "path":output_path_izq,
                "izquierdo":[{
                    "name":"va_izq",
                    "value":angulo_VA_izq
                }]
        }]
    

    angulo_VA_der=[{
                "name": "Derecho",
                "path":output_path_der,
                "izquierdo":[{
                    "name":"va_der",
                    "value":angulo_VA_der
                }]
        }]

    return angulo_VA_izq,angulo_VA_der