import numpy as np
from CentroBordeLateral.Utils import esCorteEcuatorialCoronal
from PreprocesamientoDeCorte import preprocesarCoronal, preprocesarCoronalSegmentado
from matplotlib import pyplot as plt
import cv2
from SectorAcetabular.Angulos import aasa


def graficar(coronal_slice, x, y, x_opuesto, y_opuesto, x_final, y_final, lado):

    # Dibujar línea verde entre (x, y) y (x_opuesto, y_opuesto)
    cv2.line(coronal_slice, (x, y), (x_opuesto, y_opuesto), (0, 255, 0), 2)
    cv2.circle(coronal_slice, (x, y), 1, (0, 255, 0), 2)  # Punto verde en (x, y)
    cv2.circle(coronal_slice, (x_opuesto, y_opuesto), 1, (0, 255, 0), 2)  # Punto verde en (x_opuesto, y_opuesto)

    # Dibujar línea verde entre (x, y) y (x_final, y_final)
    cv2.line(coronal_slice, (x, y), (x_final, y_final), (0, 255, 0), 1)

    # Calcular dirección de la línea perpendicular
    dx = x_opuesto - x
    dy = y_opuesto - y

    # La dirección perpendicular
    perp_dx = -dy
    perp_dy = dx

    # Normalizar la dirección perpendicular
    length = np.sqrt(perp_dx**2 + perp_dy**2)
    perp_dx /= length
    perp_dy /= length

    # Si el lado es "derecho", invertir la dirección de la línea perpendicular
    if lado == "derecho":
        perp_dx = -perp_dx
        perp_dy = -perp_dy

    # Extender la línea perpendicular
    length_perpendicular = 100  # Longitud de la línea perpendicular
    start_point = (x, y)
    end_point = (int(start_point[0] + perp_dx * length_perpendicular), 
                 int(start_point[1] + perp_dy * length_perpendicular))

    # Dibujar la línea perpendicular en verde
    cv2.line(coronal_slice, start_point, end_point, (0, 255, 0), 1)

    return coronal_slice


def detectar(cabezas_femur_axiales,tomografia_original,tomografia_segmentada):
    tomografia_segmentada_invertida = tomografia_segmentada[:, ::-1, :, :]


    # Coordenadas del centro del círculo en la vista axial
    coordenadas_izq = cabezas_femur_axiales["ecuatorial"]["izquierdo"]["coordenadas"]
    numero_corte_izq=cabezas_femur_axiales["ecuatorial"]["izquierdo"]["numero_corte"]
    x_izq,y_izq,z_izq=esCorteEcuatorialCoronal.obtenerCentroide(tomografia_original,coordenadas_izq,numero_corte_izq)
    coronal_slice_izq_segmentado=tomografia_segmentada_invertida[:,z_izq,:,5]
    coronal_slice_izq_segmentado= preprocesarCoronalSegmentado.procesarCorte(coronal_slice_izq_segmentado)
    print(coordenadas_izq)
    #Forzado a mano - BORRAR
    #z_izq=z_izq+8
    coronal_slice_izq=tomografia_original[:,z_izq,:]
    coronal_slice_izq= preprocesarCoronal.procesarCorte(coronal_slice_izq)
    
    coordenadas_der = cabezas_femur_axiales["ecuatorial"]["derecho"]["coordenadas"]
    numero_corte_der=cabezas_femur_axiales["ecuatorial"]["derecho"]["numero_corte"]
    x_der,y_der,z_der=esCorteEcuatorialCoronal.obtenerCentroide(tomografia_original,coordenadas_der,numero_corte_der)
    #Forzado a mano - BORRAR
    print(coordenadas_der)
    #z_der=z_der+12
    coronal_slice_der_segmentado=tomografia_segmentada_invertida[:,z_der,:,4]
    coronal_slice_der_segmentado= preprocesarCoronalSegmentado.procesarCorte(coronal_slice_der_segmentado)
    coronal_slice_der=tomografia_original[:,z_der,:]
    coronal_slice_der= preprocesarCoronal.procesarCorte(coronal_slice_der)
   
    #Forzado a mano - BORRAR
    #y_izq = y_izq -4
    #y_der = y_der - 10

    
    x_final_izq, y_final_izq, angulo_CBL_izq=aasa.detectar(coronal_slice_izq_segmentado,x_izq,y_izq,x_der,y_der,1,"izquierdo")
    x_final_der, y_final_der, angulo_CBL_der=aasa.detectar(coronal_slice_der_segmentado,x_der,y_der,x_izq,y_izq,1,"derecho")



    corte_CBL_izquierdo=graficar(coronal_slice_izq,x_izq,y_izq,x_der,y_der,x_final_izq,y_final_izq,"izquierdo")
    corte_CBL_derecho= graficar(coronal_slice_der,x_der,y_der,x_izq,y_izq,x_final_der,y_final_der,"derecho") 


        # Cortar cada imagen por la mitad
    mitad_derecha_izq = corte_CBL_izquierdo[:, corte_CBL_izquierdo.shape[1] // 2:]
    mitad_izquierda_der = corte_CBL_derecho[:, :corte_CBL_derecho.shape[1] // 2]


    # Combinar las mitades
    composicion = np.hstack((mitad_izquierda_der, mitad_derecha_izq))


    # Mostrar la imagen sin título ni ejes
    plt.figure(figsize=(10, 7))
    plt.imshow(composicion, cmap="gray", aspect='auto')
    plt.axis('off')  # Desactiva los ejes
    output_path = r"C:\Users\Usuario\anaconda3\envs\monailabel-env\Hip-Pal_v2\Mediciones\CentroBordeLateral.png"
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()

    #Resto 90 porque el CentroBordeLateral se mide contra la perpendicular de la plomada.
    angulo_CBL_izq= angulo_CBL_izq - 90
    angulo_CBL_der = (-1*angulo_CBL_der) - 90
    angulos_CBL={
                "path":output_path,
                "izquierdo":{
                    "cbl":angulo_CBL_izq
                },
                "derecho":{
                    "cbl":angulo_CBL_der
                }
        }
    return angulos_CBL
