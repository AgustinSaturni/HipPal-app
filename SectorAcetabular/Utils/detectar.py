import cv2
from matplotlib import pyplot as plt
import numpy as np
from Excepciones.Excepciones import ErrorDetectandoAngulos
from PreprocesamientoDeCorte import preprocesarAxialOriginal, preprocesarAxialSegmentado
from SectorAcetabular.Angulos import aasa, pasa




def detectar(json,tomografia_original,tomografia_segmentada):
    base_path=r"C:\Users\Usuario\anaconda3\envs\monailabel-env\Hip-Pal_v2\Mediciones"
    try:
        
        angulos_proximal = calcularAngulos(tomografia_segmentada,json["proximal"]["izquierdo"]["coordenadas"],json["proximal"]["derecho"]["coordenadas"],json["proximal"]["izquierdo"]["numero_corte"],json["proximal"]["derecho"]["numero_corte"])
        composicion_proximal=obtenerCorteCombinado(tomografia_original,tomografia_segmentada,json["proximal"]["izquierdo"]["numero_corte"],json["proximal"]["derecho"]["numero_corte"])
        ruta_proximal = graficar(composicion_proximal,json["proximal"]["izquierdo"]["coordenadas"],json["proximal"]["derecho"]["coordenadas"],angulos_proximal,"Ecuador Proximal",base_path)

        angulos_intermedial = calcularAngulos(tomografia_segmentada,json["intermedial"]["izquierdo"]["coordenadas"],json["intermedial"]["derecho"]["coordenadas"],json["intermedial"]["izquierdo"]["numero_corte"],json["intermedial"]["derecho"]["numero_corte"])
        composicion_intermedial=obtenerCorteCombinado(tomografia_original,tomografia_segmentada,json["intermedial"]["izquierdo"]["numero_corte"],json["intermedial"]["derecho"]["numero_corte"])
        ruta_intermedial=graficar(composicion_intermedial,json["intermedial"]["izquierdo"]["coordenadas"],json["intermedial"]["derecho"]["coordenadas"],angulos_intermedial,"Intermedial Axial",base_path)

        angulos_ecuatorial = calcularAngulos(tomografia_segmentada,json["ecuatorial"]["izquierdo"]["coordenadas"],json["ecuatorial"]["derecho"]["coordenadas"],json["ecuatorial"]["izquierdo"]["numero_corte"],json["ecuatorial"]["derecho"]["numero_corte"])
        composicion_ecuatorial=obtenerCorteCombinado(tomografia_original,tomografia_segmentada,json["ecuatorial"]["izquierdo"]["numero_corte"],json["ecuatorial"]["derecho"]["numero_corte"])
        ruta_cuatorial=graficar(composicion_ecuatorial,json["ecuatorial"]["izquierdo"]["coordenadas"],json["ecuatorial"]["derecho"]["coordenadas"],angulos_ecuatorial,"Ecuador Axial",base_path)


        angulos={
            "Proximal":{
                "path":ruta_proximal,
                "izquierdo":{
                    "aasa":angulos_proximal["izquierdo"]["aasa"],
                    "pasa":angulos_proximal["izquierdo"]["pasa"],
                    "hasa":angulos_proximal["izquierdo"]["hasa"],
                },
                "derecho":{
                    "aasa":angulos_proximal["derecho"]["aasa"],
                    "pasa":angulos_proximal["derecho"]["pasa"],
                    "hasa":angulos_proximal["derecho"]["hasa"],
                }
            },
            "Intermedial":{
                "path":ruta_intermedial,
                "izquierdo":{
                    "aasa":angulos_intermedial["izquierdo"]["aasa"],
                    "pasa":angulos_intermedial["izquierdo"]["pasa"],
                    "hasa":angulos_intermedial["izquierdo"]["hasa"],
                },
                "derecho":{
                    "aasa":angulos_intermedial["derecho"]["aasa"],
                    "pasa":angulos_intermedial["derecho"]["pasa"],
                    "hasa":angulos_intermedial["derecho"]["hasa"],
                }
            },
            "Ecuatorial":{
                "path":ruta_cuatorial,
                "izquierdo":{
                    "aasa":angulos_ecuatorial["izquierdo"]["aasa"],
                    "pasa":angulos_ecuatorial["izquierdo"]["pasa"],
                    "hasa":angulos_ecuatorial["izquierdo"]["hasa"],
                },
                "derecho":{
                    "aasa":angulos_ecuatorial["derecho"]["aasa"],
                    "pasa":angulos_ecuatorial["derecho"]["pasa"],
                    "hasa":angulos_ecuatorial["derecho"]["hasa"],
                }
            }
        }

        return angulos

    except: 
        raise ErrorDetectandoAngulos(
            "Ocurrio un error inesperado en la deteccion de los angulos Sector Acetabular.")

def obtenerCorteCombinado(tomografia_original,tomografia_segmentada,numero_corte_izquierdo,numero_corte_derecho):
    
        #Original
        corte_original_acetabulo_izquierdo = tomografia_original[:,:,numero_corte_izquierdo]
        corte_original_acetabulo_izquierdo= preprocesarAxialOriginal.procesarCorte(corte_original_acetabulo_izquierdo) 
        
        corte_original_acetabulo_derecho = tomografia_original[:,:,numero_corte_derecho]
        corte_original_acetabulo_derecho= preprocesarAxialOriginal.procesarCorte(corte_original_acetabulo_derecho) 


        #Segmentado
        corte_segmentado_acetabulo_izquierdo = tomografia_segmentada[:,:,numero_corte_izquierdo,5]
        corte_segmentado_acetabulo_izquierdo= preprocesarAxialSegmentado.procesarCorte(corte_segmentado_acetabulo_izquierdo) 

        corte_segmentado_acetabulo_derecho = tomografia_segmentada[:,:,numero_corte_derecho,4]
        corte_segmentado_acetabulo_derecho= preprocesarAxialSegmentado.procesarCorte(corte_segmentado_acetabulo_derecho)   



        # Cortar cada imagen por la mitad
        mitad_derecha_izq = corte_original_acetabulo_izquierdo[:, corte_original_acetabulo_izquierdo.shape[1] // 2:]
        mitad_izquierda_der = corte_original_acetabulo_derecho[:, :corte_original_acetabulo_derecho.shape[1] // 2]


        # Combinar las mitades
        composicion = np.hstack((mitad_izquierda_der, mitad_derecha_izq))

        return composicion


def calcularAngulos(tomografia_segmentada,cabeza_izq,cabeza_der,numero_corte_izquierdo,numero_corte_derecho):

    
    #Segmentado
    corte_segmentado_acetabulo_izquierdo = tomografia_segmentada[:,:,numero_corte_izquierdo,5]
    corte_segmentado_acetabulo_izquierdo= preprocesarAxialSegmentado.procesarCorte(corte_segmentado_acetabulo_izquierdo) 

    corte_segmentado_acetabulo_derecho = tomografia_segmentada[:,:,numero_corte_derecho,4]
    corte_segmentado_acetabulo_derecho= preprocesarAxialSegmentado.procesarCorte(corte_segmentado_acetabulo_derecho)   

    x_der,y_der,r_der = cabeza_der
    x_izq,y_izq,r_izq =cabeza_izq

    #Detecto PASA   
    x_pasa_der, y_pasa_der, angulo_pasa_der  = pasa.detectar(corte_segmentado_acetabulo_derecho,x_der,y_der,x_izq,y_izq,r_der,"derecho")
    x_pasa_izq, y_pasa_izq, angulo_pasa_izq  = pasa.detectar(corte_segmentado_acetabulo_izquierdo,x_izq,y_izq,x_der,y_der,r_izq,"izquierdo")

    #Detecto AASA
    x_aasa_der, y_aasa_der, angulo_aasa_der  = aasa.detectar(corte_segmentado_acetabulo_derecho,x_der,y_der,x_izq,y_izq,r_der,"derecho")
    x_aasa_izq, y_aasa_izq, angulo_aasa_izq  = aasa.detectar(corte_segmentado_acetabulo_izquierdo,x_izq,y_izq,x_der,y_der,r_izq,"izquierdo")

    angulo_hasa_izq=angulo_aasa_izq + (angulo_pasa_izq*-1)
    angulo_hasa_der=(angulo_aasa_der*-1) + (angulo_pasa_der)

    angulos={
        "izquierdo":{
                "aasa":angulo_aasa_izq,
                "x_aasa":x_aasa_izq,
                "y_aasa":y_aasa_izq,
                "pasa":-1*angulo_pasa_izq,
                "x_pasa":x_pasa_izq,
                "y_pasa":y_pasa_izq,
                "hasa":angulo_hasa_izq
            },
        "derecho":{
                "aasa":-1*angulo_aasa_der,
                "x_aasa":x_aasa_der,
                "y_aasa":y_aasa_der,
                "pasa":angulo_pasa_der,
                "x_pasa":x_pasa_der,
                "y_pasa":y_pasa_der,
                "hasa":angulo_hasa_der
            }
    }


    return angulos


def graficar(composicion,coordenadas_izq,coordenadas_der,angulos,title,base_path):
        #Dibujo cabezas
        x_izq,y_izq,r_izq = coordenadas_izq
        cv2.circle(composicion, (x_izq, y_izq), r_izq, (0, 255, 0), 1)  # Círculo verde
        cv2.circle(composicion, (x_izq, y_izq), 1, (0, 255, 0), 1)  # Círculo verde

        x_der,y_der,r_der = coordenadas_der
        cv2.circle(composicion, (x_der, y_der), r_der, (0, 255, 0), 1)  # Círculo verde
        cv2.circle(composicion, (x_der, y_der), 1, (0, 255, 0), 1)  # Círculo verde

        # Dibujo la plomada
        cv2.line(composicion, (x_der, y_der), (x_izq, y_izq), (0, 255, 0), 2)  # Línea azul

        
        cv2.line(composicion, (x_der, y_der), (angulos["derecho"]["x_pasa"], angulos["derecho"]["y_pasa"]), (0, 255, 0), 1)  # Línea verd
        cv2.line(composicion, (x_izq, y_izq), (angulos["izquierdo"]["x_pasa"], angulos["izquierdo"]["y_pasa"]), (0, 255, 0), 1)  # Línea verd

        cv2.line(composicion, (x_der, y_der), (angulos["derecho"]["x_aasa"], angulos["derecho"]["y_aasa"]), (0, 255, 0), 1)  # Línea verd
        cv2.line(composicion, (x_izq, y_izq), (angulos["izquierdo"]["x_aasa"], angulos["izquierdo"]["y_aasa"]), (0, 255, 0), 1)  # Línea verd

     
        plt.figure(figsize=(18, 10))
        plt.subplot(1, 1, 1)
        plt.imshow(composicion, cmap='gray')
        plt.title('off')
        output_path = f"{base_path}/{title}.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.show()
        return output_path