import cv2
from Excepciones.Excepciones import ErrorIntermedialNotFound, ErrorProximalEcuatorialNotFound
import PreprocesamientoDeCorte.preprocesarAxialSegmentado as preprocesarAxialSegmentado
from SectorAcetabular.Utils import esCorteEcuatorialAxial, esCorteIntermedialAxial
from SectorAcetabular.Utils import esCorteProximalAxial 



def detectar(tomografia_segmentada):

    encontre_ecuatorial_derecho=False
    cabeza_ecuatorial_der=None
    corte_cabeza_ecuatorial_der=0
    radio_cabeza_ecuatorial_der=0
    sali_cabeza_ecuatorial_der=0

    encontre_ecuatorial_izquierdo=False
    cabeza_ecuatorial_izq=None
    corte_cabeza_ecuatorial_izq=0
    radio_cabeza_ecuatorial_izq=0
    sali_cabeza_ecuatorial_izq=0


    encontre_proximal_izquierdo=False
    cabeza_proximal_izq=None
    corte_cabeza_proximal_izq=0

    encontre_proximal_derecho=False
    cabeza_proximal_der=None
    corte_cabeza_proximal_der=0


    #Deteccion
    for i in range(tomografia_segmentada.shape[2] - 1, -1, -1):

        #Obtengo corte del archivo segmentado
        femur_derecho_segmentado = tomografia_segmentada[:, :, i,6]
        femur_izquierdo_segmentado = tomografia_segmentada[:, :, i,7]
    
        femur_derecho_segmentado = preprocesarAxialSegmentado.procesarCorte(femur_derecho_segmentado)
        femur_izquierdo_segmentado = preprocesarAxialSegmentado.procesarCorte(femur_izquierdo_segmentado)

        # Convertir a escala de grises para la detección de contornos
        femur_derecho_segmentado_gris = cv2.cvtColor(femur_derecho_segmentado, cv2.COLOR_RGB2GRAY)
        femur_izquierdo_segmentado_gris = cv2.cvtColor(femur_izquierdo_segmentado, cv2.COLOR_RGB2GRAY)
        
        if(sali_cabeza_ecuatorial_der==10):
            encontre_ecuatorial_derecho=True


        if not encontre_ecuatorial_derecho:
            coordenadas,flag_contorno,flag_cabeza = esCorteEcuatorialAxial.detectar(femur_derecho_segmentado,femur_derecho_segmentado_gris,radio_cabeza_ecuatorial_der)
            
            if (not flag_contorno and radio_cabeza_ecuatorial_der != 0):
                sali_cabeza_ecuatorial_der+=1

            if(flag_cabeza):
                _,_,r_cabeza_ecuatorial_der_hallada=coordenadas
                cabeza_ecuatorial_der=coordenadas
                corte_cabeza_ecuatorial_der=i
                radio_cabeza_ecuatorial_der=r_cabeza_ecuatorial_der_hallada


        if not encontre_proximal_derecho:
            coordenadas_proximal,flag_cabeza = esCorteProximalAxial.detectar(femur_derecho_segmentado,femur_derecho_segmentado_gris)
            if(flag_cabeza):
                cabeza_proximal_der=coordenadas_proximal
                corte_cabeza_proximal_der=i
                encontre_proximal_derecho=True

##----------------------------------------------------------------------------------------------------------------------------


        if(sali_cabeza_ecuatorial_izq==10):
            encontre_ecuatorial_izquierdo=True


        if not encontre_ecuatorial_izquierdo:
            coordenadas,flag_contorno,flag_cabeza = esCorteEcuatorialAxial.detectar(femur_izquierdo_segmentado,femur_izquierdo_segmentado_gris,radio_cabeza_ecuatorial_izq)
            
            if (not flag_contorno and radio_cabeza_ecuatorial_izq != 0):
                sali_cabeza_ecuatorial_izq+=1

            if(flag_cabeza):
                _,_,r_cabeza_ecuatorial_izq_hallada=coordenadas
                cabeza_ecuatorial_izq=coordenadas
                corte_cabeza_ecuatorial_izq=i
                radio_cabeza_ecuatorial_izq=r_cabeza_ecuatorial_izq_hallada


        if not encontre_proximal_izquierdo:
            coordenadas_proximal,flag_cabeza = esCorteProximalAxial.detectar(femur_izquierdo_segmentado,femur_izquierdo_segmentado_gris)
            if(flag_cabeza):
                cabeza_proximal_izq=coordenadas_proximal
                corte_cabeza_proximal_izq=i
                encontre_proximal_izquierdo=True


    #Verifico que encontre resultados y calculo el intermedial
    if(encontre_ecuatorial_izquierdo and encontre_ecuatorial_derecho and encontre_proximal_derecho and encontre_proximal_izquierdo):

        #Busco intermedial.
        corte_cabeza_intermedial_izq = int((corte_cabeza_proximal_izq - corte_cabeza_ecuatorial_izq)/2) + corte_cabeza_ecuatorial_izq

        corte_cabeza_intermedial_der = int((corte_cabeza_proximal_der - corte_cabeza_ecuatorial_der)/2) + corte_cabeza_ecuatorial_der

        cabeza_intermedial_izq,cabeza_intermedial_der=buscarIntermedialAxial(tomografia_segmentada,corte_cabeza_ecuatorial_izq,corte_cabeza_ecuatorial_der)


        #Armo resultados

        femurs={
            "proximal":{
                "izquierdo":{
                    "coordenadas":cabeza_proximal_izq,
                    "numero_corte":corte_cabeza_proximal_izq,
                },
                "derecho":{
                    "coordenadas":cabeza_proximal_der,
                    "numero_corte":corte_cabeza_proximal_der,
                }
            },
            "intermedial":{
                "izquierdo":{
                    "coordenadas":cabeza_intermedial_izq,
                    "numero_corte":corte_cabeza_intermedial_izq,
                },
                "derecho":{
                    "coordenadas":cabeza_intermedial_der,
                    "numero_corte":corte_cabeza_intermedial_der,
                }
            },
            "ecuatorial":{
                "izquierdo":{
                    "coordenadas":cabeza_ecuatorial_izq,
                    "numero_corte":corte_cabeza_ecuatorial_izq,
                },
                "derecho":{
                    "coordenadas":cabeza_ecuatorial_der,
                    "numero_corte":corte_cabeza_ecuatorial_der,
                }
            }
        }

        return femurs
    
    else:
        raise ErrorProximalEcuatorialNotFound(
            "No se pudo hallar uno o ambos cortes Proximales y Axiales.",
            detalles=f"Encontre_ecuatorial_izquierdo: {encontre_ecuatorial_izquierdo}, Encontre_ecuatorial_derecho: {encontre_ecuatorial_derecho},Encontre_proximal_derecho:{encontre_proximal_derecho}, Encontre_proximal_izquierdo:{encontre_proximal_izquierdo}"
        )


def buscarIntermedialAxial(tomografia_segmentada,corte_cabeza_intermedial_izq,corte_cabeza_intermedial_der):
    
    #Obtengo corte del archivo segmentado
    femur_derecho_segmentado = tomografia_segmentada[:, :, corte_cabeza_intermedial_der,6]
    femur_izquierdo_segmentado = tomografia_segmentada[:, :, corte_cabeza_intermedial_izq,7]

    femur_derecho_segmentado = preprocesarAxialSegmentado.procesarCorte(femur_derecho_segmentado)
    femur_izquierdo_segmentado = preprocesarAxialSegmentado.procesarCorte(femur_izquierdo_segmentado)

    # Convertir a escala de grises para la detección de contornos
    femur_derecho_segmentado_gris = cv2.cvtColor(femur_derecho_segmentado, cv2.COLOR_RGB2GRAY)
    femur_izquierdo_segmentado_gris = cv2.cvtColor(femur_izquierdo_segmentado, cv2.COLOR_RGB2GRAY)

    cabeza_intermedial_der,flag_cabeza_der = esCorteIntermedialAxial.detectar(femur_derecho_segmentado,femur_derecho_segmentado_gris)

    cabeza_intermedial_izq,flag_cabeza_izq = esCorteIntermedialAxial.detectar(femur_izquierdo_segmentado,femur_izquierdo_segmentado_gris)

    if(flag_cabeza_der and flag_cabeza_izq):
        return cabeza_intermedial_izq,cabeza_intermedial_der
    else:
        raise ErrorIntermedialNotFound("No se pudo hallar uno o ambos cortes Intermediales.",
            detalles=f"Encontre_intermedial_izquierdo: {flag_cabeza_izq}, Encontre_intermedial_derecho: {flag_cabeza_der}"
        )
