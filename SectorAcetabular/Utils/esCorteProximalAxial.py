import SectorAcetabular.Utils.esContornoCircular as esContornoCircular
import CabezaFemur.cabezaFemurProximalAxial as cabezaFemurProximalAxial


def detectar(femur_segmentado,femur_segmentado_gris):

    #Tratamiento especifico para cada femur
    existe_contorno_circular,contorno_mas_grande = esContornoCircular.detectar(femur_segmentado_gris,0.81)

    if existe_contorno_circular:
        flag_cabeza_femur,coordenadas = cabezaFemurProximalAxial.detectar(femur_segmentado,femur_segmentado_gris,contorno_mas_grande,None)
            
        if (flag_cabeza_femur):
            return coordenadas,True
        else:
            return None,False
    else:
        return None,False