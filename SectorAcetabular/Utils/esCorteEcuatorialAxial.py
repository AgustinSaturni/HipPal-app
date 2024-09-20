import SectorAcetabular.Utils.esContornoCircular as esContornoCircular
import CabezaFemur.cabezaFemurEcuatorialAxial as cabezaFemurEcuatorialAxial




def detectar(femur_segmentado,femur_segmentado_gris,radio_maximo):

    #Tratamiento especifico para cada femur
    existe_contorno_circular,contorno_mas_grande = esContornoCircular.detectar(femur_segmentado_gris,0.80)

    if existe_contorno_circular:
        flag_cabeza_femur,coordenadas = cabezaFemurEcuatorialAxial.detectar(femur_segmentado,femur_segmentado_gris,contorno_mas_grande,None)
            
        if (flag_cabeza_femur):
            _,_,r=coordenadas
            if( r >= radio_maximo):
                return coordenadas,True,True
            else:
                return None,True,False
        else:
            return None,True,False
    else:
        return None,False,False