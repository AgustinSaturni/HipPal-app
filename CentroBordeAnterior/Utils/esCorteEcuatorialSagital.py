








def obtenerCentroide(tomografia_original,coordenadas,numero_corte):
    
    _,_,eje_ordenadas=tomografia_original.shape
    x_axial,y_axial,_=coordenadas
    x=y_axial
    y=eje_ordenadas-numero_corte
    z=x_axial

    return  x,y,z
