






def obtenerCentroide(tomografia_original,coordenadas,numero_corte):
    
    _,profundidad,eje_ordenadas=tomografia_original.shape
    x,z,_=coordenadas
    y=eje_ordenadas-numero_corte
    z=profundidad-z
    return x,y,z
