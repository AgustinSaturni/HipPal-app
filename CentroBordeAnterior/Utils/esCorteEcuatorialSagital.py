






def obtenerCentroide(tomografia_original,coordenadas,numero_corte):
    print("tamaño:" + str(tomografia_original.shape))
    _,_,eje_ordenadas=tomografia_original.shape
    x_axial,y_axial,_=coordenadas
    x=y_axial
    y=eje_ordenadas-numero_corte
    z=x_axial
    print(x)
    return  x,y,z
