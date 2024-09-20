class ErrorCantidadEtiquetas(Exception):

    """Excepción lanzada cuando la tomografía no tiene el formato esperado."""
    def __init__(self, mensaje, detalles=None):
        super().__init__(mensaje)
        self.detalles = detalles

class ErrorProximalEcuatorialNotFound(Exception):
    """Excepción lanzada cuando la tomografía no tiene el formato esperado."""
    def __init__(self, mensaje, detalles=None):
        super().__init__(mensaje)
        self.detalles = detalles

class ErrorIntermedialNotFound(Exception):
    """Excepción lanzada cuando la tomografía no tiene el formato esperado."""
    def __init__(self, mensaje, detalles=None):
        super().__init__(mensaje)
        self.detalles = detalles

class ErrorDetectandoAngulos(Exception):
    """Excepción lanzada cuando la tomografía no tiene el formato esperado."""
    def __init__(self, mensaje, detalles=None):
        super().__init__(mensaje)
        self.detalles = detalles