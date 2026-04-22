import numpy as np

def calcular_iou_y_contencion(box1, box2):
    """Calcula tanto el IoU clásico como la contención (caja dentro de caja)."""
    x_izq = max(box1[0], box2[0])
    y_arriba = max(box1[1], box2[1])
    x_der = min(box1[2], box2[2])
    y_abajo = min(box1[3], box2[3])

    if x_der < x_izq or y_abajo < y_arriba:
        return 0.0, 0.0

    area_interseccion = (x_der - x_izq) * (y_abajo - y_arriba)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    area_union = area_box1 + area_box2 - area_interseccion

    # IoU tradicional (para solapamientos laterales)
    iou = area_interseccion / float(area_union)
    
    # Contención: Qué porcentaje de la caja más pequeña está dentro de la grande
    area_minima = min(area_box1, area_box2)
    io_min = area_interseccion / float(area_minima) if area_minima > 0 else 0

    return iou, io_min

def eliminar_repetidos_nms(detecciones, umbral_iou=0.2, umbral_contencion=0.6):
    """Filtra detecciones usando IoU y comprobando si hay cajas anidadas."""
    if len(detecciones) == 0:
        return []

    # Ordenar detecciones de mayor a menor score
    detecciones = sorted(detecciones, key=lambda x: x['score'], reverse=True)
    detecciones_finales = []

    while len(detecciones) > 0:
        mejor_det = detecciones.pop(0)
        detecciones_finales.append(mejor_det)

        detecciones_restantes = []
        for det in detecciones:
            iou, io_min = calcular_iou_y_contencion(mejor_det['box'], det['box'])
            
            # Mantenemos la caja SOLO si no solapa mucho (IoU) Y no está atrapada dentro de otra (io_min)
            if iou < umbral_iou and io_min < umbral_contencion:
                detecciones_restantes.append(det)
        
        detecciones = detecciones_restantes

    return detecciones_finales