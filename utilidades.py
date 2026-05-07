def calcular_iou_y_contencion(box1, box2):
    """
    Calcula el IoU entre dos cajas y el grado de contención de la caja menor
    dentro de la mayor.
    """
    x_izq = max(box1[0], box2[0])
    y_arriba = max(box1[1], box2[1])
    x_der = min(box1[2], box2[2])
    y_abajo = min(box1[3], box2[3])

    if x_der <= x_izq or y_abajo <= y_arriba:
        return 0.0, 0.0

    area_interseccion = (x_der - x_izq) * (y_abajo - y_arriba)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if area_box1 <= 0 or area_box2 <= 0:
        return 0.0, 0.0

    area_union = area_box1 + area_box2 - area_interseccion
    iou = area_interseccion / float(area_union)

    area_minima = min(area_box1, area_box2)
    contencion = area_interseccion / float(area_minima)

    return iou, contencion


def eliminar_repetidos_nms(detecciones, umbral_iou=0.2, umbral_contencion=0.6):
    """
    Elimina detecciones repetidas manteniendo las cajas con mayor puntuación.
    Se descartan cajas con alto solapamiento o contenidas dentro de otra.
    """
    if len(detecciones) == 0:
        return []

    detecciones = sorted(detecciones, key=lambda x: x['score'], reverse=True)
    detecciones_finales = []

    while len(detecciones) > 0:
        mejor_det = detecciones.pop(0)
        detecciones_finales.append(mejor_det)

        detecciones_restantes = []

        for det in detecciones:
            iou, contencion = calcular_iou_y_contencion(mejor_det['box'], det['box'])

            if iou < umbral_iou and contencion < umbral_contencion:
                detecciones_restantes.append(det)

        detecciones = detecciones_restantes

    return detecciones_finales