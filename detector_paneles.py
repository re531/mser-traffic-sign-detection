import cv2
import numpy as np
from utilidades import eliminar_repetidos_nms

'''
Creamos una clase DetectorPanles para encapsular todo el sistema
Esta clase tiene un método final detectar() que llama a todas las funciones que necesita
'''
class DetectorPaneles:
    #Constructor de la clase DetectorPaneles
    def __init__(self):
        '''
        Algoritmo MSER:
        Buscamos regiones estables dentro de la imagen porque los paneles
        tienen una zona azul homogéne que contrasta respecto al borde y el fondo
        '''
        self.mser = cv2.MSER_create(delta=5, min_area=600, max_area=90000)
        self.tamano_base = (80, 40)
        self.lower_blue = np.array([100, 130, 40])
        self.upper_blue = np.array([140, 255, 255])

    def buscar_rectangulos_grandes(self, imagen):
        """
        Algoritmo alternativo al MSER:
        Pensado para cuando el panel es muy grande, el texto o imágen está borroso o tiene niebla.
        Se basa en:
        - Bordes
        - Contornos
        """
        detecciones_extra = []
        alto_img, ancho_img = imagen.shape[:2]

        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        #saca bordes
        bordes = cv2.Canny(blur, 30, 100)
        #saca contornos cerrados sobre esos bordes
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #descartamos contornos demasiado pequeños
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area > 4000:
                x, y, w, h = cv2.boundingRect(cnt)
                extent = area / float(w * h)

                # Evitamos contornos pegados a los bordes de la imagen (suelen ser ruido)
                if extent > 0.60 and x > 5 and y > 5:
                    relacion_aspecto = w / float(h)

                    if 0.8 < relacion_aspecto < 3.5 and y < alto_img * 0.6:
                        roi = imagen[y:y + h, x:x + w]
                        roi_resized = cv2.resize(roi, self.tamano_base)
                        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)

                        lower_blue_relajado = np.array([90, 40, 40])
                        mask = cv2.inRange(hsv, lower_blue_relajado, self.upper_blue)

                        score = cv2.countNonZero(mask) / float(self.tamano_base[0] * self.tamano_base[1])

                        if score > 0.20:
                            detecciones_extra.append({
                                'box': [x, y, x + w, y + h],
                                'score': score
                            })
        return detecciones_extra

    def buscar_paneles_niebla(self, imagen):
        """
        Algoritmo alternativo a MSER:
        Lo usamos para casos difíciles:
        - Niebla
        - Poca saturación
        - Paneles lavados de color
        - Situaciones donde MSER y el azul estricto fallan
        """
        detecciones_niebla = []
        alto_img, ancho_img = imagen.shape[:2]

        limite_y = int(alto_img * 0.5)
        roi_superior = imagen[0:limite_y, 0:ancho_img]

        lower_blue_fog = np.array([95, 15, 50])
        upper_blue_fog = np.array([135, 110, 220])

        hsv = cv2.cvtColor(roi_superior, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue_fog, upper_blue_fog)

        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if 1500 < area < 40000:
                x, y, w, h = cv2.boundingRect(cnt)
                extent = area / float(w * h)

                if extent > 0.65 and x > 5 and y > 5:
                    relacion_aspecto = w / float(h)

                    if 0.8 < relacion_aspecto < 3.5:
                        # SOLUCIÓN NUBES: Obligamos a comprobar la densidad real de la máscara
                        # en lugar de regalar el score. Si la caja está hueca por dentro (nube), la rechaza.
                        roi_mask = mask[y:y + h, x:x + w]
                        densidad_solida = cv2.countNonZero(roi_mask) / float(w * h)

                        if densidad_solida > 0.50:
                            # Le asignamos la densidad real como score
                            detecciones_niebla.append({
                                'box': [x, y, x + w, y + h],
                                'score': densidad_solida
                            })
        return detecciones_niebla

    def detectar(self, imagen):
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((3, 3), np.uint8)
        gray = cv2.erode(gray, kernel, iterations=1)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        regiones, _ = self.mser.detectRegions(gray)
        detecciones_iniciales = []
        alto_img, ancho_img = imagen.shape[:2]

        # --- PLAN A: MSER CLÁSICO ---
        for region in regiones:
            x, y, w, h = cv2.boundingRect(region)

            # SOLUCIÓN SALPICADERO: Subimos el suelo de descarte.
            # Ignora cualquier cosa que asome por debajo del 65% de la pantalla.
            if y > alto_img * 0.65:
                continue

            relacion_aspecto = w / float(h)

            if 0.8 < relacion_aspecto < 4.0:
                margen_w = int(w * 0.08)
                margen_h = int(h * 0.08)

                x1 = max(0, x - margen_w)
                y1 = max(0, y - margen_h)
                x2 = min(ancho_img, x + w + margen_w)
                y2 = min(alto_img, y + h + margen_h)

                roi = imagen[y1:y2, x1:x2]
                if roi.size == 0: continue

                roi_resized = cv2.resize(roi, self.tamano_base)
                hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

                score = cv2.countNonZero(mask) / float(self.tamano_base[0] * self.tamano_base[1])

                if score > 0.40:
                    detecciones_iniciales.append({
                        'box': [x1, y1, x2, y2],
                        'score': score
                    })

        # --- RECOPILACIÓN DE PLANES ---
        detecciones_extra = self.buscar_rectangulos_grandes(imagen)
        detecciones_niebla = self.buscar_paneles_niebla(imagen)

        detecciones_totales = detecciones_iniciales + detecciones_extra + detecciones_niebla

        # --- LIMPIEZA FINAL NMS ---
        detecciones_finales = eliminar_repetidos_nms(detecciones_totales, umbral_iou=0.2, umbral_contencion=0.6)

        return detecciones_finales