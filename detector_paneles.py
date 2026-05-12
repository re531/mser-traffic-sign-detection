import cv2
import numpy as np
from utilidades import eliminar_repetidos_nms

class DetectorPaneles:
    def __init__(self):
        # MSER
        # Inicializamos el detector MSER. Ajustamos delta y las áreas para no detectar
        # cosas diminutas ni gigantes que no sean paneles.
        self.mser = cv2.MSER_create(delta=5, min_area=600, max_area=90000)
        
        # Tamaño de referencia para calcular el score de azul.
        self.tamano_base = (80, 40)
        
        # Rango de color azul en HSV. Buscamos azules saturados típicos de señal de tráfico.
        self.lower_blue = np.array([100, 130, 40])
        self.upper_blue = np.array([140, 255, 255])

    def buscar_rectangulos_grandes(self, imagen):
    
        # Paneles gigantes y borrosos.
        # El MSER a veces falla si el panel está muy cerca o borroso. 
        # Usamos Canny y contornos para sacar estas cajas no detectadas.
        
        detecciones_extra = []
        alto_img, ancho_img = imagen.shape[:2]
        
        # Pasamos a gris y suavizamos para que Canny no saque ruido del asfalto.
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        bordes = cv2.Canny(blur, 30, 100)
        contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            # Solo nos interesan contornos muy grandes (paneles cercanos).
            if area > 4000:
                x, y, w, h = cv2.boundingRect(cnt)
                extent = area / float(w * h)
                
                # Extent > 0.60 nos asegura que el contorno rellena la caja (es rectangular)
                # x e y > 5 evita ruido pegado a los bordes de la cámara
                if extent > 0.60 and x > 5 and y > 5:  
                    relacion_aspecto = w / float(h)
                    
                    # Filtramos por proporciones maximo 3.2 de ancho
                    # y < alto_img * 0.6 evita que detecte parte del coche
                    if 0.8 < relacion_aspecto < 3.2 and y < alto_img * 0.6:
                        roi = imagen[y:y+h, x:x+w]
                        roi_resized = cv2.resize(roi, self.tamano_base)
                        hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
                        
                        # Bajamos la saturación del azul porque los paneles grandes
                        # y borrosos suelen perder color
                        lower_blue_relajado = np.array([90, 40, 40]) 
                        mask = cv2.inRange(hsv, lower_blue_relajado, self.upper_blue)
                        score = cv2.countNonZero(mask) / float(self.tamano_base[0] * self.tamano_base[1])
                        
                        # Si tiene algo de azul razonable, lo guardamos
                        if score > 0.20:
                            detecciones_extra.append({'box': [x, y, x + w, y + h], 'score': score})
        return detecciones_extra

    def buscar_paneles_niebla(self, imagen):
    
        # Paneles con niebla.
        # La niebla desatura mucho los colores, así que buscamos manchas sólidas 
        # en la mitad superior de la imagen con un rango HSV adaptado.
    
        detecciones_niebla = []
        alto_img, ancho_img = imagen.shape[:2]
        
        # Cortamos la imagen a la mitad porque los paneles en niebla siempre están en la parte superior.
        roi_superior = imagen[0:int(alto_img * 0.5), 0:ancho_img]

        # Rango de azul muy poco saturado y oscuro.
        lower_blue_fog = np.array([95, 30, 50])
        upper_blue_fog = np.array([135, 110, 220])

        hsv = cv2.cvtColor(roi_superior, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue_fog, upper_blue_fog)

        # Operaciones morfológicas para limpiar el ruido y dejar la mancha del panel sólida.
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            # Filtro de tamaño medio para señales entre la niebla
            if 1500 < area < 40000:
                x, y, w, h = cv2.boundingRect(cnt)
                extent = area / float(w * h)
                
                # Buscamos un buen llenado rectangular y proporciones correctas
                if extent > 0.65 and 0.8 < (w/float(h)) < 3.2:
                    roi_mask = mask[y:y+h, x:x+w]
                    densidad = cv2.countNonZero(roi_mask) / float(w * h)
                    
                    # Si la máscara es lo bastante sólida, la damos por buena
                    if densidad > 0.50:
                        detecciones_niebla.append({'box': [x, y, x + w, y + h], 'score': densidad})
        return detecciones_niebla

    def detectar(self, imagen):
        
        # Flujo principal del detector. 
        # Combina la detección MSER con los planes de respaldo (Canny y Niebla).
        
        alto_img, ancho_img = imagen.shape[:2]
        
        # 1. Normalización Min-Max.
        # Estiramos el histograma para que MSER detecte mejor los bordes en zonas con sombras.
        gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Suavizamos antes de pasar MSER para no coger texturas innecesarias
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        regiones, _ = self.mser.detectRegions(gray_blur)
        detecciones_iniciales = []

        # BÚSQUEDA MSER
        for region in regiones:
            x, y, w, h = cv2.boundingRect(region)
            
            # Filtros de posición: evitamos el parasol de arriba y el salpicadero
            if y < alto_img * 0.05 or y > alto_img * 0.65: continue
            
            # Filtro por relleno: si es < 0.45 suele ser una montaña (triangular), no un panel
            if (len(region) / float(w * h)) < 0.45: continue

            relacion_aspecto = w / float(h)
            
            # Un panel no debe ser hiper alargado (max 3.2)
            if 0.8 < relacion_aspecto < 3.2:
                # Agrandamos la caja un 8% para incluir el marco blanco de la señal,
                # mejorando así el IoU con el Ground Truth.
                mw, mh = int(w * 0.08), int(h * 0.08)
                x1, y1 = max(0, x-mw), max(0, y-mh)
                x2, y2 = min(ancho_img, x+w+mw), min(alto_img, y+h+mh)

                roi = imagen[y1:y2, x1:x2]
                if roi.size == 0: continue

                # Puntuamos la cantidad de azul saturado (score)
                roi_resized = cv2.resize(roi, self.tamano_base)
                hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
                score = cv2.countNonZero(mask) / float(self.tamano_base[0] * self.tamano_base[1])

                if score > 0.40:
                    detecciones_iniciales.append({'box': [x1, y1, x2, y2], 'score': score})

        # Juntamos lo sacado por MSER con lo de Canny y Niebla
        totales = detecciones_iniciales + self.buscar_rectangulos_grandes(imagen) + self.buscar_paneles_niebla(imagen)
        
        # Filtro de densidad de bordes
        # Al mejorar el contraste general, a veces detectamos manchas en el cielo o nubes.
        # Un panel real tiene texto o flechas.
        validadas = []
        for det in totales:
            x1, y1, x2, y2 = det['box']
            roi = imagen[y1:y2, x1:x2]
            if roi.size == 0: continue
            
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bordes = cv2.Canny(roi_gray, 30, 100)
            
            # Calculamos qué porcentaje del parche son bordes
            densidad = cv2.countNonZero(bordes) / float(roi.shape[0] * roi.shape[1])
            
            # Si hay más de un 1.2% de bordes, asumimos que tiene texto y lo validamos
            if densidad > 0.012:
                validadas.append(det)

        # Finalmente, limpiamos las cajas repetidas que detectan lo mismo
        return eliminar_repetidos_nms(validadas, umbral_iou=0.2, umbral_contencion=0.6)