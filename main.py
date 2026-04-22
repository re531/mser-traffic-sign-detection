import argparse
import os
import cv2
from detector_paneles import DetectorPaneles

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Trains and executes a given detector over a set of testing images')
    parser.add_argument(
        '--detector', type=str, nargs="?", default="", help='Detector string name')
    parser.add_argument(
        '--train_path', default="", help='Select the training data dir')
    parser.add_argument(
        '--test_path', default="", help='Select the testing data dir')

    args = parser.parse_args()

    # ==========================================
    # Load training data
    # ==========================================
    # Nota: Con algoritmos clásicos (MSER, HSV) no entrenamos un modelo.
    # Podrías usar esta sección para cargar recortes del 'train_path' y 
    # extraer el color azul ideal dinámicamente si quisieras automatizarlo.
    print(f"Directorio de entrenamiento: {args.train_path}")

    # ==========================================
    # Create the detector
    # ==========================================
    print(f"Inicializando detector: {args.detector}")
    detector = DetectorPaneles()

    # ==========================================
    # Load testing data
    # ==========================================
    print(f"Cargando imágenes desde: {args.test_path}")
    if os.path.exists(args.test_path):
        archivos_test = sorted(os.listdir(args.test_path))
        # Filtramos para quedarnos solo con imágenes
        imagenes_test = [f for f in archivos_test if f.lower().endswith(('.png', '.jpg'))]
    else:
        imagenes_test = []
        print(f"Error: La ruta de test '{args.test_path}' no existe.")

    # ==========================================
    # Evaluate detections
    # ==========================================
    # 1. Crear la carpeta para guardar las imágenes resultantes
    dir_resultados_imgs = "resultado_imgs"
    if not os.path.exists(dir_resultados_imgs):
        os.makedirs(dir_resultados_imgs)

    # 2. Abrir el archivo de texto donde guardaremos los resultados
    ruta_resultado_txt = "resultado.txt"
    f_txt = open(ruta_resultado_txt, "w")

    print("\nProcesando detecciones...")
    
    # 3. Bucle principal sobre cada imagen
    for nombre_img in imagenes_test:
        ruta_img = os.path.join(args.test_path, nombre_img)
        img = cv2.imread(ruta_img)
        
        if img is None:
            continue

        # Llamada a nuestra clase detectora
        detecciones = detector.detectar(img)

        # Dibujar cajas y escribir en el txt
        for det in detecciones:
            x1, y1, x2, y2 = det['box']
            score = det['score']

            # Formato estricto: <nombre_fichero>;<x1>;<y1>;<x2>;<y2>;<tipo>;<score>
            linea_txt = f"{nombre_img};{x1};{y1};{x2};{y2};1;{score:.3f}\n"
            f_txt.write(linea_txt)

            # Requisito: Cuadrado rojo (0,0,255 en BGR) y texto amarillo (0,255,255)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Guardar la imagen pintada en la carpeta de resultados
        ruta_guardado = os.path.join(dir_resultados_imgs, nombre_img)
        cv2.imwrite(ruta_guardado, img)
        print(f" -> {nombre_img}: {len(detecciones)} paneles encontrados.")

    # Cerrar archivo de texto al terminar
    f_txt.close()
    print(f"\n¡Proceso completado! Archivo '{ruta_resultado_txt}' generado y guardado.")