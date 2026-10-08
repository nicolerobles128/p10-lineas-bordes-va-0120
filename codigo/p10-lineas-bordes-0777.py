import cv2
import numpy as np
import os

# Nicole Robles - NC: 0120

directorio = os.path.dirname(os.path.abspath(__file__))
carpeta_img = os.path.join(directorio, 'imagenes')
carpeta_res = os.path.join(directorio, 'resultados')

os.makedirs(carpeta_res, exist_ok=True)

# Buscar automáticamente la imagen en la carpeta 'imagenes'
ruta_img = None
if os.path.exists(carpeta_img):
    for archivo in os.listdir(carpeta_img):
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            ruta_img = os.path.join(carpeta_img, archivo)
            break

img = cv2.imread(ruta_img) if ruta_img else None

if img is None:
    print(f"Error: No se encontró ninguna imagen en la carpeta: {carpeta_img}")
else:
    # Redimensionar para que encaje bien
    img = cv2.resize(img, (800, 600))
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 1. ORIGINAL (Líneas Rojas - HoughLines) ---
    img_orig = img.copy()
    bordes1 = cv2.Canny(gris, 50, 150)
    lineas1 = cv2.HoughLines(bordes1, 1, np.pi/180, 200)

    if lineas1 is not None:
        for linea in lineas1:
            rho, theta = linea[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_orig, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # --- 2. EXPERIMENTO (Líneas Verdes - HoughLinesP) ---
    img_exp = img.copy()
    bordes2 = cv2.Canny(gris, 100, 200)
    lineas2 = cv2.HoughLinesP(bordes2, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

    if lineas2 is not None:
        for linea in lineas2:
            x1, y1, x2, y2 = linea.ravel()
            cv2.line(img_exp, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # --- GUARDAR RESULTADOS EN CARPETA ---
    cv2.imwrite(os.path.join(carpeta_res, 'resultado_ejemplo1_hough.jpg'), img_orig)
    cv2.imwrite(os.path.join(carpeta_res, 'resultado_experimento_houghp.jpg'), img_exp)
    cv2.imwrite(os.path.join(carpeta_res, 'resultado_bordes_canny.jpg'), bordes2)

    # --- MOSTRAR DE UNA EN UNA ---
    
    # 1. Primera ventana: Original (Roja)
    print("1. Mostrando Original (Líneas Rojas)... Presiona cualquier tecla para continuar.")
    cv2.imshow('1. Original - Hough (Rojo)', img_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2. Segunda ventana: Experimento (Verde)
    print("2. Mostrando Experimento (Líneas Verdes)... Presiona cualquier tecla para continuar.")
    cv2.imshow('2. Experimento - HoughLinesP (Verde)', img_exp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3. Tercera ventana: Bordes (Negra)
    print("3. Mostrando Bordes Canny (Negra)... Presiona cualquier tecla para finalizar.")
    cv2.imshow('3. Bordes Canny (Negra)', bordes2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Nicole Robles NC 0120 - Proceso completado exitosamente.")