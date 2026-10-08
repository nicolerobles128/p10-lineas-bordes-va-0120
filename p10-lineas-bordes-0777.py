import cv2
import numpy as np
import os

# Nombre del alumno
# Nicole Robles - NC: 0120

# Cargar imagen desde tu subcarpeta 'imagenes'
directorio_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(directorio_script, 'imagenes', 'GARZA-0120.jpeg')

imagen = cv2.imread(ruta_imagen)

# Detectar bordes con Canny
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(gris, 50, 150)

# Aplicar la Transformada de Hough (Idéntico al del enlace)
lineas = cv2.HoughLines(bordes, 1, np.pi/180, 200)

# Dibujar las líneas
if lineas is not None:
    for linea in lineas:
        rho, theta = linea[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Mostrar resultado
cv2.imshow('Lineas Detectadas', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Nicole Robles NC 0120")


# Nicole Robles - NC: 0120
# Ejemplo 1 + Experimentación (HoughLinesP con ajuste de Canny)

import cv2
import numpy as np
import os

# Ejemplo 1 + Experimentación (HoughLinesP con ajuste de Canny)

# 1. Ruta a la imagen dentro de la subcarpeta 'imagenes'
directorio_script = os.path.dirname(os.path.abspath(__file__))
ruta_imagen = os.path.join(directorio_script, 'imagenes', 'GARZA-0120.jpeg')

imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("Error: No se encontró la imagen en la carpeta 'imagenes'.")
else:
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # EXPERIMENTO 1: Ajuste de umbrales de Canny para filtrar ruido
    bordes = cv2.Canny(gris, 100, 200)

    # EXPERIMENTO 2: Transformada de Hough Probabilística (HoughLinesP)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    # Dibujar segmentos detectados
    if lineas is not None:
        for linea in lineas:
            # Corrección aquí: se aplana el arreglo para extraer x1, y1, x2, y2 directamente
            x1, y1, x2, y2 = linea.ravel()
            cv2.line(imagen, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Mostrar ventanas
    cv2.imshow('Bordes Filtrados - Canny', bordes)
    cv2.imshow('Experimento - HoughLinesP (Segmentos)', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Nicole Robles NC 0120 - Practica y Experimento completados")