import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog

# Cargar el modelo preentrenado
model = tf.saved_model.load('C:\Users\henry\OneDrive\Documentos\Proyectos de programación\Python\MapCounterApp\Images')

# Función para cargar y preprocesar la imagen
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = np.expand_dims(img, axis=0)
    return img

# Función para realizar la detección
def detect_objects(image_path):
    img = load_image(image_path)
    detections = model(img)
    return detections

# Función para dibujar las detecciones en la imagen
def draw_detections(image_path, detections):
    img = cv2.imread(image_path)
    for detection in detections:
        # Extraer coordenadas y dibujar rectángulos
        x, y, w, h = detection['box']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Detections', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Función para abrir el archivo y procesar la imagen
def open_file():
    file_path = filedialog.askopenfilename()
    detections = detect_objects(file_path)
    draw_detections(file_path, detections)

# Crear la interfaz de usuario
root = tk.Tk()
root.title("Contador de Casas")
button = tk.Button(root, text="Abrir Imagen", command=open_file)
button.pack()
root.mainloop()
