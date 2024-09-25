import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import keras
print(keras.__version__)
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from keras_preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Función para preprocesar la imagen
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# Función para crear el modelo
def create_model(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Función para entrenar el modelo (simulada para este ejemplo)
def train_model(model):
    # En un escenario real, aquí cargarías y preprocesarías tus imágenes de entrenamiento
    # Por ahora, usaremos datos aleatorios para la demostración
    X_train = np.random.rand(100, 224, 224, 3)
    y_train = np.random.randint(1, 20, size=(100, 1))  # Simulando conteos de casas

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=5)
    return model

# Función para contar casas
def count_houses():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = preprocess_image(file_path)
        prediction = model.predict(image)[0][0]
        result_label.config(text=f"Número estimado de casas: {int(round(prediction))}")
        
        # Mostrar la imagen
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Crear y entrenar el modelo
model = create_model()
model = train_model(model)

# Crear la ventana principal
root = tk.Tk()
root.title("Contador de Casas")

# Botón para cargar imagen
load_button = tk.Button(root, text="Cargar Imagen", command=count_houses)
load_button.pack(pady=10)

# Etiqueta para mostrar el resultado
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Etiqueta para mostrar la imagen
image_label = tk.Label(root)
image_label.pack(pady=10)

root.mainloop()