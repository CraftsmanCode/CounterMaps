import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense #Línea con error (requiere cambio)
from tensorflow.python.keras.preprocessing.image import img_to_array #Línea con error (requiere cambio)


# Creación del modelo simplificado
def create_simple_model():
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')  # Salida: número estimado de objetos
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Entrenamiento simple del modelo (con datos aleatorios)
def train_simple_model(model):
    X_train = np.random.rand(100, 224, 224, 3)
    y_train = np.random.randint(1, 20, size=(100, 1))  # Simulando conteos de objetos
    model.fit(X_train, y_train, epochs=5, verbose=0)
    return model

# Crear y guardar el modelo
simple_model = create_simple_model()
simple_model = train_simple_model(simple_model)
simple_model.save('object_detection_model.h5')

class ObjectCounterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Contador de Objetos")
        self.master.geometry("800x600")

        self.image = None
        self.photo = None
        self.draw_mode = "auto"
        self.objects = []

        # Cargar el modelo
        self.model = load_model('object_detection_model.h5')

        self.create_widgets()

    def create_widgets(self):
        # Frame principal
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas para la imagen
        self.canvas = tk.Canvas(main_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Frame para controles
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Botones
        tk.Button(control_frame, text="Cargar Imagen", command=self.load_image).pack(fill=tk.X, padx=5, pady=5)
        tk.Button(control_frame, text="Detección Automática", command=self.auto_detect).pack(fill=tk.X, padx=5, pady=5)
        tk.Button(control_frame, text="Limpiar Detecciones", command=self.clear_detections).pack(fill=tk.X, padx=5, pady=5)

        # Radio buttons para modo de dibujo
        self.draw_var = tk.StringVar(value="auto")
        tk.Radiobutton(control_frame, text="Auto", variable=self.draw_var, value="auto").pack(anchor=tk.W, padx=5)
        tk.Radiobutton(control_frame, text="Manual", variable=self.draw_var, value="manual").pack(anchor=tk.W, padx=5)

        # Etiqueta para mostrar el conteo
        self.count_label = tk.Label(control_frame, text="Objetos: 0")
        self.count_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = Image.open(file_path)
            self.image.thumbnail((700, 500))  # Redimensionar para ajustar al canvas
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.photo.width(), height=self.photo.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            self.clear_detections()

    def auto_detect(self):
        if self.image is None:
            messagebox.showwarning("Advertencia", "Por favor, carga una imagen primero.")
            return

        # Preprocesar la imagen para el modelo
        img_array = img_to_array(self.image.resize((224, 224)))
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalización simple

        # Realizar la detección
        prediction = self.model.predict(img_array)
        num_objects = int(round(prediction[0][0]))

        # Simular la ubicación de los objetos
        for _ in range(num_objects):
            x = np.random.randint(0, self.photo.width())
            y = np.random.randint(0, self.photo.height())
            self.draw_object(x, y)

        self.update_count()

    def on_canvas_click(self, event):
        if self.draw_var.get() == "manual":
            self.draw_object(event.x, event.y)
            self.update_count()

    def draw_object(self, x, y):
        object_id = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red")
        self.objects.append(object_id)

    def clear_detections(self):
        for obj in self.objects:
            self.canvas.delete(obj)
        self.objects.clear()
        self.update_count()

    def update_count(self):
        self.count_label.config(text=f"Objetos: {len(self.objects)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectCounterApp(root)
    root.mainloop()
