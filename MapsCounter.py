import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input

class HouseCounterApp:
    def __init__(self, master):
        self.master = master
        master.title("Contador de Casas")
        master.geometry("800x600")

        self.model = self.create_model()

        self.create_widgets()

    def create_model(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        output = Dense(1, activation='linear')(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # Frame izquierdo para la imagen
        left_frame = ttk.Frame(main_frame, borderwidth=2, relief="groove", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Frame derecho para controles y resultados
        right_frame = ttk.Frame(main_frame, borderwidth=2, relief="groove", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Widgets del frame izquierdo
        self.canvas = tk.Canvas(left_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Widgets del frame derecho
        ttk.Button(right_frame, text="Cargar Imagen", command=self.load_image).pack(pady=5, fill=tk.X)
        ttk.Button(right_frame, text="Contar Casas", command=self.count_houses).pack(pady=5, fill=tk.X)

        self.result_var = tk.StringVar()
        ttk.Label(right_frame, textvariable=self.result_var, font=('Arial', 12, 'bold')).pack(pady=10)

        self.confidence_var = tk.StringVar()
        ttk.Label(right_frame, textvariable=self.confidence_var).pack(pady=5)

        # Barra de ajuste de sensibilidad
        ttk.Label(right_frame, text="Sensibilidad:").pack(pady=5)
        self.sensitivity = ttk.Scale(right_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, value=1.0, command=self.update_sensitivity)
        self.sensitivity.pack(pady=5, fill=tk.X)

        # Lista de objetos detectados
        self.tree = ttk.Treeview(right_frame, columns=('Objeto', 'Cantidad'), show='headings')
        self.tree.heading('Objeto', text='Objeto')
        self.tree.heading('Cantidad', text='Cantidad')
        self.tree.pack(pady=10, fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.display_image()

    def display_image(self):
        # Redimensionar la imagen para que quepa en el canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_width, img_height = self.original_image.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        self.displayed_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.displayed_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)

    def count_houses(self):
        if hasattr(self, 'original_image'):
            # Preprocesar la imagen
            img = self.original_image.resize((224, 224))
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Predicción
            prediction = self.model.predict(img_array)[0][0]
            adjusted_prediction = prediction * self.sensitivity.get()

            # Actualizar resultados
            self.result_var.set(f"Casas detectadas: {int(round(adjusted_prediction))}")
            self.confidence_var.set(f"Confianza: {min(100, int(100 * (1 - abs(adjusted_prediction - round(adjusted_prediction)) / adjusted_prediction)))}%")

            # Actualizar lista de objetos
            self.tree.delete(*self.tree.get_children())
            self.tree.insert('', 'end', values=('Casas', int(round(adjusted_prediction))))

    def update_sensitivity(self, value):
        if hasattr(self, 'original_image'):
            self.count_houses()

if __name__ == "__main__":
    root = tk.Tk()
    app = HouseCounterApp(root)
    root.mainloop()