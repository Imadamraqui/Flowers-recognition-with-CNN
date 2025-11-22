import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle
model_path = "C:/Users/pc/Desktop/flowers recognition/cnn_model.h5"
model = load_model(model_path)

# Classes de fleurs
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Fonction pour prédire la classe
def predict_flower(image_path):
    try:
        # Charger l'image et la prétraiter
        img = Image.open(image_path).resize((224, 224))  # Redimensionner à la taille d'entrée du modèle
        img_array = np.array(img) / 255.0  # Normaliser les pixels
        img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
        
        # Prédiction
        prediction = model.predict(img_array) # Renvoie un tableau de probabilités pour chaque classe
        predicted_class = class_names[np.argmax(prediction)] # Classe avec la probabilité maximale
        confidence = np.max(prediction) * 100 # Confiance associée à la prédiction en pourcentage
        return predicted_class, confidence
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur lors de la prédiction : {e}")
        return None, None

# Fonction pour charger une image
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".jpg;.png;*.jpeg")])
    if file_path:
        try:
            # Charger l'image
            img = Image.open(file_path)
            
            # Calculer un redimensionnement proportionnel
            max_size = 300
            img_width, img_height = img.size
            scale = min(max_size / img_width, max_size / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Afficher l'image redimensionnée
            img_tk = ImageTk.PhotoImage(img)
            img_label.config(image=img_tk, width=max_size, height=max_size)
            img_label.image = img_tk
            
            # Prédire la classe
            predicted_class, confidence = predict_flower(file_path)
            if predicted_class:
                result_label.config(
                    text=f"Classe prédite : {predicted_class}\nConfiance : {confidence:.2f}%",
                    bg="#D5F5E3", fg="#1E8449"
                )
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement de l'image : {e}")


# Interface Tkinter
root = tk.Tk()
root.title("Classification des Fleurs")
root.geometry("600x700")
root.configure(bg="#E9F7EF")  # Vert clair

# Styles globaux
title_font = ("Helvetica", 24, "bold")
button_font = ("Helvetica", 14)
result_font = ("Helvetica", 16)

# Widgets
title_label = tk.Label(root, text="Classification des Fleurs", font=title_font, bg="#E9F7EF", fg="#1E8449")
title_label.pack(pady=20)

img_frame = tk.Frame(root, bg="#ABEBC6", bd=2, relief="groove")
img_frame.pack(pady=20)

img_label = tk.Label(img_frame, bg="#F9E79F", width=40, height=15)
img_label.pack()

result_label = tk.Label(
    root, 
    text="Chargez une image pour commencer", 
    font=result_font, 
    bg="#E9F7EF", 
    fg="#1E8449", 
    wraplength=500, 
    justify="center"
)
result_label.pack(pady=30)

load_button = tk.Button(
    root, 
    text="Charger une image", 
    command=load_image, 
    font=button_font, 
    bg="#28B463", 
    fg="white", 
    activebackground="#1D8348", 
    activeforeground="white"
)
load_button.pack(pady=10)

exit_button = tk.Button(
    root, 
    text="Quitter", 
    command=root.quit, 
    font=button_font, 
    bg="#C0392B", 
    fg="white", 
    activebackground="#922B21", 
    activeforeground="white"
)
exit_button.pack(pady=10)

# Lancer l'application
root.mainloop()