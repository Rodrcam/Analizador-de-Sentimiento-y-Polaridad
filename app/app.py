# app/app.py

import pandas as pd
import pickle
import re
import os
from flask import Flask, render_template, request

# --- Configuración de Flask ---
app = Flask(__name__, template_folder='templates')

# --- Rutas de Archivos (Ajustadas para la app) ---
# Subimos un nivel para buscar 'models/' desde 'app/'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model_clasificador_v1.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer_v1.pkl')

# --- Función de Preprocesamiento ---
# Debe ser IDÉNTICA a la usada en entrenamiento y predicción
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^RT @\w+:\s?', 'RT : ', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r't\.co/\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Carga Global del Modelo ---
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Modelo y vectorizador cargados con éxito para la aplicación web.")
except Exception as e:
    print(f"❌ ERROR al cargar modelos: {e}")
    # Es crucial que la app no inicie si no puede cargar los modelos
    model = None
    vectorizer = None

# --- Rutas de la Aplicación Flask ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_text = ""

    if request.method == 'POST':
        input_text = request.form.get('tweet_text')
        
        if model and vectorizer and input_text:
            # 1. Limpieza
            cleaned_text = clean_text(input_text)
            
            # 2. Vectorización (Se espera un array/lista, por eso [cleaned_text])
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # 3. Predicción
            prediction = model.predict(text_vectorized)[0]
            
            # 4. Resultado a mostrar
            prediction_result = f"El tono detectado es: {prediction}"
        else:
            prediction_result = "Error: El modelo no está cargado o el texto está vacío."
    
    # Renderizamos la plantilla HTML, pasando los resultados
    return render_template('index.html', result=prediction_result, original_text=input_text)

if __name__ == '__main__':
    # Ejecuta la aplicación en modo desarrollo
    app.run(debug=True)