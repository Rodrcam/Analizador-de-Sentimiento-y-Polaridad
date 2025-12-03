import pandas as pd
import pickle
import re
import os
from flask import Flask, render_template, request

# --- Configuración de Flask ---
app = Flask(__name__, template_folder='templates')

# --- Rutas de Archivos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. Ruta de los Modelos (Subimos un nivel)
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model_clasificador_v1.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer_v1.pkl')

# 2. Ruta del Dataset (¡CORREGIDA a la ruta del corpus etiquetado!)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'corpus_etiquetado.csv')

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

# --- Carga Global del Modelo y Datos ---

# 1. Carga del Modelo
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Modelo y vectorizador cargados con éxito para la aplicación web.")
except Exception as e:
    print(f"❌ ERROR al cargar modelos: {e}")
    model = None
    vectorizer = None

# 2. Carga del Dataset de Tweets
try:
    # Cargar el CSV. Asume que la columna de texto se llama 'tweet_text'
    # Nota: Si la columna de texto tiene otro nombre en 'corpus_etiquetado.csv', cámbialo aquí.
    df_tweets = pd.read_csv(DATA_PATH) 
    
    # Asumimos que la columna se llama 'tweet_text' o 'text'. Usaremos 'text' para el ejemplo.
    # Si la columna se llama diferente en tu CSV real, por favor ajústalo.
    TEXT_COLUMN = 'texto_procesado' # <-- ¡Ajusta este nombre de columna si es necesario!
    
    # Aplicar la limpieza de texto a la columna de tweets para búsquedas
    df_tweets['cleaned_text'] = df_tweets[TEXT_COLUMN].apply(clean_text)
    
    # Preparamos una lista simple de los tweets originales para facilitar el retorno
    GLOBAL_TWEETS = df_tweets[[TEXT_COLUMN, 'cleaned_text']].rename(columns={TEXT_COLUMN: 'tweet_text'}).to_dict('records')
    print(f"✅ Dataset de tweets cargado y limpio. Total: {len(GLOBAL_TWEETS)} tweets.")
except Exception as e:
    print(f"❌ ERROR al cargar el dataset de tweets: {e}")
    GLOBAL_TWEETS = []


# --- Rutas de la Aplicación Flask ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Ruta principal para la clasificación de sentimiento."""
    prediction_result = None
    input_text = ""
    
    if request.method == 'POST' and 'tweet_text' in request.form:
        input_text = request.form.get('tweet_text')
        
        if model and vectorizer and input_text:
            # 1. Limpieza
            cleaned_text = clean_text(input_text)
            
            # 2. Vectorización
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # 3. Predicción
            prediction_label = model.predict(text_vectorized)[0]
            
            # 4. Resultado a mostrar
            # Nota: Asegúrate de que tu modelo devuelva 1, 0, -1 o la etiqueta de texto directamente
            prediction_result = f"El tono detectado es: {prediction_label}"
        else:
            prediction_result = "Error: El modelo no está cargado o el texto está vacío."
    
    # Renderizamos la plantilla HTML, pasando los resultados de predicción
    return render_template('index.html', result=prediction_result, original_text=input_text, search_results=None, search_query="")

@app.route('/search_tweets', methods=['POST'])
def search_tweets():
    """Nueva ruta para el buscador de tweets usando 're' sobre la base de datos."""
    search_query = request.form.get('search_query', '')
    matching_tweets = []
    
    if search_query and GLOBAL_TWEETS:
        # 1. Escapar la consulta para usarla de forma segura en una expresión regular (case-insensitive)
        pattern = re.compile(re.escape(search_query), re.IGNORECASE)
        
        # 2. Iterar sobre la base de datos de tweets
        for tweet_data in GLOBAL_TWEETS:
            # Buscar en el texto LIMPIO del tweet para resultados más precisos
            cleaned_text = tweet_data['cleaned_text']
            
            if pattern.search(cleaned_text):
                # Si hay coincidencia, añadimos el texto ORIGINAL a los resultados
                matching_tweets.append(tweet_data['tweet_text'])
    
    # Retornar a la plantilla principal, pasando los resultados de búsqueda
    return render_template('index.html', 
                           search_results=matching_tweets, 
                           search_query=search_query, 
                           result=None, 
                           original_text="")

if __name__ == '__main__':
    app.run(debug=True)
