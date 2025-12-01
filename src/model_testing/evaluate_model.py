# src/model_testing/evaluate_model.py

import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# --- Rutas de Archivos (DEBEN COINCIDIR con train_model.py) ---
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'corpus_etiquetado.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_clasificador_v1.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer_v1.pkl')

def evaluate_model():
    """
    Carga el modelo y el vectorizador entrenados, y evalúa su rendimiento
    en el conjunto de prueba (test set).
    """
    print("Iniciando la evaluación del modelo...")

    # 1. Cargar el modelo y el vectorizador
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Modelo y vectorizador cargados con éxito.")
    except FileNotFoundError:
        print("\n❌ ERROR: Asegúrate de que los archivos .pkl existan en la carpeta 'models'.")
        return

    # 2. Cargar y preparar datos (DEBE SER IDÉNTICO AL ENTRENAMIENTO)
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=['etiqueta_tono', 'texto_procesado'], inplace=True)
    X = df['texto_procesado']
    y = df['etiqueta_tono']

    # 3. Dividir el conjunto de datos (EL MISMO random_state y test_size)
    # Se debe replicar exactamente la división del entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Evaluando con {len(X_test)} ejemplos de prueba.")

    # 4. Vectorización del conjunto de prueba
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Predicción y Evaluación
    y_pred = model.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n========================================================")
    print(f"✅ Precisión General del Modelo en Test Set: {accuracy:.2f}")
    print("========================================================")
    print("\n--- INFORME DE CLASIFICACIÓN (Métricas Clave por Clase) ---")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    evaluate_model()