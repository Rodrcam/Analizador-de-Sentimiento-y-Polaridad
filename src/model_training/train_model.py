# src/model_training/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Importamos LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import re

# --- Rutas de Archivos ---
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'corpus_etiquetado.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_clasificador_v1.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer_v1.pkl')
# -------------------------

# La función de limpieza debe ser la misma usada en todo el proyecto
def clean_text(text):
    """Función de preprocesamiento de texto."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'^RT @\w+:\s?', 'RT : ', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r't\.co/\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_model():
    """
    Carga los datos etiquetados, entrena un clasificador de texto (Logistic Regression) 
    con balanceo de clases, y guarda el modelo y el vectorizador.
    """
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Archivo de etiquetado no encontrado en {INPUT_FILE}.")
        return

    # 1. Cargar y Filtrar Datos Etiquetados
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=['etiqueta_tono', 'texto_original'], inplace=True)
    
    # Aseguramos que el texto_procesado exista y esté limpio
    df['texto_procesado'] = df['texto_original'].apply(clean_text)

    if len(df) < 50:
        print(f"ADVERTENCIA: Solo se han cargado {len(df)} tuits. Se recomienda un mínimo de 100 para estabilidad.")

    X = df['texto_procesado']  # Características (texto limpio)
    y = df['etiqueta_tono']    # Objetivo (tono)

    # 2. Dividir el conjunto de datos (Usaremos el 30% para prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nDatos de Entrenamiento: {len(X_train)} | Datos de Prueba: {len(X_test)}")
    
    # 3. Vectorización (TF-IDF)
    # Reutilizamos el parámetro max_features=500 y ngram_range=(1, 2)
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    
    # Ajustamos y transformamos los datos de entrenamiento
    X_train_vec = vectorizer.fit_transform(X_train)
    # Solo transformamos los datos de prueba
    X_test_vec = vectorizer.transform(X_test)

    # 4. Entrenamiento del Modelo (Logistic Regression con class_weight)
    print("Iniciando entrenamiento del modelo Logistic Regression con balanceo de clases...")
    
    # LogisticRegression soporta class_weight='balanced' para compensar el desequilibrio.
    model = LogisticRegression(
        class_weight='balanced',  # Parámetro clave para corregir el sesgo
        max_iter=1000, 
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    # 5. Evaluación del Modelo
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Precisión del modelo en el conjunto de prueba: {accuracy:.2f}")
    print("\n--- Informe de Clasificación (Detalle por etiqueta) ---")
    print(classification_report(y_test, y_pred))

    # 6. Guardar Modelo y Vectorizador
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

    print(f"\n🎉 ¡Entrenamiento Completo!")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Vectorizador guardado en: {VECTORIZER_PATH}")


if __name__ == '__main__':
    train_model()