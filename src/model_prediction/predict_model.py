import pandas as pd
import pickle
import re
import os

# --- Rutas de Archivos ---
# Las rutas deben coincidir con las usadas en el entrenamiento y los datos nuevos
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'nuevos_tweets.csv') # Archivo de entrada de tuits nuevos
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_clasificador_v1.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer_v1.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'predictions', 'tweets_clasificados.csv')
# -------------------------

def clean_text(text):
    """
    Función de preprocesamiento de texto.
    DEBE SER IDÉNTICA a la utilizada en el script de entrenamiento/preprocesamiento.
    """
    if not isinstance(text, str):
        return ""
    # 1. Eliminar RTs (adaptado para tu preprocesamiento)
    text = re.sub(r'^RT @\w+:\s?', 'RT : ', text)
    # 2. Eliminar menciones (@usuario)
    text = re.sub(r'@\w+', '', text)
    # 3. Eliminar URLs (http:// o https://)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 4. Eliminar el texto de la URL de twitter (t.co)
    text = re.sub(r't\.co/\w+', '', text)
    # 5. Eliminar saltos de línea y espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_new_data():
    """Carga el modelo y el vectorizador para clasificar tuits nuevos."""
    print("Iniciando la predicción de nuevos tuits...")

    # 1. Cargar el modelo y el vectorizador
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        print("Modelo y vectorizador cargados con éxito.")
    except FileNotFoundError:
        print("\n❌ ERROR: Faltan archivos clave.")
        print(f"Asegúrate de ejecutar 'train_model.py' y que los archivos .pkl existan en la carpeta 'models'.")
        print(f"Falta: {MODEL_PATH} y/o {VECTORIZER_PATH}")
        return

    # 2. Cargar los nuevos datos
    try:
        # Cargamos solo los datos crudos, sin etiquetas, para clasificarlos
        df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8')
        print(f"Cargados {len(df)} tuits para clasificar desde {RAW_DATA_PATH}")
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró el archivo de datos raw en: {RAW_DATA_PATH}")
        print("Asegúrate de crear 'nuevos_tweets.csv' con tuits no etiquetados en 'data/raw'.")
        return

    # 3. Preprocesar y Vectorizar el texto
    df['texto_procesado'] = df['texto_original'].apply(clean_text)
    
    # Transformamos el texto nuevo usando el vectorizador entrenado
    X_new = vectorizer.transform(df['texto_procesado'])
    print("Datos preprocesados y transformados (vectorizados).")

    # 4. Realizar la predicción
    predictions = model.predict(X_new)

    # 5. Guardar los resultados
    df['etiqueta_tono'] = predictions
    df['etiqueta_intencion'] = '' # La intención se deja vacía o se asigna por un modelo secundario
    
    # Asegurarse de que la carpeta de salida exista
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Seleccionar las columnas en el orden deseado para la salida
    output_df = df[['id_tuit', 'texto_original', 'texto_procesado', 'etiqueta_tono', 'etiqueta_intencion']]
    output_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')

    print(f"\n🎉 Predicciones completadas y guardadas en: {OUTPUT_PATH}")
    print("\nResumen de las clases de tono predichas:")
    print(df['etiqueta_tono'].value_counts())


if __name__ == "__main__":
    predict_new_data()