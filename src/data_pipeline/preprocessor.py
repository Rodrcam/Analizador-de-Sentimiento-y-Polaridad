import pandas as pd
import json
import re
import os

def clean_text(text):
    """
    Función de limpieza: elimina elementos ruidosos de Twitter.
    """
    # 1. Eliminar URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 2. Eliminar menciones de usuarios (@usuario)
    text = re.sub(r'@\w+', '', text)
    # 3. Eliminar Hashtags (#hashtag), pero manteniendo el texto si es deseable
    # Aquí elegimos eliminar la marca #, manteniendo el texto del hashtag
    text = re.sub(r'#', '', text)
    # 4. Eliminar saltos de línea y espacios extra
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Eliminar emojis (método básico, puede ser más complejo)
    # Este regex elimina la mayoría de los caracteres que no son letras, números o puntuación común
    # Opcional: Podrías querer conservar la puntuación.
    
    return text

def preprocess_data(input_filename='tweets_raw_ES.json', output_filename='corpus_etiquetado.csv'):
    
    input_path = os.path.join('data', 'raw', input_filename)
    output_path = os.path.join('data', 'processed', output_filename)
    
    if not os.path.exists(input_path):
        print(f"ERROR: Archivo de entrada no encontrado en {input_path}. Ejecuta collector.py primero.")
        return

    # 1. Cargar datos brutos
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Cargados {len(df)} tuits para pre-procesamiento.")

    # 2. Aplicar limpieza
    df['texto_procesado'] = df['texto_original'].apply(clean_text)

    # 3. Preparar columnas para etiquetado (HUMANO)
    # Estas son las columnas que tú o un equipo deberán llenar manualmente.
    df['etiqueta_tono'] = ''         # Positivo / Negativo
    df['etiqueta_intencion'] = ''    # Crítica_Destructiva / Elogio / Neutral
    
    # Seleccionar solo las columnas necesarias para el etiquetado manual
    df_export = df[['id_tuit', 'texto_original', 'texto_procesado', 'etiqueta_tono', 'etiqueta_intencion']]

    # 4. Guardar el archivo para etiquetado manual (en formato CSV, ideal para Excel/Hojas de cálculo)
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    df_export.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Pre-procesamiento completado. Archivo para etiquetado guardado en: {output_path}")
    print("Siguiente paso: Abrir el archivo CSV y rellenar las columnas 'etiqueta_tono' e 'etiqueta_intencion'.")


if __name__ == '__main__':
    preprocess_data()