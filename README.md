📊 Analizador de Sentimiento y Polaridad (NLP)
Este proyecto implementa un clasificador de texto en español utilizando técnicas de Procesamiento de Lenguaje Natural (NLP) y Machine Learning. El objetivo es determinar la polaridad emocional (Positivo, Negativo o Neutro) de una frase o comentario.
La aplicación se sirve mediante un micro-framework web (Flask) para ofrecer una interfaz intuitiva con un diseño estilo buscador.
🎯 Objetivo Principal
El modelo fue entrenado específicamente para resolver un problema de desequilibrio de clases, donde la clase "Negativa" era dominante. Mediante la implementación de la ponderación de clases (class_weight='balanced'), se ha logrado una alta precisión y, crucialmente, un Recall equilibrado en las clases minoritarias (Positivo y Neutro).
🚀 Tecnologías
Python 3.x
Machine Learning: scikit-learn (Regresión Logística y TfidfVectorizer)
Web: Flask
⚙️ Instalación del Proyecto
Sigue estos pasos para configurar el entorno virtual e instalar todas las dependencias necesarias.

1. Crear y Activar el Entorno Virtual
Recomendamos usar un entorno virtual para aislar las dependencias:
# Crear el entorno virtual (solo la primera vez)
python -m venv .venv

# Activar el entorno (Windows)
.\.venv\Scripts\activate.ps1

# Activar el entorno (Linux/macOS)
source .venv/bin/activate


2. Instalar Dependencias
Asegúrate de que estás en el entorno virtual ((.venv)) y ejecuta:
pip install -r requirements.txt


🛠️ Uso del Proyecto
El proyecto está dividido en tres fases principales: entrenamiento, evaluación y servicio web.
1. (Opcional) Re-Entrenar el Modelo
Si deseas re-entrenar el modelo con la configuración actual (incluyendo el balanceo de clases), utiliza el script de entrenamiento:
python src/model_training/train_model.py


Este comando genera y guarda los archivos model.pkl y vectorizer.pkl en la carpeta artifacts/.
2. (Opcional) Evaluar el Rendimiento
Para verificar el rendimiento del modelo sobre el conjunto de prueba y obtener el informe de clasificación (Precision, Recall, F1-Score):
python src/model_testing/evaluate_model.py


El resultado mostrará cómo la ponderación de clases mejoró el Recall de las clases Positivo y Neutro.
3. Ejecutar la Aplicación Web (Servicio)
El script principal de Flask carga el modelo entrenado (artifacts/model.pkl) y el vectorizador, y lo expone a través de una interfaz web.
python app/app.py


Una vez que el servidor se inicie, accede a la aplicación desde tu navegador:
➡️ Acceso: http://127.0.0.1:5000
📂 Estructura del Proyecto
.
├── app/
│   ├── app.py              # Lógica del servidor Flask y predicción.
│   └── templates/
│       └── index.html      # Interfaz web (HTML, CSS, Jinja2).
├── artifacts/
│   ├── model.pkl           # Modelo de Regresión Logística ya entrenado.
│   └── vectorizer.pkl      # Objeto TfidfVectorizer (vocabulario y pesos).
├── src/
│   ├── data_cleaning/      # (No implementado) scripts de limpieza.
│   ├── model_testing/
│   │   └── evaluate_model.py # Script para evaluar métricas.
│   └── model_training/
│       └── train_model.py  # Script para entrenar y guardar el modelo.
└── requirements.txt        # Dependencias de Python.
