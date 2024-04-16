Python version: Python 3.11.8

# Estructura del Proyecto Fraud Detection

Este documento describe la estructura del repositorio `fraud_detection_test`, proporcionando una visión general de los directorios y archivos clave, con un enfoque especial en los directorios `src` y `nb`, que son cruciales para el proyecto.

## Detalles Adicionales

### Directorio `src`
El directorio `src` contiene varias clases y scripts que son fundamentales para el funcionamiento del proyecto:

- **`group_time_series_split.py`**: Proporciona una clase que ayuda a dividir conjuntos de datos de series temporales de forma que sea adecuada para la validación cruzada en el contexto temporal.
- **`pipeline.py`**: Este es el archivo más crucial, ya que gestiona el flujo de orquestación completo del proyecto. Todo el procesamiento de datos, la preparación, el entrenamiento del modelo y las evaluaciones se coordinan a través de este script.
- **`plot_utils.py`**: Incluye funciones para crear visualizaciones estandarizadas, útiles para análisis y presentación de resultados.
- **`dataframe_analyzer.py`**: Herramientas para realizar análisis exploratorios y diagnósticos en pandas DataFrames.

### Directorio `nb`
Este directorio alberga el Jupyter Notebook que proporciona un entorno interactivo para ejecutar código, visualizar resultados y realizar pruebas. El notebook **`fraud_detection.ipynb`** es especialmente importante, ya que responde directamente a las preguntas de la prueba técnica y documenta el flujo de trabajo que sustenta cada argumento.


## Estructura de Directorios y Archivos

```plaintext
fraud_detection_test/
├── data/                   # Contiene los conjuntos de datos utilizados en el proyecto.
├── mlflow-artifacts/       # Almacena los artefactos generados por MLflow durante el entrenamiento de modelos.
├── nb/                     # Directorio de Jupyter Notebooks utilizados para análisis exploratorios y pruebas.
│   └── fraud_detection.ipynb  # Notebook principal que responde a las preguntas de la prueba y muestra el flujo de trabajo.
├── src/                    # Scripts de código fuente que forman la lógica principal del proyecto.
│   ├── group_time_series_split.py    # Clase utilitaria para dividir series temporales en entrenamiento y prueba.
│   ├── pipeline.py                   # Clase principal que contiene el flujo de orquestación para el procesamiento de datos y modelado.
│   ├── pipeline_first_approach.py    # Enfoque inicial del pipeline, usado para versiones anteriores.
│   ├── plot_utils.py                 # Utilidades para generar gráficos y visualizaciones.
│   ├── __pycache__/                  # Carpeta generada por Python para almacenar bytecode compilado.
│   └── dataframe_analyzer.py         # Herramientas para analizar DataFrames.
├── .gitignore              # Especifica intencionalmente archivos no rastreados para ignorar.
├── evaluacion_tecnica.docx # Documento con la evaluación técnica del proyecto.
├── fraud_detection.txt     # Archivo de texto relacionado con detalles del proyecto.
├── mlflow.db               # Base de datos local utilizada por MLflow para registrar experimentos.
├── README.md               # Archivo Markdown con la descripción del proyecto.
└── requirements.txt        # Dependencias necesarias para ejecutar el proyecto.