# Paso 1. Agregar dependencias del requirements.txt
- pip install  -r requirements.txt
# Paso 2. Instalar las dependencias restantes
- python -m spacy download es_core_news_sm

## Para que nltk.corpus.stopwords funcione, necesitas asegurarte de que las stopwords estén descargadas:
- import nltk nltk.download('stopwords')
# Ejecutar el codigo 
- main_version1.py: Version simplificada  
- main_version2.py: Version con un diccionario mas amplio 
