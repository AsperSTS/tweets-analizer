import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import csv
import re
import string
import spacy
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import seaborn as sns
from datetime import datetime

from config import *

# Load Spanish NLP model
nlp = spacy.load("es_core_news_sm")
# Ensure stopwords are available
stop_words = set(stopwords.words('spanish'))

# Additional stopwords for tweets
STOP_WORDS_ADICIONALES = {
    # Basics and common expressions
    'rt', 'vs', 'jaja', 'jajaja', 'jajajaja', 'ja', 'je', 'ji', 'hahaha', '6de6',
    
    # Auxiliary/modal verbs
    'poder', 'querer', 'deber', 'hacer', 'decir', 'saber', 'creer', 'mirar', 
    'conocer', 'tratar', 'seguir', 'poner', 'llamar', 'continuar', 'esperar',
    
    # Temporal adverbs
    'hoy', 'día', 'año', 'mes', 'siempre', 'ahora', 'tiempo', 'momento',
    
    # Courtesy expressions and communication
    'gracias', 'agradecer', 'felicitar', 'saludar', 'desear', 'celebrar',
    
    # Common uninformative adjectives
    'buen', 'bueno', 'gran', 'grande', 'nuevo', 'solo', 'mayor', 'principal',
    'único', 'normal', 'claro', 'positivo',
    
    # Pronouns and demonstratives
    'este', 'ese', 'aquel', 'él', 'usted', 'tal', 'mío', 'tuyo', 'suyo',
    
    # Common Mexican political names with little specific semantic meaning
    'país', 'méxico', 'presidente', 'gobierno', 'político',
    
    # Additional connectors
    'pues', 'asi', 'aunque', 'mientras', 'entonces',
    
    # Other frequent terms in the corpus
    'estar', 'ser', 'haber', 'tener', 'ir', 'pre', 'ver', 'dar', 'así',
    
    # Additional auxiliary verbs and connectors
    'hacer', 'poder', 'querer', 'decir', 'saber', 'mirar', 'creer',
    'seguir', 'continuar', 'resultar', 'quedar', 'parecer',
    
    # More pronouns and demonstratives
    'todo', 'nada', 'algo', 'alguien', 'nadie',
    
    # Common adverbs
    'bien', 'mal', 'solo', 'muy', 'más', 'menos', 'ahora', 'siempre', 'nunca',
    'ahí', 'aquí', 'allí', 'entonces', 'después', 'antes',
    
    # Generic adjectives
    'buen', 'gran', 'mayor', 'menor', 'mejor', 'peor', 'primero', 'último',
    
    # Temporal words
    'día', 'mes', 'año', 'tiempo', 'hora', 'rato', 'momento',
    
    # Mexican colloquialisms
    'wey', 'chingar', 'madre', 'oro', 'vivo',
    
    # Specific words in the political context that do not provide semantic value
    'persona', 'país', 'gente', 'cosa', 'punto', 'modo', 'vez',
    
    # Social network words
    'tweet', 'tuitear', 'jajajajaja', 'hahahaha',
    
    # Generic political words
    'político', 'gobierno', 'presidente', 'votar', 'elección', 'campaña',
    'democracia', 'reunión', 'desarrollo', 'proyecto', 'cambio',
    'mensaje', 'administración', 'opinión', 'palabra', 'trabajo', 'apoyo'
}

# Update stopwords with additional ones
stop_words.update(STOP_WORDS_ADICIONALES)

# Dictionaries for positive and negative words
PALABRAS_POSITIVAS = set([
    # Positively charged political words
    'excelente', 'victoria', 'triunfo', 'ganar', 'mejor', 'apoyo', 'progreso',
    'avance', 'logro', 'éxito', 'beneficio', 'favorable', 'acuerdo', 'solución',
    'confianza', 'transparencia', 'honesto', 'justo', 'eficiente', 'respeto',
    'compromiso', 'esperanza', 'oportunidad', 'seguridad', 'bienestar', 'prosperidad',
    'acierto', 'defender', 'verdad', 'representar', 'construir', 'competente',
    'capacitado', 'responsable', 'líder', 'visionario', 'convincente', 'prometedor',
    'reconciliación', 'diálogo', 'participación', 'democrático', 'inclusivo',
    'satisfacción', 'celebrar', 'impulsar', 'fortalecer', 'consolidar',
    'estabilidad', 'crecimiento', 'dedicación', 'honestidad', 'honorable',
    'integridad', 'coherente', 'auténtico', 'legítimo', 'admiración', 'credibilidad',
    'congruente', 'ejemplar', 'respaldo', 'unidad', 'victoria', 'honor',
    'competitividad', 'adaptabilidad', 'visión', 'apertura'
])

PALABRAS_NEGATIVAS = set([
    # Negatively charged political words
    'corrupto', 'fraude', 'mentira', 'fracaso', 'engaño', 'escándalo', 'nepotismo',
    'impunidad', 'incompetente', 'traición', 'abuso', 'manipulación', 'autoritario',
    'dictadura', 'derrota', 'debacle', 'decadencia', 'crisis', 'conflicto', 'negligencia',
    'amiguismo', 'clientelismo', 'despilfarro', 'ineficiente', 'ilegal', 'ilícito',
    'irregularidad', 'imposición', 'controversial', 'polémico', 'cuestionable',
    'incapaz', 'hipócrita', 'deshonesto', 'inconsistente', 'injusto', 'inepto',
    'desconfianza', 'inseguridad', 'estancamiento', 'retroceso', 'deterioro',
    'privilegio', 'desigualdad', 'represión', 'censura', 'intransigencia',
    'polarización', 'radicalización', 'populismo', 'demagogia', 'falso', 'falsedad',
    'descontento', 'indignación', 'protesta', 'rechazo', 'repudio', 'oposición',
    'discordia', 'confrontación', 'disputa', 'contradicción', 'inestabilidad',
    'peligroso', 'amenaza', 'riesgo', 'violencia', 'opacidad', 'oscuro'
])

def cargar_diccionario_personalizado():
    """
    Intenta cargar diccionarios personalizados desde archivos si existen,
    si no, utiliza los valores predefinidos
    """
    try:
        palabras_positivas = set()
        palabras_negativas = set()
        
        # Intenta cargar diccionario de palabras positivas
        if os.path.exists('diccionario_positivo.txt'):
            with open('diccionario_positivo.txt', 'r', encoding='utf-8') as f:
                for linea in f:
                    palabra = linea.strip()
                    if palabra:
                        palabras_positivas.add(palabra)
        else:
            palabras_positivas = PALABRAS_POSITIVAS
            
        # Intenta cargar diccionario de palabras negativas
        if os.path.exists('diccionario_negativo.txt'):
            with open('diccionario_negativo.txt', 'r', encoding='utf-8') as f:
                for linea in f:
                    palabra = linea.strip()
                    if palabra:
                        palabras_negativas.add(palabra)
        else:
            palabras_negativas = PALABRAS_NEGATIVAS
            
        return palabras_positivas, palabras_negativas
    
    except Exception as e:
        print(f"Error al cargar diccionarios personalizados: {e}")
        return PALABRAS_POSITIVAS, PALABRAS_NEGATIVAS

def guardar_diccionario_global(diccionario, nombre_archivo="diccionario_global.txt"):
    """
    Guarda el desglose de palabras en un archivo de texto
    
    Args:
        diccionario (dict): Diccionario con pares palabra:significado
        nombre_archivo (str): Nombre del archivo donde guardar el diccionario
    """
    try:
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            for palabra, significado in sorted(diccionario.items()):
                f.write(f"{palabra} : {significado}\n")
        print(f"Diccionario guardado en {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar diccionario global: {e}")

def procesar_tweet(tweet):
    """
    Extrae el nombre de usuario del retweet, lematiza el tweet en español,
    elimina stopwords, signos de puntuación, menciones, números y caracteres extraños.
    
    Args:
        tweet (str): El texto del tweet a procesar
        
    Returns:
        tuple: (username, tokens_lematizados) donde username es el nombre de usuario 
               del retweet (o None si no es un retweet) y tokens_lematizados es 
               una lista de palabras procesadas
    """
    # Verificar si tweet es una Serie de pandas u objeto string
    if hasattr(tweet, 'iloc'):
        # Si es una Serie, toma el primer elemento
        tweet = tweet.iloc[0] if len(tweet) > 0 else ""
    elif not isinstance(tweet, str):
        # Si no es string ni Serie, conviértelo a string
        tweet = str(tweet)
    
    # Extraer nombre de usuario si es un retweet
    username = None
    rt_match = re.search(r'RT @(\w+):', tweet)
    if rt_match:
        username = rt_match.group(1)
        tweet_sin_rt = tweet[rt_match.end():].strip()
    else:
        tweet_sin_rt = tweet
    
    # Eliminar URLs
    tweet_sin_urls = re.sub(r'https?://\S+|www\.\S+', '', tweet_sin_rt)
    
    # Eliminar menciones (@usuario)
    tweet_sin_menciones = re.sub(r'@\w+', '', tweet_sin_urls)
    
    # Eliminar hashtags
    tweet_sin_hashtags = re.sub(r'#\w+', '', tweet_sin_menciones)
    
    # Eliminar caracteres Unicode extraños
    tweet_limpio = re.sub(r'\\u[0-9a-fA-F]{4}', '', tweet_sin_hashtags)
    
    # Eliminar números
    tweet_sin_numeros = re.sub(r'\b\d+\b', '', tweet_limpio)
    
    # Eliminar puntos suspensivos y caracteres sueltos
    tweet_sin_puntos = re.sub(r'\.{2,}', '', tweet_sin_numeros)
    
    # Eliminar símbolos comunes en tweets
    tweet_sin_simbolos = re.sub(r'[""''…]', '', tweet_sin_puntos)
    
    # Tokenización y lematización con spaCy
    doc = nlp(tweet_sin_simbolos.lower())
    
    # Procesar tokens: eliminar puntuación, stopwords, números y lematizar
    tokens_procesados = []
    for token in doc:
        # Verificar que no sea puntuación, stopword, número o mención
        if (token.text not in string.punctuation and 
            token.text not in stop_words and 
            token.lemma_ not in stop_words and
            not token.is_space and 
            len(token.text) > 2 and  # Ignorar caracteres sueltos
            not token.like_num and   # Ignorar números
            not token.text.startswith('@') and  # Ignorar menciones
            not token.text.startswith('#') and  # Ignorar hashtags
            not re.match(r'^[_\W]+$', token.text) and  # Ignorar tokens que solo contienen caracteres especiales
            not token.lemma_ == '-PRON-'):  # Ignorar pronombres marcados por spaCy
            
            tokens_procesados.append(token.lemma_)  # Usar el lema en lugar del token original
    
    if username is None:
        username = "Anonimo"
    
    return username, tokens_procesados

def seleccionar_columna_tweets_esp(dataframe):
    """
    Selecciona la primera columna de un dataframe que contiene tweets
    
    Args:
        dataframe (pd.DataFrame): DataFrame que contiene los tweets
        
    Returns:
        pd.DataFrame: DataFrame con solo la primera columna seleccionada
    """
    # Guardar el nombre de la primera columna
    nombre_columna_original = dataframe.columns[0]
    
    # Seleccionar solo la primera columna
    df_resultado = dataframe.iloc[:, [0]].copy()
    
    # Añadir el nombre de la columna original como primera fila
    df_resultado.loc[-1] = [nombre_columna_original]
    df_resultado.index = df_resultado.index + 1
    df_resultado = df_resultado.sort_index()
    
    # Renombrar la columna
    df_resultado = df_resultado.rename(columns={nombre_columna_original: 'columna_1'})
    
    return df_resultado

def analizar_sentimiento_tweet(palabras_procesadas, palabras_positivas, palabras_negativas):
    """
    Analiza el sentimiento de un tweet basado en las palabras positivas y negativas
    
    Args:
        palabras_procesadas (list): Lista de palabras procesadas del tweet
        palabras_positivas (set): Conjunto de palabras positivas para comparar
        palabras_negativas (set): Conjunto de palabras negativas para comparar
        
    Returns:
        str: 'positivo', 'negativo' o 'neutro' según el análisis
        list: Palabras positivas encontradas
        list: Palabras negativas encontradas
    """
    # Contar palabras positivas y negativas
    palabras_pos_encontradas = [p for p in palabras_procesadas if p in palabras_positivas]
    palabras_neg_encontradas = [p for p in palabras_procesadas if p in palabras_negativas]
    
    # Determinar sentimiento basado en cantidad de palabras
    if len(palabras_pos_encontradas) > len(palabras_neg_encontradas):
        return 'positivo', palabras_pos_encontradas, palabras_neg_encontradas
    elif len(palabras_neg_encontradas) > len(palabras_pos_encontradas):
        return 'negativo', palabras_pos_encontradas, palabras_neg_encontradas
    else:
        if len(palabras_pos_encontradas) > 0 or len(palabras_neg_encontradas) > 0:
            # Si hay igual cantidad de palabras positivas y negativas (pero no cero)
            return 'neutro', palabras_pos_encontradas, palabras_neg_encontradas
        else:
            # Si no hay palabras positivas ni negativas
            return 'sin_sentimiento', [], []

def procesar_dataframe_tweets(dataframe, palabras_positivas, palabras_negativas):
    """
    Procesa todos los tweets en un DataFrame y los analiza
    
    Args:
        dataframe (pd.DataFrame): DataFrame con tweets
        palabras_positivas (set): Conjunto de palabras positivas
        palabras_negativas (set): Conjunto de palabras negativas
        
    Returns:
        dict: Diccionario con análisis de los tweets
    """
    # Preparar estructura para almacenar resultados
    resultados = {
        'usuarios': [],
        'palabras_procesadas': [],
        'sentimientos': [],
        'palabras_positivas': [],
        'palabras_negativas': []
    }
    
    # Procesar todas las filas del DataFrame
    dataframe_tmp = seleccionar_columna_tweets_esp(dataframe)
    
    # Saltar la primera fila que es el nombre de la columna
    for index, fila in dataframe_tmp[1:].iterrows():
        try:
            tweet_text = fila['columna_1']
            usuario, palabras_procesadas = procesar_tweet(tweet_text)
            
            # Filtrar solo palabras relevantes (sin menciones, sin números)
            palabras_filtradas = [palabra for palabra in palabras_procesadas 
                                if not palabra.startswith('@') and not re.match(r'^\d+$', palabra)]
            
            # Analizar sentimiento
            sentimiento, palabras_pos, palabras_neg = analizar_sentimiento_tweet(
                palabras_filtradas, palabras_positivas, palabras_negativas
            )
            
            # Guardar resultados
            resultados['usuarios'].append(usuario)
            resultados['palabras_procesadas'].append(palabras_filtradas)
            resultados['sentimientos'].append(sentimiento)
            resultados['palabras_positivas'].append(palabras_pos)
            resultados['palabras_negativas'].append(palabras_neg)
            
        except Exception as e:
            print(f"Error procesando fila {index}: {e}")
    
    return resultados

def guardar_palabras_por_sentimiento(directorio, tipo, resultados):
    """
    Guarda las palabras positivas o negativas de cada usuario en un archivo CSV
    
    Args:
        directorio (str): Directorio donde guardar el archivo
        tipo (str): 'Positivo' o 'Negativo'
        resultados (dict): Diccionario con resultados del análisis
    """
    try:
        # Crear nombre de archivo según el tipo
        nombre_archivo = os.path.join(directorio, f"palabras{tipo}s.txt")
        
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            # Escribir encabezado
            f.write("Usuario,Palabras\n")
            
            # Determinar qué conjunto de palabras usar
            indice_palabras = 'palabras_positivas' if tipo == 'Positivo' else 'palabras_negativas'
            
            # Escribir cada usuario con sus palabras
            for i, usuario in enumerate(resultados['usuarios']):
                palabras = resultados[indice_palabras][i]
                if palabras:  # Solo escribir si hay palabras
                    f.write(f"{usuario},{','.join(palabras)}\n")
        
        print(f"Archivo de palabras {tipo}s guardado en {nombre_archivo}")
    except Exception as e:
        print(f"Error al guardar palabras {tipo}s: {e}")

def crear_diccionario_palabras(resultados):
    """
    Crea un diccionario global de palabras con su significado (positivo/negativo)
    
    Args:
        resultados (dict): Diccionario con resultados del análisis
        
    Returns:
        dict: Diccionario con pares palabra:significado
    """
    diccionario_global = {}
    
    # Procesar palabras positivas
    for lista_palabras in resultados['palabras_positivas']:
        for palabra in lista_palabras:
            if palabra not in diccionario_global:
                diccionario_global[palabra] = "positivo"
    
    # Procesar palabras negativas
    for lista_palabras in resultados['palabras_negativas']:
        for palabra in lista_palabras:
            if palabra not in diccionario_global:
                diccionario_global[palabra] = "negativo"
            elif diccionario_global[palabra] == "positivo":
                # Si ya estaba como positiva, marcar como ambigua
                diccionario_global[palabra] = "ambiguo"
    
    return diccionario_global

def analizar_partido(partido, rutas_csv, palabras_positivas, palabras_negativas):
    """
    Analiza todos los tweets de un partido y genera estadísticas
    
    Args:
        partido (str): Nombre del partido político
        rutas_csv (list): Lista de rutas a archivos CSV del partido
        palabras_positivas (set): Conjunto de palabras positivas
        palabras_negativas (set): Conjunto de palabras negativas
        
    Returns:
        dict: Estadísticas del partido
    """
    stats = {
        'total_tweets': 0,
        'tweets_positivos': 0,
        'tweets_negativos': 0,
        'tweets_neutros': 0,
        'tweets_sin_sentimiento': 0,
        'palabras_mas_frecuentes': Counter(),
        'palabras_positivas_frecuentes': Counter(),
        'palabras_negativas_frecuentes': Counter(),
        'tendencia_temporal': defaultdict(lambda: {'pos': 0, 'neg': 0, 'neu': 0, 'sin': 0}),
    }
    
    for ruta_csv in rutas_csv:
        try:
            # Extraer fecha del directorio
            partes_ruta = ruta_csv.split(os.path.sep)
            fecha_str = None
            
            # Buscar un patrón de fecha en las partes de la ruta
            for parte in partes_ruta:
                # Verificar si la parte tiene formato de fecha (DDMMYY)
                if re.match(r'\d{6}', parte):
                    fecha_str = parte
                    break
            
            # Si no se encontró un formato de fecha directo, buscar en el nombre del directorio
            if not fecha_str and len(partes_ruta) > 1:
                directorio = partes_ruta[-2]  # Directorio padre
                # Buscar un patrón numérico que podría ser una fecha
                match = re.search(r'(\d{2})(\d{2})(\d{2})', directorio)
                if match:
                    fecha_str = match.group(0)
            
            # Cargar el CSV
            df = pd.read_csv(ruta_csv)
            
            # Procesar los tweets
            resultados = procesar_dataframe_tweets(df, palabras_positivas, palabras_negativas)
            
            # Actualizar estadísticas
            num_tweets = len(resultados['usuarios'])
            stats['total_tweets'] += num_tweets
            
            # Contar sentimientos
            contador_sentimientos = Counter(resultados['sentimientos'])
            stats['tweets_positivos'] += contador_sentimientos.get('positivo', 0)
            stats['tweets_negativos'] += contador_sentimientos.get('negativo', 0)
            stats['tweets_neutros'] += contador_sentimientos.get('neutro', 0)
            stats['tweets_sin_sentimiento'] += contador_sentimientos.get('sin_sentimiento', 0)
            
            # Palabras más frecuentes
            todas_palabras = [palabra for lista in resultados['palabras_procesadas'] for palabra in lista]
            stats['palabras_mas_frecuentes'].update(todas_palabras)
            
            # Palabras positivas y negativas más frecuentes
            palabras_pos = [palabra for lista in resultados['palabras_positivas'] for palabra in lista]
            palabras_neg = [palabra for lista in resultados['palabras_negativas'] for palabra in lista]
            stats['palabras_positivas_frecuentes'].update(palabras_pos)
            stats['palabras_negativas_frecuentes'].update(palabras_neg)
            
            # Tendencia temporal
            if fecha_str:
                try:
                    # Intentar interpretar la fecha
                    fecha_obj = datetime.strptime(fecha_str, '%d%m%y')
                    mes = fecha_obj.strftime('%B')  # Nombre del mes
                    
                    # Actualizar estadísticas por mes
                    stats['tendencia_temporal'][mes]['pos'] += contador_sentimientos.get('positivo', 0)
                    stats['tendencia_temporal'][mes]['neg'] += contador_sentimientos.get('negativo', 0)
                    stats['tendencia_temporal'][mes]['neu'] += contador_sentimientos.get('neutro', 0)
                    stats['tendencia_temporal'][mes]['sin'] += contador_sentimientos.get('sin_sentimiento', 0)
                except ValueError:
                    # Si no se puede interpretar la fecha, usar la ruta como clave
                    stats['tendencia_temporal'][ruta_csv]['pos'] += contador_sentimientos.get('positivo', 0)
                    stats['tendencia_temporal'][ruta_csv]['neg'] += contador_sentimientos.get('negativo', 0)
                    stats['tendencia_temporal'][ruta_csv]['neu'] += contador_sentimientos.get('neutro', 0)
                    stats['tendencia_temporal'][ruta_csv]['sin'] += contador_sentimientos.get('sin_sentimiento', 0)
            
            # Guardar palabras en el directorio correspondiente
            directorio = os.path.dirname(ruta_csv)
            guardar_palabras_por_sentimiento(directorio, 'Positivo', resultados)
            guardar_palabras_por_sentimiento(directorio, 'Negativo', resultados)
            
            # Crear diccionario de palabras y guardarlo
            diccionario = crear_diccionario_palabras(resultados)
            guardar_diccionario_global(diccionario, f"{directorio}/diccionario_palabras.txt")
            
        except Exception as e:
            print(f"Error al procesar {ruta_csv}: {e}")
    
    return stats

def crear_graficas_partido(partido, stats, directorio_salida="graficas"):
    """
    Crea gráficas de análisis para un partido político
    
    Args:
        partido (str): Nombre del partido
        stats (dict): Estadísticas del partido
        directorio_salida (str): Directorio donde guardar las gráficas
    """
    # Crear directorio de salida si no existe
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    try:
        # 1. Gráfico de pastel de sentimientos
        plt.figure(figsize=(10, 6))
        labels = ['Positivos', 'Negativos', 'Neutros', 'Sin sentimiento']
        sizes = [
            stats['tweets_positivos'],
            stats['tweets_negativos'],
            stats['tweets_neutros'],
            stats['tweets_sin_sentimiento']
        ]
        colors = ['#66b3ff', '#ff9999', '#99ff99', '#ffcc99']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'Distribución de sentimientos en tweets sobre {partido}')
        plt.savefig(f"{directorio_salida}/{partido}_sentimientos_pie.png")
        plt.close()
        
        # 2. Gráfico de barras de palabras más frecuentes
        plt.figure(figsize=(12, 8))
        palabras_frecuentes = stats['palabras_mas_frecuentes'].most_common(15)
        palabras = [p[0] for p in palabras_frecuentes]
        frecuencias = [p[1] for p in palabras_frecuentes]
        
        plt.bar(palabras, frecuencias, color='#66b3ff')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Palabras')
        plt.ylabel('Frecuencia')
        plt.title(f'Palabras más frecuentes en tweets sobre {partido}')
        plt.tight_layout()
        plt.savefig(f"{directorio_salida}/{partido}_palabras_frecuentes.png")
        plt.close()
        
        # 3. Gráfico de barras comparativo para palabras positivas y negativas
        plt.figure(figsize=(14, 8))
        
        palabras_pos = stats['palabras_positivas_frecuentes'].most_common(10)
        palabras_neg = stats['palabras_negativas_frecuentes'].most_common(10)
        
        # Crear un DataFrame para facilitar la visualización
        df_palabras = pd.DataFrame({
            'Palabra': [p[0] for p in palabras_pos],
            'Frecuencia': [p[1] for p in palabras_pos],
            'Tipo': ['Positiva'] * len(palabras_pos)
        })
        
        df_temp = pd.DataFrame({
            'Palabra': [p[0] for p in palabras_neg],
            'Frecuencia': [p[1] for p in palabras_neg],
            'Tipo': ['Negativa'] * len(palabras_neg)
        })
        
        df_palabras = pd.concat([df_palabras, df_temp])
        
        # Crear gráfico con seaborn
        sns.barplot(x='Palabra', y='Frecuencia', hue='Tipo', data=df_palabras, palette=['#66b3ff', '#ff9999'])
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Palabras positivas y negativas más frecuentes - {partido}')
        plt.tight_layout()
        plt.savefig(f"{directorio_salida}/{partido}_palabras_pos_neg.png")
        plt.close()
        
        # 4. Gráfico de tendencia temporal
        if stats['tendencia_temporal']:
            plt.figure(figsize=(12, 7))
            
            meses = list(stats['tendencia_temporal'].keys())
            meses.sort()  # Ordenar los meses cronológicamente
            
            pos_vals = [stats['tendencia_temporal'][mes]['pos'] for mes in meses]
            neg_vals = [stats['tendencia_temporal'][mes]['neg'] for mes in meses]
            neu_vals = [stats['tendencia_temporal'][mes]['neu'] for mes in meses]
            
            x = np.arange(len(meses))
            width = 0.25
            
            plt.bar(x - width, pos_vals, width, label='Positivos', color='#66b3ff')
            plt.bar(x, neg_vals, width, label='Negativos', color='#ff9999')
            plt.bar(x + width, neu_vals, width, label='Neutros', color='#99ff99')
            
            plt.xlabel('Mes')
            plt.ylabel('Cantidad de tweets')
            plt.title(f'Tendencia de sentimientos por mes - {partido}')
            plt.xticks(x, meses)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{directorio_salida}/{partido}_tendencia_temporal.png")
            plt.close()
        
        print(f"Gráficas para {partido} guardadas en {directorio_salida}")
    except Exception as e:
        print(f"Error al crear gráficas para {partido}: {e}")

def calcular_probabilidad_victoria(stats_partidos):
    """
    Calcula una probabilidad simplificada de victoria basada en el análisis de sentimientos
    
    Args:
        stats_partidos (dict): Diccionario con estadísticas de todos los partidos
        
    Returns:
        dict: Probabilidades de victoria para cada partido
    """
    probabilidades = {}
    total_positivos = sum(stats['tweets_positivos'] for stats in stats_partidos.values())
    total_negativos = sum(stats['tweets_negativos'] for stats in stats_partidos.values())
    
    for partido, stats in stats_partidos.items():
        # Fórmula simplificada: (Positivos_partido / Total_positivos) - (Negativos_partido / Total_negativos)
        if total_positivos > 0 and total_negativos > 0:
            ratio_positivo = stats['tweets_positivos'] / total_positivos if total_positivos > 0 else 0
            ratio_negativo = stats['tweets_negativos'] / total_negativos if total_negativos > 0 else 0
            
            # Calcular un score combinado (entre -1 y 1)
            score = ratio_positivo - ratio_negativo
            
            # Convertir a probabilidad (0 a 100%)
            probabilidades[partido] = max(0, min(100, (score + 1) * 50))
        else:
            probabilidades[partido] = 0
    
    return probabilidades

def crear_grafica_probabilidades(probabilidades, directorio_salida="graficas"):
    """
    Crea una gráfica con las probabilidades de victoria de cada partido
    
    Args:
        probabilidades (dict): Diccionario con probabilidades por partido
        directorio_salida (str): Directorio donde guardar la gráfica
    """
    try:
        # Ordenar partidos por probabilidad
        partidos_ordenados = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)
        partidos = [p[0] for p in partidos_ordenados]
        probs = [p[1] for p in partidos_ordenados]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(partidos, probs, color=plt.cm.viridis(np.linspace(0, 1, len(partidos))))
        
        # Añadir etiquetas con porcentajes
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.xlabel('Partidos Políticos')
        plt.ylabel('Probabilidad de Victoria (%)')
        plt.title('Probabilidad de Victoria por Partido Político Basada en Análisis de Sentimientos')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.tight_layout()
        
        plt.savefig(f"{directorio_salida}/probabilidades_victoria.png")
        plt.close()
        
        print(f"Gráfica de probabilidades guardada en {directorio_salida}/probabilidades_victoria.png")
    except Exception as e:
        print(f"Error al crear gráfica de probabilidades: {e}")

def crear_reporte_global(stats_partidos, probabilidades, directorio_salida="reportes"):
    """
    Crea un reporte global con los resultados del análisis
    
    Args:
        stats_partidos (dict): Estadísticas de todos los partidos
        probabilidades (dict): Probabilidades de victoria
        directorio_salida (str): Directorio donde guardar el reporte
    """
    # Crear directorio si no existe
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        
    try:
        # Ordenar partidos por probabilidad
        partidos_ordenados = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)
        
        with open(f"{directorio_salida}/reporte_global.txt", 'w', encoding='utf-8') as f:
            # Encabezado
            f.write("=" * 80 + "\n")
            f.write("ANÁLISIS DE OPINIÓN PÚBLICA EN TWITTER SOBRE PARTIDOS POLÍTICOS\n")
            f.write("=" * 80 + "\n\n")
            
            # Fecha de generación
            f.write(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            
            # Resumen de probabilidades
            f.write("PROBABILIDADES DE VICTORIA\n")
            f.write("-" * 40 + "\n")
            for partido, prob in partidos_ordenados:
                f.write(f"{partido}: {prob:.2f}%\n")
            f.write("\n")
            
            # Partido con mayor probabilidad
            ganador = partidos_ordenados[0][0]
            prob_ganador = partidos_ordenados[0][1]
            f.write(f"El partido con mayor probabilidad de victoria es: {ganador} con {prob_ganador:.2f}%\n\n")
            
            # Información detallada por partido
            f.write("INFORMACIÓN DETALLADA POR PARTIDO\n")
            f.write("=" * 80 + "\n\n")
            
            for partido, stats in stats_partidos.items():
                f.write(f"PARTIDO: {partido}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total de tweets analizados: {stats['total_tweets']}\n")
                f.write(f"Tweets positivos: {stats['tweets_positivos']} ({stats['tweets_positivos']/stats['total_tweets']*100:.1f}% del total)\n")
                f.write(f"Tweets negativos: {stats['tweets_negativos']} ({stats['tweets_negativos']/stats['total_tweets']*100:.1f}% del total)\n")
                f.write(f"Tweets neutros: {stats['tweets_neutros']} ({stats['tweets_neutros']/stats['total_tweets']*100:.1f}% del total)\n")
                
                # Palabras más frecuentes
                f.write("\nPalabras más frecuentes:\n")
                for palabra, freq in stats['palabras_mas_frecuentes'].most_common(10):
                    f.write(f"  - {palabra}: {freq} veces\n")
                
                # Palabras positivas más frecuentes
                f.write("\nPalabras positivas más frecuentes:\n")
                for palabra, freq in stats['palabras_positivas_frecuentes'].most_common(5):
                    f.write(f"  - {palabra}: {freq} veces\n")
                
                # Palabras negativas más frecuentes
                f.write("\nPalabras negativas más frecuentes:\n")
                for palabra, freq in stats['palabras_negativas_frecuentes'].most_common(5):
                    f.write(f"  - {palabra}: {freq} veces\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Conclusiones
            f.write("CONCLUSIONES\n")
            f.write("-" * 40 + "\n")
            
            # Ordenar partidos por porcentaje de tweets positivos
            partidos_por_positivos = sorted(
                [(p, s['tweets_positivos']/s['total_tweets']*100) for p, s in stats_partidos.items() if s['total_tweets'] > 0],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Ordenar partidos por porcentaje de tweets negativos
            partidos_por_negativos = sorted(
                [(p, s['tweets_negativos']/s['total_tweets']*100) for p, s in stats_partidos.items() if s['total_tweets'] > 0],
                key=lambda x: x[1],
                reverse=True
            )
            
            f.write(f"El partido con mejor percepción positiva es {partidos_por_positivos[0][0]} con {partidos_por_positivos[0][1]:.1f}% de tweets positivos.\n")
            f.write(f"El partido con mayor crítica negativa es {partidos_por_negativos[0][0]} con {partidos_por_negativos[0][1]:.1f}% de tweets negativos.\n\n")
            
            f.write(f"Según el análisis de sentimientos en Twitter, {ganador} tiene la mayor probabilidad de victoria con {prob_ganador:.2f}%.\n")
            f.write("Este análisis se basa en la proporción de tweets positivos y negativos para cada partido.\n\n")
            
            f.write("NOTA: Este análisis es una aproximación simplificada y no considera factores como el alcance real de los tweets,\n")
            f.write("la representatividad de la muestra, o elementos como la intención de voto real. Los resultados deben\n")
            f.write("interpretarse como tendencias de opinión en redes sociales, no como predicciones electorales precisas.\n")
            
        print(f"Reporte global guardado en {directorio_salida}/reporte_global.txt")
    except Exception as e:
        print(f"Error al crear reporte global: {e}")

def main():
    """Función principal del programa"""
    print("=== ANÁLISIS DE TWEETS DE PARTIDOS POLÍTICOS ===")
    
    # Cargar diccionarios de palabras
    palabras_positivas, palabras_negativas = cargar_diccionario_personalizado()
    print(f"Palabras positivas cargadas: {len(palabras_positivas)}")
    print(f"Palabras negativas cargadas: {len(palabras_negativas)}")
    
    # Obtener todas las rutas de archivos CSV por partido
    rutas_archivos_csv = {partido: [] for partido in PARTIDOS}
    
    for mes in MESES:
        for raiz, subcarpetas, archivos in os.walk(mes):
            for archivo in archivos:
                if ".csv" in archivo and archivo.replace(".csv","") in ARCHIVOS:
                    for partido in PARTIDOS:
                        if partido in raiz.split(os.path.sep):                                
                            ruta_archivo = os.path.join(raiz, archivo)
                            rutas_archivos_csv[partido].append(ruta_archivo)
    
    # Mostrar resumen de archivos encontrados
    print("\nResumen de archivos encontrados:")
    for partido, rutas in rutas_archivos_csv.items():
        print(f"  - {partido}: {len(rutas)} archivos")
    
    # Analizar cada partido y almacenar estadísticas
    stats_partidos = {}
    for partido, rutas in rutas_archivos_csv.items():
        if rutas:  # Solo procesar si hay archivos
            print(f"\nAnalizando partido: {partido} ({len(rutas)} archivos)")
            stats_partidos[partido] = analizar_partido(partido, rutas, palabras_positivas, palabras_negativas)
            
            # Crear gráficas para el partido
            crear_graficas_partido(partido, stats_partidos[partido])
    
    # Calcular probabilidades de victoria
    if stats_partidos:
        probabilidades = calcular_probabilidad_victoria(stats_partidos)
        
        # Crear gráfica de probabilidades
        crear_grafica_probabilidades(probabilidades)
        
        # Crear reporte global
        crear_reporte_global(stats_partidos, probabilidades)
        
        # Mostrar resumen de probabilidades
        print("\nProbabilidades de victoria basadas en análisis de sentimientos:")
        for partido, prob in sorted(probabilidades.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {partido}: {prob:.2f}%")
    else:
        print("\nNo se encontraron datos suficientes para realizar el análisis.")
    
    print("\n=== ANÁLISIS COMPLETADO ===")

if __name__ == "__main__":
    main()