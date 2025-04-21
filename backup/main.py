import matplotlib.pyplot as plt
import pandas as pd
import os
import csv

from config import *

import re
import string
import spacy
from nltk.corpus import stopwords
# import nltk

# Descargar recursos necesarios (ejecutar una vez)
# nltk.download('stopwords')
# nltk.download('punkt')
# python -m spacy download es_core_news_sm
""" 
    REQUERIMIENTOS

    1.- LIMPIEZA DE LOS TWEETS
    2.- DICCIONARIO DE PALABRAS BUENAS Y MALAS
    3.- CONTABILIDAD DE LAS PALABRAS (PROBABLEMENTE GENERAR UN CSV) NOTA: LOS ARCHIVOS SE GENERAN EN CADA CARPETA
        3.1 TXT DONDE SE PONGA EL USUARIO, Y LAS PALABRAS POSITIVAS O NEGATIVAS DEL TWEET (EN TOTAL DOS CSV PORQUE SON NEGATIVOS Y POSITIVIOS)
        3.2 TXT DICCIONARIO DEL DESGLOCE DE LAS PALABRAS(GLOBAL) EX. PALABRA : SIGNIFICADO
    4.- ANALISIS DE LOS RESULTADOS DE TWEETS (EN BASE A LAS PALABRAS POSITIVAS O NEGATIVAS DE LAS SUBCARPETAS DE CADA PARTIDO, POR EJEMPLO ANALISIS MENSUAL)
        4.1- EN BASE A ESTE ANALISIS SE DEBE DE TERMINAR QUE PARTIDO TENDRIA MAS POSIBILIDADES DE GANAR
    5.- GRAFICAS DE OPINION PUBLICA EN BASE A LOS TWEETS
        5.1- GRAFICAS DE PASTEL Y DE BARRAS 
"""
def juntar_csv(rutas_archivos_csv: dict, partido="Morena"):
    print("===== PRINTING===")
    """
        Funcion que itera sobre todos los csv de un partido dado y da como resultado 
        ...
        Por implementar
    """
    for element in rutas_archivos_csv[partido]:
        partes_ruta = element.split(os.path.sep)
        if len(partes_ruta) > 1:
            # Eliminar el último elemento (nombre del archivo)
            directorio = os.path.sep.join(partes_ruta[:-1])
            print(f"Directorio: {directorio}")
        else:
            print(f"La ruta '{element}' no tiene un directorio padre.")
        #  if partido in raiz.split("/", 2)
        tmp = pd.read_csv(element)
        print(tmp.head(5))
        
def ver_todos_archivos(rutas_archivos_csv: dict):
    print("===== PRINTING===")
    tmp = None
    for key, elements in rutas_archivos_csv.items():
        # print(f"{key} : {element} \n")
        for element in elements:
            # print(f"{key} : {element}")
            tmp = pd.read_csv(element)
            print(tmp[1].head(5))
            
def seleccionar_columna_tweets_esp(dataframe):
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
    
    # Cargar el modelo de spaCy para español
    nlp = spacy.load("es_core_news_sm")
    
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
    
    # Cargar stopwords en español
    stop_words = set(stopwords.words('spanish'))
    
    # Añadir stopwords adicionales específicas para tweets
    stop_words_adicionales = {  # Posible eliminacion porque puede reducir significado
        # Básicas y expresiones comunes
        'rt', 'vs', 'jaja', 'jajaja', 'jajajaja', 'ja', 'je', 'ji', 'hahaha', '6de6',
        
        # Verbos auxiliares/modales
        'poder', 'querer', 'deber', 'hacer', 'decir', 'saber', 'creer', 'mirar', 
        'conocer', 'tratar', 'seguir', 'poner', 'llamar', 'continuar', 'esperar',
        
        # Adverbios temporales
        'hoy', 'día', 'año', 'mes', 'siempre', 'ahora', 'tiempo', 'momento',
        
        # Expresiones de cortesía y comunicación
        'gracias', 'agradecer', 'felicitar', 'saludar', 'desear', 'celebrar',
        
        # Adjetivos comunes poco informativos
        'buen', 'bueno', 'gran', 'grande', 'nuevo', 'solo', 'mayor', 'principal',
        'único', 'normal', 'claro', 'positivo',
        
        # Pronombres y demostrativos
        'este', 'ese', 'aquel', 'él', 'usted', 'tal', 'mío', 'tuyo', 'suyo',
        
        # Nombres comunes en política mexicana con poca carga semántica específica
        'país', 'méxico', 'presidente', 'gobierno', 'político',
        
        # Conectores adicionales
        'pues', 'asi', 'aunque', 'mientras', 'entonces',
        
        # Otros términos frecuentes en el corpus
        'estar', 'ser', 'haber', 'tener', 'ir', 'pre', 'ver', 'dar', 'así'
    }
    # Ampliar la lista de stopwords con palabras que no aportan valor semántico
    stop_words.update([
        # Verbos auxiliares y conectores
        'hacer', 'poder', 'querer', 'decir', 'saber', 'mirar', 'creer',
        'seguir', 'continuar', 'resultar', 'quedar', 'parecer',
        
        # Pronombres y demostrativos
        'este', 'ese', 'aquel', 'tal', 'todo', 'nada', 'algo', 'alguien', 'nadie',
        
        # Adverbios comunes
        'bien', 'mal', 'solo', 'muy', 'más', 'menos', 'ahora', 'siempre', 'nunca',
        'ahí', 'aquí', 'allí', 'entonces', 'después', 'antes',
        
        # Adjetivos genéricos
        'buen', 'gran', 'mayor', 'menor', 'mejor', 'peor', 'primero', 'último',
        
        # Palabras temporales
        'día', 'mes', 'año', 'tiempo', 'hora', 'rato', 'momento',
        
        # Palabras coloquiales mexicanas
        'wey', 'chingar', 'madre', 'oro', 'vivo',
        
        # Palabras específicas del contexto político que no aportan valor semántico
        'persona', 'país', 'gente', 'cosa', 'punto', 'modo', 'vez',
        
        # Palabras de redes sociales
        'tweet', 'tuitear', 'jajajajaja', 'hahahaha'
    ])
    
    stop_words.update(stop_words_adicionales)
    
    # Palabras políticas genéricas
    palabras_politicas_genericas = [
        'político', 'gobierno', 'presidente', 'votar', 'elección', 'campaña',
        'país', 'democracia', 'reunión', 'desarrollo', 'proyecto', 'cambio',
        'mensaje', 'administración', 'opinión', 'palabra', 'trabajo', 'apoyo'
    ]

    stop_words.update(palabras_politicas_genericas)
    # Procesar tokens: eliminar puntuación, stopwords, números y lematizar
    tokens_procesados = []
    for token in doc:
        # Verificar que no sea puntuación, stopword, número o mención
        if (token.text not in string.punctuation and 
            token.text not in stop_words and 
            token.lemma_ not in stop_words and  # Verificar también el lema
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

# def guadar_diccionario_palabras(user, palabras_filtradas):
    # dataframe = 
    
def prueba_todo(dataframe):
    dataframe_tmp = seleccionar_columna_tweets_esp(dataframe)
    dataframe = dataframe_tmp[1:5]
    datos = []
    for index, fila in dataframe.iterrows():
        try:
            tweet_text = fila['columna_1']
            usuario, lemas = procesar_tweet(tweet_text)
            # Filtrar solo palabras relevantes (sin menciones, sin números)
            palabras_filtradas = [palabra for palabra in lemas if not palabra.startswith('@') and not re.match(r'^\d+$', palabra)]
            print(f"{usuario} : {palabras_filtradas}")
            datos.append([usuario] + palabras_filtradas)
        except Exception as e:
            print(f"Error procesando fila {index}: {e}")
    
    # Crear el DataFrame resultante
    if datos: # GUARDAR EN TXT EN VEZ DE CSV
        # Obtener todos los nombres de usuario para la primera columna
        columnas = ['usuario'] + [f'palabra_{i+1}' for i in range(max(len(fila) - 1 for fila in datos))]
        df_resultado = pd.DataFrame(datos, columns=columnas[:max(len(fila) - 1 for fila in datos) + 1])
        return df_resultado
    else:
        return pd.DataFrame(columns=['usuario'])
def guardar_txt_palabras_individual(nombre_archivo_csv, _datos):
    with open(nombre_archivo_csv, 'w', newline='', encoding='utf-8') as archivo_csv: 
        writer = csv.writer(archivo_csv)
    # usuario = row['autor']
    # palabras = row['palabras']
        # CAMBIAR LA LOGICA PARA TXT CON LAS PALABRAS SEPARADAS POR COMAS
        fila_csv = [usuario] + palabras
        writer.writerow(fila_csv)      
def main():
    
    rutas_archivos_csv = {partido: [] for partido in PARTIDOS}
    
    for mes in MESES:
        for raiz, subcarpetas, archivos in os.walk(mes):
            # print(f"{raiz} \n")
            for archivo in archivos:
                if ".csv" in archivo and archivo.replace(".csv","") in ARCHIVOS:
                    for partido in PARTIDOS:
                            if partido in raiz.split("/", 2) :                                
                                ruta_archivo = raiz + '/' + archivo
                                # print(f"{ruta_archivo} \n")
                                rutas_archivos_csv[partido].append(ruta_archivo)

    directorio_anterior = None
    for partido in PARTIDOS:  # RECORREMOS TODOS LOS PARTIDOS
        for element in rutas_archivos_csv[partido]:    # RECORREMOS TODOS LOS CSV
            partes_ruta = element.split(os.path.sep)    
            csv_cargado = pd.read_csv(element)
            if len(partes_ruta) > 1:    # OBTENEMOS LOS DIRECTORIOS PARA PODER COMPARAR Y HACER EL DICCIONARIO DE PALABRAS
                # Eliminar el último elemento (nombre del archivo)
                directorio = os.path.sep.join(partes_ruta[:-1])
                directorio_anterior = directorio
                
                # AQUI DEBES DE PROCESAR EL CSV COMPLETO
                _datos = prueba_todo(csv_cargado)
                # CREA UN DATAFRAME POSITIVO O NEGATIVO DEPENDIENTOD DE partes_ruta[3:], 
                # archivo_tmp = partes_ruta[3:][0]
                if "Negativo" in partes_ruta[3:][0]:
                    ruta_ = directorio+"/palabrasNegativas.csv"
                    guardar_csv_palabras_individual(ruta_, _datos)
                    # print(ruta_)
                    print(f"Negativo: {element}")
                    
                elif "Positivo" in partes_ruta[3:][0]:
                    ruta_ = directorio+"/palabrasNegativas.csv"
                    print(f"Positivo: {element}")
                     
    # print(f"Claves de diccionario: {rutas_archivos_csv.keys()}")
    # juntar_csv(rutas_archivos_csv)

    # print("\n==== POSITIVO ==== \n")
    # csv_tmp = pd.read_csv("Marzo/010323/PARTIDO_VERDE/AnalisisGeneralPositivo.csv")
    # prueba_todo(csv_tmp)
    # print("\n==== NEGATIVO ==== \n")
    # csv_tmp = pd.read_csv("Marzo/010323/PARTIDO_VERDE/AnalisisGeneralNegativo.csv")
    # prueba_todo(csv_tmp)
    
    # print(csv.iloc[:,0])

if __name__ == "__main__":
    main()