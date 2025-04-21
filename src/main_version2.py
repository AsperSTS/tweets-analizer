import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import re
import string
import spacy
from nltk.corpus import stopwords
from collections import Counter
import seaborn as sns
import numpy as np
from pathlib import Path

from config import *

# Ensure directories exist for saving results
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load spaCy model (should be downloaded already)
nlp = spacy.load("es_core_news_sm")

# Dictionary of sentiment words
PALABRAS_POSITIVAS = {
    'bueno', 'excelente', 'grandioso', 'increíble', 'maravilloso', 'genial', 'feliz',
    'esperanza', 'motivado', 'honesto', 'valiente', 'transparente', 'justo', 'respeto',
    'democracia', 'progreso', 'futuro', 'desarrollo', 'justicia', 'equidad', 'paz',
    'compromiso', 'responsable', 'eficiente', 'innovador', 'solución', 'oportunidad',
    'inteligente', 'propuesta', 'favor', 'apoyo', 'mejorar', 'beneficio', 'construir',
    'confiar', 'alegría', 'seguridad', 'eficaz', 'competente', 'trabajar', 'éxito'
}

PALABRAS_NEGATIVAS = {
    'malo', 'pésimo', 'terrible', 'horrible', 'desastroso', 'triste', 'decepcionante',
    'corrupto', 'mentiroso', 'incompetente', 'fraude', 'violencia', 'crimen', 'robo',
    'nepotismo', 'injusticia', 'impunidad', 'pobreza', 'desigualdad', 'crisis', 'guerra',
    'ineficiente', 'fracaso', 'problema', 'conflicto', 'error', 'amenaza', 'peligro',
    'enojo', 'contra', 'odio', 'falso', 'daño', 'corrupción', 'desconfianza', 'miedo',
    'inseguridad', 'deficiente', 'incompetencia', 'negligencia', 'autoritario'
}

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
    
    # Cargar stopwords en español
    stop_words = set(stopwords.words('spanish'))
    
    # Añadir stopwords adicionales específicas para tweets
    stop_words_adicionales = {
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

def categorizar_palabras(palabras):
    """
    Categoriza las palabras como positivas o negativas
    
    Args:
        palabras (list): Lista de palabras a categorizar
        
    Returns:
        tuple: (palabras_positivas, palabras_negativas) listas de palabras categorizadas
    """
    palabras_positivas = [palabra for palabra in palabras if palabra.lower() in PALABRAS_POSITIVAS]
    palabras_negativas = [palabra for palabra in palabras if palabra.lower() in PALABRAS_NEGATIVAS]
    
    return palabras_positivas, palabras_negativas

def procesar_dataframe(dataframe, limite=None):
    """
    Procesa un dataframe de tweets y extrae usuario y palabras
    
    Args:
        dataframe (pandas.DataFrame): DataFrame con tweets
        limite (int, optional): Número máximo de filas a procesar
        
    Returns:
        dict: Diccionario con usuarios y sus palabras clasificadas
    """
    dataframe_tmp = seleccionar_columna_tweets_esp(dataframe)
    if limite:
        dataframe_proc = dataframe_tmp[:limite]
    else:
        dataframe_proc = dataframe_tmp
    
    resultados = {'usuarios': [], 'palabras_positivas': [], 'palabras_negativas': []}
    
    for index, fila in dataframe_proc.iterrows():
        try:
            tweet_text = fila['columna_1']
            usuario, lemas = procesar_tweet(tweet_text)
            
            # Filtrar solo palabras relevantes (sin menciones, sin números)
            palabras_filtradas = [palabra for palabra in lemas if not palabra.startswith('@') and not re.match(r'^\d+$', palabra)]
            
            # Categorizar palabras
            positivas, negativas = categorizar_palabras(palabras_filtradas)
            
            # Guardar resultados
            resultados['usuarios'].append(usuario)
            resultados['palabras_positivas'].append(positivas)
            resultados['palabras_negativas'].append(negativas)
            
        except Exception as e:
            print(f"Error procesando fila {index}: {e}")
    
    # Crear DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)
    return df_resultados

def guardar_txt_palabras(directorio, sentimiento, datos):
    """
    Guarda las palabras en archivos TXT según sentimiento
    
    Args:
        directorio (str): Directorio donde guardar el archivo
        sentimiento (str): 'Positivas' o 'Negativas'
        datos (pandas.DataFrame): DataFrame con usuarios y palabras
    """
    ensure_dir(directorio)
    ruta_archivo = os.path.join(directorio, f"palabras{sentimiento}.txt")
    
    with open(ruta_archivo, 'w', encoding='utf-8') as archivo:
        for index, row in datos.iterrows():
            usuario = row['usuarios']
            palabras = row[f'palabras_{sentimiento.lower()}']
            if palabras:  # Solo escribir si hay palabras
                archivo.write(f"{usuario}: {', '.join(palabras)}\n")

def guardar_diccionario_global(directorio_base):
    """
    Crea y guarda un diccionario global de palabras positivas y negativas
    
    Args:
        directorio_base (str): Directorio base donde guardar el diccionario
    """
    ensure_dir(directorio_base)
    
    # Guardar diccionario de palabras positivas
    ruta_positivas = os.path.join(directorio_base, "diccionario_positivas.txt")
    with open(ruta_positivas, 'w', encoding='utf-8') as archivo:
        for palabra in sorted(PALABRAS_POSITIVAS):
            archivo.write(f"{palabra}: Palabra con connotación positiva en el contexto político\n")
    
    # Guardar diccionario de palabras negativas
    ruta_negativas = os.path.join(directorio_base, "diccionario_negativas.txt")
    with open(ruta_negativas, 'w', encoding='utf-8') as archivo:
        for palabra in sorted(PALABRAS_NEGATIVAS):
            archivo.write(f"{palabra}: Palabra con connotación negativa en el contexto político\n")

def analizar_por_partido(resultados_por_partido):
    """
    Analiza los resultados por partido y mes
    
    Args:
        resultados_por_partido (dict): Diccionario con resultados por partido
        
    Returns:
        dict: Estadísticas de análisis
    """
    analisis = {}
    
    for partido, datos_meses in resultados_por_partido.items():
        analisis[partido] = {}
        
        for mes, datos in datos_meses.items():
            total_positivas = sum(len(p) for p in datos['palabras_positivas'])
            total_negativas = sum(len(p) for p in datos['palabras_negativas'])
            total_palabras = total_positivas + total_negativas
            
            # Evitar división por cero
            if total_palabras > 0:
                ratio_positivo = total_positivas / total_palabras
            else:
                ratio_positivo = 0
            
            # Contar palabras únicas
            palabras_positivas_unicas = set()
            palabras_negativas_unicas = set()
            
            for lista in datos['palabras_positivas']:
                palabras_positivas_unicas.update(lista)
            
            for lista in datos['palabras_negativas']:
                palabras_negativas_unicas.update(lista)
            
            # Palabras más frecuentes
            todas_positivas = [palabra for sublist in datos['palabras_positivas'] for palabra in sublist]
            todas_negativas = [palabra for sublist in datos['palabras_negativas'] for palabra in sublist]
            
            palabras_positivas_freq = Counter(todas_positivas).most_common(10) if todas_positivas else []
            palabras_negativas_freq = Counter(todas_negativas).most_common(10) if todas_negativas else []
            
            # Guardar resultados
            analisis[partido][mes] = {
                'total_tweets': len(datos['usuarios']),
                'total_palabras_positivas': total_positivas,
                'total_palabras_negativas': total_negativas,
                'palabras_positivas_unicas': len(palabras_positivas_unicas),
                'palabras_negativas_unicas': len(palabras_negativas_unicas),
                'ratio_positivo': ratio_positivo,
                'palabras_positivas_freq': palabras_positivas_freq,
                'palabras_negativas_freq': palabras_negativas_freq
            }
    
    return analisis

def generar_graficas(analisis, directorio_salida):
    """
    Genera gráficas de análisis de sentimiento por partido
    
    Args:
        analisis (dict): Resultados del análisis
        directorio_salida (str): Directorio donde guardar las gráficas
    """
    ensure_dir(directorio_salida)
    
    # 1. Gráfica de barras: Ratio positivo por partido y mes
    plt.figure(figsize=(12, 8))
    partidos = list(analisis.keys())
    
    # Verificar si tenemos datos para ambos meses
    meses_disponibles = []
    for partido in partidos:
        meses_disponibles.extend(list(analisis[partido].keys()))
    meses_disponibles = sorted(set(meses_disponibles))
    
    # Crear gráfica de barras agrupadas
    x = np.arange(len(partidos))
    width = 0.35 / len(meses_disponibles)
    
    for i, mes in enumerate(meses_disponibles):
        ratios = []
        for partido in partidos:
            if mes in analisis[partido]:
                ratios.append(analisis[partido][mes]['ratio_positivo'])
            else:
                ratios.append(0)
        
        plt.bar(x + i*width, ratios, width, label=mes)
    
    plt.ylabel('Ratio Positivo')
    plt.title('Ratio de sentimiento positivo por partido político')
    plt.xticks(x, partidos, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directorio_salida, 'ratio_positivo_partidos.png'))
    plt.close()
    
    # 2. Gráfica de pastel: Proporción de palabras positivas vs negativas por partido
    for partido in partidos:
        plt.figure(figsize=(10, 10))
        
        # Sumar datos de todos los meses
        total_positivas = sum(analisis[partido][mes]['total_palabras_positivas'] for mes in analisis[partido])
        total_negativas = sum(analisis[partido][mes]['total_palabras_negativas'] for mes in analisis[partido])
        
        # Crear gráfica de pastel
        etiquetas = ['Positivas', 'Negativas']
        sizes = [total_positivas, total_negativas]
        colores = ['#66b3ff', '#ff9999']
        
        plt.pie(sizes, labels=etiquetas, colors=colores, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title(f'Proporción de palabras positivas vs negativas - {partido}')
        plt.tight_layout()
        plt.savefig(os.path.join(directorio_salida, f'proporcion_sentimiento_{partido}.png'))
        plt.close()
    
    # 3. Gráfica de barras: Palabras más frecuentes por partido
    for partido in partidos:
        # Combinar datos de todos los meses
        todas_positivas = []
        todas_negativas = []
        
        for mes in analisis[partido]:
            todas_positivas.extend([palabra for palabra, freq in analisis[partido][mes]['palabras_positivas_freq']])
            todas_negativas.extend([palabra for palabra, freq in analisis[partido][mes]['palabras_negativas_freq']])
        
        # Contar frecuencias combinadas
        positivas_freq = Counter(todas_positivas).most_common(10)
        negativas_freq = Counter(todas_negativas).most_common(10)
        
        # Gráfica palabras positivas
        if positivas_freq:
            plt.figure(figsize=(12, 8))
            palabras, frecuencias = zip(*positivas_freq)
            plt.barh(palabras, frecuencias, color='green')
            plt.xlabel('Frecuencia')
            plt.title(f'Palabras positivas más frecuentes - {partido}')
            plt.tight_layout()
            plt.savefig(os.path.join(directorio_salida, f'palabras_positivas_{partido}.png'))
            plt.close()
        
        # Gráfica palabras negativas
        if negativas_freq:
            plt.figure(figsize=(12, 8))
            palabras, frecuencias = zip(*negativas_freq)
            plt.barh(palabras, frecuencias, color='red')
            plt.xlabel('Frecuencia')
            plt.title(f'Palabras negativas más frecuentes - {partido}')
            plt.tight_layout()
            plt.savefig(os.path.join(directorio_salida, f'palabras_negativas_{partido}.png'))
            plt.close()
    
    # 4. Gráfica de líneas: Evolución del ratio positivo por mes
    plt.figure(figsize=(12, 8))
    
    for partido in partidos:
        meses = sorted(analisis[partido].keys())
        ratios = [analisis[partido][mes]['ratio_positivo'] for mes in meses]
        plt.plot(meses, ratios, marker='o', label=partido)
    
    plt.ylabel('Ratio Positivo')
    plt.title('Evolución del sentimiento positivo por partido')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(directorio_salida, 'evolucion_sentimiento.png'))
    plt.close()

def generar_reporte_ganador(analisis, directorio_salida):
    """
    Genera un reporte sobre qué partido tendría más posibilidades de ganar
    basado en el análisis de sentimiento
    
    Args:
        analisis (dict): Resultados del análisis
        directorio_salida (str): Directorio donde guardar el reporte
    """
    # Calcular ratio positivo global por partido
    ratios_partido = {}
    
    for partido in analisis:
        total_positivas = 0
        total_negativas = 0
        
        for mes in analisis[partido]:
            total_positivas += analisis[partido][mes]['total_palabras_positivas']
            total_negativas += analisis[partido][mes]['total_palabras_negativas']
        
        total = total_positivas + total_negativas
        if total > 0:
            ratios_partido[partido] = total_positivas / total
        else:
            ratios_partido[partido] = 0
    
    # Ordenar partidos por ratio positivo
    partidos_ordenados = sorted(ratios_partido.items(), key=lambda x: x[1], reverse=True)
    
    # Generar reporte
    ruta_reporte = os.path.join(directorio_salida, "reporte_prediccion.txt")
    
    with open(ruta_reporte, 'w', encoding='utf-8') as archivo:
        archivo.write("ANÁLISIS DE POSIBILIDADES DE VICTORIA POR PARTIDO\n")
        archivo.write("==============================================\n\n")
        
        archivo.write("Ranking de partidos según percepción positiva en redes sociales:\n\n")
        
        for i, (partido, ratio) in enumerate(partidos_ordenados, 1):
            archivo.write(f"{i}. {partido}: {ratio:.2%} de percepción positiva\n")
        
        archivo.write("\nCONCLUSIÓN:\n")
        
        ganador, ratio_ganador = partidos_ordenados[0]
        segundo, ratio_segundo = partidos_ordenados[1] if len(partidos_ordenados) > 1 else (None, 0)
        
        archivo.write(f"Según el análisis de sentimiento en tweets, {ganador} muestra la mejor percepción pública ")
        archivo.write(f"con un {ratio_ganador:.2%} de menciones positivas.\n\n")
        
        if segundo:
            diferencia = (ratio_ganador - ratio_segundo) * 100
            archivo.write(f"La ventaja sobre el segundo partido ({segundo}) es de {diferencia:.1f} puntos porcentuales.\n\n")
        
        archivo.write("Factores considerados en el análisis:\n")
        archivo.write("- Cantidad de menciones positivas vs. negativas\n")
        archivo.write("- Evolución del sentimiento a lo largo del tiempo\n")
        archivo.write("- Términos positivos y negativos más frecuentes\n\n")
        
        archivo.write("ADVERTENCIA: Este análisis se basa únicamente en datos de redes sociales ")
        archivo.write("y no considera otros factores importantes como encuestas formales, ")
        archivo.write("alcance de campañas, demografía de votantes, etc.\n")

def main():
    print("Iniciando análisis de tweets políticos...")
    
    # Almacenar resultados por partido y mes
    resultados_por_partido = {partido: {} for partido in PARTIDOS}
    
    # Crear directorio para diccionario global
    directorio_base = "Resultados"
    ensure_dir(directorio_base)
    
    # Guardar diccionario global
    print("Generando diccionario global de palabras...")
    guardar_diccionario_global(directorio_base)
    
    # Procesar cada archivo CSV
    for mes in MESES:
        print(f"Procesando tweets del mes de {mes}...")
        
        for partido in PARTIDOS:
            resultados_por_partido[partido][mes] = {'usuarios': [], 'palabras_positivas': [], 'palabras_negativas': []}
            
            # Buscar archivos para este partido y mes
            for raiz, subcarpetas, archivos in os.walk(mes):
                for archivo in archivos:
                    if ".csv" in archivo and archivo.replace(".csv","") in ARCHIVOS:
                        if partido in raiz.split(os.path.sep):
                            ruta_archivo = os.path.join(raiz, archivo)
                            print(f"  Procesando archivo: {ruta_archivo}")
                            
                            try:
                                # Cargar CSV
                                csv_cargado = pd.read_csv(ruta_archivo)
                                
                                # Determinar tipo de sentimiento desde la ruta
                                es_positivo = "Positivo" in archivo
                                
                                # Procesar dataframe
                                resultados = procesar_dataframe(csv_cargado)
                                
                                # Acumular resultados
                                resultados_por_partido[partido][mes]['usuarios'].extend(resultados['usuarios'])
                                resultados_por_partido[partido][mes]['palabras_positivas'].extend(resultados['palabras_positivas'])
                                resultados_por_partido[partido][mes]['palabras_negativas'].extend(resultados['palabras_negativas'])
                                
                                # Guardar resultados en TXT del directorio correspondiente
                                directorio_salida = os.path.dirname(ruta_archivo)
                                if es_positivo:
                                    guardar_txt_palabras(directorio_salida, "Positivas", resultados)
                                else:
                                    guardar_txt_palabras(directorio_salida, "Negativas", resultados)
                                    
                            except Exception as e:
                                print(f"Error procesando {ruta_archivo}: {e}")
    
    # Analizar resultados
    print("Analizando resultados...")
    analisis = analizar_por_partido(resultados_por_partido)
    
    # Generar gráficas
    print("Generando gráficas...")
    directorio_graficas = os.path.join(directorio_base, "Graficas")
    generar_graficas(analisis, directorio_graficas)
    
    # Generar reporte de predicción
    print("Generando reporte de predicción...")
    generar_reporte_ganador(analisis, directorio_base)
    
    print("Análisis completo. Resultados guardados en el directorio 'Resultados'.")

if __name__ == "__main__":
    main()