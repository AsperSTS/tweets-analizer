import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from config import *

"""
    1.- LIMPIEZA DE LOS TWEETS
    2.- DICCIONARIO DE PALABRAS BUENAS Y MALAS
    3.- CONTABILIDAD DE LAS PALABRAS (PROBABLEMENTE GENERAR UN CSV) NOTA: LOS ARCHIVOS SE GENERAN EN CADA CARPETA
        3.1 CSV DONDE SE PONGA EL USUARIO, Y LAS PALABRAS POSITIVAS O NEGATIVAS DEL TWEET (EN TOTAL DOS CSV PORQUE SON NEGATIVOS Y POSITIVIOS)
        3.2 TXT DICCIONARIO DEL DESGLOCE DE LAS PALABRAS(GLOBAL) EX. PALABRA : SIGNIFICADO
    4.- ANALISIS DE LOS RESULTADOS DE TWEETS (EN BASE A LAS PALABRAS POSITIVAS O NEGATIVAS DE LAS SUBCARPETAS DE CADA PARTIDO, POR EJEMPLO ANALISIS MENSUAL)
        4.1- EN BASE A ESTE ANALISIS SE DEBE DE TERMINAR QUE PARTIDO TENDRIA MAS POSIBILIDADES DE GANAR
    5.- GRAFICAS DE OPINION PUBLICA EN BASE A LOS TWEETS
        5.1- GRAFICAS DE PASTEL Y DE BARRAS 
"""
def main():
    
    
    for mes in MESES:
        for raiz, subcarpetas, archivos in os.walk(mes):
            # print(f"{raiz} \n")
            if 
            # for
        # for elements in PARTIDOS:
        #     os.
        #     print(elements)

if __name__ == "__main__":
    main()