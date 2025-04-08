#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys  



from googletrans import Translator
from pattern.en import sentiment
import csv
import time
reload(sys)  
sys.setdefaultencoding('utf8')

traductor = Translator()
dic = open("Obrador1.csv")
r1 = csv.reader(dic, delimiter=',')
dic=[]
for pal_dic in r1:
	dic.append(pal_dic[1])
	
dicf = open("AnalisisGeneral.csv", "w")
dicp = open("AnalisisGeneralPositivo.csv", "w")
dicn = open("AnalisisGeneralNegativo.csv", "w")


for pala in dic:
	arr=[]
	arr.append(pala)
	try:
		resultado = traductor.translate(pala, dest="en")
	except:
		pass
	if sentiment(str(resultado.text))[0]==0:
		arr.append(resultado.text)
		arr.append(sentiment(str(resultado.text))[0])
		writer = csv.writer(dicf)
		writer.writerow(arr)
	if sentiment(str(resultado.text))[0]>0:
		arr.append(resultado.text)
		arr.append(sentiment(str(resultado.text))[0])
		writer = csv.writer(dicp)
		writer.writerow(arr)
	if sentiment(str(resultado.text))[0]<0:
		arr.append(resultado.text)
		arr.append(sentiment(str(resultado.text))[0])
		writer = csv.writer(dicn)
		writer.writerow(arr)	
	
	
