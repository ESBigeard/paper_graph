#!/usr/bin/python
# -*- coding:utf-8 -*-
""" one-time script to lemmatize the mesh"""

import csv
import spacy
nlp=spacy.load("en_core_web_sm")


def lemmatise(s):
	s=s.lower() #spacy tends to leave capitalised words as-it
	spacy_text=nlp(s)
	out=[]
	for token in spacy_text:
		out.append(token.lemma_)
	return " ".join(out)

with open("synonymes.txt",mode="r",encoding="utf-8") as f1, open("synonyms_lem.txt",mode="w",encoding="utf-8") as f2:
	for l in f1:
		l=l.strip()
		l_out=[]
		for label in l.split(","):
			label2=lemmatise(label)
			if label2 not in l_out:
				l_out.append(label2)
			if label not in l_out:
				l_out.append(label)#we keep the raw token as a possible synonym. we keep the lemmatised version as pref label, so we add this at the end
		out=",".join(l_out)
		f2.write(out+"\n")




