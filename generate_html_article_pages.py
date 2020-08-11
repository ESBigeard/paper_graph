#!/usr/bin/python
# -*- coding:utf-8 -*-
"""didn't test after separating fom generate_gephi_csv.py ! might not work !
used to generate the folder with each paper as an html page, where you can read the contents of the paper with calculated keywords and most important sentences in yellow"""

import unicodedata
import re
import sys
import os
import pickle
import csv
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from collections import OrderedDict
from rake_nltk import Rake
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#import local scripts
import utilsperso

import spacy
nlp=spacy.load("en_core_web_sm")


rake=Rake(max_length=4)

idf=utilsperso.finish_idf("27_07_tei_idf.pickle")


def print_html_keywords_keysentences(fname):
	"""for a single grobit xml file outputs an html containing the text of the article separated by sections. each sections is printed with its 5 top keywords and its text with top sentences highlighted in yellow. keywords and top sentences are computed by tf-idf and rake, and are shown side by side (tf-idf on the left, rake on the right)"""

	with open(fname,mode="r",encoding="utf-8") as f:


		fname=fname.split("/")[-1]
		output="<html><head><title>"+fname+"</title>"
		output+='<link rel="stylesheet" href="style.css">'
		output+="</head><body>"
		output+="<p>"+fname+"</p>"
		fname2=re.split("\.",fname)[0]
		output+="<p><a href='http://elisebigeard.yo.fr/gephi_files/pdf_files/"+fname2+".pdf' class='pdf_link'>See PDF</a></p>"

		lines="".join(f.readlines())
		soup=BeautifulSoup(lines,"xml")

		article_title=soup.find("titleStmt").getText()
		output+="<h1>"+article_title+"</h1>"

		#tags to remove
		for tags in [soup("ref")]:
			for tag in tags:
				tag.decompose()

	
		current_main_title="other"
		title_print=" "
		sections_sentences=defaultdict(list) #key=main section title. value=list of sentences
		for tag in soup.find_all(["head","p"]): #iterate through head and p tags in order
			if tag.name=="head":
				#title_print=tag.getText()
				if True: #temporarily disabling selection of sections. doesn't seem to work properly

					title_clean=utilsperso.process_section_title(tag.getText()) 
					if title_clean:
						if title_clean == "other": #random title in the body
							pass
						else: #a maintitle such as "introduction"
							current_main_title=title_clean
							title_print=tag.getText()
					else: #title of something we don't want to keep
						current_main_title="trash"

			else: #p tag, we want to get the text
				if current_main_title=="trash":
					continue #we don't keep this part at all
				else:
					sentences=utilsperso.preprocess_text(tag,keep_all_words=True,separate_sentences=True) #we need the stopwords for rake
					sections_sentences[title_print]+=sentences


		#output section by section the keywords + highlighted text

		for section in sections_sentences:

			output_tf_keywords=""
			output_tf_sentences=""
			output_rake_keywords=""
			output_rake_sentences=""



			#get lemmatised sentences and flat text
			lem_text_flat=[] #all lemmatized words, flat
			lem_text_sent=[] #all lemmatized words, sentence by sentence, in nested lists
			for sentence in sections_sentences[section]:

				current_sent=[]
				for token in sentence:
					if token.pos_=="NUM":
						lem_text_flat.append("NUM")
						current_sent.append("NUM")
					else:
						lem_text_flat.append(token.lemma_)
						current_sent.append(token.lemma_)

				lem_text_sent.append(current_sent)


			#how many sentences from this section we want to highlight
			total_sentences=len(sections_sentences[section])
			if total_sentences<5:
				how_many=1 
			elif total_sentences<11:
				how_many=3
			else:
				how_many=int(total_sentences/10)



			#compute tf-idf of the section
			#TODO check when I last did that idf and if it's still good
			#TODO mmmmh does this make sense ??
			tf_dic=utilsperso.count_text_tfidf(lem_text_flat,idf)

			#tf-idf keywords
			tf=utilsperso.count_text_tfidf(lem_text_flat,idf)
			top_keywords=[]
			for i,word in enumerate(sorted(tf, key=tf.get, reverse=True)):
				if i<6:
					top_keywords.append(word)
			output_tf_keywords=" ; ".join(top_keywords)

			#compute tf-idf score of each sentence
			tf_sentences=[]
			for sentence in lem_text_sent:
				current_sent_score=0
				for token in sentence:
					current_sent_score+=tf_dic[token]
				tf_sentences.append(current_sent_score)

			#compute min tf for a sentence to be highlighted
			cutoff_score=sorted(tf_sentences,reverse=True)[how_many-1]



			#print sentences with highlighs
			for i,sent in enumerate(sections_sentences[section]):
				score=tf_sentences[i]
				text=[]
				for token in sent:
					text.append(token.text)
				text=" ".join(text)
				if score>=cutoff_score:
					output_tf_sentences+="<p style='background:yellow;'>"+text+"</p>"
				else:
					output_tf_sentences+="<p>"+text+"</p>"


			#keywords by rake
			rake=Rake(max_length=4)
			rake.extract_keywords_from_text(" ".join(lem_text_flat))
			keyphrases=rake.get_ranked_phrases_with_scores()


			#print top keywords
			top=[x[1] for x in keyphrases[:5]]
			output_rake_keywords=top=" ; ".join(top)


			#identify sentences with highest score
			d_phrases={}
			for score,phrase in keyphrases:
				d_phrases[phrase]=score

			sent_scores=[]
			for sentence in lem_text_sent:
				sent_score=0
				for i,word in enumerate(sentence):
					window=" ".join(sentence[i:i+4])
					for top_phrase in d_phrases:
						if window in top_phrase:
							sent_score+=d_phrases[top_phrase]
							#TODO bug une phrase len<4 peut se répéter
						
				sent_scores.append(sent_score)

			#min score for a sentence to be highlighted
			cutoff_score=d_phrases[sorted(d_phrases, key=d_phrases.get, reverse=True)[how_many-1]] #gets the Nth highest score (d_phrases values) where Nth is how_many-1


			
			#get all sentences in order, whith highlight of higest score
			for i,sent in enumerate(sections_sentences[section]):
				score=sent_scores[i]
				text=[]
				for token in sent:
					text.append(token.text)
				text=" ".join(text)
				if score>=cutoff_score:
					output_rake_sentences+="<p style='background:yellow;'>"+text+"</p>"
				else:
					output_rake_sentences+="<p>"+text+"</p>"

			#print output

			if False: #output 2 colonnes tfidf/rake
				output+="<h2>"+section.upper()+"</h2>"
				output+="<table><tr><th>TF-IDF</th><th>RAKE</th></tr>"
				output+="<tr style='border-bottom:solid 2px black;'><td><strong>Keywords : </strong>"+output_tf_keywords+"</td><td><strong>Keywords : </strong>"+output_rake_keywords+"</td></tr>"
				output+="<tr><td>"+output_tf_sentences+"</td><td>"+output_rake_sentences+"</td></tr>"
				output+="</table>"

			if True: #output clean user
				output+="<h2>"+section+"</h2>"
				output+="<p><strong>Keywords (single word) : </strong>"+output_tf_keywords+"</p>"
				output+="<p><strong>Keywords (multi words) : </strong>"+output_rake_keywords+"</p>"
				output+="<p>"+output_tf_sentences+"</p>"


		output+="</body></html>"
		return output

if __name__=="__main__":

	out_directory="paper_pages"
	directory="fulltext_tei"
	for fname in os.listdir(directory):
		if fname.endswith(".xml"):
			print("processing "+fname)
			html=print_html_keywords_keysentences(directory+"/"+fname)
			fname=fname.replace(" ","")
			with open(out_directory+"/"+fname[:-7]+"html",mode="w",encoding="utf-8") as f:
				f.write(html)
	
