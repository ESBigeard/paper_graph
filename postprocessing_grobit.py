#!/usr/bin/python
# -*- coding:utf-8 -*-

import unicodedata
import re
import sys
import os
import pickle
from bs4 import BeautifulSoup
from collections import defaultdict
from rake_nltk import Rake

#import local scripts
import utilsperso

with open("dropbox_tei.pickle",mode="rb") as f:
	idf=pickle.load(f)
idf=utilsperso.finish_idf(idf)

def clean_string(s):
	"""given a string in a tei file created by grobit, cleans up gorbit mistakes. puts diacritics back on their letter, removes diaresis (ex : wo-rd)
	diacritics is done. diaresis not yet"""

	#binds the diacritics to the correct letters, and does some other unicode normalisation.
	#unicodedata.normalize has a bug and introduces a space before each diacritic+letter character
	#at this step it looks like the diactitics binds to the n+1 letter instead of the correct letter. but once we remove the n-1 space at the next step it will look right
	s2=unicodedata.normalize("NFKD",s)

	#removes the space unicodata.normalize introduced before the diacritic+letter characters
	s3=""
	for char in s2:
		if unicodedata.category(char)=="Mn" and s3[-1]==" ":
			s3=s3[:-1]
		s3+=char

	return s3

def process_article(fname):

	with open(fname,mode="r",encoding="utf-8") as f:

		lines="".join(f.readlines())
		soup=BeautifulSoup(lines,"xml")

		article_title=soup.find("titleStmt").getText()
		print("<h1>"+article_title+"</h1>")

		#tags to remove
		for tags in [soup("ref")]:
			for tag in tags:
				tag.decompose()

	
		current_main_title="other"
		title_print=""
		sections_sentences=defaultdict(list) #key=main section title. value=list of sentences
		for tag in soup.find_all(["head","p"]):
			if tag.name=="head":
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
				sentences=utilsperso.preprocess_text(tag,keep_all_words=True) #we need the stopwords for rake
				if current_main_title=="trash":
					continue
				else:
					sections_sentences[title_print]+=sentences

		

		#output section by section the keywords + highlighted text

		for section in sections_sentences:

			output_tf_keywords=""
			output_tf_sentences=""
			output_rake_keywords=""
			output_rake_sentences=""



			#get lemmatised sentences and flat text
			lem_text_flat=[]
			lem_text_sent=[]
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
			tf_dic=utilsperso.count_text_tfidf(lem_text_flat)

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
			
			#print all sentences, whith highlight of higest score
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

			print("<h2>"+section+"</h2>")
			print("<table><tr><th>TF-IDF</th><th>RAKE</th></tr>")
			print("<tr style='border-bottom:solid 2px black;'><td><strong>Keywords : </strong>"+output_tf_keywords+"</td><td><strong>Keywords : </strong>"+output_rake_keywords+"</td></tr>")
			print("<tr><td>"+output_tf_sentences+"</td><td>"+output_rake_sentences+"</td></tr>")
			print("</table>")
				





	return soup


if __name__=="__main__":
	directory="/media/ezi/Backup1/dl_nancy/Canceropole_dropbox/fulltext1_tei"
	print("<html><body>")
	for fname in os.listdir(directory):
		if fname.endswith(".xml"):
			sys.stderr.write(fname+"\n")
			process_article(directory+"/"+fname)
			print("<p></p>")
			print("<p></p>")
			print("<hr>")
			print("<p></p>")
	print("</html></body>")
