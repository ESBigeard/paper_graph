#!/usr/bin/python
# -*- coding:utf-8 -*-

import unicodedata
import re
import sys
import os
import pickle
import csv
import pandas as pd
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import utilsperso
import spacy
#nlp=spacy.load("en_core_web_sm")

#idf=utilsperso.finish_idf("27_07_tei_idf.pickle") #put here the .pickle file containing an idf

#mesh_hierarchy_up, mesh_hierarchy_down=utilsperso.load_mesh()

	
mesh_hierarchy_up=defaultdict(list)
mesh_hierarchy_down=defaultdict(list)
mesh_keywords=set([])
keyword_exclude=set([])
with open("mesh/exclude_list.txt",mode="r",encoding="utf-8") as f:
	for l in f:
		l=l.strip()
		l=l.lower()
		keyword_exclude.add(l)

	
with open("mesh/mesh_lem.txt",mode="r",encoding="utf-8") as f:
	reader=csv.reader(f)
	next(reader)
	for row in reader:
		label1,relation,label2=row
		label1=label1.lower()
		label2=label2.lower() #todo more nettoyage
		if label1 in keyword_exclude or label2 in keyword_exclude :
			continue
		else:
			mesh_keywords.add(label1)
			mesh_keywords.add(label2)
			if label1==label2:
				#sys.stderr.write("There an entry in the mesh hierarchy with this term as both hyponym and hyperonym : "+label1+" This entry has been ignored but removing it from the data is recommended.\n")
				continue
			mesh_hierarchy_up[label2].append(label1)
			mesh_hierarchy_down[label1].append(label2)
	

mesh_synonyms={} #dict[alt_label]=pref_label. several alt_label may point to the same pref_label.
with open("mesh/synonyms_lem.txt",mode="r",encoding="utf-8") as f:
	for l in f:
		l=l.strip()
		l=l.split(",")
		pref_label=l[0]
		if pref_label in keyword_exclude:
			continue
		else:
			for label in l[1:]:
				mesh_synonyms[label]=pref_label
		mesh_synonyms[pref_label]=pref_label #so the pref label is also in the dict, and the dict can be used as a list of possible mesh terms


def clean_string(s):
	"""given a string in a tei file created by grobit, cleans up gorbit mistakes. puts diacritics back on their letter, removes diaresis (ex : wo-rd)
	diacritics is done. diaresis not yet
	needed for some documents, not all of them"""

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

def recursive_add_hyperonyms_mesh(label,latest_id,nb_columns):
	"""given a mesh label present in mesh_hierarchy_up, fetches info for all of its hyperonyms, recursively. adds edges and nodes to global variables "nodes" and "edges". "keywords_id_dict" is also globally edited. Yes it's messy, sorry future me
	mesh_hierarchy_up is a dict[label]=[label_up,label_up] where label_up is an hyperonym of label"""

	global nodes
	global edges
	global keywords_id_dict

	#print("start recurse",label,latest_id)
	if label in mesh_hierarchy_up:
		hyperonyms=mesh_hierarchy_up[label]
		for label_up in hyperonyms:

			if label_up==label:
				raise ValueError("Error in mesh hierarchy data. This term makes a loop: "+label)

			#add node and edges connected to current hyponym
			if label_up not in keywords_id_dict:
				latest_id+=1
				keywords_id_dict[label_up]=latest_id
				row=[latest_id,label_up,"keyword"]
				row += [''] * (nb_columns - len(row)) #pads empty columns
				nodes.append(row)
			id_1=keywords_id_dict[label]
			id_2=keywords_id_dict[label_up]
			weight=0.1
			edge=[id_1,id_2,"directed",0.1]
			if edge not in edges:
				edges.append(edge)

			#recurse toward higher hyperonym
			try:
				latest_id=recursive_add_hyperonyms_mesh(label_up,latest_id,nb_columns)
			except RecursionError:
				raise RecursionError("Maximum recursion depth reached. It's likely the mesh hierarchy data makes a loop. Investigate these nodes : "+label+" ; "+label_up)
			#print("out recursive",label,label_up,latest_id)

	return(latest_id)

def list_titles(nodes_file="august_all_nodes.csv"):
	"""from a nodes.csv file, lists the titles of all articles inside"""

	titles=[]
	with open(nodes_file,mode="r",encoding="utf-8") as f:
		reader=csv.reader(f)
		next(reader)
		for l in reader:
			label=l[1]
			type_=l[2]
			if type_=="article":
				label=label.strip()
				label=label.lower()
				titles.append(label)

	return titles

def copy_index_aman(titles_list=list_titles(),aman_dat_file="aman git/acm/id_title.dat"):
	"""given a .dat file from aman, extracts the correspondence between article titles and article id"""

	with open(aman_dat_file,mode="r",encoding="utf-8") as f:
		for l in f:
			l=l.strip()
			id_,title=l.split("\t")
			title=title.lower()
			if title not in titles_list:
				print(title)

def separate_sections_article(soup,keep_all_words=True):
	"""used to separate an article into several nodes, each node being a part of the article (introduction, methods, etc)"""

	current_main_title="trash"
	title_clean="trash"
	output=defaultdict(list) #key=main section title. value=list of sentences
	output["abstract"]=utilsperso.preprocess_text(soup.abstract,keep_all_words=keep_all_words,separate_sentences=False)
	output["introduction"]=[""] #initalised to prevent a bug downstream if those sections don't exist
	output["material and method"]=[""]
	output["conclusion"]=[""]
	soup.abstract.decompose()
	for tag in soup.find_all(["head","p"]): #iterate through head and p tags in the same order they come through the file. head=section title. p=content of the section. So we read the title of a section, followed by its content
		if tag.name=="head":
			#detect if this is a section title we are interested in (introduction, methods...)

			title_clean=utilsperso.process_section_title(tag.getText()) #title_clean is either something like "introduction" if it's an important section, "other" if it's something not recognized, or is empty if it's a trash section such as "acknowledgments"
			if title_clean:
				if title_clean == "other": #random title in the body. it may be a subtitle inside a main section, so we want to keep it attached to the current title_clean
					pass
				else: #a maintitle such as "introduction"
					current_main_title=title_clean
			else: #title of something we don't want to keep
				current_main_title="trash"

		else: #p tag, we want to get the text
			if current_main_title=="trash":
				continue #we don't keep this part at all
			else:
				text=utilsperso.preprocess_text(tag,keep_all_words=keep_all_words,separate_sentences=False) #we need the stopwords for rake
				output[current_main_title]+=text
	return output

def extract_acm(folder="aman git/acm"):
	"""main function if the corpus is acm"""
	pass

def idf_acm():
	"""use this once at the start to generate the idf for this particular corpus"""

	documents=[]

	with open("aman git/acm/id_abstract",mode="r",encoding="utf-8") as f:
		for l in f:
			l=re.sub("^\n+ '","",l) #removes the numerical id at the start of each line + opening '
			l=l[:-1] #removes ending ' at the end of the line
			l=l.strip()

	utils_perso.edit_idf(documents,filetype="raw_text",idf_file="acm_idf.pickle")
	return

def extract_canceropole(root_directory):
	"""main function if the corpus is canceropole"""

	id_article=-1
	authors_id_dict={}
	author_edges=[] #list of lists [id_article, id_author, id_author, id_author]
	titles={}
	option_used_text="abstract"
	abstracts_matrix=[]
	fulltexts_matrix=[]
	mainsections_matrix=[]
	abstracts_raw=[]

	idf=utilsperso.finish_idf("27_07_tei_idf.pickle") #put here the .pickle file containing an idf
	word_matrix=idf.keys()

	tmp_missing_titles={}
	with open("missing_titles.txt",mode="r",encoding="utf-8") as f:
		for l in f:
			l=l.strip()
			l=l.split("\t")
			fname=l[0].lower()
			fname=fname.replace(" ","")
			tmp_missing_titles[fname]=l[1]

	try:
		for path, dirs, files in os.walk(root_directory):
			for fname in files: #one file = one article
				if fname.endswith(".xml"):
					print("processing "+fname)
					with open(path+"/"+fname,mode="r",encoding="utf-8") as f:
						lines="".join(f.readlines())
						soup=BeautifulSoup(lines,"xml")

					#article id and basic infos
					article_title=soup.find("titleStmt").getText()
					if len(article_title)<3:
						a=fname[:-8].lower()
						a=a.replace(" ","")
						article_title=tmp_missing_titles[a]
					id_article+=1
					article_title=article_title.strip()
					titles[id_article]=article_title
					abstract=soup.abstract.getText()
					abstract=abstract.strip()
					abstracts_raw.append(abstract)


					#authors
					article_authors_list=[]
					for author_block in soup.sourceDesc.find_all("author"):
						try :
							author=author_block.persName.getText(separator=" ")
						except AttributeError: #empty author
							continue

						#add node if the author doesn't already exist
						if author not in authors_id_dict:
							try:
								id_author=max(authors_id_dict.values())+1
							except ValueError:#first time, dict is empty
								id_author=0
							authors_id_dict[author]=id_author
						else:
							id_author=authors_id_dict[author]

						article_authors_list.append(id_author)
					#authors edges
					author_edges.append([id_article]+article_authors_list)



					#process the text

					for text_mode in ["abstract","fulltext","main_sections"]:

						#select which parts of the article we want to take into account
						if text_mode =="fulltext":
							# all, including the abstract
							#soup.abstract.decompose() #removes the abstract from the text
							text=utilsperso.preprocess_text(soup,keep_all_words=False)
						elif text_mode=="abstract":
							text=utilsperso.preprocess_text(soup.abstract,keep_all_words=False)
							abstract_raw=utilsperso.preprocess_text(soup.abstract,keep_all_words=True)
						elif text_mode=="main_sections":
							#abstract + intro + material and methods + conclusion
							text=utilsperso.preprocess_text(soup.abstract,keep_all_words=False) #add abstract
							sections=separate_sections_article(soup)
							for section in sections:
								if section in ["introduction","material and method","conclusion"]:
									text+=sections[section]
						else:
							raise ValueError


						try:
							text=[x.lemma_ for x in text]
						except AttributeError: #spacy being annoying
							text2=[]
							for token in text:
								try:
									token=token.lemma_
								except AttributeError:
									token=token
								text2.append(token)
							text=text2


						#convert mesh terms
						text2=[]
						for word in text:
							if word in mesh_synonyms:
								word=mesh_synonyms[word]
							text2.append(word)
						text=text2


						#get score
						tf=utilsperso.count_text_tfidf(" ".join(text),idf)
						article_matrix=[id_article]
						for word in word_matrix:
							if word in text:
								score=tf[word]
							else:
								score=0
							article_matrix.append(score)

						if text_mode=="abstract":
							abstracts_matrix.append(article_matrix)
						elif text_mode=="fulltext":
							fulltexts_matrix.append(article_matrix)
						elif text_mode=="main_sections":
							mainsections_matrix.append(article_matrix)


						continue



						#add the keywords we found to the nodes/edges
						for keyword in keywords_found:
							

							#get the printable/clean version of the keyword
							keyword_propre=keyword
							if keyword in mesh_synonyms:
								#print("mesh synonym detected",keyword,mesh_synonyms[keyword])
								keyword_propre=mesh_synonyms[keyword]

							if keyword_propre in keyword_exclude: #this keyword is garbage, ignore it
								continue

							#add node if doesn't exist
							if keyword_propre not in keywords_id_dict:
								previous_id+=1
								keywords_id_dict[keyword_propre]=previous_id
								keyword_id=previous_id
								row=[keyword_id,keyword_propre,"keyword"]
								row += [''] * (nb_columns - len(row)) #pads empty columns
								nodes.append(row)
							else:
								keyword_id=keywords_id_dict[keyword_propre]

							#add edge
							weight=keywords_found[keyword]
							if option_separate_paper_parts:
								#connect to the paper section node
								edges.append([section_node_id,keyword_id,"undirected",weight])
							else:
								#connect the the whole paper node
								edges.append([article_id,keyword_id,"undirected",weight])


	except KeyboardInterrupt:
		pass #used to break the loop when we have enough files processed
	

	with open("aman_output/id_authorname.dat",mode="w",encoding="utf-8") as f:
		for author_name in authors_id_dict:
			f.write(str(authors_id_dict[author_name])+" "+author_name+"\n")
	
	with open("aman_output/id_author.dat",mode="w",encoding="utf-8") as f:
		for entry in author_edges:
			entry=[str(x) for x in entry] #won't " ".join if it's int
			f.write(" ".join(entry)+"\n")

	with open("aman_output/id_title.dat",mode="w",encoding="utf-8") as f:
		for article in titles:
			f.write(str(article)+" "+titles[article]+"\n")

	with open("aman_output/id_words.dat",mode="w",encoding="utf-8") as f:
		for i,word in enumerate(word_matrix):
			f.write(str(i)+" "+word+"\n")
	
	with open("aman_output/id_abstract.dat",mode="w",encoding="utf-8") as f:
		for article in abstracts_raw:
			f.write(article+"\n")

	for variable,fname in ([abstracts_matrix,"abstract"],[fulltexts_matrix,"fulltext"],[mainsections_matrix,"mainsections"]):

		with open("aman_output/id_"+fname+"_score.dat",mode="w",encoding="utf-8") as f:
			for article in variable:
				article=[str(x) for x in article]
				f.write(" ".join(article)+"\n")


		with open("aman_output/id_"+fname+"_binary.dat",mode="w",encoding="utf-8") as f:
			for article in variable:
				article=[1 if x!=0 else x for x in article ]
				article=[str(x) for x in article]
				f.write(" ".join(article)+"\n")



if __name__=="__main__":
	#extract_canceropole("fulltext_tei/all")
	extract_acm("aman_git/acm")
	
