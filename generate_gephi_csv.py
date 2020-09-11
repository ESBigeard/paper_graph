#!/usr/bin/python
# -*- coding:utf-8 -*-
""" takes a folder of tei xml generated by grobid, output nodes.csv and edges.csv for gephi """

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
#from rake_nltk import Rake
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
import numpy as np

#import local scripts
import utilsperso

import spacy
nlp=spacy.load("en_core_web_sm")

#global variables
nodes=[]
edges=[]
keywords_id_dict={}

#rake=Rake(max_length=4)

idf=utilsperso.finish_idf("27_07_tei_idf.pickle") #put here the .pickle file containing an idf

mesh_hierarchy_up=defaultdict(list)
mesh_hierarchy_down=defaultdict(list)
mesh_keywords=set([])
keyword_exclude=set([])

tmp_missing_titles={}
with open("missing_titles.txt",mode="r",encoding="utf-8") as f:
	for l in f:
		l=l.strip()
		l=l.split("\t")
		fname=l[0].lower()
		fname=fname.replace(" ","")
		tmp_missing_titles[fname]=l[1]

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


idfs={}
def load_idf():
	global idfs
	text_sections=["material and method","result","materials and method","introduction","conclusion","abstract","background","discussion"]
	idfs={}
	for section in text_sections:
		fname=section+".idf.pickle"
		with open(fname,mode="rb") as f:
			idf=pickle.load(f)
			if section=="materials and method":
				for word in idf:
					occurences=idf[word]
					idfs["material and method"][word]+=occurences
			else:
				idfs[section]=idf
	for section in idfs:
		idf=idfs[section]
		idf=utilsperso.finish_idf(idf)
		idfs[section]=idf
	

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

def get_clean_text_from_directory(directory,key="article_title",text_section="fulltext"):
	""" used by cluster_documents which is currently unused and i'm not sure what i used to do with it. so probably not useful to keep ?
	from a directory of xml tei generated by grobit, returns a dictionary where the key is the article title and the data is the flat preprocessed text of one xml file
	by default the key of the dictionnary is the article title. but you can change it to the file path instead with 'path'
	by default returns the full text, but option output can be changed for 'abstract' or 'aic' for abstract+intro+ccl"""

	if text_section=="article_title":
		text_section="title"

	documents={}
	try:
		for fname in os.listdir(directory):
			#for fname in os.listdir(directory+"/"+subdir): #WARNING made changes here to work with subdirectories
			if True:
				if fname.endswith(".xml"):
					sys.stderr.write("processing "+fname+"\n")
					#with open(directory+"/"+subdir+"/"+fname,mode="r",encoding="utf-8") as f:
					with open(directory+"/"+fname,mode="r",encoding="utf-8") as f:
						lines="".join(f.readlines())
						soup=BeautifulSoup(lines,"xml")
					#fname=subdir[-1]+"/"+fname

					
					article_title=soup.find("titleStmt").getText()
					if text_section=="title":
						textlem=article_title
					else:


						for tags in [soup("ref")]:
							for tag in tags:
								tag.decompose()

						#selects what text we want
						if text_section=="fulltext":
							fulltext=utilsperso.preprocess_text(soup,keep_all_words=False)
						elif text_section=="abstract":
							fulltext=utilsperso.preprocess_text(soup.abstract,keep_all_words=False)
						elif text_section=="aic":
							pass #TODO do stuff
						else:
							raise Exception("bad argument")


						textlem=[]
						for token in fulltext:
							textlem.append(token.lemma_)
						textlem=" ".join(textlem)

					if key=="article_title":
						documents[article_title]=textlem
					else:
						documents[fname]=textlem
	except KeyboardInterrupt:
		pass #break the loop when we have enough data
	
	return documents

def cluster_documents(directory):

	documents=get_clean_text_from_directory(directory,key="path",text_section="title")

	sys.stderr.write("Vectorizing...\n")


	vectorizer=TfidfVectorizer()
	vectorized_text=vectorizer.fit_transform(documents.values())

	#data2=np.hstack((data,vectorized_text.toarray()))

		

	sys.stderr.write("Training...\n")
	kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
	kmean_indices = kmeans.fit_predict(vectorized_text)

	pca = PCA(n_components=2)
	scatter_plot_points = pca.fit_transform(vectorized_text.toarray())

	colors = ["r", "b", "c" ,"#666666","#000000" ]

	x_axis = [o[0] for o in scatter_plot_points]
	y_axis = [o[1] for o in scatter_plot_points]
	fig, ax = plt.subplots(figsize=(20,10))

	ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

	for i, txt in enumerate(documents.keys()):
		ax.annotate(txt, (x_axis[i], y_axis[i]))
	plt.show()


def test_keywords(root_directory):
	"""print juste l'abstract/fulltext de chaque article ET les keywords détectés. used for debugging"""

	for path, dirs, files in os.walk(root_directory):
		for fname in files: #one file = one article
			if fname.endswith(".xml"):
				print("processing "+fname)
				with open(path+"/"+fname,mode="r",encoding="utf-8") as f:
					lines="".join(f.readlines())
					soup=BeautifulSoup(lines,"xml")

				#title
				article_title=soup.find("titleStmt").getText()
				abstract=soup.abstract.getText()


				#text
				#needed to detect keywords
				text=utilsperso.preprocess_text(soup,keep_all_words=True)
				text=[x.lemma_ for x in text]
				text=" ".join(text)
				text=" "+text+" "
				abstract=utilsperso.preprocess_text(soup.abstract,keep_all_words=True)
				abstract=[x.lemma_ for x in abstract]
				abstract=" ".join(abstract)
				abstract=" "+abstract+" "

				#keywords
				explicit_keywords=[]
				a=soup.keywords
				if a:
					for term in a.find_all("term"):
						explicit_keywords.append(term.getText())

				#mesh keywords
				fulltext_keywords=[]
				abstract_keywords=[]
				for keyword in mesh_keywords:
					if " "+keyword+" " in text:
						fulltext_keywords.append(keyword)
					if " "+keyword+" " in abstract:
						abstract_keywords.append(keyword)

				#print
				print(fname)
				print()
				print("explicit keywords : ",";".join(explicit_keywords))
				print("abstract keywords : ",";".join(abstract_keywords))
				print("text keywords : ",";".join(fulltext_keywords))
				print()
				print("abstract :")
				print(abstract)
				print()
				print()
				print("full text :")
				print(text)
				print()
				print()
				print()
				print()
				print()

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


def print_csv_for_gephi(root_directory,option_used_text="fulltext",option_prune_keywords=True, option_separate_paper_parts=True,option_fname_out=False,option_communities=False):
	""" main function
	option_used_text=abstract, fulltext or main_sections. in which text do we look for keywords ? main_sections is abstract + introduction + methods + conclusion
	option_prune_keywords=True/False. do we keep all keywords found at least once in the text, or only the most relevant ?
	option_separate_paper_parts=True/False. is each article a node, or separate nodes for intro/method/ccl/etc ?
	option_fname_out : name of output csv files. default : fulltext or abstract depending of option_used_text
	option_communities : tmp for august batch, leave false in normal use"""
	


	global nodes
	global edges
	global keywords_id_dict
	#the global are necessary because a recursive function edits them. ok, not necessary, but a lot more convenient.

	if option_communities: #tmp for august batch
		comunities={}
		with open("fulltext_tei/august_batch/list.txt",mode="r",encoding="utf-8") as f:
			for l in f:
				l=l.strip()
				l=l.split(" ")
				a="".join(l[0:-1])
				comunities[a]=l[-1]

	if option_used_text not in ("fulltext", "abstract","main_sections","aman_output"):
		raise ValueError("Wrong value for argument option_used_text of function print_csv_for_gephi()")
	
	if option_separate_paper_parts:
		option_used_text="fulltext"
	


	authors_id_dict={} #dict["name of author"]=id of author
	keywords_id_dict={} #dict["text of keyword"]=id of keyword
			#the text of the author/keyword in key is used to detect duplicates
	nodes=[] #the info to output in the csv for nodes
	if option_used_text == "aman_output":
		nodes.append(["id","label","type","fulltext_link","community","abstract","introduction","methods","conclusion"])
	else:
		nodes.append(["id","label","type","fulltext_link","abstract","community"])
	nb_columns=len(nodes[0])
	previous_id=-1 #a different id for each node. articles, titles, keywords, etc, all share the same id numerotation
	edges=[] #the info to output in the csv for edges
	edges.append(["source","target","type","weight"])
	text_ids=[] #id of each article, used for text similarity (so, not used currently)
	text_text=[] #the text itself of each article, used for text similarity (so, not used currently)

	tmp_out_keywords={}#not used in normal use, makes a dict where each entry is a paper, which hosts another dict of paper sections, where the data is a list of keywords found with duplicates

	try:
		for path, dirs, files in os.walk(root_directory):
			for fname in files: #one file = one article
				if fname.endswith(".xml"):
					print("processing "+fname)
					with open(path+"/"+fname,mode="r",encoding="utf-8") as f:
						lines="".join(f.readlines())
						soup=BeautifulSoup(lines,"xml")

					#create article base node
					article_title=soup.find("titleStmt").getText()
					if len(article_title)<3:
						a=fname[:-8].lower()
						a=a.replace(" ","")
						article_title=tmp_missing_titles[a]
					previous_id+=1
					article_id=previous_id
					abstract=soup.abstract.getText() #to display to the user on the left column
					url="elisebigeard.yo.fr/gephi_files/fulltexts/"+fname[:-7]+"html"
					if option_communities:
						a=fname.replace(" ","")
						if a in comunities:
							comu=comunities[a]
						else:
							comu=""
					else:
						comu=""

					if option_used_text == "aman_output":
						pass #we don't have yet all the data we need to create the node in this mode
					else:
						nodes.append([article_id,article_title,"article",url,abstract,comu])
					#tmp_out_keywords[article_id]={}

					#author nodes, attached to article node
					for author_block in soup.sourceDesc.find_all("author"):
						try :
							author=author_block.persName.getText(separator=" ")
						except AttributeError: #empty author
							continue

						#add node if the author doesn't already exist
						if author not in authors_id_dict:
							previous_id+=1
							authors_id_dict[author]=previous_id
							author_id=previous_id
							row=[author_id,author]
							row += [''] * (nb_columns - len(row)) #pads empty columns
							nodes.append(row)
						else:
							author_id=authors_id_dict[author]

						#add edge
						edges.append([article_id,author_id,"undirected",1])

					#retrieve keywords explicitly given by the authors
					#those keywords are associated with the full paper, even if paper sections are in separate nodes
					a=soup.keywords
					if a:
						for term in a.find_all("term"): #for keyword in keywords
							keyword=term.getText()
							spacy_text=nlp(keyword)

							#lemmatize
							keyword=[x.lemma_ for x in spacy_text]
							keyword=" ".join(keyword)
							keyword=keyword.lower()

							if keyword in mesh_synonyms: #replace by a standard form
								#print("mesh synonym detected",keyword,mesh_synonyms[keyword])
								keyword=mesh_synonyms[keyword]

							if keyword in keyword_exclude: #ignore that keyword, it's junk
								continue

							#add node if doesn't exist
							if keyword not in keywords_id_dict:
								previous_id+=1
								keywords_id_dict[keyword]=previous_id
								keyword_id=previous_id
								row=[keyword_id,keyword,"keyword"]
								row += [''] * (nb_columns - len(row)) #pads empty columns
								nodes.append(row)
							else:
								keyword_id=keywords_id_dict[keyword]

							#add edge
							edges.append([article_id,keyword_id,"undirected",1])



					#process the text
					#needed to detect keywords

					#separates sections and transform into spacy text
					if option_separate_paper_parts:
						sections=separate_sections_article(soup)
					else: #put all the text in a single section called "fulltext"
						if option_used_text=="fulltext":
							#soup.abstract.decompose() #removes the abstract from the text
							text=utilsperso.preprocess_text(soup,keep_all_words=True)
						elif option_used_text=="abstract":
							text=utilsperso.preprocess_text(soup.abstract,keep_all_words=True)
						elif option_used_text=="main_sections":
							text=utilsperso.preprocess_text(soup.abstract,keep_all_words=True) #add abstract
							sections=separate_sections_article(soup)
							for section in sections:
								if section in ["introduction","material and method","conclusion"]:
									text+=sections[section]
						elif option_used_text=="aman_output":
							#we want to get the raw text of each section and add it to the node
							sections=separate_sections_article(soup,keep_all_words=True)
							for section in sections:
								if section in ["introduction","material and method","conclusion"]:
									text=sections[section]
									text=" ".join([str(x) for x in text])
									text=utilsperso.normalise_unicode(text,remove_non_ascii=True)
									sections[section]=text
							nodes.append([article_id,article_title,"article",url,comu,abstract,sections["introduction"],sections["material and method"],sections["conclusion"]])
							text=utilsperso.preprocess_text(soup,keep_all_words=True) #to detect the keywords in the fulltext
						else:
							raise ValueError
						sections={}
						sections["fulltext"]=text


						#text_text.append(text) #used for text similarity
						#text_ids.append(article_id) #used for text similarity

					#create nodes for paper sections and detect keywords
					for section_title in sections:
						#sys.stderr.write("processing "+section_title+"\n")
						text=sections[section_title]
						text=[x.lemma_ for x in text]
						text=" ".join(text)
						text=" "+text+" " #to look for full words using " "+word+" "

						if option_separate_paper_parts:
							#create node for paper section
							previous_id+=1
							section_node_id=previous_id
							title=section_title+" : "+article_title[:10]+"..."
							row=[section_node_id,title,"article_section_"+section_title]
							row += [''] * (nb_columns - len(row)) #pads empty columns
							nodes.append(row)
							edges.append([article_id,section_node_id,"undirected",1])

						#detect mesh keywords
						keywords_found=defaultdict(int)

						#remove part of the detected keywords to keep only the most relevant ones
						if option_prune_keywords: 

							i_keywords=0
							max_keywords=10 #how many keywords per article to keep
							
							if False: #among keywords that are a mesh term, keep top keywords per rake score
								rake.extract_keywords_from_text(text)
								top=rake.get_ranked_phrases() #here "phrases" mean multiword expressions, not sentences.
								for multiword in top:
									multiword=" "+multiword+" "
									multiword=multiword.lower()
									for keyword in mesh_keywords:
										if " "+keyword+" " in multiword:
											keywords_found[keyword]+=1
											i_keywords+=1
											if i_keywords>max_keywords:
												break

							else: #among keywords that are a mesh term, keep top keywords per tf-idf score
								if option_separate_paper_parts:
									#tf=utilsperso.count_text_tfidf(text,idf[section_title]) #if you want to have separate idf's for each section. didn't work well so commented out
									tf=utilsperso.count_text_tfidf(text,idf)
								else:
									tf=utilsperso.count_text_tfidf(text,idf)
								for word in sorted(tf, key=tf.get, reverse=True):
									if word in mesh_synonyms:
										i_keywords+=1
										keywords_found[word]=tf[word]
										if i_keywords>max_keywords:
											break

						#keep any mesh keyword that appear at least once in the text
						else:
							for keyword in mesh_keywords:
								if " "+keyword+" " in text.lower():
									weight=text.count(" "+keyword+" ")
									keywords_found[keyword]=weight



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
	

	#add edges between keywords according to mesh hierarchy
	for label1 in keywords_id_dict.copy(): #copy because we add new nodes during the loop, which python doesn't allow
		id_1=keywords_id_dict[label1]

		#recursively adds all levels of hyperonyms (more generic, upper-level terms)^
		# variables nodes, edges and keywords_id_dict are globally changed inside the function to add data.
		previous_id=recursive_add_hyperonyms_mesh(label1,previous_id,nb_columns)

		#old version, fetches only one level higher
		if False:
			hyperonymes=mesh_hierarchy_up[label1]
			for label2 in hyperonymes: #list of labels
				
				#fetch id or create node
				if True: #disable to avoir creating new nodes to avoid bloating the graph
					if label2 not in keywords_id_dict:
						previous_id+=1
						keywords_id_dict[label2]=previous_id
						row=[previous_id,label2,"keyword"]
						row += [''] * (nb_columns - len(row)) #pads empty columns
						nodes.append(row)
				if label2 in keywords_id_dict:
					id_2=keywords_id_dict[label2]
					edges.append([id_1,id_2,"directed",1])
		
		#fetches one label lower. should be useless normally.
		hyponymes=mesh_hierarchy_down[label1]
		for label2 in hyponymes: #list of labels
			
			#fetch id or create node
			if False: #disable to avoid creating new nodes to avoid bloating the graph
				if label2 not in keywords_id_dict:
					previous_id+=1
					keywords_id_dict[label2]=previous_id
					row=[previous_id,label2,"keyword"]
					row += [''] * (nb_columns - len(row)) #pads empty columns
					nodes.append(row)
			if label2 in keywords_id_dict:
				id_2=keywords_id_dict[label2]
				edges.append([id_2,id_1,"directed",1]) #reverse direction


	
	#text similarity
	if False:
		vectorizer=TfidfVectorizer()
		vect_text=vectorizer.fit_transform(text_text)
		pair_similarity=vect_text*vect_text.T
		pair_similarity=pair_similarity.toarray()
		for i,column in enumerate(pair_similarity):
			article_source=text_ids[i]
			for i,score in enumerate(column):
				article_target=text_ids[i]
				score=score*50
				
				if article_source==article_target:
					continue
				
				edges.append([article_source,article_target,"undirected",score])


	if option_fname_out :
		fname=option_fname_out
	else:
		if option_separate_paper_parts:
			fname="separate_paper_parts"
		elif option_used_text=="fulltext":
			fname="fulltext"
		elif option_used_text=="abstract":
			fname="abstract"
		else:
			raise Exception("Problem with options in print_csv_for_gephi()")

	#write nodes
	with open(fname+"_nodes.csv",mode="w", encoding="utf-8") as f:
		writer=csv.writer(f)
		writer.writerows(nodes)
	
	#write edges
	with open(fname+"_edges.csv",mode="w", encoding="utf-8") as f:
		writer=csv.writer(f)
		writer.writerows(edges)



def stats_keywords(dict_keywords):

	text_sections=["material and method","result","introduction","conclusion","abstract","background","discussion"] #tel que dans les data en entrée
	label_sections=["mat/meth", "results", "intro", "ccl","abstract","backg.","disc."] #tel qu'affiché sur le graph en sortie

	totaux=[]
	totaux_distinct=[]
	totaux_unique=[]

	for section in text_sections:
		totaux.append(0)
		totaux_distinct.append(0)
		totaux_unique.append(0)

	for paper in dict_keywords:
		seen=defaultdict(int)
		for section in dict_keywords[paper]:
			idf=dict_keywords[paper][section]
			tot=0
			tot_distinct=0

			for word in idf:
				tot+=idf[word] #+ le nombre de fois où le mot apparait dans la section
				tot_distinct+=1 #+1 si le word est présent dans cette section
				seen[word]+=1

			section_index=text_sections.index(section)
			totaux[section_index]+=tot
			totaux_distinct[section_index]+=tot_distinct

		for section in dict_keywords[paper]:
			idf=dict_keywords[paper][section]
			tot_unique=0
			for word in idf:
				if seen[word]==1:
					tot_unique+=1
			section_index=text_sections.index(section)
			totaux_unique[section_index]+=tot_unique
	


	x=np.arange(len(label_sections))
	width=0.25
	fig,ax=plt.subplots()
	rects1 = ax.bar(x - width/2, totaux, width, label='Total')
	rects2 = ax.bar(x + width/3, totaux_distinct, width, label='Distinct')
	rects2 = ax.bar(x + width, totaux_unique, width, label='Unique')
	ax.set_xticks(x)
	ax.set_xticklabels(label_sections)
	ax.legend()
	#autolabel(rects1)
	#autolabel(rects2)
	fig.tight_layout()
	plt.show()

if __name__=="__main__":
	directory="fulltext_tei/all" #change this as needed

	#test_keywords(directory)
	print_csv_for_gephi(directory,"aman_output",option_separate_paper_parts=False,option_prune_keywords=True,option_fname_out="august",option_communities=True)
	#stats_keywords(tmp_keywords)



