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
import bibtexparser #https://github.com/sciunto-org/python-bibtexparser
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

def separate_sections_article_canceropole(soup,keep_all_words=True):
	"""used to separate an article into several nodes, each node being a part of the article (introduction, methods, etc)
	returns a dict["name of section"]="text of section" 
	"other", "trash" etc is cleaned up inside this function. only real, major sections are returned"""

	current_main_title="trash"
	title_clean="trash"
	output=defaultdict(list) #key=main section title. value=list of sentences
	output["abstract"]=utilsperso.preprocess_text(soup.abstract,keep_all_words=keep_all_words,separate_sentences=False,return_lemmas=True)
	#output["introduction"]=[""] #initalised to prevent a bug downstream if those sections don't exist
	#output["material and method"]=[""]
	#output["conclusion"]=[""]
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

def separate_sections_article_acl(fulltext):
	"""argument must be the text with linebreaks in it"""
	
	current_main_title="trash"
	title_clean="trash"
	output=defaultdict(list) #key=main section title. value=list of sentences

	sections={}
	current_title="trash"
	for line in fulltext:
		current_section=[]
		is_title=False
		line2=utilsperso.process_section_title(line,restricted_allowed_titles=True)
		if line2==False: #stuff like "author contribution", we want to remove it from the fulltext
			current_title="trash"
			continue
		elif line2=="other": #probably not a section title, an ordinary line
			current_section.append(line)
		else: #good section title
			sections[current_title]=" ".join(current_section)


def extract_acm(folder="aman git/acm"):
	"""main function if the corpus is acm"""
	idf=utilsperso.finish_idf("acm_idf.pickle") #make it with idf_acm()
	word_matrix=idf.keys()


	with open("aman_output/acm/id_words.dat",mode="w",encoding="utf-8") as f:
		for i,word in enumerate(word_matrix):
			f.write(str(i)+" "+word+"\n")

	try :
		with open("aman git/acm/id_abstract.dat",mode="r",encoding="utf-8") as f,\
		open("aman_output/acm/id_abstract_tfidf_score.dat",mode="w",encoding="utf-8") as f_idf_score,\
		open("aman_output/acm/id_abstract_tfidf_binary.dat",mode="w",encoding="utf-8") as f_idf_binary:
			for l in f:
				id_=re.match("^\d+ ",l).group(0)[:-1]
				sys.stderr.write("processing "+id_+" ("+str(int(int(id_)/12498.0))+"%)\n")
				l=re.sub("^\d+ '","",l) #removes the numerical id at the start of each line + opening '
				l=l[:-1] #removes ending ' at the end of the line
				l=l.strip()

				#preprocessing
				text=utilsperso.preprocess_text(l,return_lemmas=True)
				tf=utilsperso.count_text_tfidf(" ".join(text),idf)

				#tf-idf score
				score_matrix=[id_]
				binary_matrix=[id_]
				for word in word_matrix:
					if word in text:
						score=str(tf[word])
						score_matrix.append(score)
						binary_matrix.append("1")
					else:
						score_matrix.append("0")
						binary_matrix.append("0")

				f_idf_score.write(" ".join(score_matrix)+"\n")
				f_idf_binary.write(" ".join(binary_matrix)+"\n")
	except KeyboardInterrupt:
		pass #manually break loop
	return


def idf_acm():
	"""use this once at the start to generate the idf for this particular corpus"""

	documents=[]

	try:
		with open("aman git/acm/id_abstract.dat",mode="r",encoding="utf-8") as f:
			for l in f:
				l=re.sub("^\n+ '","",l) #removes the numerical id at the start of each line + opening '
				l=l[:-1] #removes ending ' at the end of the line
				l=l.strip()

				#preprocessing
				text=utilsperso.preprocess_text(l)
				text2=[]
				for token in text:
					if token.pos_=="NUM":
						text2.append("NUM")
					else:
						text2.append(token.lemma_)
				text=" ".join(text2)
				documents.append(text)
	except KeyboardInterrupt:
		pass

	utilsperso.edit_idf(documents,filetype="raw_text",idf_file="acm_idf.pickle")
	return


def extract_acl_anthology(root_directory,output_directory):
	"""main function if the corpus is acl anthology"""


	#list the articles present in the cleaned full text folder. this allows to prune the .bib later and keep less things in memory
	article_ids=set([])
	for fname in os.listdir(root_directory+"/cleaned_acl_arc"):
		fname=fname[:-4] #drop .txt
		fname=fname.upper()
		article_ids.add(fname)
	

	#parse the bibtex
	#bibtex_file=root_directory+"/anthology+abstracts.bib"
	bibtex_file=root_directory+"/anthology_small.bib"
	sys.stderr.write("loading bib, this may take a while...\n")
	with open(bibtex_file) as bibtex_f:
		parser = bibtexparser.bparser.BibTexParser(common_strings=True)
		parser.customization = bibtexparser.customization.convert_to_unicode
		bib_database = bibtexparser.load(bibtex_f, parser=parser)
	
	sys.stderr.write("bib loaded !\n")
	

	#collect the bib for each article that we have the fulltext of
	bib={}
	i=0
	for entry in bib_database.entries:
		if "url" in entry.keys():
			url=entry["url"]
			id_=url.split("/")[-1].upper()
			if id_ in article_ids:
				bib[id_]=entry


	#TODO fix cut words

	#calculate idf
	#run this only once the first time
	if False:
		documents={}
		for i,id_ in enumerate(bib):
			entry_bib=bib[id_]
			with open(root_directory+"/cleaned_acl_arc/"+id_+".txt",mode="r",encoding="iso-8859-1") as f:
				text=f.readlines()
				text=" ".join(text)

				#preprocess text
				text=utilsperso.preprocess_text(text,return_lemmas=True)
				text=" ".join(text)
				documents[id_]=text

				if len(documents)%500==0:
					sys.stderr.write("Sending a batch of idf to process\n")
					#utilsperso.edit_idf(documents,filetype="acl",idf_file="acl_anthology_september.pickle")
					documents={}
					progress=float(i)/len(documents)
					sys.stderr.write(str(progress)+" % completed !\n")
	

		sys.stderr.write("Sending last batch of idf to process\n")
		#utilsperso.edit_idf(documents,filetype="acl",idf_file="acl_anthology_september.pickle")
		documents={}

	#load word matrix + idf
	idf=utilsperso.finish_idf("acl_anthology_september.pickle",prune_threshold=5)
	word_matrix=idf.keys()

	#retrieve the full text for each article and process stuff
	id_authors={}
	keys=set([])
	#all output files are written line by line, article by article. so they have to be open at all times.
	with open(output_directory+"/id_title.dat",mode="w",encoding="utf-8") as ftitle, \
	open(output_directory+"/id_title_score.dat",mode="w",encoding="utf-8") as ftitlescore, \
	open(output_directory+"/id_title_binary.dat",mode="w",encoding="utf-8") as ftitlebinary, \
	open(output_directory+"/id_author.dat",mode="w",encoding="utf-8") as fauthor, \
	open(output_directory+"/id_fulltext_score.dat",mode="w",encoding="utf-8") as ffullscore :
		for id_ in bib: #loop over articles


			#get info from the bib (author, title, etc)

			entry_bib=bib[id_]
			sys.stderr.write("processing article"+str(id_)+"\n")

			#authors
			authors=entry_bib["author"]
			authors=authors.split("and\n")
			authors2=[id_] #adding it here assures that a paper without authors will not result in a skipped line
			for author in authors:
				author=author.strip()
				#TODO more preprocessing
				if author in id_authors:
					id_author=id_authors[author]
				else:
					try:
						id_author=max(id_authors.keys())+1
					except ValueError:
						id_author=0
					id_authors[id_author]=author
				authors2.append(str(id_author))
			fauthor.write((" ".join(authors2))+"\n")

			#title
			title=entry_bib["title"]
			ftitle.write(str(id_)+" "+title+"\n")
			title=utilsperso.preprocess_text(title,return_lemmas=True)
			title_matrix=utilsperso.make_word_matrix(title,word_matrix,scores=idf,output_strings=True) #doesn't make sense to compute the tf-idf of a title, score is just the idf
			title_matrix2=" ".join(title_matrix)
			ftitlescore.write(str(id_)+" "+title_matrix2+"\n")
			title_matrix=["0" if x=="0" else "1" for x in title_matrix ]
			title_matrix=" ".join(title_matrix)
			ftitlebinary.write(str(id_)+" "+title_matrix+"\n")


			#the abstract can be either in the bib or in the full text
			abstract=False
			if "abstract" in entry_bib:
				abstract=entry_bib["abstract"]


			#get info from the full text (word matrices for each section)

			with open(root_directory+"/cleaned_acl_arc/"+id_+".txt",mode="r",encoding="iso-8859-1") as ffulltext :


				#get full text
				with open(root_directory+"/cleaned_acl_arc/"+id_+".txt",mode="r",encoding="iso-8859-1") as f:
					text=f.readlines()
					text="\n".join(text)
					
					if text.lower().split(" ",1) in ["abstract","abstract.","summary","summary."]:
						abstract=True


					#preprocess text
					text=utilsperso.preprocess_text(text,return_lemmas=True)
					tf=utilsperso.count_text_tfidf(" ".join(text),idf)
					current_word_matrix=utilsperso.make_word_matrix(text,word_matrix,tf,True)
					ffullscore.write((" ".join(current_word_matrix))+"\n")
					#TODO add binary output


	with open(output_directory+"/id_authorname.dat",mode="w",encoding="utf-8") as fauthorname :
		for id_ in id_authors:
			fauthorname.write(str(id_)+" "+id_authors[id_]+"\n")

	with open(output_directory+"/id_words.dat",mode="w",encoding="utf-8") as fwords :
		for word in word_matrix:
			fwords.write(word+"\n")



def extract_canceropole(root_directory):
	"""main function if the corpus is canceropole"""

	id_article=-1
	authors_id_dict={}
	author_edges=[] #list of lists [id_article, id_author, id_author, id_author]
	titles={}
	sections_matrices=[] #list of articles. each list element is a list, a word matrix for that article. a word matrix is a list of scores where the index gives what word is concerned, and the value gives the score for that word.
	abstracts_raw=[]

	idf=utilsperso.finish_idf("27_07_tei_idf.pickle") #put here the .pickle file containing an idf
	word_matrix=idf.keys() #this list gives the indices for words in the word matrices

	tmp_missing_titles={} #for some annoying papers in this corpus that don't process properly
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
					abstract=re.sub("\n"," ",abstract)
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

					text=utilsperso.preprocess_text(soup.abstract,keep_all_words=False,return_lemmas=True) #add abstract
					sections=separate_sections_article_canceropole(soup)
					sections["fulltext"]=utilsperso.preprocess_text(soup,keep_all_words=False,return_lemmas=True)

					current_paper_matrices={}
					for section in sections: #section = abstract, fulltext, intro, methods...

						text=sections[section]

						if True: #should not be needed anymore
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

						#save matrix
						current_word_matrix=[]
						for word in word_matrix:
							if word in text:
								score=tf[word]
							else:
								score=0
							current_word_matrix.append(score)
						current_paper_matrices[section]=current_word_matrix
					sections_matrices.append(current_paper_matrices)





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

	#list_possible_sections=set([])
	#for article in sections_matrices:
	#	for section in sections_matrices[article]:
	#		if section not in list_possible_sections:
	#			list_possible_sections.add(section)
	#print("lama",list_possible_sections)
	#exit() #HERE, trying to print everything to the correct file. below is old version.
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

def count_sections(source):
	"""counts the section titles in one whole corpus and outputs stats
	source : acl or canceropole"""

	if source not in ["canceropole","acl"]:
		raise ValueError
	
	if source=="canceropole":
		folder="fulltext_tei/all"
		encoding='utf-8'
	else:
		folder="/home/sam/work/corpora/acl/cleaned_acl_arc"
		encoding="iso-8859-1"
	i_articles=0
	titles=defaultdict(int)
	try :
		for path, dirs, files in os.walk(folder):
			for fname in files:
				with open(path+"/"+fname,mode="r",encoding=encoding) as f:
					i_articles+=1

					if source=="acl":
						for l in f:
							m=re.match("^\d+[\. ]",l)
							if m:
								l=re.sub("[\d \.]","",l)
								l=utilsperso._clean_section_title(l)
								if l:
									if len(l)>0:
										titles[l]+=1

					else:
						lines="".join(f.readlines())
						soup=BeautifulSoup(lines,"xml")
						a=soup.find_all("head")
						for title in a:
							title=utilsperso._clean_section_title(title.getText())
							if title:
								if len(title)>0:
									titles[title]+=1

				if i_articles%10==0:
					pass
					#print(str(i_articles)+" traites. "+str(len(titles.keys()))+" titres trouvés")
	except KeyboardInterrupt:
		pass

	for key in sorted(titles, key=titles.get, reverse=True):
		print(key+"\t"+str((titles[key])/float(i_articles)*100)[:2]+"%")

if __name__=="__main__":
	#extract_canceropole("fulltext_tei/all")
	#extract_acl_anthology("/home/sam/work/corpora/acl","aman_output/acl")
	extract_acm("aman git/acm")
	
