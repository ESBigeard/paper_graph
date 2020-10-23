#!/usr/bin/python
# -*- coding:utf-8 -*-
"""This module is used by several other scripts, be prudent making changes. Can also be used alone to generate an idf"""

import re
import spacy
import pickle
import os
import sys
import math
import unicodedata
from collections import defaultdict,Counter
from bs4 import BeautifulSoup

nlp=spacy.load("en_core_web_sm")
sentence_tokeniser=nlp.create_pipe("sentencizer")
nlp.add_pipe(sentence_tokeniser)

def normalise_unicode(s,keep_diacritics=True,remove_non_ascii=False):
	"""normalise a string toward a standard unicode string, w/ or w/o diacritics

	normalise une chaine vers une unicode string standard, avec ou sans diacritiques

	:param arg1: string to normalise
	:type arg1: str or unicode
	:param arg2: True to keep diacritics, False to delete them. Default : keep them
	:type arg2: bool

	:example:
	>>> normalise_unicode(u"\\xc3\\xa9ternel",True)
	u"\\xe9ternel"
	"""
	#note : if you look at the code of the example above, the double backslashes are escaped single backslashes. those are to be read as single backslashes

	if remove_non_ascii:
		keep_diacritics=False

	if keep_diacritics:
		nf=unicodedata.normalize('NFKC',s)
	else:
		nf=unicodedata.normalize('NFKD',s)
	nf=nf.replace(u'\u0153','oe')

	if False : #normaliser les whitespaces
		nf2=""
		for char in nf:
			if not re.match(u"\s",char,re.UNICODE) or char in ["\n"," "]:
				nf2+=char
		return u''.join(c for c in nf2 if not unicodedata.combining(c))

	if keep_diacritics:
		return nf
	else:
		a=u''.join(c for c in nf if not unicodedata.combining(c))
		if remove_non_ascii:
			a=a.encode("ascii","ignore")
			a=a.decode("ascii")
			return a
		else:
			return a
	

def finish_idf(idf_file,prune_threshold=False):
	"""do that to an idf generated by edit_idf (stored in a idf.pickle). edit_idf only generates a dictionnary of word occurences. this turns it into word frequencies
	argument prune_threshold : specify a minimal occurence to keep a word in the vocabulary. This makes the vocabulary smaller, useful if you have issues processing the complete vocabulary. 5 is a good threshold. By default no pruning is performed """

	if not idf_file.endswith(".pickle"):
		raise ValueError("Argument idf_file must be a pickle file")

	with open(idf_file,mode="rb") as f:
		try:
			nb_documents,idf=pickle.load(f)
		except Exception as e:
			raise Exception("Something went wrong when loading the idf file "+idf_file+" you may be trying to load an older version that is not compatible anymore") from e

	#max_freq=float(idf.most_common(1)[0][1])
	idf2={}
	for word in idf:
		if prune_threshold:
			if idf[word]>prune_threshold:
				idf2[word]=math.log(nb_documents / float(idf[word]))
		else:
			idf2[word]=math.log(nb_documents / float(idf[word]))

	return idf2

def _get_words_from_grobit_xml(filename):
	"""from a grobit xml file, extracts the list of all encountered words. returns a list of words in the order they were encountered, so they typically appear several times. stopwords etc are removed"""
	wordlist=[]

	if type(filename)==str:
		with open(filename,mode="r",encoding="utf-8") as f:
			lines="".join(f.readlines())
			soup=BeautifulSoup(lines,"xml")
	elif type(filename)==bs4.BeautifulSoup:
		soup=filename
	else:
		raise TypeError("argument filename should be a path to a file or a beautifulsoup")

	for tags in [soup("ref")]:
		for tag in tags:
			tag.decompose()

	#identify the sections of the text
	#we care about the sections to be able to exclude stuff like "acknowledgment"
	current_main_title="other"
	for tag in soup.find_all(["head","p"]):
		if tag.name=="head":
			title_clean=process_section_title(tag.getText()) 
			if title_clean:
				if title_clean == "other": #random title in the body
					pass
				else: #a maintitle such as "introduction"
					current_main_title=title_clean
			else: #title of something we don't want to keep
				current_main_title="trash"

		else: #p tag, we want to get the text
			tokens=preprocess_text(tag,keep_all_words=False,separate_sentences=False)
			if current_main_title=="trash":
				continue
			else:
				#we have clean text, we add it to the wordlist
				for token in tokens:
					if token.pos_ == "NUM":
						wordlist.append("NUM")
					wordlist.append(token.lemma_)

	return wordlist

def make_word_matrix(words,matrix,scores=False,output_strings=False):
	"""transforms a text (list of words) into a matrix (list of scores where the position in the list indicates if this word was in this text
	words : list of words of the text, or string of text to be tokenised
	matrix : list of possible words in the correct order
	scores : tf-idf scores. if False, will return 1 or 0 if the word is present/absent
	output_strings : the outputed list contains numbers as string, instead of int. saves time if you're just going to print the result anyway"""

	if type(words)==str:
		words=re.split("\W+",words)

	output=[]
	for word in matrix:
		if word in words:
			if scores:
				result=scores[word]
			else:
				result=1
		else:
			result=0
		if output_strings:
			result=str(result)
		output.append(result)
	return output

def edit_idf(documents_list,filetype="raw_text",idf_file="idf.pickle"):
	"""add the frequencies of words from new documents to an idf count. if the idf_file specified doesn't exist, a new idf count is initiated.
	in raw_text mode, and most modes, the text should be already preprocessed
	Warning : Before using the generated idf, remember to divide each entry by the max frequency ! (use finish_idf()) Raw occurences are stored, not frequencies !
	argument documents_list : list of documents to process. If filetype==raw_text it's a list of strings, where each string is the preprocessed text of the document. Otherwise, it should be a dictionnary where the key is some sort of id, and the data the text of the document. Wether it should be already preprocessed depends on the filetype.
	argument filetype : xml generated by grobit ? pubmed full text xml ? raw text ? The default behaviour is raw_text, where argument documents_list is a list where each element is the text of one document in a string. The text must be one string already preprocessed (lemmatised, tokenised...)
	argument idf_file : a pickle file containing a previously saved idf. if the files doesn't exist a new idf will be started and stored in this file
	assumes that along the idf_file "filename.pickle" stands a file "filename.files.pickle" that stores the list of paths of already processed files. This allows to interrupt and resume the processing easily. Files in this pickle will not be processed. This check is disabled if the filetype is "raw_text".
	"""

	#check input integrity
	if not filetype in ["grobit_xml","raw_text","acl"]: #list of types currently processed
		raise ValueError("Unknown value for argument filetype")

	if not idf_file.endswith(".pickle"):
		raise ValueError("Argument idf_file must be a pickle file")

	#load idf
	start_over=False
	try:
		with open(idf_file,mode="rb") as f:
			nb_documents,idf=pickle.load(f)
			
	#if we can't load the idf, we start a new one
	except FileNotFoundError:
		sys.stderr.write("Starting a new idf in file "+idf_file+"\n")
		start_over=True
		idf=Counter()
		processed_files_idf=[]
		nb_documents=0
	
	#if we want to check if a file has already been processed, we load the list of already processed files
	if filetype != "raw_text":
		idf_file2=idf_file.split(".")[0]+".files.pickle"
		if not start_over:
			try:
				with open(idf_file2,mode="rb") as f:
					processed_files_idf=pickle.load(f)
			except FileNotFoundError:
				raise FileNotFoundError("There should be a file named "+idf_file2+" next to the idf file. If this file is missing it's better to start over with a new idf. Remove your current idf file or specify a new path.")
	

	#start processing the files
	try:
		for i,fname in enumerate(documents_list):

			if filetype!="raw_text":
				if fname in processed_files_idf:
					continue
			
			nb_documents+=1

			if filetype=="raw_text":
				sys.stderr.write("Calculating idf\n")
			else:
				sys.stderr.write("Calculating idf of "+fname+"\n")
			
			if filetype=="grobit_xml":

				wordlist=_get_words_from_grobit_xml(fname)
				for word in wordlist:
					idf[word]+=1
			elif filetype=="acl": 
				text=documents_list[fname]
				for word in re.split("\W+",text):
					if len(word)>1:
						idf[word]+=1
				
			elif filetype=="raw_text":
				#assuming the text is already preprocessed
				for word in re.split("\W+",fname):
					if len(word)>1:
						idf[word]+=1
			else:
				sys.stderr.write("You have specified an invalid value for argument filetype\n")

			if filetype!="raw_text":
				processed_files_idf.append(fname)
		
			if i%20==0 and filetype !="raw_text":
				sys.stderr.write("Reminder : You can KeyboardInterrupt at any time. This will save the current state of the IDF to disk.\n")
	
	except KeyboardInterrupt: #this breaks the iteration over entry files. Allows the user to stop processing new files and proceed with the rest of the script
		pass


	with open(idf_file,mode="wb") as f:
		pickle.dump((nb_documents,idf),f)
	if filetype!="raw_text":
		with open(idf_file2,mode="wb") as f:
			pickle.dump(processed_files_idf,f)
	sys.stderr.write("IDF saved !\n")

	return

def count_text_tfidf(text,idf=False):
	""" calculates the tf-idf of each word of a text, given an already computed idf
	text argument : either a text or a list of words from a text already tokenised. words should be already preprocessed : lemmatised, stop-words filtered...
	idf argument : a word frequency dictionnary. If not specified, will look for a idf.pickle file in the working directory. remember to pass your idf through finish_idf()
	text_format : one string of text, or a list of words
	output : tf/idf dictionnary. dict["word"]=score"""


	if not idf:
		idf=finish_idf("idf.pickle")

	if type(text)==str:
		text=re.split("\W+",text)
	elif type(text)==list:
		pass
	else:
		raise TypeError
	
	tf=defaultdict(int)
	tot_words=0
	for word in text:
		#word=word.lower() #preprocessing should have been done outside of this function, but can't hurt to do something very basic just in case. just in case of uppercase. haha. sorry.
		#actually don't do that cause NUM is uppercase
		if len(word)>1:
			tf[word]+=1
			tot_words+=1
	
	for word in tf:
		tf[word]=tf[word]/tot_words
		if word in idf:
			tf[word]=tf[word]*idf[word]
		else:
			tf[word]=0 #may happen if the text we're processing wasn't in the corpus to build the idf

	return tf

def _clean_section_title(string):
	"""does the text cleaning for paper section titles, such as introduction, conclusion...
	lowercase, removes number, removes plural, does a little spellcheck and normalises synonyms
	this function does not try to recognise good/common titles from bad/uncommon titles. this is the job of _process_section_title
	this function is typically called by _process_section_title but can be used as a standalone"""
	
	if not string:
		return
	if len(string)<1:
		return
	string=string.strip()
	if len(string)<1:
		return
	string=string.lower()

	words=re.split("\W+",string)
	words2=[]
	for word in words:
		if len(word)>0:
			if word[-1]=="s" :# fix plural variability
				word=word[:-1]
		words2.append(word)
	string=" ".join(words2)

	#string=string.replace("’","'") 
	string=re.sub("[^\w]","",string) #removes non-alphanumeric characters
	string=re.sub("\d","",string) #removes numbers
	string=re.sub("&","and",string)
	string=string.strip()

	if string=="acknowledgment":
		string="acknowledgement"
	
	string=re.sub("futurework","future work")
	
	if string=="summary":
		string="abstract"

	return string

def process_section_title(raw_title,restricted_allowed_titles=False):
	"""argument : raw section title
	returns the clean/preprocessed title
	good titles such as "conclusion" will be returned in a normalised form ("Conclusions" becomes "conclusion")
	bad titles such as "acknowledgment" will return False
	uncommon titles will return "other"
	returns None if the title is empty
	a good title is a common title, found in most papers (introduction, method, conclusion...)
	a bad title is the title of a section we typically want to exclude from the full text article (acknowledgement, author's contribution...)
	option restricted_allowed_titles : limits good titles to a smaller list"""

	if restricted_allowed_titles:
		good_titles=["introduction","conclusion","related work","result","discussion","material and method","abstract"]
	else:
		good_titles=["introduction","method","conclusion","finding","result","discussion","background","material and method","related work","discussion","evaluation","experiment","summary"]
	bad_titles=["acknowledgement","source of funding","funding","competing interest","author contribution","authors contribution","supplementary information","supplementary material","figure","fig","table","conflict of interest","abbreviation"]
	title_clean=_clean_section_title(raw_title) #synonyms are handled here
	if not title_clean:
		return "other"
	if len(title_clean)>1:
		if title_clean in good_titles :
			if title_clean == "method":
				title_clean="material and method"
			return title_clean
		elif title_clean in bad_titles:
			return False
		else:
			return "other"


def preprocess_text(input_text,keep_all_words=False,separate_sentences=False,return_lemmas=False):
	""" this function should be used for all text preprocessing for uniformity. it lemmatises, removes stop words and does some normalisation
	argument input_text : string of text OR beautifulsoup containing text. can be a section of soup. extract and processes all text inside that soup.
	returns list of spacy sentences
	if argument keep_all_words=True : returns list of spacy sentences with all words left, including stopwords
	argument separate_sentences : if False returns a flat list of words. if True returns a list of sentences, where each list is a list of words
	argument returns_lemmas : if True returns a list of strings, where each string is one word in its lemmatised form (with numbers normalised to "NUM"). if False returns a list of spacy tokens. """
	
	try:
		text=input_text.getText(separator=" ") #if the input is a soup
	except AttributeError:
		text=input_text
	
	if len(text)<1:
		return [""]

	try:
		text=re.sub("\. ?\. ",". ",text) #removes a soup artifact ". .. . ... . "etc replaced by ". ". Also replaces "..." by ". " spacy is going to treat those as empty sentences and it's annoying
		text=re.sub("et al \.","et al ",text) #same, spacy is annoying with dots
		text=re.sub(" +"," ",text) #removes multiple spaces
	except TypeError: #one of those steps reduced the input to an empty string
		return [""]

	spacy_text=nlp(text)
	sentences=[]
	for sentence in spacy_text.sents:
		
		#select which tokens to keep
		tokens=[]
		for token in sentence:

			if keep_all_words:
				tokens.append(token)
			else:
				if not token.is_stop:
					if token.pos_ in ["NOUN","VERB","PROPN","ADJ","NUM"]:
						tokens.append(token)

		#format output
		current_sent=[]
		for token in tokens:
			if return_lemmas: #replaces the spacy token object by a string containing only the lemma
				if token.pos_=="NUM":
					token="NUM"
				else:
					token=token.lemma_
					token=token.lower() #spacy capitalises lemmas of proper nouns or abreviations
			if separate_sentences:
				current_sent.append(token)
			else:
				sentences.append(token)
		if separate_sentences:
			sentences.append(current_sent)
	return sentences

def load_mesh():
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
					continue #the same term is both hyponym and hyperonym. ignore it.
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
			mesh_synonyms[pref_label]=pref_label #to avoid keyerror, and the dict can be used as a list of possible mesh terms

	return mesh_hierarchy_up,mesh_hierarchy_down

def output_multi_corpus():
	"""outputs the preprocessed text of ALL known corpora. used to train embeddings and the like. use this function as a standalone.
	"""

	use_canceropole=True
	use_acm=True
	use_acl=True
	use_text8=True #this is going to take a while é_è
	keep_all_words=False
	documents=[]


	if use_canceropole:
		try:
			for path, dirs, files in os.walk("fulltext_tei/all"):
				for fname in files: #one file = one article
					if fname.endswith(".xml"):
						print("processing "+fname)
						with open(path+"/"+fname,mode="r",encoding="utf-8") as f:
							lines="".join(f.readlines())
							soup=BeautifulSoup(lines,"xml")

						text=preprocess_text(soup,keep_all_words=keep_all_words,return_lemmas=True)
						text=" ".join(text)

						documents.append(text)
		except KeyboardInterrupt:
			pass #manually break loop
	
	if use_acm:
		try:
			with open("aman git/acm/id_abstract.dat",mode="r",encoding="utf-8") as f:
				for l in f:
					id_=re.match("^\d+ ",l).group(0)[:-1]
					sys.stderr.write("processing "+id_+" ("+str(int((int(id_)/12498.0)*100))+"%)\n")
					l=re.sub("^\d+ '","",l) #removes the numerical id at the start of each line + opening '
					l=l[:-1] #removes ending ' at the end of the line
					l=l.strip()

					#preprocessing
					text=preprocess_text(l,keep_all_words=keep_all_words,return_lemmas=True)
					text=" ".join(text)
					documents.append(text)
		except KeyboardInterrupt:
			pass #manually break loop

	if use_acl:
		try:
			directory="/home/sam/work/corpora/acl/cleaned_acl_arc/"
			for fname in os.listdir(directory):
				print("processing "+fname)
				with open(directory+fname,mode="r",encoding="iso-8859-1") as f:
					text=f.readlines()
					text="\n".join(text)
					text=preprocess_text(l,keep_all_words=keep_all_words,return_lemmas=True)
					text=" ".join(text)
					documents.append(text)
		except KeyboardInterrupt:
			pass #manually break loop
	

	if use_text8:
		try:
			fname="/home/sam/work/glove/GloVe/text8"
			i=0
			with open(fname,mode="r",encoding="utf-8") as f:
				for l in f:
					l=l.split(" ")
					chunk=[]
					for word in l:
						chunk.append(word)
						if len(chunk)>1000:
							try:
								i+=1
								print("processing chunk "+str(i))
								chunk=" ".join(chunk)
								text=preprocess_text(chunk,keep_all_words=keep_all_words,return_lemmas=True)
								text=" ".join(text)
								documents.append(text)
								chunk=[]
							except Exception:
								chunk=[]


					try:
						text=preprocess_text(chunk,keep_all_words=keep_all_words,return_lemmas=True)
						text=" ".join(text)
						documents.append(text)
					except Exception:
						pass
			print("done")

		except KeyboardInterrupt:
			pass #manually break loop

	#past this point, adapt the output to your needs
	documents=" ".join(documents)
	with open("full_corpora.txt",mode="w",encoding="utf-8") as f:
		f.write(documents)


if __name__=="__main__":

	#generate the full text of all known corpora, for training purposes
	#output_multi_corpus()

	#generate idf for a folder
	#directory="fulltext_tei"
	#files=[directory+"/"+x for x in os.listdir(directory)]

	with open("/home/sam/work/corpora/acm/id_title_abstract_prep.dat",mode="r",encoding="utf-8") as f:
		files=f.readlines()
	edit_idf(files,filetype="raw_text",idf_file="22_10_acm_abstract_title.pickle")
