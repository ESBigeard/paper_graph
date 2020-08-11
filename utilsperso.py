#!/usr/bin/python
# -*- coding:utf-8 -*-
"""This module is used by several other scripts, be prudent making changes. Can also be used alone to generate an idf"""

import re
import spacy
import pickle
import os
import sys
import math
from collections import defaultdict,Counter
from bs4 import BeautifulSoup


nlp=spacy.load("en_core_web_sm")
sentence_tokeniser=nlp.create_pipe("sentencizer")
nlp.add_pipe(sentence_tokeniser)

def finish_idf(idf_file):
	"""do that to an idf generated by edit_idf (stored in a idf.pickle). edit_idf only generates a dictionnary of word occurences. this turns it into word frequencies"""

	if not idf_file.endswith(".pickle"):
		raise ValueError("Argument idf_file must be a pickle file")

	with open(idf_file,mode="rb") as f:
		try:
			nb_documents,idf=pickle.load(f)
		except Exception as e:
			raise Exception("Something went wrong when loading the idf file "+idf_file+" you may be trying to load an older version that is not compatible anymore") from e

	#max_freq=float(idf.most_common(1)[0][1])
	for word in idf:
		idf[word]=math.log(nb_documents / float(idf[word]))

	return idf

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


def edit_idf(documents_list,filetype="",idf_file="idf.pickle"):
	"""add the frequencies of words from new documents to an idf count. if the idf_file specified doesn't exist, a new idf count is initiated.
	Warning : Before using the generated idf, remember to divide each entry by the max frequency ! (use finish_idf()) Raw occurences are stored, not frequencies !
	argument documents_list : list of paths of files to process
	argument filetype : xml generated by grobit ? pubmed full text xml ? raw text ?
	argument idf_file : a pickle file containing a previously saved idf. if the files doesn't exist a new idf will be started and stored in this file
	assumes that along the idf_file "filename.pickle" stands a file "filename.files.pickle" that stores the list of paths of already processed files. This allows to interrupt and resume the processing easily. Files in this pickle will not be processed.
	"""

	#check input integrity
	if not filetype in ["grobit_xml","raw_text"]: #list of types currently processed
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
			
			nb_documents+=1

			if filetype=="raw_text":
				sys.stderr.write("Calculating idf\n")
			else:
				sys.stderr.write("Calculating idf of "+fname+"\n")
			
			if filetype=="grobit_xml":

					wordlist=_get_words_from_grobit_xml(fname)
					for word in wordlist:
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
	""" calculates the tf of a text, given an already computed idf
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
		word=word.lower() #preprocessing should have been done outside of this function, but can't hurt to do something very basic just in case. just in case of uppercase. haha. sorry.
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
	
	if len(string)<1:
		return string
	string=string.lower()

	words=string.split("\W+")
	words2=[]
	for word in words:
		if word[-1]=="s" :# fix plural variability
			word=word[:-1]
		words2.append(word)
	string=" ".join(words2)

	#string=string.replace("’","'") 
	string=re.sub("[^ \w]","",string) #removes non-alphanumeric characters
	string=re.sub("\d","",string) #removes numbers
	string=re.sub("&","and",string)
	string=string.strip()

	return string

def process_section_title(raw_title):
	"""argument : raw section title
	returns the clean/preprocessed title
	good titles such as "conclusion" will be returned in a normalised form ("Conclusions" becomes "conclusion")
	bad titles such as "acknowledgment" will return False
	uncommon titles will return "other"
	returns None if the title is empty
	a good title is a common title, found in most papers (introduction, method, conclusion...)
	a bad title is the title of a section we typically want to exclude from the full text article (acknowledgement, author's contribution...)"""

	good_titles=["introduction","method","conclusion","finding","result","discussion","background","materials and method","material and method"]
	bad_titles=["acknowledgement","acknowledgment","source of funding","competing interest","author contribution","authors contribution","supplementary information","figure","fig","table","conflict of interest","conflicts of interest"]
	title_clean=_clean_section_title(raw_title)
	if len(title_clean)>1:
		if title_clean in good_titles :
			if title_clean in ["materials and method","method"]:
				title_clean="material and method"
			return title_clean
		elif title_clean in bad_titles:
			return False
		else:
			return "other"


def preprocess_text(soup,keep_all_words=False,separate_sentences=False):
	""" this function should be used for all text preprocessing for uniformity
	argument soup : beautifulsoup containing text. can be a section of soup. extract and processes all text inside that soup.
	returns list of spacy sentences with only words that were kept for the idf.
	if argument keep_all_words=True : returns list of spacy sentences with all words left, including stopwords
	argument separate_sentences : if False returns a flat list of words. if True returns a list of sentences, where each list is a list of words
	warning : when calculating the idf I usually switch numbers for NUM. Here the numbers are returned as untounched tokens. Numbers should be detected by token.pos_=="NUM" """
	
	text=soup.getText(separator=" ")
	text=re.sub("\. ?\. ",". ",text)
	text=re.sub("et al \.","et al ",text)
	text=re.sub(" +"," ",text)
		

	spacy_text=nlp(text)
	sentences=[]
	for sentence in spacy_text.sents:
		
		current_sent=[]
		for token in sentence:

			if keep_all_words:
				if separate_sentences:
					current_sent.append(token)
				else:
					sentences.append(token)

			else:
				if not token.is_stop:

					#warning, for pos NUM when calculating the idf we switched the lemma for "NUM"
					if  token.pos_ in ["NOUN""VERB","PROPN","ADJ","NUM"]:
						if separate_sentences:
							current_sent.append(token)
						else:
							sentences.append(token)
		if separate_sentences:
			sentences.append(current_sent)
	return sentences


if __name__=="__main__":

	#generate idf for a folder
	directory="fulltext_tei"
	files=[directory+"/"+x for x in os.listdir(directory)]
	edit_idf(files,filetype="grobit_xml",idf_file="27_07_tei_idf.pickle")
