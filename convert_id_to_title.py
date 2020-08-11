#!/usr/bin/python
# -*- coding:utf-8 -*-
"""used to convert in the nodes.csv the info about the most similar nodes. The provided id is converted into the corresponding article title"""

import csv,sys,os,re
from bs4 import BeautifulSoup

fname_in=sys.argv[1]
fname_out="2"+fname
csv_in="updated_nodes_fulltext.csv" #only needed for xml mode

mode="xml" #"csv" or "xml". csv to edit a csv, xml to edit a gexf file directly

#1ere passe, on enregistre les correspondances id/label

dict_label={}
if True:
	if mode=="csv":
		in_file=fname_in
	else:
		in_file=csv_in
	with open(in_file,mode="r",encoding="utf-8") as f:
		reader=csv.reader(f)
		header=next(reader)
		for row in reader:
			id_=row[0]
			id_=int(id_)
			label=row[1]
			dict_label[id_]=label


if mode =="csv":
	#2eme passe, on modifie les lignes du csv au fur et à mesure
	with open(fname_in,mode="r",encoding="utf-8") as f, open(fname_out,mode="w",encoding="utf-8") as fout:
		reader=csv.reader(f)
		writer=csv.writer(fout)

		header=next(reader)
		writer.writerow(header)

		for row in reader:
			row_before=row[0:5]
			row_sim=row[5:15]
			row_after=row[15:]
			row_sim2=[]
			for id_ in row_sim:
				id_=re.sub("\.0$","",id_)
				id_=int(id_)
				id_=id_-1 #correct aman error
				if id_=="0":
					print("zéro")
				try:
					label=dict_label[id_]
				except KeyError:
					print("skipped ",id_)
					c2ontinue
				row_sim2.append(label)
			row=row_before+row_sim2+row_after
			writer.writerow(row)

elif mode=="xml":
	
	with open(fname_in,mode="r",encoding="utf-8") as f, open(fname_out,mode="w",encoding="utf-8") as fout:
		content=" ".join(f.readlines())
		soup=BeautifulSoup(content,"xml")
		for attribute in soup.find_all("attvalue"):
			if attribute["for"].startswith("sim_"):
				id_=attribute["value"]
				id_=int(float(id_))
				id_=id_-1 #correct aman error
				text_value=dict_label[id_]
				attribute["value"]=text_value
				

		fout.write(soup.prettify())
