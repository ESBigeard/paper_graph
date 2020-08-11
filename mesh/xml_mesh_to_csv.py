#!/usr/bin/python
# -*- coding:utf-8 -*-
"""sort une hiérarchie basique/simplifiée à partir du mesh xml de l'atilf"""

import csv
from rdflib import Graph
g = Graph()
g.parse("MeSH.xml") #prend du temps


label_id_dict={} #dict["name of author"]=id of author
nodes=[] #the info to output in the csv for nodes
nodes.append(["id","label","type"])
previous_id=-1 #a different id for each node. articles, titles, keywords, etc, all share the same id numerotation
edges=[] #the info to output in the csv for edges
edges.append(["source","target","type","weight"])

q=g.query("""
select ?lab1 ?lab2 where{
?a skos:prefLabel ?lab1.
?b skos:prefLabel ?lab2.
?a skos:narrower ?b.
FILTER(langMatches(lang(?lab1),"EN"))
FILTER(langMatches(lang(?lab2),"EN"))
}
""")

for row in q:
	label_larger=row[0]
	label_narrower=row[1]

	#add node if not exist
	if label_larger not in label_id_dict:
		previous_id+=1
		label_id_dict[label_larger]=previous_id
		nodes.append([previous_id,label_larger,"concept"])
	if label_narrower not in label_id_dict:
		previous_id+=1
		label_id_dict[label_narrower]=previous_id
		nodes.append([previous_id,label_narrower,"concept"])
	
	#add edge
	id_larger=label_id_dict[label_larger]
	id_narrower=label_id_dict[label_narrower]
	edges.append([id_larger,id_narrower,"directed",1])
	

q=g.query("""
select ?preflab ?altlab where{
?a skos:altLabel ?altlab.
?a skos:prefLabel ?preflab.
FILTER(langMatches(lang(?preflab),"EN"))
FILTER(langMatches(lang(?altlab),"EN"))
}""")

for row in q:
	label_pref=row[0]
	label_alt=row[1]

	#add node if not exist
	if label_pref not in label_id_dict:
		previous_id+=1
		label_id_dict[label_pref]=previous_id
		nodes.append([previous_id,label_pref,"concept"])
	if label_alt not in label_id_dict:
		previous_id+=1
		label_id_dict[label_alt]=previous_id
		nodes.append([previous_id,label_alt,"alt_label"])
	
	#add edge
	id_pref=label_id_dict[label_pref]
	id_alt=label_id_dict[label_alt]
	edges.append([id_pref,id_alt,"undirected",1])


#write nodes
fname="mesh_test"
with open(fname+"_nodes.csv",mode="w", encoding="utf-8") as f:
	writer=csv.writer(f)
	writer.writerows(nodes)

#write edges
with open(fname+"_edges.csv",mode="w", encoding="utf-8") as f:
	writer=csv.writer(f)
	writer.writerows(edges)

