#!/usr/bin/python
# -*- coding:utf-8 -*-

import os,sys,csv
from collections import defaultdict

#nodes=[]
#edges=[]
#keywords_id_dict={}


def recursive_add_hyperonyms_mesh(label,latest_id):
	"""given a mesh label present in mesh_hierarchy_up, fetches info for all of its hyperonyms, recursively. adds edges and nodes to global variables "nodes" and "edges". "keywords_id_dict" is also globally edited. Yes it's messy, sorry future me
	mesh_hierarchy_up is a dict[label]=[label_up,label_up] where label_up is an hyperonym of label"""

	#global nodes
	#global edges
	#global keywords_id_dict

	#print("start recurse",label,latest_id)
	if label in mesh_hierarchy_up:
		hyperonyms=mesh_hierarchy_up[label]
		for label_up in hyperonyms:

			if label_up==label:
				raise ValueError("Error in mesh hierarchy data. This term makes a loop: "+label)

			#add node and edges connected to current hyponym
			#if label_up not in keywords_id_dict:
			#	latest_id+=1
			#	keywords_id_dict[label_up]=latest_id
				#nodes.append([latest_id,label_up,"keyword","",""])
			#id_1=keywords_id_dict[label]
			#id_2=keywords_id_dict[label_up]
			#edge=[id_1,id_2,"directed",1]
			#if edge not in edges:
			#	edges.append(edge)

			#recurse toward higher hyperonym
			try:
				latest_id=recursive_add_hyperonyms_mesh(label_up,latest_id)
			except RecursionError:
				raise RecursionError("Maximum recursion depth reached. It's likely the mesh hierarchy data makes a loop. Investigate these nodes : "+label_up+" ; "+label)
			#print("out recursive",label,label_up,latest_id)

	return(latest_id)



mesh_hierarchy_up=defaultdict(list)
mesh_hierarchy_down=defaultdict(list)
mesh_keywords=set([])
with open("mesh_lem.txt",mode="r",encoding="utf-8") as f:
	reader=csv.reader(f)
	next(reader)
	for row in reader:
		label1,relation,label2=row
		label1=label1.lower()
		label2=label2.lower() #todo more nettoyage
		mesh_keywords.add(label1)
		mesh_keywords.add(label2)
		if label1==label2:
			#sys.stderr.write("There an entry in the mesh hierarchy with this term as both hyponym and hyperonym : "+label1+" This entry has been ignored but removing it from the data is recommended.\n")
			continue
		mesh_hierarchy_up[label2].append(label1)
		mesh_hierarchy_down[label1].append(label2)

n_recursive_errors=0
for i,keyword in enumerate(mesh_keywords):
	recursive_add_hyperonyms_mesh(keyword,i)
print("total recursive errors ",n_recursive_errors)
