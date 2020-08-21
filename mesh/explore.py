#!/usr/bin/python
# -*- coding:utf-8 -*-
"""little script to explore manually the mesh"""
import csv
from collections import defaultdict


nodes=[]
edges=[]
keywords_id_dict={}

def recursive_add_hierarchy_mesh(label,direction="hyperonym"):
	"""given a mesh label present in mesh_hierarchy_up, fetches info for all of its hyper/hypo-nyms, recursively. adds edges and nodes to global variables "nodes" and "edges". "keywords_id_dict" is also globally edited. Yes it's messy, sorry future me
	mesh_hierarchy_up is a dict[label]=[label_up,label_up] where label_up is an hyperonym of label. same for mesh_hierarchy_down"""

	global nodes
	global edges
	global keywords_id_dict

	output=set([])

	#print("start recurse",label,latest_id)
	if direction in ("hyperonym","up"):
		mesh_hierarchy=mesh_hierarchy_up
	else:
		mesh_hierarchy=mesh_hierarchy_down
		
	if label in mesh_hierarchy:
		hyperonyms=mesh_hierarchy[label]
		for label_up in hyperonyms:

			if label_up==label:
				raise ValueError("Error in mesh hierarchy data. This term makes a loop: "+label)

			output.add(label_up)

			#add node and edges connected to current hyponym
			if label_up not in keywords_id_dict:
				keywords_id_dict[label_up]="osef"
				nodes.append(["osef",label_up,"keyword","",""])
			id_1="osef"
			id_2="osef"
			weight=0.1
			edge=[id_1,id_2,"directed",0.1]
			if edge not in edges:
				edges.append(edge)

			#recurse toward higher hyperonym
			try:
				output=output.union(recursive_add_hierarchy_mesh(label_up,direction))
			except RecursionError:
				raise RecursionError("Maximum recursion depth reached. It's likely the mesh hierarchy data makes a loop. Investigate these nodes : "+label+" ; "+label_up)
			#print("out recursive",label,label_up,latest_id)

	return(output)


### load
if True:
	mesh_hierarchy_up=defaultdict(list)
	mesh_hierarchy_down=defaultdict(list)
	mesh_keywords=set([])
	keyword_exclude=set([])

	with open("exclude_list.txt",mode="r",encoding="utf-8") as f:
		for l in f:
			l=l.strip()
			l=l.lower()
			keyword_exclude.add(l)

		
	with open("mesh_lem.txt",mode="r",encoding="utf-8") as f:
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

### stuff

l=(recursive_add_hierarchy_mesh("amino acid peptide and protein","down"))
for e in l:
	print(e)
exit()

top_keywords=defaultdict(int)
for term in mesh_keywords:
	top_keywords[term]=len(recursive_add_hierarchy_mesh(term,"down"))

for key in sorted(top_keywords, key=top_keywords.get, reverse=False):
	print(key,top_keywords[key])

