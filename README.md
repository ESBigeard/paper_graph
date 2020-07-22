# Paper Graph
Dev/tools repo for a project about scientific papers mining to construct graphs

## References

### Closest to our work

* [BioWordVec, improving biomedical word embeddings with subword information and MeSH](https://www.nature.com/articles/s41597-019-0055-0.pdf) **word embedding, terminologies, resource, not about graphs**
   2019.
   
   2 methods combined : subword embedding, and mesh. Subword embedding with fastText n-grams. Mesh hierarchy is translated into sentences where each word is a mesh term, and the word order represents a hierarchical sequence. We simply add those sentences to the corpus (pubmed articles) and learn everything together. The direction and type of edges are ignored. code available https://github.com/ncbi-nlp/BioWordVec pretrained word embeddings available
   
* [Paper2vec: Combining Graph and Text Information for Scientific Paper Representation](https://researchweb.iiit.ac.in/~soumyajit.ganguly/papers/P2v_1.pdf) **word embedding, document embedding, full-text**
   2017
   
   about citations networks, 2 tasks : discover missing/possible citations (edge prediction) ; and tag papers (node classification). learns word vectors with skip-gram + skip-gram negative sampling, then use the word vectors to build document vectors. svm for node classification. logistic regression for link prediction.
   
* [Network Representation Learning with Rich Text Information (TADW)](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) **textual node representation**
   2015 
   
   
   https://github.com/albertyang33/TADW code in matlab
   Describes a network representation method to incorporate the tf-idf matrix of a text into its graph node representation. technical math paper, I didn't understand much of the method, but they provide the code. eval : multiclass vertex classification, with svm. data : citation network or wikipedia article links. text data : title+abstract for the papers, full article for wikipedia.

* [On Joint Representation Learning of Network Structur eand Document Content](https://hal.inria.fr/hal-01677137/document) **textual node representation**
2017
The paper says itself that it's not as good as it's 2 competitors, paper2vec and TADW. But it explains word2vec, random walk and several other embedding things very well, i recommend reading the paper for a quick overview of the field in 2017. The paper states that paper2vec and TADW are the only attempts in 2017 of learning from the textual content of the node AND the graph structure.

* [DocCit2Vec: Citation Recommendation viaEmbedding of Content and Structural Contexts](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9123859) 2020

For the task of citation recommendation in citation networks, content of the text is more important than graph data.

### other

* [BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/abs/1810.09302) **sentence embedding, resource, not about graphs**
    2018
    
    trained sent2vec on a corpus composed of pubmed articles and clinical notes (MIMIC-III), and made the resulting embeddings available
    
* [Weighted PageRank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.5022&rep=rep1&type=pdf) **not about graphs, can be used for link prediction in citation networks**
    2004

    classic paper. used on internet info retrieval, can be used in the context of a citation network to rank influential papers. Not sure this will be useful for our work, but I've seen it used in a few papers ( including [this other paper in our bib](https://openreview.net/forum?id=W3Dzaik1ipL)) so keeping it on hand
