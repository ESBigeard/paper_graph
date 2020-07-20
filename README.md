# Paper Graph
Dev/tools repo for a project about scientific papers mining to construct graphs

## References


* [BioWordVec, improving biomedical word embeddings with subword information and MeSH](https://www.nature.com/articles/s41597-019-0055-0.pdf) **word embedding, terminologies, resource, not about graphs**
   2019. 2 methods combined : subword embedding, and mesh. Subword embedding with fastText n-grams. Mesh hierarchy is translated into sentences where each word is a mesh term, and the word order represents a hierarchical sequence. We simply add those sentences to the corpus (pubmed articles) and learn everything together. The direction and type of edges are ignored. code available https://github.com/ncbi-nlp/BioWordVec pretrained word embeddings available
   
* /!\ [Paper2vec: Combining Graph and Text Information for Scientific Paper Representation](https://researchweb.iiit.ac.in/~soumyajit.ganguly/papers/P2v_1.pdf) **word embedding, document embedding, full-text**
   2017
   about citations networks, 2 tasks : discover missing/possible citations (edge prediction) ; and tag papers (node classification). learns word vectors with skip-gram + skip-gram negative sampling, then use the word vectors to build document vectors. svm for node classification. logistic regression for link prediction.
   
* [Network Representation Learning with Rich Text Information](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) **textual node representation**
   2015 
   https://github.com/albertyang33/TADW code in matlab
   Describes a network representation method to incorporate the tf-idf matrix of a text into its graph node representation. technical math paper, I didn't understand much of the method, but they provide the code. eval : multiclass vertex classification, with svm. data : citation network or wikipedia article links. text data : title+abstract for the papers, full article for wikipedia.

* [BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/abs/1810.09302) **sentence embedding, resource, not about graphs**
    2018
    trained sent2vec on a corpus composed of pubmed articles and clinical notes (MIMIC-III), and made the resulting embeddings available. evaluated on sentence similarity on a set of manually annotated sentences.
