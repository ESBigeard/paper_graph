
# Paper Graph
Dev/tools repo for a project about scientific papers mining to construct graphs

## References

### Closest to our work

* GOOD [BioWordVec, improving biomedical word embeddings with subword information and MeSH](https://www.nature.com/articles/s41597-019-0055-0.pdf) **word embedding, terminologies, resource, not about graphs** 2019.
   
   2 methods combined : subword embedding, and mesh. Subword embedding with fastText n-grams. Mesh hierarchy is translated into sentences where each word is a mesh term, and the word order represents a hierarchical sequence. We simply add those sentences to the corpus (pubmed articles) and learn everything together. The direction and type of edges are ignored. code available https://github.com/ncbi-nlp/BioWordVec pretrained word embeddings available
   
* GOOD [Paper2vec: Combining Graph and Text Information for Scientific Paper Representation](https://researchweb.iiit.ac.in/~soumyajit.ganguly/papers/P2v_1.pdf) **word embedding, document embedding, full-text**
   2017
   
   about citations networks, 2 tasks : discover missing/possible citations (edge prediction) ; and tag papers (node classification). learns word vectors with skip-gram + skip-gram negative sampling, then use the word vectors to build document vectors. svm for node classification. logistic regression for link prediction.
   [paper doesn't give code but open implementations are available](https://github.com/tianhan4/paper2vec)
   
* GOOD [Network Representation Learning with Rich Text Information (TADW)](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) **textual node representation, full-text**
   2015 

   [code available](https://github.com/albertyang33/TADW)
   Describes a network representation method to incorporate the tf-idf matrix of a text into its graph node representation. Based on deepwalk (Text Associated Deep Walk). eval : multiclass vertex classification, with svm. data : citation network or wikipedia article links. text data : title+abstract for the papers, full article for wikipedia.

* [On Joint Representation Learning of Network Structur eand Document Content](https://hal.inria.fr/hal-01677137/document) **textual node representation**
2017
The paper says itself that it's not as good as it's 2 competitors, paper2vec and TADW. But it explains word2vec, random walk and several other embedding things very well, i recommend reading the paper for a quick overview of the field in 2017. The paper states that paper2vec and TADW are the only attempts in 2017 of learning from the textual content of the node AND the graph structure.


### citation recommendation

*[Citation Recommendation: Approaches and Datasets](https://arxiv.org/abs/2002.06961) 2020 **review**

   citation recommendation = given a short text from a paper you are writing, suggest papers to cite in this short text.

*[Scientific Paper Recommendation: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8598708) 2019 **review**

   tf-idf. random walk. paperRank (based on pageRank)

* GOOD [DocCit2Vec: Citation Recommendation via Embedding of Content and Structural Contexts](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9123859) 2020

   makes embeddings between citations links and the sentence where the citation appears. embedding is based on doc2vec and hyperdoc2vec. testing datasets : ACL anthology and DBLP. Takes into account papers that are already cited in your paper, and only recommends papers you don't know already.

   they tested that, for the task of citation recommendation in citation networks, content of the text is more important than graph data.

   methods cited as classical/older in introduction, not as good as embeddings : using seed papers as input, pagerank, metadata, topic modeling, "translation" of sentences into references.

* [HybridCite: A Hybrid Model for Context-Aware Citation Recommendation](https://arxiv.org/abs/2002.06406) 2020

   Focuses on local citation recommendation : input is only a few sentences from a paper the user would be currently writing. based on hyperdoc2vec. Proposes to mix algorithms not by weights, but by something inspired from genetic algorithms. They mix one algo based on text content, and one algo based on graph structure. The idea can be adapted to any 2 aglo that can take a text context and suggest ranked papers.

   If paper A cites paper B, the citation context in paper A tells more about paper B than paper A.

   dataset : microsoft academic graph (220 million papers in 2019, grows each week) doesn't include fulltext, but includes abstracts and citation contexts ; ACL anthology ; arxiv

* [Linked Document Embedding for Classification](https://dl.acm.org/doi/10.1145/2983323.2983755) 2016

* GOOD [hyperdoc2vec: Distributed Representations of Hypertext Documents](https://www.aclweb.org/anthology/P18-1222.pdf) 2018

   uses the text content of a link (more relevant for internet links than citations) and the context of the link. a problem to tackle is that new papers will have no links toward them, but we want new papers. semantics of "evaluated by [citation]" or "evaluated on corpus [citation]". dual embeddings of papers : in and out.

"context as content" : if paper A cites paper B, we take the text in A surrounding the citation (context) and append it to the text of paper B (content)

   corpus : DBLP, ACL anthology and NIPS (one machine learning conference, fulltext)


### other

* [Not All Contexts Are Created Equal:Better Word Representations with Variable Attention](https://www.researchgate.net/publication/301446021_Not_All_Contexts_Are_Created_Equal_Better_Word_Representations_with_Variable_Attention) 2015 **word embeddings*

   word embeddings with attention

* [BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/abs/1810.09302) **sentence embedding, resource, not about graphs** 2018
    
    trained sent2vec on a corpus composed of pubmed articles and clinical notes (MIMIC-III), and made the resulting embeddings available
    
* [Weighted PageRank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.5022&rep=rep1&type=pdf) **not about graphs, can be used for link prediction in citation networks** 2004

    classic paper. used on internet info retrieval, can be used in the context of a citation network to rank influential papers. Not sure this will be useful for our work, but I've seen it used in a few papers ( including [this other paper in our bib](https://openreview.net/forum?id=W3Dzaik1ipL)) so keeping it on hand
