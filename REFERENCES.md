## References

**Summary**

Outside of domain-specific stuff, **paper2vec** (2017) and **tadw** (2015) are the state of the art to include textual content and graph information into the node's embedding. Methods that use the text and the graph to create embeddings usually test against methods that use one or the other, making it difficult to make comparisons.

**BioWordVec** (2019) has trained embeddings available, and the code to generate them available as well. Uses mesh. We should use them, and so choose a method that uses word embeddings. Paper2vec starts from word embeddings, which makes it possible to combine with biowordvec. I'm unsure the same can be said of TADW ? The only problem is that we don't have a standard corpus of medical papers, so it will be difficult to compare or method to others if we want to leverage biowordvec.

Going into domain-specific tasks, **citation recommendation** is very close to what we want. It consists of, given a very small text from a paper, suggesting citations to go with this specific part of the paper. It's intended to be used by authors as they write an article. All methods use either the full text or a window of context around citations in the full text. **DocCit2Vec** (2020) is promising. **Hyperdoc2vec** (2018) is a standard, intended more to crawl the internet but used in several papers for citation networks.

Datasets mostly used : ACL anthology and DBLP. Mimic is used to train medical word embeddings, but it's not scientific papers.

**Overall suggestion** : Use methods from citation recommendation to make fine-grained paper sections embeddings. Use BioWordVec to include MESH into the word embeddings, also some kind of attention mechanism.

(from aman but keeping here to not forget about it https://arxiv.org/abs/1907.00962 detect parts of the abstract that contain a claim. manually annotated dataset available)

### Closest to our work

* GOOD [BioWordVec, improving biomedical word embeddings with subword information and MeSH](https://www.nature.com/articles/s41597-019-0055-0.pdf) **word embedding, terminologies, resource, not about graphs** 2019.
   
   2 methods combined : subword embedding, and mesh. Subword embedding with fastText n-grams. Mesh hierarchy is translated into sentences where each word is a mesh term, and the word order represents a hierarchical sequence. We simply add those sentences to the corpus (pubmed articles) and learn everything together. The direction and type of edges are ignored. code available https://github.com/ncbi-nlp/BioWordVec pretrained word embeddings available
   
* GOOD [Paper2vec: Combining Graph and Text Information for Scientific Paper Representation](https://researchweb.iiit.ac.in/~soumyajit.ganguly/papers/P2v_1.pdf) **word embedding, document embedding, full-text**
   2017
   
   about citations networks. uses doc2vec as a first step to learn only from text, and use them as pre-trained weight. uses text similarity to add edges. they learn from the graph using some kind of random walk. svm for node classification. logistic regression for link prediction.
   [paper doesn't give code but open implementations are available](https://github.com/tianhan4/paper2vec)
   evaluated on 2 tasks : discover missing/possible citations (edge prediction) ; and tag papers (node classification).
   dataset : CORA and DBLP
   
* GOOD [Network Representation Learning with Rich Text Information (TADW)](https://www.ijcai.org/Proceedings/15/Papers/299.pdf) **textual node representation, full-text**
   2015 

   [code available](https://github.com/albertyang33/TADW)
   Describes a network representation method to incorporate the tf-idf matrix of a text into its graph node representation. Based on deepwalk (Text Associated Deep Walk). eval : multiclass vertex classification, with svm. data : citation network or wikipedia article links. text data : title+abstract for the papers, full article for wikipedia.

* [On Joint Representation Learning of Network Structur eand Document Content](https://hal.inria.fr/hal-01677137/document) **textual node representation**
2017
The paper says itself that it's not as good as it's 2 competitors, paper2vec and TADW. But it explains word2vec, random walk and several other embedding things very well, i recommend reading the paper for a quick overview of the field in 2017. The paper states that paper2vec and TADW are the only attempts in 2017 of learning from the textual content of the node AND the graph structure.


### citation/paper recommendation

citation recommendation = focus on a short part of a paper, only a few sentences, and suggest papers that might be cited here. Intented to be used by researchers while writing a paper

paper recommendation = Input is a whole paper, or several papers, or more. Intended to suggest papers to a researcher in a more general context

* [DBLP + abstracts citation network corpus](https://www.aminer.org/dblp_citation) **corpus**

* [Citation Recommendation: Approaches and Datasets](https://arxiv.org/abs/2002.06961) 2020 **review**


* [Scientific Paper Recommendation: A Survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8598708) 2019 **review**

   tf-idf. random walk. paperRank (based on pageRank). hybrid methods.

   content-based : input is a researcher's papers, output is papers they will want to read.

* GOOD [DocCit2Vec: Citation Recommendation via Embedding of Content and Structural Contexts](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9123859) 2020

   makes embeddings between citations links and the sentence where the citation appears. embedding is based on doc2vec and hyperdoc2vec. testing datasets : ACL anthology and DBLP. Takes into account papers that are already cited in your paper, and only recommends papers you don't know already.

   they tested that, for the task of citation recommendation in citation networks, content of the text is more important than graph data.

   methods cited as classical/older in introduction, not as good as embeddings : using seed papers as input, pagerank, metadata, topic modeling, "translation" of sentences into references.

* [HybridCite: A Hybrid Model for Context-Aware Citation Recommendation](https://arxiv.org/abs/2002.06406) 2020

   based on hyperdoc2vec. Proposes to mix algorithms not by weights, but by something inspired from genetic algorithms. They mix one algo based on text content, and one algo based on graph structure. The idea can be adapted to any 2 aglo that can take a text context and suggest ranked papers.

   If paper A cites paper B, the citation context in paper A tells more about paper B than paper A.

   dataset : microsoft academic graph (220 million papers in 2019, grows each week) doesn't include fulltext, but includes abstracts and citation contexts ; ACL anthology ; arxiv

* [Linked Document Embedding for Classification](https://dl.acm.org/doi/10.1145/2983323.2983755) 2016

* GOOD [hyperdoc2vec: Distributed Representations of Hypertext Documents](https://www.aclweb.org/anthology/P18-1222.pdf) 2018 **citation rec**

   uses the text content of a link (more relevant for internet links than citations) and the context of the link. a problem to tackle is that new papers will have no links toward them, but we want new papers. semantics of "evaluated by [citation]" or "evaluated on corpus [citation]". dual embeddings of papers : in and out.

   "context as content" : if paper A cites paper B, we take the text in A surrounding the citation (context) and append it to the text of paper B (content)

   corpus : DBLP, ACL anthology and NIPS (one machine learning conference, fulltext)

* GOOD [Docear software](https://www.researchgate.net/publication/265412702_The_Architecture_and_Datasets_of_Docear%27s_Research_Paper_Recommender_System) 2014 **paper rec**
    
    Citation manager software that uses the user's library, including the organisation (categories, custom tags, summary/main ideas written by the user...) to recommend new papers. It's a clever way to get the user to do the work. The user can also highlight and annotate specific parts of the pdf in their library. Nodes can be : a paper, category, tag, annotation... The user can also rate the recommendations.

    The idea is super interesting. Developping something similar would be far out of the scope of the project, but we should keep it in mind.

* [Scientific Article Recommendation: Exploiting Common Author Relations and Historical Preferences](http://thealphalab.org/papers/ScientificArticleRecommendationExploitingCommonAuthorRelationsandHistoricalPreferences.pdf) 2016

    Author-based system. dataset : CiteULike, an online citation manager that makes its data available.

* [Cross-domain item recommendation based on user similarity](https://www.semanticscholar.org/paper/Cross-domain-item-recommendation-based-on-user-Xu-Jiang/797ed887d40cdcd3ef83b2781d979c351787dea5) 2016

   another author-based system

* [PaperRank](https://ieeexplore.ieee.org/document/5171002) 2009

   based on [Weighted PageRank](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.5022&rep=rep1&type=pdf) (not about graphs, can be used for link prediction in citation networks, 2004)

    PageRank : classic paper. used on internet info retrieval, can be used in the context of a citation network to rank influential papers. Not sure this will be useful for our work, but I've seen it used in a few papers ( including [this other paper in our bib](https://openreview.net/forum?id=W3Dzaik1ipL)) so keeping it on hand

* [VOPRec: Vector Representation Learning of Papers with Text Information and Structural Identity for Recommendation](https://www.researchgate.net/publication/324787467_VOPRec_Vector_Representation_Learning_of_Papers_with_Text_Information_and_Structural_Identity_for_Recommendation) 2018 **paper rec**


### other

* [Not All Contexts Are Created Equal:Better Word Representations with Variable Attention](https://www.researchgate.net/publication/301446021_Not_All_Contexts_Are_Created_Equal_Better_Word_Representations_with_Variable_Attention) 2015 **word embeddings*

   word embeddings with attention

* [BioSentVec: creating sentence embeddings for biomedical texts](https://arxiv.org/abs/1810.09302) **sentence embedding, resource, not about graphs** 2018
    
    trained sent2vec on a corpus composed of pubmed articles and clinical notes (MIMIC-III), and made the resulting embeddings available
    

* [Citation-Enhanced Keyphrase Extraction from Research Papers: A Supervised Approach](https://www.aclweb.org/anthology/D14-1150/) 2014

* [corpus pubmed fulltext](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) [download link](https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/)
