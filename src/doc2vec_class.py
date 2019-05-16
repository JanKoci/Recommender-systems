""" 
Author: Jan Koci

Implementation of classes Doc2VecInput and Doc2VecClass
The Doc2VecInput class is used for creating input of the Doc2Vec 
model implemented in the gensim library. 

Usage:
    >>> from doc2vec_class import Doc2VecInput, Doc2VecClass
    >>> doc_input = Doc2VecInput(metadata)
    >>> doc2vec = Doc2VecClass(doc_input)
    >>> doc2vec.train()
    >>> page = doc2vec.url_2_id[url]
    >>> page_vector = doc2vec.model.docvecs.vectors_docs(page)

"""
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk import word_tokenize
import numpy as np



class Doc2VecInput(object):
    """The Doc2VecInput class is used to create list of TaggedDocunt objects 
    from articles metadata in json format as the needed input 
    for the Doc2Vecgensim  method
    
    Attributes:
        docs_json     : json metadata loaded in a dictionary
        input         : a list of TaggedInput objects created from docs_json
        url_2_id      : dictionary mapping urls to internal Ids of the model
        
    """
    
    def __init__(self, docs_json=None):
        assert type(docs_json == dict)
        
        if docs_json:
            self.docs_json = docs_json
        else:
            self.docs_json = dict()
            
        self.input = None
        self.__url_2_id = dict()
        self.__id_2_url = dict()
        
        
    @property
    def url_2_id(self):
        return self.__url_2_id

    @property
    def id_2_url(self):
        return self.__id_2_url
        
        
    def add_json(self, docs_json):
        self.docs_json = {**self.docs_json, **docs_json}
        
        
    def fit_transform(self, url_2_id=None, id_2_url=None):
        """Transforms the json metadata stored in docs_json to a list
        of TaggedDocument objects
        
        Arguments:
            url_2_id    : if specified only pages that can be found in the 
                          mapping are included in the created input. 
                          If None, all pages are included and a new 
                          url_2_id mapping is created
        """
        if url_2_id and id_2_url:
            self.__url_2_id = url_2_id
            self.__id_2_url = id_2_url
            self.__create_input()
        else:
            self.__create_ids_input()
            
        
    def __create_ids_input(self):
        if self.url_2_id:
            self.__url_2_id = dict()
            self.__id_2_url = dict()
        
        tagged_docs = []
        i = 0
        self.removed = []
        
        for url, data in self.docs_json.items():
            doc_tokenized = self.__tokenize_document(data)
            if doc_tokenized:
                tagged_docs.append((i, doc_tokenized))
                self.__url_2_id[url] = i
                self.__id_2_url[i] = url
                i += 1
            else:
                self.removed.append(url)
        
        self.input = [TaggedDocument(tokens, [i]) for (i, tokens) in tagged_docs]
        
    

    def __create_input(self):
        tagged_docs = []
        
        for url, data in self.docs_json.items():
            if url in self.url_2_id:
                doc_tokenize = self.__tokenize_document(data)
                if doc_tokenize:
                    tagged_docs.append((self.url_2_id[url], doc_tokenize))
        
        self.input = [TaggedDocument(tokens, [i]) for (i, tokens) in tagged_docs]
    
    
    def __tokenize_document(self, doc_json):        
        if not doc_json['description'][0]:
            return ''
        doc_str = doc_json['title'][0] + ' ' + doc_json['description'][0]
        tokens = word_tokenize(doc_str.lower())
        tokens = tokens[:-1]
        return tokens
    
    

class Doc2VecClass(object):
    """The Doc2VecClass creates a wrapper of the Doc2Vec gensim model
    
    Attributes:
        tagged_input  : instance of Doc2VecInput class containing the metadata
        model         : the created gensim Doc2Vec model
        
    """    

    def __init__(self, tagged_input):
        self.input = tagged_input
        self.url_2_id = tagged_input.url_2_id
        self.id_2_url = tagged_input.id_2_url
        self.model = None
        self.doc_vectors = None
        
        
    def get_item(self, id):
        url = self.id_2_url[id]
        return self.input.docs_json[url]
        
        
    def get_item_title(self, id):
        url = self.id_2_url[id]
        return self.input.docs_json[url]['title'][0]
    
    
    def get_item_tags(self, id):
        url = self.id_2_url[id]
        return self.input.docs_json[url]['tags'][0]
    
    
    def get_item_description(self, id):
        url = self.id_2_url[id]
        return self.input.docs_json[url]['description'][0]
        
    
    
    def train(self, vector_size=5, alpha=0.025, max_epochs=50):
        """
        Train the model locally on the passed tagged input documents
        """
        min_alpha = 0.00025
        
        self.model = Doc2Vec(vector_size=vector_size, 
                             alpha=alpha, 
                             min_alpha=min_alpha)
        
        self.model.build_vocab(self.input.input)
        
        for epoch in range(max_epochs):
            print('Epoch {0} of {1}'.format(epoch+1, max_epochs))
            self.model.train(self.input.input, 
                             total_examples=self.model.corpus_count,
                             epochs=self.model.epochs)
            self.model.alpha -= 0.0002
            self.model.min_alpha = self.model.alpha
        self.doc_vectors = self.model.docvecs.vectors_docs
        
        
        
    def save_vecs_and_labels(self, labels_file='item_labels.tsv', 
                             vectors_file='doc2vec.tsv'):
        
        np.savetxt(vectors_file, self.doc_vectors, delimiter="\t")
        
        with open(labels_file, 'w') as file:
            file.write('id')
            file.write('\t')
            file.write('tokens')
            file.write('\n')
            for i in range(len(self.doc_vectors)):
                file.write(str(i))
                file.write('\t')
                file.write(self.get_item_title(i))
                file.write('\n')
        
        