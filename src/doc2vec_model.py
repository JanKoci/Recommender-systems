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
import numpy as np
from abstract.recommender import RecommenderAbstract
from doc2vec_class import Doc2VecClass
from my_sparse.my_sparse import make_sparse
from data_utils import RecommenderDataFrame
from gensim.models import Doc2Vec



class Doc2VecModel(RecommenderAbstract):
    """Doc2VecModel class wrapps the Doc2VecClass to create a recommender 
    implementing the API defined by RecommenderAbstract
    
    Attributes:
        dataframe    : object of RecommenderDataFrame class creater from 
                       the passed dataset of user-item interactions
        model        : instance of the Doc2VecClass
        user_vectors : created user vectors
        doc_vectors  : created document vectors
    
    """
    
    def __init__(self, df, input_model):
        self.__input = input_model
        self.model = None
        self.__user_vectors = None
        self.__doc_vectors = None
        
        # create tagged documents
        self.__input.fit_transform()
        self.__url_2_id = self.__input.url_2_id
        self.__id_2_url = self.__input.id_2_url
        
        # make user mappings
        self.dataframe = RecommenderDataFrame(df, url_2_id=self.__url_2_id)
        self.__uid_2_id = self.dataframe.uid_2_id
        self.__id_2_uid = self.dataframe.id_2_uid
        
        
    @property
    def url_2_id(self):
        return self.__url_2_id

    @property
    def uid_2_id(self):
        return self.dataframe.uid_2_id

    @property
    def id_2_url(self):
        return self.__id_2_url

    @property
    def id_2_uid(self):
        return self.dataframe.id_2_uid
    
    @property
    def doc_vectors(self):
        return self.__doc_vectors
    
    @property
    def user_vectors(self):
        return self.__user_vectors
        
    
    def train(self, train_df=None, # dummy variable :)
                    vector_size=20, 
                    epochs=50, 
                    alpha1=0.25, 
                    alpha2=None, 
                    epsilon=None,
                    metric='bin'):
        """
        This method creates the Doc2VecClass object to train the model locally 
        only on the passed tagged input
        Attributes:
            alpha1  : hyperparameter used in gensim Doc2Vec model
            alpha2  : hyperparameter for log confidence
            epsilon : hyperparameter for log confidence
            metric  : determines the metric for creating user vectors,
                      possible values are log, lin, bin
        """
        self.__user_vectors = np.zeros(shape=(self.dataframe.df.uid.unique().shape[0], 
                                              vector_size))

        self.model = Doc2VecClass(self.__input)
        self.model.train(vector_size, alpha1, epochs)
        self.__doc_vectors = self.model.model.docvecs.vectors_docs
        print('Creating user vectors')
        if metric == 'bin':
            self.create_user_vectors_bin()
        else:
            self.create_user_vectors(alpha2, epsilon, metric=metric)
        print('Done')
        
        
    def load_pretrained(self, model_path='../data/processed/enwiki_dbow/doc2vec.bin', model_dim=300):
        """
        This method loads a pre-trained Doc2Vec model and infers document 
        vectors for the given tagged input. It can only use the bin metric 
        for creating user vectors.
        
        Attributes:
            model_path : path to the model
            model_dim  : dimensionality of its word vectors
            
        """
        model = Doc2Vec.load(model_path)
        docvecs = np.zeros((len(self.__input.url_2_id), model_dim))
        for text in self.__input.input:
            docvecs[text[1]] = model.infer_vector(text[0])
        self.__doc_vectors = docvecs
        self.model = Doc2VecClass(self.__input)
        self.model.doc_vectors = docvecs
        self.create_user_vectors_bin()
        
         
    def recommend(self, uid, n=None, exclude_seen_pages=False):
        """
        Creates list of recommendations for user uid in format: [(score, url),]
        containing the computed score and url of the recommended item.
        
        Attributes:
            uid : identifier of the user
            n   : number of tuples (recommendations) that will be returned
            
        Returns:
            the created list of recommendations
            
        """
        user_vector = self.__user_vectors[self.uid_2_id[uid]]
        scores = self.doc_vectors.dot(user_vector)
        
        if exclude_seen_pages:
            seen_pages = self.dataframe.df.loc[self.dataframe.df.uid==uid].page_url.unique()
            page_ids = [self.url_2_id[url] for url in seen_pages \
                                            if url in self.url_2_id]
        else:
            page_ids = []
        
        scores = [(self.id_2_url[id], score) for id, score in enumerate(scores) \
                                            if id not in page_ids]
        scores = sorted(scores, reverse=True, key=lambda tup: tup[1])
        return scores[:n]
    
    
    def predict(self, uid, url):
        """
        Predict the score between a user and an item
        
        Attributes:
            uid  : the user identifier
            url  : url of the item
            
        Returns:
            the predicted score
            
        """
        user_vector = self.__user_vectors[self.uid_2_id[uid]]
        page_vector = self.doc_vectors[self.url_2_id[url]]
        return user_vector.dot(page_vector)
    
    
    def nearest_items(self, url, n=None):
        """
        Find nearest items for item specified by its url
        
        Attributes:
            url  : item
            n    : number of nearest items that will be returned
        
        Returns:
            list of n tuples (url, score)
            
        """
        page_vector = self.doc_vectors[self.url_2_id[url]]
        scores = self.doc_vectors.dot(page_vector)
        scores = [(self.id_2_url[id], score) for id, score in enumerate(scores)]
        scores = sorted(scores, reverse=True, key=lambda tup: tup[1])
        return scores[:n]
    
    def nearest_users(self, uid, n=None):
        """
        Find nearest users for user specified by his uid
        
        Attributes:
            uid  : user identifier
            n    : number of nearest users that will be returned
        
        Returns:
            list of n tuples (uid, score)
            
        """
        user_vector = self.user_vectors[self.uid_2_id[uid]]
        scores = self.user_vectors.dot(user_vector)
        scores = [(self.id_2_uid[id], score) for id, score in enumerate(scores)]
        scores = sorted(scores, reverse=True, key=lambda tup: tup[1])
        return scores[:n]
    
    
    def create_user_vectors(self, alpha, epsilon=None, metric='log'):
        """
        Create user vectors as a weighted average of the document vectors
        of the items they interacted with. The weight is the confidence of the
        interaction that is computed using the log, or lin metric.
        
        Attributes:
            alpha, epsilon  : hyperparameters for computing the confidence
            metric  : metric for computing the confidence (log, or lin)
        
        """
        sparse = make_sparse(self.dataframe.df, self.url_2_id, self.uid_2_id)
        if metric == 'log':
            sparse = sparse.to_coo(alpha=alpha, epsilon=epsilon, metrics='log')
        elif metric == 'lin':
            sparse = sparse.to_coo(alpha=alpha, metrics='lin')
        sparse = sparse.toarray()
        
        confidence_sums = sparse.sum(axis=1)
        confidence_sums = confidence_sums.reshape((len(confidence_sums), 1))
        # ommit zero division :) 
        confidence_sums[confidence_sums == 0] = 1
        
        user_matrix = sparse.dot(self.doc_vectors)
        user_matrix = user_matrix / confidence_sums
        
        self.__user_vectors = user_matrix
        
        
    def create_user_vectors_bin(self):
        """
        Creates user vectors as a sum of their document vectors, meaning
        document vectors of items the interacted with.
        """
        sparse = make_sparse(self.dataframe.df, self.url_2_id, self.uid_2_id)
        sparse = sparse.to_coo(metrics='bin')
        sparse = sparse.toarray()
        
        self.__user_vectors = sparse.dot(self.doc_vectors)
        
            
            
            
            
            