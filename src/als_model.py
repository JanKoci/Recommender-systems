"""
Author: Jan Koci

Implementation of class ImplicitALS

Usage:
    >>> from als_model import ImplicitALS
    >>> model = ImplicitALS(df)
    >>> model.train(train_df, factors=10, iterations=20)
    >>> model.recommend(uid=112, n=5)

"""
import helpers
from my_sparse.my_sparse import make_sparse
import implicit
import numpy as np
from abstract.recommender import RecommenderAbstract



class ImplicitALS(RecommenderAbstract):
    """This class implements a collaborative filtering approach using matrix 
    factorization with the Alternating Least Squares (ALS) algorithm.
    
    Attributes:
        df            : pandas DataFrame object containing the dataset 
                        of user-item interactions
        model         : created ALS model
        user_factors  : created user vectors
        item_factors  : created item vectors
        url_2_id      : dictionary mapping urls to internal Ids of the model
        uid_2_id      : dictionary mapping users to internal Ids of the model
    """

    def __init__(self, df):
        self.__df = df
        self.__url_2_id, self.__id_2_url = helpers.make_item_mappings(df)
        self.__uid_2_id, self.__id_2_uid = helpers.make_user_mappings(df)
        self.__model = None
        self.__user_factors = None
        self.__item_factors = None
        self.__train_df = None


    @property
    def df(self):
        return self.__df

    @property
    def model(self):
        return self.__model

    @property
    def user_factors(self):
        return self.__user_factors

    @property
    def item_factors(self):
        return self.__item_factors

    @property
    def url_2_id(self):
        return self.__url_2_id

    @property
    def uid_2_id(self):
        return self.__uid_2_id

    @property
    def id_2_url(self):
        return self.__id_2_url

    @property
    def id_2_uid(self):
        return self.__id_2_uid


    def train(self, train_df, factors=None, iterations=None,
              alpha=None, epsilon=None, metric='log'):
        print("##################### ALS model #####################")
        self.__train_df = train_df
        print("[1]  Creating sparse interaction matrix")
        sparse = make_sparse(train_df, self.__url_2_id, self.__uid_2_id, users_in_rows=False)
        
        if metric == 'log':
            sparse = sparse.to_coo(metrics='log', alpha=alpha, epsilon=epsilon)
        elif metric == 'lin':
            sparse = sparse.to_coo(metrics='lin', alpha=alpha)
        elif metric == 'bin':
            sparse = sparse.to_coo(metrics='bin')
        else:
            return
        sparse = sparse.tocsc()
        self.__model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                            iterations=iterations,
                                                            num_threads=0)
        print("\n[2]  Fitting the matrix to the model")
        self.__model.fit(sparse)
        self.__user_factors = self.model.user_factors
        self.__item_factors = self.model.item_factors
        print("\n[DONE]  ALS successfull")


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
        if self.model == None:
            return
        user_index = self.__uid_2_id[uid]
        user_factor = self.user_factors[user_index]
        scores = self.item_factors.dot(user_factor)
        
        if exclude_seen_pages:
            seen_pages = self.__train_df.loc[self.__train_df.uid==uid].page_url.unique()
            page_ids = [self.url_2_id[url] for url in seen_pages]
        else:
            page_ids = []

        scores = ([(self.id_2_url[id], value) for id, value in enumerate(scores)
                                            if id not in page_ids])
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
        return np.dot(self.user_factors[self.uid_2_id[uid]],
                      self.item_factors[self.url_2_id[url]])
        
    def nearest_items(self, url, n=None):
        """
        Find nearest items for item specified by its url
        
        Attributes:
            url  : item
            n    : number of nearest items that will be returned
        
        Returns:
            list of n tuples (url, score)
            
        """
        item_vector = self.item_factors[self.url_2_id[url]]
        scores = self.item_factors.dot(item_vector)
        scores = [(self.id_2_url[id], value) for id, value in enumerate(scores)]
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
        user_vector = self.user_factors[self.uid_2_id[uid]]
        scores = self.user_factors.dot(user_vector)
        scores = [(self.id_2_uid[id], value) for id, value in enumerate(scores)]
        scores = sorted(scores, reverse=True, key=lambda tup: tup[1])
        return scores[:n]
        


    def __get_rank_ui(self, page_id, recommendations):
        page_position = [self.url_2_id[tup[0]] for tup in recommendations].index(page_id)
        return (page_position / (len(recommendations)-1)) * 100
















