"""
Author: Jan Koci

Implementation of class SVDModel

Usage:
    >>> from svd_model import SVDModel
    >>> model = SVDModel(df)
    >>> model.train(train_df)
    >>> model.recommend(uid=112, n=5)

"""
import helpers
from my_sparse.my_sparse import make_sparse
import numpy as np
from abstract.recommender import RecommenderAbstract


class SVDModel(RecommenderAbstract):
    """SVDModel class implements a recommender system using 
    Singular Value Decomposition (SVD).
    
    Attributes:
        df  : dataset
        u   : users in rows, orthogonal matrix containing eigenvectors of X*X^T
        s   : vector of singular values
        vh  : items in columns, transposed orthogonal matrix containing containing
              eigenvectors of X^T*X
"""

    def __init__(self, df):
        self.__df = df
        self.__train_df = None
        self.__url_2_id, self.__id_2_url = helpers.make_item_mappings(df)
        self.__uid_2_id, self.__id_2_uid = helpers.make_user_mappings(df)
        self.__u = None
        self.__s = None
        self.__vh = None


    @property
    def df(self):
        return self.__df

    @property
    def url_2_id(self):
        return self.__url_2_id

    @property
    def id_2_url(self):
        return self.__id_2_url

    @property
    def uid_2_id(self):
        return self.__uid_2_id

    @property
    def id_2_uid(self):
        return self.__id_2_uid

    @property
    def u(self):
        return self.__u

    @property
    def s(self):
        return self.__s

    @property
    def vh(self):
        return self.__vh

    @property
    def train_df(self):
        return self.__train_df


    def train(self, train_df, alpha=None, epsilon=None, min_interactions=1, metric='bin'):
        print("##################### SVD model #####################")
        self.__train_df = train_df
        if min_interactions != 1:
            train_df = helpers.df_min_interactions(df=train_df,
                                                   min_interactions=min_interactions)
        print("[1]  Making sparse interaction matrix")
        sparse = make_sparse(df=train_df,
                             url_2_id=self.url_2_id,
                             uid_2_id=self.uid_2_id)

        if metric == 'log':
            sparse = sparse.to_coo(metrics='log', alpha=alpha, epsilon=epsilon)
        elif metric == 'bin':
            sparse = sparse.to_coo(metrics='bin')
        elif metric == 'lin':
            sparse = sparse.to_coo(metrics='lin', alpha=alpha)

        print("[2]  Performing SVD on interaction matrix")
        self.__u, self.__s, self.__vh = np.linalg.svd(sparse.toarray(),
                                                     full_matrices=False)
        print("[DONE]  SVD successfull")



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
        user_position = self.uid_2_id[uid]

        scores = np.dot(self.u[user_position], self.vh.transpose())
        
        if exclude_seen_pages:
            seen_pages = self.train_df.loc[self.train_df.uid==uid].page_url.unique()
            page_ids = [self.url_2_id[url] for url in seen_pages]
        else:
            page_ids = []

        scores = ([(self.id_2_url[id], value) for id, value in enumerate(scores)
                                              if id not in page_ids])
        scores = sorted(scores, reverse=True, key=lambda tup: tup[1])
        return scores[:n]


    def predict(self, uid, url):
        smat = np.diag(self.s)
        scores = np.dot(np.dot(self.u[self.uid_2_id[uid]], smat), self.vh)
        return scores[self.url_2_id[url]]



    def save_matrix(self, filename, matrix='u', delimiter="\t"):
        matrix_to_save = None
        if matrix == 'u':
            matrix_to_save = self.__u

        elif matrix == 's':
            matrix_to_save = self.__s

        elif matrix == 'vh':
            matrix_to_save = self.__vh

        else:
            raise AttributeError("matrix attribute has to be one of the SVD "
                                 "matrices 'u', 's', 'vh'")

        np.savetxt(filename, matrix_to_save, delimiter=delimiter)
