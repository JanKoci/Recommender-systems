"""
Author: Jan Koci

This file implements the Evaluatior class.

"""
from progress.bar import Bar
from my_sparse.my_sparse import make_sparse
from abstract.recommender import RecommenderAbstract


class Evaluator(object):
    """ The Evaluator class is used for evaluatin the recommenders.
    It provides three evaluation metrics: RANK, Precision@k, Recall@k
    
    Attributes:
        model : the recommender to be evaluated
    
    """

    def __init__(self, model):
        assert isinstance(model, RecommenderAbstract)
        self.__model = model


    @property
    def model(self):
        return self.__model


    def rank_evaluation(self, test_set):
        """
        Returns the RANK evaluation value computed using the test_set
        """
        if self.model == None:
            return # ASSERT of EXCEPTION
        test_set = test_set.groupby(['uid', 'page_url'])['time'].sum()
        user_iterator = None
        rank = 0
        bar = Bar('Evaluating', max=test_set.shape[0])
        self.dropped_urls = []

        for (user, url), time in test_set.items():
            if url not in self.model.url_2_id:
                test_set.drop((user, url), inplace=True)
                self.dropped_urls.append(url)
                continue
            
            if (user != user_iterator):
                recommendations = self.model.recommend(user)
                user_iterator = user
                
            try:
                rank_ui = time * self.__get_rank_ui(url, recommendations)
            except ValueError:
                test_set.drop((user, url), inplace=True)
                self.dropped_urls.append(url)
                continue
            
            rank += rank_ui
            bar.next()

        bar.finish()
        return rank / test_set.sum()


    def recall_at_k(self, test_df, k=5):
        """
        Returns the Recall@k value computed using the test_df
        """
        recall_sum = 0
        iteration = 0
        test_df = test_df.groupby('uid')['page_url'].unique()
        bar = Bar('Evaluating', max=test_df.shape[0])

        for uid, urls in test_df.items():
            scores = self.model.recommend(uid, k)
            scores = [tup[0] for tup in scores]
            recall_ui = len(set(scores) & set(urls)) / len(urls)
            recall_sum += recall_ui
            iteration += 1
            bar.next()

        bar.finish()
        return recall_sum / iteration


    def precision_at_k(self, test_df, k=5):
        """
        Returns the Precision@k computed using the test_df
        """
        precision_sum = 0
        iteration = 0
        test_df = test_df.groupby('uid')['page_url'].unique()
        bar = Bar('Evaluating', max=test_df.shape[0])

        for uid, urls in test_df.items():
            scores = self.model.recommend(uid, k)
            scores = [tup[0] for tup in scores]
            precision_ui = len(set(scores) & set(urls)) / k
            precision_sum += precision_ui
            iteration += 1
            bar.next()

        bar.finish()
        return precision_sum / iteration



    def mse(self, test_df):
        iteration = 0
        mse_sum = 0
        print('[1] Creating sparse matrix for whole dataset')
        sparse = make_sparse(df=test_df, url_2_id=self.model.url_2_id,
                             uid_2_id=self.model.uid_2_id)
        sparse = sparse.to_coo()
        sparse = sparse.toarray()
        test_df = test_df.groupby('uid')['page_url'].unique()
        bar = Bar('Processing', max=test_df.shape[0])

        for uid, urls in test_df.items():
            for url in urls:
                real = sparse[self.model.uid_2_id[uid]][self.model.url_2_id[url]]
                score = self.model.predict(uid, url)
                mse_ui = real - score
                mse_ui = mse_ui**2
                mse_sum += mse_ui
                iteration += 1
            bar.next()
        bar.finish()
        return mse_sum / iteration



    def __get_rank_ui(self, page_url, recommendations):
        page_position = [tup[0] for tup in recommendations].index(page_url)
        return (page_position / (len(recommendations)-1)) * 100
