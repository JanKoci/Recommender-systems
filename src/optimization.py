"""
Author: Jan Koci

This file contains the Optimizer class.

"""
from skopt import forest_minimize
from evaluation import Evaluator
from hyperopt import fmin, tpe

        
    
class Optimizer(object):
    """
    The Optimizer class is used to find optimal hyperparameters of 
    the recommenders models.
    
    Attributes:
        model : the recommender to be optimized
        params : list of hyperparameters
        space  : search space
        optimized_params : found optimal parameters
        
    """
    def __init__(self, model, params, space):
        self.__model = model
        self.__params = params
        self.__space = space
        self.__evaluator = Evaluator(self.__model)
        self.__train_df = None
        self.__test_df = None
        
        
    @property
    def optimized_params(self):
        return self.__optimized_params
    
    @property
    def space(self):
        return self.__space
    
    @property
    def params(self):
        return self.__params
    
    
    def optimized_params_dict(self):
        return dict(zip(self.params, self.optimized_params.x))
        
        
    def optimize_forest(self, train_df, 
                       test_df, 
                       verbose=True, 
                       random_state=42, 
                       n_calls=50,
                       objective='rank'):
        """
        Finds optimal parameters using the skopt.forrest_minimize function
        
        Attributes:
            train_df   : dataset for training
            test_df    : test dataset
            n_calls    : maximum number of training the model
            objective  : the metric to be minimized
            
        """
        self.__train_df = train_df
        self.__test_df = test_df
        
        if objective == 'rank':
            objective_func = self.__objective_rank
        elif objective == 'recall':
            objective_func = self.__objective_recall
        elif objective == 'precision':
            objective_func = self.__objective_precision
            
        self.__optimized_params = forest_minimize(objective_func,
                                                  self.__space,
                                                  n_calls=n_calls,
                                                  verbose=verbose,)
                                                  #random_state=random_state
        print("_________OPTIMIZATION FINISHED_________")
        print('optimal parameters:')
        for name, value in zip(self.params, self.__optimized_params.x):
            print(name, '=', value)
            
    
    def optimize_hyperopt(self, train_df, test_df, n_calls=50, objective='rank'):
        """
        Finds optimal parameters using hyperopt.fmin function
        
        Attributes:
            train_df   : dataset for training
            test_df    : test dataset
            n_calls    : maximum number of training the model
            objective  : the metric to be minimized
            
        """
        self.__train_df = train_df
        self.__test_df = test_df
        
        if objective == 'rank':
            objective_func = self.__objective_rank
        elif objective == 'recall':
            objective_func = self.__objective_recall
        elif objective == 'precision':
            objective_func = self.__objective_precision
        
        self.__optimized_params = fmin(objective_func, 
                                       space=self.__space,
                                       algo=tpe.suggest,
                                       max_evals=n_calls)
        
    
    def __objective_rank(self, search_space):
        params_dict = dict(zip(self.__params, search_space))
        self.__model.train(self.__train_df, **params_dict)
        return self.__evaluator.rank_evaluation(self.__test_df)
    
    
    def __objective_recall(self, search_space):
        params_dict = dict(zip(self.__params, search_space))
        self.__model.train(self.__train_df, **params_dict)
        return (- self.__evaluator.recall_at_k(self.__test_df, k=10))
    
    def __objective_precision(self, search_space):
        params_dict = dict(zip(self.__params, search_space))
        self.__model.train(self.__train_df, **params_dict)
        return (- self.__evaluator.precision_at_k(self.__test_df, k=10))
        
        
        
        
        
        
        
        
        
        
