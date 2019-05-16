"""
Author: Jan Koci

File with optimal parameters of our recommenders. It also defined seach spaces
that were used for optimization

"""
from hyperopt import hp

#######################################################################
# ALS
#######################################################################
# 13
als_rank_bin = {'factors': 13, 'iterations': 76}
# 10.34
als_rank_log = {'factors': 4, 'iterations': 76, 'alpha': 11, 'epsilon': 0.0009337325561958652}
# 10.61
als_rank_log_minutes = {'factors': 4, 'iterations': 80, 'alpha': 12, 'epsilon': 0.00033263101791867285}
# 12.12
als_rank_lin_minutes = {'factors': 4, 'iterations': 78, 'alpha': 17}

# 0.28
als_recall_log = {'factors': 44, 
                  'iterations': 64,
                  'alpha': 8, 
                  'epsilon': 9.281032059054776e-08}
# 0.28
als_recall_log_minutes = {'factors': 29,
                          'iterations': 60,
                          'alpha': 6,
                          'epsilon': 2.3944944176762766e-07}
# 0.18
als_recall_bin = {'factors': 22, 'iterations': 33}

# 0.12
als_precision_log = {'factors': 36,
                     'iterations': 58,
                     'alpha': 12,
                     'epsilon': 1.1872582083530007e-07}
# 0.2728
als_recall_lin = {'factors': 49, 'iterations': 58, 'alpha': 22}

params_als = ['factors', 'iterations', 'alpha', 'epsilon']
space_als = [(1, 50), # factors
             (5, 80), # iterations
             (1, 50), # alpha
             (1, 50), # alpha1
             (10**-9, 10**-2, 'log-uniform'), # epsilon
             (120, 1200) # threshold
             ]



#######################################################################
# Doc2Vec
#######################################################################
# 26.63
doc2vec_rank_optimum_log = {'vector_size': 13,
                             'epochs': 91,
                             'alpha1': 0.014168018793577233,
                             'alpha2': 26,
                             'epsilon': 0.0070501000450619924}

# 27.72
doc2vec_rank_optimum_bin = {'vector_size': 17, 'epochs': 26, 'alpha1': 0.031130269077638238}

# 0.11
doc2vec_recall_log = {'vector_size': 60,
                              'epochs': 26,
                              'alpha1': 0.08039269127820824,
                              'alpha2': 18,
                              'epsilon': 0.002188277627389094}
# 0.096
doc2vec_recall_bin = {'vector_size': 28, 'epochs': 26, 'alpha1': 0.018803079426143548}

# 0.043
doc2vec_precision_log = {'vector_size': 36,
                         'epochs': 34,
                         'alpha1': 0.023183610672843313,
                         'alpha2': 16,
                         'epsilon': 0.00606472254717809}
# 0.043
doc2vec_precision_bin = {'vector_size': 31, 'epochs': 27, 'alpha1': 0.028214045570393524}

params_doc2vec = ['vector_size', 'epochs', 'alpha1', 'alpha2', 'epsilon']
params_doc2vec_bin = ['vector_size', 'epochs', 'alpha1']

space_doc2vec = [(10, 60),                          # vector_size
                 (20, 100),                         # epochs
                 (10**-6, 10**-1),   # alpha1
                 (1, 30),                           # alpha2
                 (10**-9, 10**-2, 'log-uniform')    # epsilon
                 ]
space_doc2vec_bin = [(5, 60),                          # vector_size
                     (10, 80),                         # epochs
                     (10**-6, 10**-1),   # alpha1
                     ]


#######################################################################
# SkipGram
#######################################################################
skip_gram_optimum = {'epochs': 20,
                     'num_tags': 50,
                     'tag_dimension': 60,
                     'hidden_dimension': 200,
                     'out_dimension': 150,
                     'learning_rate': 0.001,
                     'num_negatives': 20}

params_skip_gram = ['epochs', 'num_tags', 'tag_dimension', 'hidden_dimension', 
                    'out_dimension', 'learning_rate', 'num_negatives']

space = [hp.quniform('epochs', 10, 100, 1),
         hp.quniform('num_tags', 20, 80, 1),
         hp.quniform('tag_dimension', 60, 200, 1),
         hp.quniform('hidden_dimension', 50, 300, 1),
         hp.quniform('out_dimension', 50, 300, 1),
         hp.loguniform('learning_rate', -7, 0),
         hp.quniform('num_negatives', 2, 20, 1)]

space_skip_gram = [(5, 50),
                   (10, 50),
                   (10, 50),
                   (10, 100),
                   (5, 80),
                   (-7, 0, 'log-uniform'),
                   (1, 10)]