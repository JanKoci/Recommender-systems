# Recommender Systems for web articles
This repository was created as a part of my bachelor's thesis. \
To be able to work with the pre-trained Doc2Vec model (the __load_pretrained__ method in the __Doc2VecModel__ class) it is necessary to first download the pre-trained model: 
- [English Wikipedia DBOW (1.4GB)](https://ibm.ent.box.com/s/3f160t4xpuya9an935k84ig465gvymm2)

The model was retrieved from the following GitHub repository:
- [jhlau/doc2vec](https://github.com/jhlau/doc2vec)

After downloading the model you have to unzip its content and place the __enwiki_dbow__ directory it into the __data/processed__ folder. After that the pretrained binary model should be availabe at __data/processed/enwiki_dbow/doc2vec.bin__.

The __experiments.ipynb__ ipython notebook contains the experiments performed with our implementer recommender models. It  explains the usage of all models and provides an easy and straightforward way of reviewing their capabilities.

This repository consists of the following contents:

- __data__: directory with datasets
  - _json_: contains metadata in JSON
  - _processed_: contains processed datasets and other needed structures
  - _tests_: contains test files for testing __UserMappings__ class
- __doc___: contains pdf file of this thesis
- ___src_doc___: contains latex source files of the this thesis
- ___src___: contains all models and other source files
  - _abstract_: module with the __RecommenderAbstract__ class
  - _my_sparse_: module with the __IncrementalSparseMatrix__ class
  - _user_mappings_: module with the __UserMappings__ class
  - _als_model.py_: implements the __ALS__ recommender
  - _data_utils.py_: implements functions and classes working with data
  - _doc2vec_class.py_: implements the __Doc2VecInput__ and __Doc2VecClass__ classes
  - _doc2vec_model.py_: implements the __Doc2VecModel__ recommender
  - _evaluation.py_: implements the __Evaluator__ class used for evaluating the models
  - _experiments.ipynb_: ipython notebook showing the performed experiments
  - _helpers.py_: implements helper functions
  - _optimal_parameters.py_: contains optimal parameters of the recommenders
  - _optimization.py_: implements the __Optimizer__ class used for finding optimal hyperparameters of our recommenders
  - _skip_gram_recommender.py_: implements the __SkipGramModel__ and __SkipGramRecommender__ classes
  - _svd_model.py_: implements the __SVDModel__ class
- _requirements.txt_: contains a list of all required libraries
