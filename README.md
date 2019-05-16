# Recommender Systems for web articles
This repository was created as a part of my bachelor's thesis. Recommender Systems were the topic of my thesis and its main objective was to create several models recommending web articles in the domain developers.redhat.com. In total four different models were created. They use traditional methods, such as the __Singular value decomposition__ (SVD) or collaborative filtering with the __Alternating Least Squares__ (ALS) method, and also propose some rather less common approaches using the __Doc2Vec__ and a __SkipGram negative sampling__ inspired methods. Besides the source files of all implemented recommender models, this repository also includes all required datasets and latex documentation of the thesis text.

### Downloading pretrained Doc2Vec model
! To be able to work with the pre-trained Doc2Vec model (used in the __load_pretrained__ method in __Doc2VecModel__ class) it is necessary to first download the pre-trained model from: 
- [English Wikipedia DBOW (1.4GB)](https://ibm.ent.box.com/s/3f160t4xpuya9an935k84ig465gvymm2)

The model was retrieved from the following GitHub repository:
- [jhlau/doc2vec](https://github.com/jhlau/doc2vec)

After downloading the model you have to unzip its content and place the __enwiki_dbow__ directory into the __data/processed__ folder. After that the pretrained binary model should be availabe at __data/processed/enwiki_dbow/doc2vec.bin__.

### Performing experiments
To review the experiments performed with our models refer to the __experiments.ipynb__ file. This ipython notebook explains the usage of our models and describes the process of their evaluation. It also shows some of the experiments created in the course of this thesis and enables one to further play with the models and study their abilities.

### Directory structure
Besides the implemented recommenders this repository also includes all required datasets and latex source files used to create its pdf documentation. The repository forms the following structure:

- __data__: directory containing our datasets
  - _json_: contains metadata of articles in JSON
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
