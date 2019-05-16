# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class RecommenderAbstract(ABC):
    
    @property
    @abstractmethod
    def url_2_id(self):
        pass

    @property
    @abstractmethod
    def uid_2_id(self):
        pass

    @property
    @abstractmethod
    def id_2_url(self):
        pass

    @property
    @abstractmethod
    def id_2_uid(self):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def recommend(self, uid, n=None):
        pass

    @abstractmethod
    def predict(self, uid, url):
        pass