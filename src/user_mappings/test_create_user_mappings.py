"""
Tests for create_user_mappings function.
The create_user_mappings function maps users to unique identifiers. It uses
UserMappings class to create the mappings of the desired dataframe.

Usage:      $ pytest test_create_user_mappings.py

"""
import pytest
import pandas as pd
import os.path
from user_mappings.user_mappings import create_user_mappings


data_path = '../../data/tests'


class TestCreateUserMappings(object):
    test_dataset_file = os.path.join(data_path, 'easy_test_1.csv')
    correct_mappings = {'11111' : 0, '22222' : 1, '23' : 2}
    correct_pointers = {'33333' : ['23']}

    @classmethod
    def setup_class(cls):
        cls.df = pd.read_csv(cls.test_dataset_file, dtype={'visitor_id' : object,
                                                            'user_id' : object})
        cls.df = cls.df.fillna('')

    def test_mappings_correct(self):
        user_mappings = create_user_mappings(self.df)
        assert(self.correct_mappings == user_mappings.mappings)
        assert(self.correct_pointers == user_mappings.pointers)


class TestAddUserID(TestCreateUserMappings):
    test_dataset_file = os.path.join(data_path, 'easy_test_2.csv')
    correct_mappings = {'21' : 0, '22' : 1, '23' : 2}
    correct_pointers = {'33333' : ['23'], '11111' : ['21'], '22222' : ['22']}


class TestAddVisitorId(TestCreateUserMappings):
    test_dataset_file = os.path.join(data_path, 'easy_test_3.csv')
    correct_mappings = {'21' : 0, '22' : 1, '23' : 2}
    correct_pointers = {'33333' : ['23'], '11111' : ['21'], '22222' : ['22'],
                        '55555' : ['21'], '66666' : ['22']}


class TestBeforeMerge(TestCreateUserMappings):
    test_dataset_file = os.path.join(data_path, 'before_merge_test.csv')
    correct_mappings = {'21' : 0, '22' : 1, '23' : 2, '44444' : 3}
    correct_pointers = {'33333' : ['23'], '11111' : ['21'], '22222' : ['22'],
                        '55555' : ['21'], '66666' : ['22']}


class TestMerge(TestCreateUserMappings):
    test_dataset_file = os.path.join(data_path, 'merge_test.csv')
    correct_mappings = {'21' : 0, '22' : 1, '23' : 2}
    correct_pointers = {'33333' : ['23'], '11111' : ['21'], '22222' : ['22'],
                        '55555' : ['21'], '66666' : ['22'], '44444' : ['23']}


class TestLong(TestCreateUserMappings):
    test_dataset_file = os.path.join(data_path, 'long_test.csv')
    correct_mappings = {'21' : 0, '22' : 1, '23' : 2, '24' : 3, '25' : 4, '26' : 5}
    correct_pointers = {'33333' : ['23'], '11111' : ['21', '22', '26'], '22222' : ['22'],
                        '55555' : ['21'], '66666' : ['22'], '44444' : ['23'],
                        '77777' : ['24'], '88888' : ['25'], '99999' : ['26']}
