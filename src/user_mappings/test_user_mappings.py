"""
Tests for UserMappings class methods

Usage:  $ pytest test_user_mappings.py

"""
import pytest
from user_mappings.user_mappings import UserMappings


class TestMappings(object):

    @classmethod
    def setup_class(cls):
        cls.mappings = UserMappings()

    def test_simple(self):
        self.mappings.add_user('1111')
        self.mappings.add_user('2222')
        self.mappings.add_user('3333', '3')
        assert({'1111' : 0, '2222' : 1, '3' : 2} == self.mappings.mappings)
        assert({'3333' : ['3']} == self.mappings.pointers)
        assert(self.mappings.get_user('3333', '3') == 2)
        assert(self.mappings.get_user('1111') == 0)
        assert(self.mappings.get_user('3333', '4') == None)


    def test_actualize_user_id(self):
        assert({'1111' : 0, '2222' : 1, '3' : 2} == self.mappings.mappings)
        assert({'3333' : ['3']} == self.mappings.pointers)

        self.mappings.actualize_user('1111', '1')
        self.mappings.actualize_user('2222', '2')
        assert({'1' : 0, '2' : 1, '3' : 2} == self.mappings.mappings)
        assert({'3333' : ['3'], '1111' : ['1'], '2222' : ['2']} == self.mappings.pointers)


    def test_add_visitor_id(self):
        assert({'1' : 0, '2' : 1, '3' : 2} == self.mappings.mappings)
        assert({'3333' : ['3'], '1111' : ['1'], '2222' : ['2']} == self.mappings.pointers)

        self.mappings.actualize_user('5555', '1')
        self.mappings.actualize_user('6666', '2')
        assert({'1' : 0, '2' : 1, '3' : 2} == self.mappings.mappings)
        assert({'3333' : ['3'], '1111' : ['1'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2']} == self.mappings.pointers)


    def test_merge(self):
        assert({'1' : 0, '2' : 1, '3' : 2} == self.mappings.mappings)
        assert({'3333' : ['3'], '1111' : ['1'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2']} == self.mappings.pointers)
        self.mappings.add_user('4444')
        assert({'1' : 0, '2' : 1, '3' : 2, '4444' : 3} == self.mappings.mappings)
        assert({'3333' : ['3'], '1111' : ['1'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2']} == self.mappings.pointers)

        assert(self.mappings.get_user('4444', '2') == 1)
        self.mappings.actualize_user('4444', '2')
        assert({'1' : 0, '2' : 1, '3' : 2} == self.mappings.mappings)
        assert({'3333' : ['3'], '1111' : ['1'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2'], '4444' : ['2']} == self.mappings.pointers)


    def test_add_user_id(self):
        assert({'1' : 0, '2' : 1, '3' : 2} == self.mappings.mappings)
        assert({'3333' : ['3'], '1111' : ['1'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2'], '4444' : ['2']} == self.mappings.pointers)

        self.mappings.add_user('3333', '9')
        self.mappings.actualize_user('1111', '3')
        assert({'1' : 0, '2' : 1, '3' : 2, '9' : 3} == self.mappings.mappings)
        assert({'3333' : ['3', '9'], '1111' : ['1', '3'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2'], '4444' : ['2']} == self.mappings.pointers)


    def test_get_user(self):
        assert({'1' : 0, '2' : 1, '3' : 2, '9' : 3} == self.mappings.mappings)
        assert({'3333' : ['3', '9'], '1111' : ['1', '3'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2'], '4444' : ['2']} == self.mappings.pointers)

        assert(self.mappings.get_user('0000', '9') == 3)
        assert(self.mappings.get_user('0000') == None)
        assert(self.mappings.get_user('4444') == 1)
        assert(self.mappings.get_user('3333', '3') == 2)
        assert(self.mappings.get_user('3333') == ['3', '9'])
        assert(self.mappings.get_user('4242', '42') == None)
        assert(self.mappings.get_user('1111', '3') == 2)


    def test_add_existing_user(self):
        assert({'1' : 0, '2' : 1, '3' : 2, '9' : 3} == self.mappings.mappings)
        assert({'3333' : ['3', '9'], '1111' : ['1', '3'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2'], '4444' : ['2']} == self.mappings.pointers)

        self.mappings.add_user('3333')
        self.mappings.add_user('1234', '3')
        self.mappings.add_user('1111', '9')
        self.mappings.add_user('1111', '1')
        assert({'1' : 0, '2' : 1, '3' : 2, '9' : 3} == self.mappings.mappings)
        assert({'3333' : ['3', '9'], '1111' : ['1', '3'], '2222' : ['2'], '5555' : ['1'],
                '6666' : ['2'], '4444' : ['2']} == self.mappings.pointers)
