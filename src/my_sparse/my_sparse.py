from scipy.sparse import coo_matrix
import numpy as np
import math


def make_sparse(df, url_2_id, uid_2_id, users_in_rows=True):
    if users_in_rows:
        shape = (len(uid_2_id), len(url_2_id))
    else:
        shape = (len(url_2_id), len(uid_2_id))

    matrix = IncrementalSparseMatrix(shape=shape)

    for _, row in df.iterrows():
        user = uid_2_id[row.uid]            
        if row.page_url not in url_2_id:
            continue
        item = url_2_id[row.page_url]
        time = row.time

        if users_in_rows:
            matrix.add(user, item, time)
        else:
            matrix.add(item, user, time)

    return matrix


class IncrementalSparseMatrix(object):

    def __init__(self, shape=None):
        self.__entries = dict()
        self.__rows = list()
        self.__cols = list()
        self.__data = list()
        self.__shape = shape

    @property
    def rows(self):
        return self.__rows

    @property
    def cols(self):
        return self.__cols

    @property
    def data(self):
        return self.__data

    @property
    def entries(self):
        return self.__entries

    @property
    def shape(self):
        return self.__shape


    def __confidence_linear(self, values, alpha=12):
        preference = np.array(values).sum()
        preference = preference // 60 # make it be in minutes
        preference = preference if preference > 0 else 1
        confidence = 1 + (alpha * preference)

        return confidence

    def __confidence_log(self, values, alpha, epsilon):
        preference = np.array(values).sum()
        preference = preference // 60 # make it be in minutes
        preference = preference if preference > 0 else 1
        confidence = 1 + (alpha * math.log(1 + (preference/epsilon)))
        return confidence


    def add(self, row, col, data):
        if not (row, col) in self.__entries:
            self.__entries[(row, col)] = list()

        self.__entries[(row, col)].append(data)


    def to_coo(self, metrics='log', alpha=13, epsilon=4.0e-08, alpha1=None, threshold=None):
        if self.__data:
            return coo_matrix((self.__data, (self.__rows, self.__cols)))

        for (key, values) in self.__entries.items():
            self.__rows.append(key[0])
            self.__cols.append(key[1])
            if metrics == 'log':
                self.__data.append(self.__confidence_log(values, alpha, epsilon))
            elif metrics == 'bin':
                self.__data.append(1)
            elif metrics == 'lin':
                self.__data.append(self.__confidence_linear(values, alpha))

        return coo_matrix((self.__data, (self.__rows, self.__cols)), shape=self.shape)


    def get_weights(self):
        weights = {}
        for (key, values) in self.__entries.items():
            weights[key] = np.array(values).sum()

        return weights
