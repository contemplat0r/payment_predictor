
# coding: utf-8

# In[162]:
import sys
import warnings
import pickle

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from numpy.testing import assert_almost_equal, assert_array_equal

from sklearn.datasets import load_boston
from sklearn import cross_validation
from sknn.mlp import Regressor, Layer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.testing import (assert_raises, assert_greater,
                                   assert_equal, assert_false)



dairy_data = pd.read_csv('dairy.csv')


dairy_min_cols = dairy_data[['FARM', 'YEAR', 'MILK', 'FEED']]


dairy_milk_only = dairy_min_cols.drop('FEED', axis=1)

dairy_panel = dairy_milk_only.pivot(index='FARM', columns='YEAR', values='MILK')

#dairy_panel_milk_feed = dairy_min_cols.pivot(index='FARM', columns='YEAR')


dairy_panel_matrix = dairy_panel.as_matrix()
dairy_panel_matrix = dairy_panel_matrix.astype(np.float64)


def train_dairy_regression_predictor(train_data, train_labels, learning_rule='sgd', learning_rate=0.002, n_iter=20, units=4):
    print learning_rule
    mlp = Regressor(layers=[Layer('Rectifier', units=units),
                            Layer('Linear')],
                   learning_rule=learning_rule,
                   learning_rate=learning_rate,
                   n_iter=n_iter)
    mlp.fit(train_data, train_labels)
    print mlp.score(train_data, train_labels)
    return mlp



independed_vars_values = dairy_panel_matrix[0:,:4]
depended_vars_values = dairy_panel_matrix[0:, 4]


train_data, test_data, train_labels, test_labels = cross_validation.train_test_split(independed_vars_values, depended_vars_values, test_size=0.3)

train_data = StandardScaler().fit_transform(train_data)
test_data = StandardScaler().fit_transform(test_data)


#dairy_mlp = train_dairy_regression_predictor(train_data, train_labels, learning_rule='nesterov', learning_rate=0.0000002, n_iter=80, units=10)
dairy_mlp = train_dairy_regression_predictor(train_data, train_labels, learning_rule='sgd', learning_rate=0.00000002, n_iter=80, units=20)


pickle.dump(dairy_mlp, open('dairy_mlp_7_sgd.pkl', 'wb'))
dairy_mlp.score(test_data, test_labels)


real_data = dairy_panel_matrix[0:, 1:5]
real_labels = dairy_panel_matrix[0:, 5]

scaler = StandardScaler()
real_data = scaler.fit_transform(real_data)


dairy_mlp.score(real_data, real_labels)


for i, (train_indices, test_indices) in enumerate(cross_validation.KFold(depended_vars_values.size, n_folds=3, shuffle=True)):
    train_data = independed_vars_values[train_indices]
    test_data = independed_vars_values[test_indices]
    train_labels = depended_vars_values[train_indices]
    test_labels = depended_vars_values[test_indices]
    train_data = StandardScaler().fit_transform(train_data)
    test_data = StandardScaler().fit_transform(test_data)
    dairy_mlp = train_dairy_regression_predictor(train_data, train_labels, learning_rule='sgd', learning_rate=0.00000002, n_iter=80, units=20)
    pickle.dump(dairy_mlp, open('dairy_mlp_cv_%s_sgd.pkl' % str(i), 'wb'))
    print dairy_mlp.score(test_data, test_labels)

#print dairy_mlp.score(real_data, real_labels)

