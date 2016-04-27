
# coding: utf-8

import sys
import pickle

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from numpy.testing import assert_almost_equal, assert_array_equal

from sklearn import cross_validation
from sknn.mlp import Regressor, Layer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.grid_search import  GridSearchCV
from sklearn.utils.testing import (assert_raises, assert_greater,
                                   assert_equal, assert_false)


np.seterr(all='warn')
scaler = StandardScaler()


payment_data = pd.read_csv('user_payments.csv', index_col=0)
payment_data.head(4)


payment_data_matrix = payment_data.as_matrix()
payment_data_matrix = payment_data_matrix.astype(np.float64)


def train_regression_predictor(train_x, train_y, learning_rule='sgd', learning_rate=0.002, n_iter=20, units=4):
    mlp = Regressor(layers=[Layer('Rectifier', units=units),
                            Layer('Linear')],
                   learning_rule=learning_rule,
                   learning_rate=learning_rate,
                   n_iter=n_iter)
    mlp.fit(train_x, train_y)
    print mlp.score(train_x, train_y)
    return mlp

independed_vars_values = payment_data_matrix[0:,:4]
depended_var_values = payment_data_matrix[0:, 4]

train_x, test_x, train_y, test_y = cross_validation.train_test_split(independed_vars_values, depended_var_values, test_size=0.3)

train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

payment_mlp = train_regression_predictor(train_x, train_y, learning_rule='sgd', learning_rate=0.0002, n_iter=20, units=4)

score = payment_mlp.score(test_x, test_y)

real_x = payment_data_matrix[0:, 1:5]
real_x = scaler.fit_transform(real_x)
real_y = payment_data_matrix[0:, 5]


print 'Score on real data', payment_mlp.score(real_x, real_y)

best_mlp = payment_mlp

for i, (train_indices, test_indices) in enumerate(cross_validation.KFold(depended_var_values.size, n_folds=3, shuffle=True)):
    train_x = independed_vars_values[train_indices]
    test_x = independed_vars_values[test_indices]
    train_y = depended_var_values[train_indices]
    test_y = depended_var_values[test_indices]
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)
    payment_mlp = train_regression_predictor(train_x, train_y, learning_rule='sgd', learning_rate=0.0002, n_iter=20, units=4)
    pickle.dump(payment_mlp, open('payment_mlp_cv_0%s_sgd.pkl' % str(i), 'wb'))
    crossvalidation_test_score =  payment_mlp.score(test_x, test_y)

    if crossvalidation_test_score > score:
        score = crossvalidation_test_score
        best_mlp = payment_mlp


print 'real data score: ', best_mlp.score(real_x, real_y)
pickle.dump(best_mlp, open('best_mlp.pkl', 'wb'))

predicted_y = best_mlp.predict(real_x)
predicted_y = predicted_y.reshape(predicted_y.shape[0],)


predicted_and_true_values = DataFrame({'predicted' : list(predicted_y.round(decimals=2)), 'real' : list(real_y)})
print predicted_and_true_values.head(20)
predicted_and_true_values.to_csv('predicted_and_true_values.csv')

