from DNN.model import Model
from sklearn.model_selection import ShuffleSplit, GridSearchCV

import tensorflow as tf
import numpy as np


class Optimization:
    def __init__(self):
        self.grid_search = None

    # hyper paramters tuning from grid searching
    def grid_search_cv(self):
        # network structure parameters
        hidden_layers = [2]  # [1, 2, 4]
        hidden_nodes = [128]  # [32, 64, 128, 256]

        # model parameters
        activation = ['relu', 'tanh', 'sigmoid']  # hard_sigmoid, linear, softmax, softplus, softsign, PReLU
        init = ['lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal']
        optimizer = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

        # model optimization
        dropout_rate = [0.0]  # , 0.1, 0.2, 0.5]
        learn_rate = [0.001, 0.01, 0.1, 0.3]
        momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

        # experiment design
        epochs = [100]  # add 50, 100, 150 etc
        batch_size = [100]  # , 500] # add 5, 10, 20, 40, 60, 80, 100 etc

        model = tf.keras.wrappers.scikit_learn.KerasRegressor(build_fn=Model.create_model, epochs=100, verbose=0)

        param_grid = dict(hidden_layers=hidden_layers, hidden_nodes=hidden_nodes, epochs=epochs, batch_size=batch_size,
                          # activation=activation, dropout_rate=dropout_rate, init=init, optimizer=optimizer,
                          # learn_rate=learn_rate, momentum=momentum
                          )
        cv = ShuffleSplit(n_splits=5)
        self.grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv,
                                        scoring='neg_mean_squared_error', return_train_score=True)
        return self.grid_search


def show_best_model(grid_search):
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)


def show_all_model_performance(grid_search):
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
