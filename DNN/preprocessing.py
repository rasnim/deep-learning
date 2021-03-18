import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *


# Load .csv File
class DataLoad:
    def __init__(self):
        self.dataset = None

    def load_csv_data(self, in_path, in_file):
        csv_path = os.path.join(in_path, in_file)
        self.dataset = pd.read_csv(csv_path)


class FeatureSetting:
    def __init__(self, dataset):
        self.dataset = dataset
        self.X = None
        self.y = None

    def set_x_cols(self, col_indexes):
        self.X = self.dataset.iloc[:, col_indexes]

    def set_y_cols(self, col_indexes):
        self.y = self.dataset.iloc[:, col_indexes]


class FeatureNormalizing:
    def __init__(self, x, y):
        self.X = x
        self.y = y
        self.x_norm = None
        self.y_norm = None
        self.y_index = None

    # Pipeline : Sequentially apply a list of transforms (scaler) and a final estimator(ML/DL model).
    # The purpose of the pipeline is to assemble several steps that can be cross-validated together
    # while setting different parameters.
    def getPipeline(self):
        # z-normalization
        standard_scaler = StandardScaler()

        # min-max normalization
        minmax_scaler = MinMaxScaler()
        return Pipeline([('minmax_scaler', minmax_scaler)])

    def scaling_X(self):
        scaler_X = self.getPipeline()
        self.x_norm = scaler_X.fit_transform(self.X)
        return self.x_norm

    def scaling_y(self):
        scaler_y = self.getPipeline()
        self.y_norm = scaler_y.fit_transform(self.y)
        return self.y_norm, scaler_y

    def set_y(self, y_index):
        self.y_index = y_index
        return self.y_index


class DatasetSplitting:
    def __init__(self, x, y, y_index):
        self.X = x
        self.y = y
        self.y_index = y_index
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train_test_split(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y[:, self.y_index], test_size=test_size, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test