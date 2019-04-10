import os
import math
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from datetime import datetime

# 데이터 변환
class DnnAnalyzer:
    # def __init__(self):

    def load_csv_data(self, in_path, in_file):
        csv_path = os.path.join(in_path, in_file)
        self.data = pd.read_csv(csv_path)

    def set_X_cols(self, col_indexes):
        self.X0 = self.data.iloc[:, col_indexes]

    def set_Y_cols(self, col_indexes):
        self.y0 = self.data.iloc[:, col_indexes]

    def getPipeline(self):
        return Pipeline([
            #         ('std_scaler', StandardScaler()),  #z-normalization
            ('minmax_scaler', MinMaxScaler()),  # min-max normalization
        ])

    def norm_X(self):
        self.scaler_X = self.getPipeline()
        X_norm = self.scaler_X.fit_transform(self.X0)
        self.X = pd.DataFrame(X_norm)

    def norm_y(self):
        self.scaler_y = self.getPipeline()
        self.y = self.scaler_y.fit_transform(self.y0)

    def set_y(self, y_index):
        self.y_index = y_index

    def train_test_split(self, test_size=0.3, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y[:,self.y_index], test_size=0.3, random_state=42)

    def create_model(self, hidden_layers=1, hidden_nodes=128,
                     activation='relu', dropout_rate=0.0,
                     init='uniform', optimizer='adam'):
        # Initialize the constructor
        model = Sequential()
        # Add an input layer
        model.add(Dense(5, activation=activation, input_shape=(5,)))

        # 모델 구조 정의하기
        model = Sequential()  # 순차적 계층화 준비
        model.add(
            Dense(hidden_nodes, activation=activation, input_shape=(6,)))  # 입력 6개로부터 전달받는 hidden_nodes개 노드의 layer 생성
        model.add(Dropout(dropout_rate))  # dropout ratio=10% (훈련시 10% arc를 랜덤으로 무시)

        for i in range(hidden_layers):  # hidden_nodes개 노드의 layer를 hidden_layers개 반복 생성
            model.add(Dense(hidden_nodes, activation=activation))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1))
        model.add(Activation(
            'linear'))  # 회귀(regression)을 위해 linear 함수 사용 ('linear'가 default activation function이므로 회귀시에 생략해도 됨)

        # 모델 구축하기
        # classification 모델용
        # model.compile(
        #     loss='categorical_crossentropy',  #다중 교차엔트로피
        #     optimizer="rmsprop",   #최적화 기법 중 하나
        #     metrics=['accuracy'])  #정확도 측정
        # regression 모델용
        model.compile(loss='mse',
                      optimizer=optimizer,  # 최적화 기법 중 하나 선택
                      metrics=['mae'])
        return model

    def gridSearchCV(self):
        # hyperparamters 및 grid search 설계
        # network structure
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

        model = KerasRegressor(build_fn=self.create_model, epochs=100, verbose=0)
        param_grid = dict(
                          hidden_layers=hidden_layers, hidden_nodes=hidden_nodes,
                          epochs=epochs, batch_size=batch_size,
                          #                  activation=activation, dropout_rate=dropout_rate,
                          #                  init=init, optimizer=optimizer
                          #                  learn_rate=learn_rate, momentum=momentum,
                          )
        cv = ShuffleSplit(n_splits=5)

        self.grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv,
                                        scoring='neg_mean_squared_error', return_train_score=True)

    def fit(self):
        # best hyperparameter 조합을 찾기 위하여 CV로 Training 수행 (실제 실행되는 부분!!!)
        self.grid_search.fit(self.X_train, self.y_train,
                        #         batch_size=100,  #100개에 한 번씩 업데이터 실행
                        #         epochs=100,       #훈련 데이터셋을 총 10회 반복 실험. 단, 조기중지될 수 있음
                        # validation_split=0.2,
                        # callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                        verbose=1)  # 전 과정을 화면에 출력(1) 또는 미출력(0) 모드

    def show_best_model(self):
        print(self.grid_search.best_params_)
        print(self.grid_search.best_estimator_)

    def show_all_model_performance(self):
        cvres = self.grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

    def predict(self):
        self.y_pred = self.grid_search.predict(self.X_test)

        mse = metrics.mean_squared_error(self.y_test, self.y_pred);
        print("mse: ", mse);
        print("rmse: ", math.sqrt(mse))
        mae = metrics.mean_absolute_error(self.y_test, self.y_pred);
        print("mae: ", mae)

    def draw_graph(self, label="F1", min=0.0, max=1.0):
        # 예측 데이터 상세확인 및 그래프 가시화
        y_temp = np.zeros(shape=(len(self.y_test), 2))
        y_temp[:, self.y_index] = np.transpose(self.y_test)
        y_temp = self.scaler_y.inverse_transform(y_temp)
        y_actual = y_temp[:, self.y_index]

        y_temp2 = np.zeros(shape=(len(self.y_test), 2))
        y_temp2[:, self.y_index] = self.y_pred
        y_temp2 = self.scaler_y.inverse_transform(y_temp2)
        y_predicted = y_temp2[:, self.y_index]
        print("y_predicted : ", y_predicted)

        plt.figure(figsize=(5, 5))
        plt.scatter(y_actual, y_predicted, label=label, s=3)
        # plt.title('Title')
        plt.xlabel('y_actual')
        plt.ylabel('y_predicted')
        plt.legend()
        plt.xlim((min, max))
        plt.ylim((min, max))
        today = datetime.today().strftime("%Y%m%d")
        self.save_fig("prediction_dnn_"+label, today)
        plt.show()

    def save_fig(self, fig_id, dir, tight_layout=True, fig_extension="png", resolution=300):
        images_path = os.path.join("images", dir)
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        images_path = os.path.join(images_path, fig_id + "." + fig_extension)
        print("Saving figure", images_path)
        #if tight_layout:
        #   plt.tight_layout()
        plt.savefig(images_path, format=fig_extension, dpi=resolution)


IN_PATH = "bmt1"
IN_FILE = "bmt1.csv"
OUT_PATH = "bmt1_dnn"
OUT_FILE = "bmt1_norm.csv"

bmt1 = DnnAnalyzer()
bmt1.load_csv_data(IN_PATH, IN_FILE)
bmt1.set_X_cols(range(6))
bmt1.set_Y_cols(range(6,8))
bmt1.norm_X()
bmt1.norm_y()

#'F1' 모델 분석
bmt1.set_y(0)
bmt1.train_test_split()
bmt1.gridSearchCV()
bmt1.fit()
bmt1.show_best_model()
bmt1.show_all_model_performance()
bmt1.predict()
bmt1.draw_graph("F1", -7000, 2000)

#'F2' 모델 분석
bmt1.set_y(1)
bmt1.train_test_split()
bmt1.gridSearchCV()
bmt1.fit()
bmt1.show_best_model()
bmt1.show_all_model_performance()
bmt1.predict()
bmt1.draw_graph("F2", -2000, 5000)
