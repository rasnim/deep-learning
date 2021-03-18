import tensorflow as tf
import math
from sklearn.metrics import *


def create_model(hidden_layers=4, hidden_nodes=128, activation='relu',
                 dropout_rate=0.1, optimizer='adam'):
    # Initialize the constructor
    model = tf.keras.models.Sequential()

    # Add an input layer
    # 입력 6개로부터 전달받는 hidden_nodes개 노드의 layer 생성
    model.add(
        tf.keras.layers.Dense(hidden_nodes, activation=activation, input_shape=(6,)))

    # Add hidden layers
    # hidden_nodes개 노드의 layer를 hidden_layers개 반복 생성
    for i in range(hidden_layers):
        model.add(tf.keras.layers.Dense(hidden_nodes, activation=activation))
        # dropout ratio=10% (훈련시 10% arc를 랜덤으로 무시)
        model.add(tf.keras.layers.Dropout(dropout_rate))

    # Add output layer
    # 회귀(regression)을 위해 linear 함수 사용 ('linear'가 default activation function이므로 회귀시에 생략해도 됨)
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    # model compile

    '''
    classification model compiler
    
    model.compile(
        loss='categorical_crossentropy',  #다중 교차엔트로피
        optimizer="rmsprop",   #최적화 기법 중 하나
        metrics=['accuracy'])  #정확도 측정
    '''

    # regression model compiler
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


# Model training method
def fit(grid_search, X_train, y_train):
    # best hyperparameter 조합을 찾기 위하여 CV로 Training 수행 (실제 실행되는 부분!!!)
    grid_search.fit(X_train, y_train,
                    #         batch_size=100,  #100개에 한 번씩 업데이터 실행
                    #         epochs=100,       #훈련 데이터셋을 총 10회 반복 실험. 단, 조기중지될 수 있음
                    # validation_split=0.2,
                    # callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
                    verbose=1)  # 전 과정을 화면에 출력(1) 또는 미출력(0) 모드
    return grid_search


def predict(grid_search, X_test, y_test):
    y_pred = grid_search.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"mse: {mse}")
    print(f"rmse: {math.sqrt(mse)}")
    print(f"mae: {mae}")
    return y_pred
