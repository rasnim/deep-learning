from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import os


def save_fig(fig_id, dir, fig_extension="png", resolution=300):
    images_path = os.path.join("images", dir)
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    images_path = os.path.join(images_path, fig_id + "." + fig_extension)
    print("Saving figure", images_path)
    plt.savefig(images_path, format=fig_extension, dpi=resolution)


class Visualize:
    def __init__(self, scaler_y, y_pred, y_test, y_index):
        self.scaler_y = scaler_y
        self.y_pred = y_pred
        self.y_test = y_test
        self.y_index = y_index

    def draw_graph(self, label="F1", min=0.0, max=1.0):
        # 예측 데이터 상세확인 및 그래프 가시화
        # Actual Values
        y_temp = np.zeros(shape=(len(self.y_test), 2))
        y_temp[:, self.y_index] = np.transpose(self.y_test)
        y_temp = self.scaler_y.inverse_transform(y_temp)
        y_actual = y_temp[:, self.y_index]

        # Predicted Values
        y_temp2 = np.zeros(shape=(len(self.y_test), 2))
        y_temp2[:, self.y_index] = self.y_pred
        y_temp2 = self.scaler_y.inverse_transform(y_temp2)
        y_predicted = y_temp2[:, self.y_index]

        plt.figure(figsize=(5, 5))
        plt.scatter(y_actual, y_predicted, label=label, s=3)
        plt.xlabel('y_actual')
        plt.ylabel('y_predicted')
        plt.legend()
        plt.xlim((min, max))
        plt.ylim((min, max))
        today = datetime.today().strftime("%Y%m%d")

        save_fig("prediction_dnn_" + label, today)
        plt.show()
