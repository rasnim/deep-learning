from preprocessing import *
from model import *
from visualization import *
from tuning import *

IN_PATH = "bmt1"
IN_FILE = "bmt1.csv"
OUT_PATH = "bmt1_dnn"
OUT_FILE = "bmt1_norm.csv"


def model_analysis(x_norm, y_norm, y_index, scaler_y, graph_label, graph_min, graph_max):
    # Data spliting
    bmt1_x_train, bmt1_x_test, bmt1_y_train, bmt1_y_test = \
        DatasetSplitting(x_norm, y_norm, y_index).train_test_split(0.3, 42)

    # 'F1' model analysis
    bmt1_optimizer = Optimization()
    bmt1_grid_search = bmt1_optimizer.grid_search_cv()
    bmt1_grid_search = fit(bmt1_grid_search, bmt1_x_train, bmt1_y_train)
    show_best_model(bmt1_grid_search)
    show_all_model_performance(bmt1_grid_search)

    bmt1_y_pred = predict(bmt1_grid_search, bmt1_x_test, bmt1_y_test)

    # Result visualization
    Visualize(scaler_y, bmt1_y_pred, bmt1_y_test, y_index).draw_graph(graph_label, graph_min, graph_max)


if __name__ == '__main__':
    # Data load
    bmt1_dataset = DataLoad().load_csv_data(IN_PATH, IN_FILE)

    # Data preprocessing
    bmt1_featuring = FeatureSetting(bmt1_dataset)

    # X, Y splitting
    bmt1_x = bmt1_featuring.set_x_cols(range(6))
    bmt1_y = bmt1_featuring.set_y_cols(range(6, 8))

    # Data normalization
    bmt1_normalizing = FeatureNormalizing(bmt1_x, bmt1_y)

    bmt1_x_norm = bmt1_normalizing.scaling_X()
    bmt1_y_norm, scaler_y = bmt1_normalizing.scaling_y()

    # 'F1' model analysis
    bmt1_y_index = bmt1_normalizing.set_y(0)
    model_analysis(bmt1_x_norm, bmt1_y_norm, bmt1_y_index, scaler_y, "F1", -7000, 2000)

    # 'F2' model analysis
    bmt1_y_index_2 = bmt1_normalizing.set_y(1)
    model_analysis(bmt1_x_norm, bmt1_y_norm, bmt1_y_index_2, scaler_y, "F2", -2000, 5000)