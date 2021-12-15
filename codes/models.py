# authors_name = 'Preetham Ganesh, Harsha Vardhini Vasu'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com, harshavardhini2019@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.metrics import median_absolute_error as mdae_score
from sklearn.metrics import explained_variance_score as evs_score
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
import os
import numpy as np


def polynomial_feature_transformation(train_district_data_input: pd.DataFrame,
                                      test_district_data_input: pd.DataFrame):
    return 0, 0


def rmse_score(actual_values: np.ndarray,
               predicted_values: np.ndarray):
    return (mse_score(actual_values, predicted_values)) ** 0.5


def calculate_metrics(actual_values: np.ndarray,
                      predicted_values: np.ndarray):
    return {'mse_score': mse_score(actual_values, predicted_values),
            'rmse_score': rmse_score(actual_values, predicted_values),
            'mae_score': mae_score(actual_values, predicted_values),
            'mdae_score': mdae_score(actual_values, predicted_values),
            'evs_score': evs_score(actual_values, predicted_values),
            'r2_score': r2_score(actual_values, predicted_values)}


def model_training_testing(train_district_data_input: np.ndarray,
                           train_district_data_target: np.ndarray,
                           test_district_data_input: np.ndarray,
                           test_district_data_target: np.ndarray,
                           chosen_model_name: str,
                           parameter: int):
    if chosen_model_name == 'polynomial_regression':
        model = 0
    elif chosen_model_name == 'decision_tree_regression':
        model = DecisionTreeRegressor(max_depth=parameter)
    else:
        model = 0
    model.fit(train_district_data_input, train_district_data_target)
    train_district_data_predict = model.predict(train_district_data_input)
    test_district_data_predict = model.predict(test_district_data_input)


def per_district_model_training_testing(district_name: str,
                                        parameters: list,
                                        chosen_model_name: str):
    district_data = pd.read_csv('{}/{}'.format('../data/min_max_normalized_data', district_name))
    repeated_kfold = RepeatedKFold(n_repeats=10, n_splits=10)
    for train_index, test_index in repeated_kfold.split(district_data):
        train_district_data = district_data.iloc[train_index]
        test_district_data = district_data.iloc[test_index]
        train_district_data_input = np.array(train_district_data.drop(columns=['district', 'rainfall']))
        train_district_data_target = np.array(train_district_data['rainfall'])
        test_district_data_input = np.array(test_district_data.drop(columns=['district', 'rainfall']))
        test_district_data_target = np.array(test_district_data['rainfall'])
        if chosen_model_name == 'polynomial_regression':
            train_district_data_input, test_district_data_input = polynomial_feature_transformation(
                train_district_data_input, test_district_data_input)
        print(train_district_data_input.shape)
        print(train_district_data_target.shape)
        print(type(train_district_data_input))
        break


def retrieve_hyperparameters(chosen_model_name: str):
    if chosen_model_name == 'polynomial_regression':
        parameters = [2, 3, 4, 5]
    elif chosen_model_name == 'decision_tree_regression':
        parameters = [2, 3, 4, 5, 6, 7]
    else:
        parameters = ['linear', 'poly', 'rbf', 'sigmoid']
    return parameters


def district_model_training_testing(district_names: list,
                                    chosen_model_name: str):
    parameters = retrieve_hyperparameters(chosen_model_name)
    per_district_model_training_testing(district_names[0], parameters, chosen_model_name)


def choose_model():
    print()
    print('Choose the model to be trained:')
    model_names = ['polynomial_regression', 'decision_tree_regression', 'support_vector_regression']
    for i in range(len(model_names)):
        print('{}. {}'.format(str(i), model_names[i]))
    print()
    chosen_model_number = input()
    return model_names[int(chosen_model_number)]


def main():
    district_names = os.listdir('../data/min_max_normalized_data')
    district_names.sort()
    chosen_model_name = choose_model()
    district_model_training_testing(district_names, chosen_model_name)


if __name__ == '__main__':
    main()

