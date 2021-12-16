# authors_name = 'Preetham Ganesh, Harsha Vardhini Vasu'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com, harshavardhini2019@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.metrics import median_absolute_error as mdae_score
from sklearn.metrics import explained_variance_score as evs_score
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
import os
import numpy as np
from sklearn.utils import shuffle


def polynomial_feature_transformation(train_district_data_input: pd.DataFrame,
                                      test_district_data_input: pd.DataFrame,
                                      parameter: int):
    return 0, 0


def rmse_score(actual_values: np.ndarray,
               predicted_values: np.ndarray):
    """Calculates Root Mean Squared Error based on the actual_values and predicted_values for the currently trained
    model.

        Args:
            actual_values: Actual rainfall values in the dataset
            predicted_values: Rainfall values predicted by the currently trained model

        Returns:
            Floating point value containing RMSE value calculated using the given input
    """
    # rmse_score = mse_score ** 0.5
    return (mse_score(actual_values, predicted_values)) ** 0.5


def calculate_metrics(actual_values: np.ndarray,
                      predicted_values: np.ndarray):
    """Using actual_values, predicted_values calculates metrics such as MSE, RMSE, MAE, MDAE, EVS, and R2 scores.

        Args:
            actual_values: Actual rainfall values in the dataset
            predicted_values: Rainfall values predicted by the currently trained model

        Returns:
            Dictionary contains keys as score names and values as scores which are floating point values.
    """
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
                           current_model_name: str,
                           parameter: int):
    """Creates an object for the model using the input and performs training and testing of the models using the given
    training and testing datasets.

        Args:
            train_district_data_input: Input training data for the district
            train_district_data_target: Target training data for the district
            test_district_data_input: Input testing data for the district
            test_district_data_target: Target testing data for the district
            current_model_name: Name of the model currently expected to be trained
            parameter: Hyperparameter value for optimizing the regression model

        Returns:
            List containing metrics for the training and testing dataset computed using the currently trained model
    """
    # Based on the current_model_name the scikit-learn object is initialized using the hyperparameter (if necessary)
    if current_model_name == 'polynomial_regression' or current_model_name == 'linear_regression':
        model = LinearRegression()

        # if current_model_name is polynomial_regression then the training_input and testing_input should be transformed
        # based on the hyperparameter which is the number of degrees
        if current_model_name == 'polynomial_regression':
            train_district_data_input, test_district_data_input = polynomial_feature_transformation(
                train_district_data_input, test_district_data_input, parameter)

    elif current_model_name == 'decision_tree_regression':
        model = DecisionTreeRegressor(max_depth=parameter)
    else:
        model = SVR(kernel=parameter)

    # Train the object created for the model using the training input and target
    model.fit(train_district_data_input, train_district_data_target)

    # Using the trained model, predict the rainfall values for the training and testing inputs
    train_district_data_predict = model.predict(train_district_data_input)
    test_district_data_predict = model.predict(test_district_data_input)

    # Calculate the metrics for the predicted rainfall values for the training and testing inputs
    train_metrics = calculate_metrics(train_district_data_target, train_district_data_predict)
    test_metrics = calculate_metrics(test_district_data_target, test_district_data_predict)
    return [train_metrics, test_metrics]


def calculate_metrics_mean_repeated_kfold(parameters_metrics: pd.DataFrame,
                                          repeated_kfold_metrics: pd.DataFrame,
                                          parameter: int,
                                          metrics_features: list):
    """Calculate the mean of the metrics computed in every iteration of Repeated K-fold Cross Validation

        Args:
            parameters_metrics: A dataframe containing the mean of all the metrics for all the hyperparameters
            repeated_kfold_metrics: A dataframe containing the metrics for all the iterations in the Repeated K-fold
                                    Cross Validation
            parameter: Current hyperparameter used for optimizing the regression model
            metrics_features: List containing the acronyms of the metrics used for evaluating the trained models

        Returns:
            An updated dataframe containing the mean of all the metrics for all the hyperparameters including the
            current hyperparameter
    """
    # Calculates mean of all the metrics computed in every iteration of Repeated K-fold Cross Validation
    repeated_kfold_metrics_mean = {metrics_features[i]: np.mean(repeated_kfold_metrics[metrics_features[i]]) for i in
                                   range(len(metrics_features))}
    repeated_kfold_metrics_mean['parameters'] = parameter

    # Append the current hyperparameter's mean of metrics to the parameters_metrics
    parameters_metrics = parameters_metrics.append(repeated_kfold_metrics_mean, ignore_index=True)
    return parameters_metrics


def per_district_model_training_testing(district_name: str,
                                        parameters: list,
                                        chosen_model_name: str):
    district_data = pd.read_csv('{}/{}'.format('../data/min_max_normalized_data', district_name))
    repeated_kfold = RepeatedKFold(n_repeats=10, n_splits=10)
    district_data = shuffle(district_data)
    metrics_features = ['mse_score', 'rmse_score', 'mae_score', 'mdae_score', 'evs_score', 'r2_score']
    train_parameters_metrics = pd.DataFrame(columns=['parameters'] + metrics_features)
    test_parameters_metrics = pd.DataFrame(columns=['parameters'] + metrics_features)
    for i in range(len(parameters)):
        train_repeated_kfold_metrics = pd.DataFrame(columns=metrics_features)
        test_repeated_kfold_metrics = pd.DataFrame(columns=metrics_features)
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
            train_metrics, test_metrics = model_training_testing(train_district_data_input, train_district_data_target,
                                                                 test_district_data_input, test_district_data_target,
                                                                 chosen_model_name, parameters[i])
            train_repeated_kfold_metrics = train_repeated_kfold_metrics.append(train_metrics, ignore_index=True)
            test_repeated_kfold_metrics = test_repeated_kfold_metrics.append(test_metrics, ignore_index=True)
        train_parameters_metrics = calculate_metrics_mean_repeated_kfold(train_parameters_metrics,
                                                                         train_repeated_kfold_metrics, parameters[i])
        test_parameters_metrics = calculate_metrics_mean_repeated_kfold(test_parameters_metrics,
                                                                        test_repeated_kfold_metrics, parameters[i])
    print(train_parameters_metrics.head())


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

