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
                                      parameter: str):
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
                           parameter: str):
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
            A tuple containing metrics for the training and testing dataset computed using the currently trained model
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
        model = DecisionTreeRegressor(max_depth=int(parameter))
    else:
        model = SVR(kernel=int(parameter))

    # Train the object created for the model using the training input and target
    model.fit(train_district_data_input, train_district_data_target)

    # Using the trained model, predict the rainfall values for the training and testing inputs
    train_district_data_predict = model.predict(train_district_data_input)
    test_district_data_predict = model.predict(test_district_data_input)

    # Calculate the metrics for the predicted rainfall values for the training and testing inputs
    train_metrics = calculate_metrics(train_district_data_target, train_district_data_predict)
    test_metrics = calculate_metrics(test_district_data_target, test_district_data_predict)
    return train_metrics, test_metrics


def calculate_metrics_mean_repeated_kfold(parameters_metrics: pd.DataFrame,
                                          repeated_kfold_metrics: pd.DataFrame,
                                          current_model_name: str,
                                          parameter: str,
                                          metrics_features: list):
    """Calculate the mean of the metrics computed in every iteration of Repeated K-fold Cross Validation

        Args:
            parameters_metrics: A dataframe containing the mean of all the metrics for all the hyperparameters
            repeated_kfold_metrics: A dataframe containing the metrics for all the iterations in the Repeated K-fold
                                    Cross Validation
            current_model_name: Name of the model currently trained
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
    repeated_kfold_metrics_mean['model_names'] = current_model_name

    # Append the current hyperparameter's mean of metrics to the parameters_metrics
    parameters_metrics = parameters_metrics.append(repeated_kfold_metrics_mean, ignore_index=True)
    return parameters_metrics


def retrieve_hyperparameters(current_model_name: str):
    """Based on the current_model_name return a list of hyperparameters used for optimizing the model (if necessary).

        Args:
            current_model_name: Name of the model currently expected to be trained

        Returns:
            None
    """
    # For polynomial_regression, the hyperparameter tuned is degrees.
    if current_model_name == 'polynomial_regression':
        parameters = [2, 3, 4, 5]

    # For decision_tree_regression, the hyperparameter tuned is max_depth
    elif current_model_name == 'decision_tree_regression':
        parameters = [2, 3, 4, 5, 6, 7]

    # For support_vector_regression, the hyperparameter tuned is kernel
    elif current_model_name == 'support_vector_regression':
        parameters = ['linear', 'poly', 'rbf', 'sigmoid']

    # For multiple_linear_regression, none of the hyperparameters are tuned.
    else:
        parameters = ['None']
    return parameters


def split_data_input_target(district_data: pd.DataFrame):
    """Splits district_data into input and target datasets by filtering / selecting certain columns

        Args:
            district_data: Training / Testing dataset used to split / filter certain columns

        Returns:
            A tuple containing 2 numpy ndarrays containing the input and target datasets
    """
    district_data_input = district_data.drop(columns=['district', 'rainfall'])
    district_data_target = district_data['rainfall']
    return np.array(district_data_input), np.array(district_data_target)


def district_results_export(district_name: str,
                            data_split: str,
                            metrics_dataframe: pd.DataFrame):
    """Exports the metrics_dataframe into a CSV format to the mentioned data_split folder. If the folder does not
    exist, then the folder is created.

        Args:
            district_name: Name of district in Tamil Nadu, India among the available 29 districts
            data_split: Location where the metrics has to be exported
            metrics_dataframe: A dataframe containing the mean of all the metrics for all the hyperparameters and models

        Returns:
            None
    """
    directory_path = '{}/{}'.format('../results/metrics', data_split)
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    file_path = '{}/{}.csv'.format(directory_path, district_name)
    metrics_dataframe.to_csv(file_path, index=False)


def per_district_model_training_testing(district_name: str,
                                        model_names: list):
    """Performs regression model training and testing using Repeated K-fold Cross Validation, calculation of metrics for
    every iteration and computation of mean of all the metrics for all the hyperparameters for every regression model
    (if necessary).

        Args:
            district_name: Name of district in Tamil Nadu, India among the available 29 districts
            model_names: List containing the names of the models used for developing the regression models

        Returns:
            None
    """
    district_data = pd.read_csv('{}/{}'.format('../data/min_max_normalized_data', district_name))

    # Created an object for the Repeated K-fold Cross Validation where the number of repeats and splits as 10
    repeated_kfold = RepeatedKFold(n_repeats=10, n_splits=10)

    # Shuffle the imported data before splitting it using Repeated K-fold Cross Validation
    district_data = shuffle(district_data)

    # Creating empty dataframes for the mean values of metrics for the current district's training and testing datasets
    metrics_features = ['mse_score', 'rmse_score', 'mae_score', 'mdae_score', 'evs_score', 'r2_score']
    train_models_parameters_metrics = pd.DataFrame(columns=['model_names', 'parameters'] + metrics_features)
    test_models_parameters_metrics = pd.DataFrame(columns=['model_names', 'parameters'] + metrics_features)

    # Iterates across model_names for training and testing the regression models
    for i in range(len(model_names)):
        parameters = retrieve_hyperparameters(model_names[i])

        # Iterates across the parameters for optimizing the training of the regression models
        for j in range(len(parameters)):
            train_repeated_kfold_metrics = pd.DataFrame(columns=metrics_features)
            test_repeated_kfold_metrics = pd.DataFrame(columns=metrics_features)

            # Iterates across the Repeated K-fold Cross Validation's data splits and repeats
            for train_index, test_index in repeated_kfold.split(district_data):

                # Based on the split index values, the training and testing datasets are created
                train_district_data = district_data.iloc[train_index]
                test_district_data = district_data.iloc[test_index]

                # Splits district_data into input and target datasets
                train_district_data_input, train_district_data_target = split_data_input_target(train_district_data)
                test_district_data_input, test_district_data_target = split_data_input_target(test_district_data)

                # Computes training and testing metrics for the current iteration
                train_metrics, test_metrics = model_training_testing(train_district_data_input,
                                                                     train_district_data_target,
                                                                     test_district_data_input,
                                                                     test_district_data_target, model_names[i],
                                                                     parameters[j])

                # Append training and testing metrics to Repeated K-fold dataframe
                train_repeated_kfold_metrics = train_repeated_kfold_metrics.append(train_metrics)
                test_repeated_kfold_metrics = test_repeated_kfold_metrics.append(test_metrics)

            # Computes training and testing mean values of metrics for current regression model's hyperparameter
            train_models_parameters_metrics = calculate_metrics_mean_repeated_kfold(train_models_parameters_metrics,
                                                                                    train_repeated_kfold_metrics,
                                                                                    model_names[i], parameters[j],
                                                                                    metrics_features)
            test_models_parameters_metrics = calculate_metrics_mean_repeated_kfold(test_models_parameters_metrics,
                                                                                   test_repeated_kfold_metrics,
                                                                                   model_names[i], parameters[j],
                                                                                   metrics_features)

    # Exports the training and testing metrics into CSV files
    district_results_export(district_name, 'training_metrics', train_models_parameters_metrics)
    district_results_export(district_name, 'testing_metrics', test_models_parameters_metrics)


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

