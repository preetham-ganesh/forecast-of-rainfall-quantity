# authors_name = 'Preetham Ganesh, Harsha Vardhini Vasu'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com, harshavardhini2019@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import os


def district_model_training_testing(file_names: list):
    return 0


def choose_model():
    print()
    print('Choose the model to be trained:')
    model_names = ['multiple_linear_regression', 'polynomial_regression', 'decision_tree_regression',
                   'support_vector_regression']
    for i in range(len(model_names)):
        print('{}. {}'.format(str(i), model_names[i]))
    print()
    chosen_model_name = input()
    return chosen_model_name


def main():
    file_names = os.listdir('../data/min_max_normalized_data')
    file_names.sort()
    chosen_model_name = choose_model()


if __name__ == '__main__':
    main()

