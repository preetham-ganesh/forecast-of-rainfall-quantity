# author_name = 'Preetham Ganesh'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2021@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
import os
import numpy as np


def district_feature_data_import(district_name:str,
                                 feature_name: str,
                                 data_version: str):
    """Imports feature data for a particular district based on input variables.

        Args:
            district_name: Name of district in Tamil Nadu, India
            feature_name: Name of feature among the list of 9 features
            data_version: Version of the data to be imported

        Returns:
            Pandas Dataframe contains feature's data for 102 years from 1900 - 2002.
    """
    if '.csv' not in feature_name:
        feature_name += '.csv'
    feature_location = '{}/{}/{}/{}'.format('../data', data_version, district_name, feature_name, 'csv')
    district_feature_data = pd.read_csv(feature_location, sep='\t')
    return district_feature_data


def column_name_preprocessing(column_names: list):
    """Cleans the list of features/districts by lowering the case, replacing spaces with underscores, and removing
    .csv extensions.

        Args:
            column_names: List of features/districts

        Returns:
            List of processed features/districts
    """
    new_column_names = []
    for i in range(len(column_names)):
        column_name = column_names[i].lower()
        column_name = column_name.replace('.csv', '')
        column_name = '_'.join(column_name.split(' '))
        new_column_names.append(column_name)
    return new_column_names


def feature_transformation(district_name: str,
                           feature_name: str):
    """Converts the 2D dataframe of a district's feature into a sequential list

        Args:
            district_name: Name of district in Tamil Nadu, India
            feature_name: Name of feature among the list of 9 features

        Returns:
            Sequential of list of the data for a district's feature
    """
    district_feature_data = district_feature_data_import(district_name, feature_name, 'original_data')


def data_preprocessing():
    district_names = os.listdir('../data/original_data')
    district_names.sort()
    features = os.listdir('{}/{}/'.format('../data/original_data', district_names[0]))
    district_data = district_feature_data_import(district_names[0], features[0], 'original_data')
    print(district_data.head())
    district_names_processed = column_name_preprocessing(district_names)
    features_processed = column_name_preprocessing(features)


def main():
    data_preprocessing()


if __name__ == '__main__':
    main()