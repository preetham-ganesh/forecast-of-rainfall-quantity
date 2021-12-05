# author_name = 'Preetham Ganesh'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2021@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
import os


def feature_data_import(district_name:str,
                        data_version: str,
                        feature_name: str = None):
    """Imports data for a particular district based on input variables.

        Args:
            district_name: Name of district in Tamil Nadu, India among the available 29 districts
            data_version: Version of the data to be imported
            feature_name: Name of feature among the list of 9 features

        Returns:
            Pandas dataframe for the particular district
    """
    if '.csv' not in feature_name:
        feature_name += '.csv'
    feature_location = '{}/{}/{}/{}'.format('../data', data_version, district_name, feature_name)
    district_feature_data = pd.read_csv(feature_location, sep='\t')
    return district_feature_data


def column_name_processing(column_name: str):
    """Cleans the input string by lowering the case, replacing spaces with underscores, and removing extensions.

        Args:
            column_name: String to be cleaned

        Returns:
            Processed string
    """
    new_column_name = column_name.lower()
    new_column_name = new_column_name.replace('.csv', '')
    new_column_name = '_'.join(new_column_name.split(' '))
    return new_column_name


def feature_transformation(district_name: str,
                           feature_name: str):
    """Imports the district's feature datafrome and transforms it into a sequential list

        Args:
            district_name: Name of district in Tamil Nadu, India among the available 29 districts
            feature_name: Name of feature among the list of 9 features

        Returns:
            Sequential of list of the data for a district's feature
    """
    feature_data_location = '{}/{}/{}'.format('../data/original_data', district_name, feature_name)
    original_feature_data = pd.read_csv(feature_data_location, sep='\t')
    transformed_feature_data = list()
    for i in range(len(original_feature_data)):
        transformed_feature_data.extend(list(original_feature_data.iloc[i][1:]))
    transformed_feature_data = [round(transformed_feature_data[i], 3) for i in range(len(transformed_feature_data))]
    return transformed_feature_data


def data_transformation(district_name: str,
                        features: list):
    """

        Args:
            district_name: Name of a district in Tamil Nadu, India
            features: List containing
    """
    combined_data = dict()
    feature_transformation(district_name, features[0])
    #for i in range(len())


def data_preprocessing():
    district_names = os.listdir('../data/original_data')
    district_names.sort()
    features = os.listdir('{}/{}/'.format('../data/original_data', district_names[0]))
    data_transformation(district_names[0], features)


def main():
    data_preprocessing()


if __name__ == '__main__':
    main()