# author_name = 'Preetham Ganesh'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2021@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
import os


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


def min_max_normalization(feature_values: list):
    """Performs min-max normalization on the given list of values

        Args:
            feature_values: Sequential list of floating point values for the district's feature data

        Returns:
            Sequential list of min-max normalized values for the district's feature data
    """
    min_feature_value = min(feature_values)
    max_feature_value = max(feature_values)
    return [(feature_values[i] - min_feature_value) / (max_feature_value - min_feature_value) for i in
            range(len(feature_values))]


def district_data_export(district_name: str,
                         data_version: str,
                         data: pd.DataFrame):
    """Exports the dataframe into a CSV format to the mentioned data_version folder. If the folder does not exist, then
    the folder is created.

        Args:
            district_name: Name of district in Tamil Nadu, India among the available 29 districts
            data_version: Version of the data to be imported
            data: Combined data for a district

        Returns:
            None
    """
    directory_path = '{}/{}'.format('../data', data_version)
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    file_path = '{}/{}.csv'.format(directory_path, district_name)
    data.to_csv(file_path, index=False)


def data_transformation(district_name: str,
                        features: list):
    """Perform feature transformation for every individual feature for a district and exporting the transformed
    dataframe.

        Args:
            district_name: Name of district in Tamil Nadu, India among the available 29 districts
            features: List of features for available for every district in Tamil Nadu, India

        Returns:
            None
    """
    combined_data, combined_min_max_data = dict(), dict()
    processed_district_name = column_name_processing(district_name)
    processed_features = []
    for i in range(len(features)):
        processed_feature = column_name_processing(features[i])
        transformed_feature_data = feature_transformation(district_name, features[i])
        min_max_feature_data = min_max_normalization(transformed_feature_data)
        combined_data[processed_feature] = transformed_feature_data
        combined_min_max_data[processed_feature] = min_max_feature_data
        processed_features.append(processed_feature)
    combined_dataframe = pd.DataFrame(combined_data, columns=processed_features)
    combined_min_max_dataframe = pd.DataFrame(combined_min_max_data, columns=processed_features)
    district_data_export(processed_district_name, 'combined_data', combined_dataframe)
    district_data_export(processed_district_name, 'min_max_normalized_data', combined_min_max_dataframe)


def data_preprocessing():
    """Performs data preprocessing on all the districts data in Tamil Nadu, India.

        Args:
            None

        Returns:
            None
    """
    district_names = os.listdir('../data/original_data')
    district_names.sort()
    features = os.listdir('{}/{}/'.format('../data/original_data', district_names[0]))
    for i in range(len(district_names)):
        data_transformation(district_names[i], features)


def main():
    data_preprocessing()


if __name__ == '__main__':
    main()

