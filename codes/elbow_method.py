# author_name = 'Preetham Ganesh'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
import os
from data_preprocessing import min_max_normalization
from data_preprocessing import district_data_export
from data_preprocessing import column_name_processing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def data_combine(district_names: list):
    """Uses district_names to read the datasets and combine them into a single dataset.

        Args:
            district_names: List containing names of 29 districts in Tamil Nadu, India

        Returns:
            combined_dataframe: Pandas dataframe containing the combined data from all the 29 districts in Tamil Nadu,
                                India
            processed_district_names: Processed list of district_names
    """
    directory_path = '../data/combined_data'

    # In case if the code was already executed, the combined data will be stored in a CSV file called 'all_districts',
    # since that is not an actual district in Tamil Nadu, India, the file_name is removed from the list
    if 'all_districts.csv' in district_names:
        district_names.remove('all_districts.csv')

    # Reading the first district's data into the combined_dataframe, adding a new column to the dataframe 'district',
    # which contains the current district's processed name
    combined_dataframe = pd.read_csv('{}/{}'.format(directory_path, district_names[0]))
    current_district_name = column_name_processing(district_names[0])
    combined_dataframe['district'] = list([current_district_name for i in range(len(combined_dataframe))])
    processed_district_names = [current_district_name]

    # Performs the above process iteratively on all the available district names
    for i in range(1, len(district_names)):
        district_data = pd.read_csv('{}/{}'.format(directory_path, district_names[i]))
        current_district_name = column_name_processing(district_names[i])
        district_data['district'] = list([current_district_name for _ in range(len(district_data))])
        combined_dataframe = combined_dataframe.append(district_data)
        processed_district_names.append(current_district_name)

    # Export the combined dataframe into a CSV file
    district_data_export('all_districts', 'combined_data', combined_dataframe)
    return combined_dataframe, processed_district_names


def data_preprocessing(combined_dataframe: pd.DataFrame):
    """Performs min-max normalization on the combined dataframe for all the features individually.

        Args:
            combined_dataframe: Pandas dataframe containing the combined data from all the 29 districts in Tamil Nadu,
                                India

        Returns:
            combined_min_max_dataframe: Min-max normalized Pandas dataframe containing the combined data from all the
                                        29 districts in Tamil Nadu, India
            features: Modified list of features for available for every district in Tamil Nadu, India
    """
    features = list(combined_dataframe.columns)
    combined_min_max_dataframe = pd.DataFrame(columns=features)
    combined_min_max_dataframe['district'] = combined_dataframe['district']
    if 'district' in features:
        features.remove('district')
    for i in range(len(features)):
        combined_min_max_dataframe[features[i]] = min_max_normalization(list(combined_dataframe[features[i]]))
    district_data_export('all_districts', 'min_max_normalized_data', combined_dataframe)
    return combined_min_max_dataframe, features


def compute_district_feature_median(combined_min_max_dataframe: pd.DataFrame,
                                    district_names: list,
                                    features: list):
    """Computes District-wise median for all features and stores in a Pandas DataFrame

        Args:
            combined_min_max_dataframe: Min-max normalized Pandas dataframe containing the combined data from all the
                                        29 districts in Tamil Nadu, India
            district_names: List containing names of 29 districts in Tamil Nadu, India
            features: List of features for available for every district in Tamil Nadu, India

        Returns:
            Pandas DataFrame containing District-wise median for all the features
    """
    combined_min_max_median_data = {i: list() for i in list(combined_min_max_dataframe.columns)}
    for i in range(len(district_names)):
        combined_min_max_median_data['district'].append(district_names[i])
        district_min_max_data = combined_min_max_dataframe[combined_min_max_dataframe['district'] == district_names[i]]
        for j in range(len(features)):
            combined_min_max_median_data[features[j]].append(np.median(district_min_max_data[features[j]]))
    combined_min_max_median_dataframe = pd.DataFrame(combined_min_max_median_data, columns=['district'] + features)
    return combined_min_max_median_dataframe


def elbow_method(combined_min_max_median_dataframe: pd.DataFrame):
    """Performs elbow method analysis by fitting K-means clustering model on the input for different no. of centers and
    plotting a graph which shows no. of centers vs sum of squared distances.

        Args:
            combined_min_max_median_dataframe: Pandas DataFrame containing District-wise median for all the features

        Returns:
            None
    """
    sum_of_squares = list()
    combined_min_max_median_dataframe = combined_min_max_median_dataframe.drop(columns=['district'])
    k_values = list(range(2, 29))
    for i in k_values:
        k_means_model = KMeans(n_clusters=i)
        k_means_model.fit(combined_min_max_median_dataframe)
        sum_of_squares.append(k_means_model.inertia_)
    plt.plot(k_values, sum_of_squares, 'bx-')
    plt.xlabel('Centers')
    plt.ylabel('Sum of Squared Distances')
    plt.savefig('../results/elbow_method_analysis.png')
    plt.show()


def main():
    district_names = os.listdir('../data/combined_data')
    district_names.sort()
    combined_dataframe, district_names = data_combine(district_names)
    combined_min_max_dataframe, features = data_preprocessing(combined_dataframe)
    combined_min_max_median_dataframe = compute_district_feature_median(combined_min_max_dataframe, district_names,
                                                                        features)
    elbow_method(combined_min_max_median_dataframe)


if __name__ == '__main__':
    main()

