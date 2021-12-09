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


def data_combine(district_names: list):
    """Uses district_names to read the datasets and combine them into a single dataset.

        Args:
            district_names: List containing names of 29 districts in Tamil Nadu, India

        Returns:
            Pandas dataframe containing the combined data from all the 29 districts in Tamil Nadu, India
    """
    directory_path = '../data/combined_data'
    if 'all_districts' in district_names:
        district_names.remove('all_districts')
    combined_dataframe = pd.read_csv('{}/{}'.format(directory_path, district_names[0]))
    processed_district_name = column_name_processing(district_names[0])
    combined_dataframe[['district']] = list([processed_district_name for i in range(len(combined_dataframe))])
    for i in range(1, len(district_names)):
        district_data = pd.read_csv('{}/{}'.format(directory_path, district_names[i]))
        processed_district_name = column_name_processing(district_names[i])
        district_data[['district']] = list([processed_district_name for i in range(len(district_data))])
        combined_dataframe = combined_dataframe.append(district_data)
    district_data_export('all_districts', 'combined_data', combined_dataframe)
    return combined_dataframe


def data_preprocessing(combined_dataframe: pd.DataFrame):
    """Performs min-max normalization on the combined dataframe for all the features individually.

        Args:
            combined_dataframe: Pandas dataframe containing the combined data from all the 29 districts in Tamil Nadu,
                                India
        Returns:
            Min-max normalized Pandas dataframe containing the combined data from all the 29 districts in Tamil Nadu,
            India
    """
    features = list(combined_dataframe.columns)
    combined_min_max_dataframe = pd.DataFrame(columns=features)
    for i in range(len(features)):
        combined_min_max_dataframe[features[i]] = min_max_normalization(list(combined_dataframe[features[i]]))
    return combined_min_max_dataframe


def elbow_method(combined_min_max_dataframe: pd.DataFrame):
    sum_of_squares = list()
    for i in range(2, 29):
        print(i)
        k_means_model = KMeans(n_clusters=i)
        k_means_model.fit(combined_min_max_dataframe)
        sum_of_squares.append(k_means_model.inertia_)
    print(sum_of_squares)
    plt.plot(list(range(2, 29)), sum_of_squares)
    plt.show()


def main():
    district_names = os.listdir('../data/combined_data')
    district_names.sort()
    data_combine(district_names)
    #combined_dataframe = data_combine(district_names)
    #combined_min_max_dataframe = data_preprocessing(combined_dataframe)
    #elbow_method(combined_min_max_dataframe)


if __name__ == '__main__':
    main()

