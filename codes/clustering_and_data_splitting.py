# authors_name = 'Preetham Ganesh, Harsha Vardhini Vasu'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com, harshavardhini2019@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
from elbow_method import compute_district_feature_median
from data_preprocessing import column_name_processing
from data_preprocessing import district_data_export
from elbow_method import data_min_max_preprocessing
import os
from sklearn.cluster import KMeans


def district_cluster_identification(district_names: list,
                                    n_clusters: int):
    """Performs cluster analysis on the combined_min_max_dataframe based on the result from the analysis of elbow
    method. The analysis returns a list of districts belonging to each cluster.

        Args:
            district_names: Names of districts in Tamil Nadu, India
            n_clusters: No. of clusters that should be used to train the K-Means clustering model.

        Returns:
            A dataframe containing the list of districts in Tamil Nadu, India, and the clusters they belong to.
    """
    directory_path = '../data/min_max_normalized_data'
    combined_min_max_dataframe = pd.read_csv('{}/{}.csv'.format(directory_path, 'all_districts'))
    processed_district_names = list()

    # In case if the code was already executed, the combined data will be stored in a CSV file called 'all_districts',
    # since that is not an actual district in Tamil Nadu, India, the file_name is removed from the list
    if 'all_districts.csv' in district_names:
        district_names.remove('all_districts.csv')

    # Iterates across the district names and preprocesses them.
    for i in range(len(district_names)):
        processed_district_names.append(column_name_processing(district_names[i]))
    features = list(combined_min_max_dataframe.columns)

    # Identification of the clusters for each district by using K-means clustering.
    combined_min_max_median_dataframe = compute_district_feature_median(combined_min_max_dataframe,
                                                                        processed_district_names, features)
    combined_min_max_median_dataframe = combined_min_max_median_dataframe.drop(columns=['district'])
    k_means_model = KMeans(n_clusters=n_clusters)
    k_means_model.fit(combined_min_max_median_dataframe)
    cluster_labels = list(k_means_model.labels_)

    # Convert the list into dataframe for future use
    cluster_district_data = {'district': processed_district_names, 'cluster': cluster_labels}
    cluster_district_dataframe = pd.DataFrame(cluster_district_data, columns=['district', 'cluster'])
    return cluster_district_dataframe


def cluster_based_data_combine(cluster_district_dataframe: pd.DataFrame,
                               n_clusters: int):
    """Uses cluster_district_dataframe to filter data from the combined_dataframe

        Args:
            cluster_district_dataframe: A dataframe containing the list of districts in Tamil Nadu, India, and the
                                        clusters they belong to.
            n_clusters: No. of clusters that should be used to train the K-Means clustering model.

        Returns:
            None
    """
    directory_path = '../data/combined_data'
    combined_dataframe = pd.read_csv('{}/{}.csv'.format(directory_path, 'all_districts'))

    # Iterates across the no. of clusters to perform filtration on the combined_dataframe
    for i in range(n_clusters):
        per_cluster_district_dataframe = cluster_district_dataframe[cluster_district_dataframe['cluster'] == i]

        # Filters district's data for the current cluster
        per_cluster_combined_dataframe = combined_dataframe[combined_dataframe['district'].isin(
            list(per_cluster_district_dataframe['district']))]
        current_cluster_name = '{}_{}'.format('cluster', str(i))

        # Saves current cluster's dataframe into a CSV file
        district_data_export(current_cluster_name, 'combined_data', per_cluster_combined_dataframe)

        # Converts current cluster's dataframe into a min-max normalized dataframe and saves into a CSV file
        per_cluster_combined_min_max_dataframe, _ = data_min_max_preprocessing(per_cluster_combined_dataframe,
                                                                               current_cluster_name,
                                                                               'min_max_normalized_data')


def main():
    district_names = os.listdir('../data/combined_data')
    district_names.sort()
    n_clusters = 6
    cluster_district_dataframe = district_cluster_identification(district_names, n_clusters)
    cluster_based_data_combine(cluster_district_dataframe, n_clusters)


if __name__ == '__main__':
    main()

