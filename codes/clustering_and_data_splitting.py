# author_name = 'Preetham Ganesh'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
from elbow_method import compute_district_feature_median
from data_preprocessing import column_name_processing
import os
from sklearn.cluster import KMeans


def district_cluster_identification(district_names: list,
                                    n_clusters: int):
    directory_path = '../data/min_max_normalized_data'
    combined_min_max_dataframe = pd.read_csv('{}/{}.csv'.format(directory_path, 'all_districts'))
    processed_district_names = list()
    if 'all_districts.csv' in district_names:
        district_names.remove('all_districts.csv')
    for i in range(len(district_names)):
        processed_district_names.append(column_name_processing(district_names[i]))
    features = list(combined_min_max_dataframe.columns)
    combined_min_max_median_dataframe = compute_district_feature_median(combined_min_max_dataframe,
                                                                        processed_district_names, features)
    combined_min_max_median_dataframe = combined_min_max_median_dataframe.drop(columns=['district'])
    k_means_model = KMeans(n_clusters=n_clusters)
    k_means_model.fit(combined_min_max_median_dataframe)
    cluster_labels = list(k_means_model.labels_)
    district_clusters_data = {'district': processed_district_names, 'cluster': cluster_labels}
    district_clusters_dataframe = pd.DataFrame(district_clusters_data, columns=['district', 'cluster'])
    return district_clusters_dataframe


def main():
    district_names = os.listdir('../data/combined_data')
    district_names.sort()
    n_clusters = 6
    district_clusters_dataframe = cluster_analysis(district_names, n_clusters)


if __name__ == '__main__':
    main()

