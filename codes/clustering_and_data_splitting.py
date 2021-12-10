# author_name = 'Preetham Ganesh'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
from elbow_method import compute_district_feature_median
from data_preprocessing import column_name_processing
import os


def data_preprocessing(district_names: list):
    processed_district_names = list()
    if 'all_districts.csv' in district_names:
        district_names.remove('all_districts.csv')
    for i in range(len(district_names)):
        processed_district_names.append(column_name_processing(district_names[i]))
    return processed_district_names


def cluster_analysis(district_names: list):
    directory_path = '../data/min_max_normalized_data'
    combined_min_max_dataframe = pd.read_csv('{}/{}.csv'.format(directory_path, 'all_districts'))
    processed_district_names = data_preprocessing(district_names)
    features = list(combined_min_max_dataframe.columns)
    combined_min_max_median_dataframe = compute_district_feature_median(combined_min_max_dataframe,
                                                                        processed_district_names, features)



def main():
    district_names = os.listdir('../data/combined_data')
    district_names.sort()
    cluster_analysis(district_names)



if __name__ == '__main__':
    main()

