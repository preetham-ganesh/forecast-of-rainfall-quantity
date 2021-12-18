# authors_name = 'Preetham Ganesh, Harsha Vardhini Vasu'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com, harshavardhini2019@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
import numpy as np


def compute_cluster_monthly_median(clusters: list,
                                   months: list):
    directory_path = '../data/min_max_normalized_data'
    cluster_monthly_median_data = {}
    for i in range(len(clusters)):
        cluster_monthly_median_data[clusters[i]] = []
        file_path = '{}/{}.csv'.format(directory_path, clusters[i])
        cluster_data = pd.read_csv(file_path)
        for j in range(len(months)):
            cluster_filtered_data = cluster_data.filter(items=[k + j for k in range(0, len(cluster_data), len(months))],
                                                        axis=0)
            cluster_monthly_median_data[clusters[i]].append(np.median(list(cluster_filtered_data['rainfall'])))
    cluster_monthly_median_data['months'] = months
    cluster_monthly_median_dataframe = pd.DataFrame(cluster_monthly_median_data, columns=['months'] + clusters)
    return cluster_monthly_median_dataframe


def plot_preprocessing(cluster_)


def main():
    clusters = ['cluster_{}'.format(str(i)) for i in range(0, 6)]
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    cluster_monthly_median_dataframe = compute_cluster_monthly_median(clusters, months)




if __name__ == '__main__':
    main()

