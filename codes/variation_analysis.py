# authors_name = 'Preetham Ganesh, Harsha Vardhini Vasu'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com, harshavardhini2019@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def compute_cluster_monthly_median(clusters: list,
                                   months: list):
    """Computes median for all the clusters based on months, where the median would be computed for the rainfall
    feature.

        Args:
            clusters: A list containing the cluster names used for files in the previous codes.
            months: A list containing the names of months in a year.

        Returns:
            A dataframe containing rainfall monthly median for all the clusters.
    """
    directory_path = '../data/min_max_normalized_data'
    cluster_monthly_median_data = {}

    # Iterates across the list of the clusters for computing the monthly medians.
    for i in range(len(clusters)):
        cluster_monthly_median_data[clusters[i]] = []
        file_path = '{}/{}.csv'.format(directory_path, clusters[i])
        cluster_data = pd.read_csv(file_path)

        # Iterates across the months for filtering the data and computing the median.
        for j in range(len(months)):

            # Filters the cluster data based on the index from the iteration of months
            cluster_filtered_data = cluster_data.filter(items=[k + j for k in range(0, len(cluster_data), len(months))],
                                                        axis=0)

            # Computes median for rainfall feature data and saves to the dictionary.
            cluster_monthly_median_data[clusters[i]].append(np.median(list(cluster_filtered_data['rainfall'])))
    cluster_monthly_median_data['months'] = months

    # Converts a dictionary to a dataframe.
    cluster_monthly_median_dataframe = pd.DataFrame(cluster_monthly_median_data, columns=['months'] + clusters)
    return cluster_monthly_median_dataframe


def plot_preprocessing(cluster_month_median_dataframe: pd.DataFrame,
                       clusters: list,
                       months: list):
    months_numbers = list(range(len(months)))
    months_numbers_linspace = np.linspace(min(months_numbers), max(months_numbers), 500)
    cluster_smooth_values = {}
    cluster_smooth_values_mean = {}
    for i in range(len(clusters)):
        smoothing_function_current_cluster = interp1d(months_numbers, list(cluster_month_median_dataframe[clusters[i]]),
                                                      kind='quadratic')
        cluster_smooth_values[clusters[i]] = smoothing_function_current_cluster(months_numbers_linspace)
        cluster_smooth_values_mean[clusters[i]] = [np.mean(cluster_smooth_values[clusters[i]]) for _ in
                                                   range(len(cluster_smooth_values[clusters[i]]))]
    return cluster_smooth_values, cluster_smooth_values_mean, months_numbers_linspace


def generate_plot(cluster_smooth_values: dict,
                  cluster_smooth_values_mean: dict,
                  months_numbers_linspace: np.ndarray,
                  clusters: list,
                  cluster_groups: list,
                  months: list,
                  cluster_colors: list,
                  plot_name: str):
    font = {'size': 28}
    plt.rc('font', **font)
    figure(num=None, figsize=(20, 10))
    for i in cluster_groups:
        plt.plot(months_numbers_linspace, cluster_smooth_values[clusters[i]], color=cluster_colors[i],
                 label='cluster_{}'.format(str(i)), alpha=0.4, linewidth=3)
        plt.plot(months_numbers_linspace, cluster_smooth_values_mean[clusters[i]], '--',
                 color=cluster_colors[i], linewidth=3)
    plt.xlabel('months')
    plt.ylabel('normalized_rainfall')
    plt.legend(loc='upper left')
    plt.xticks(range(0, len(months)), months)
    plt.grid(color='black', linestyle='-.', linewidth=2, alpha=0.3)
    plt.savefig('../results/variation_analysis_{}.png'.format(plot_name))
    plt.show()


def main():
    clusters = ['cluster_{}'.format(str(i)) for i in range(0, 6)]
    cluster_colors = ['red', 'orange', 'green', 'blue', 'magenta', 'black']
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    cluster_group_1 = [0, 2, 3]
    cluster_group_2 = [1, 4, 5]
    cluster_monthly_median_dataframe = compute_cluster_monthly_median(clusters, months)
    cluster_smooth_values, cluster_smooth_values_mean, months_numbers_linspace = plot_preprocessing(
        cluster_monthly_median_dataframe, clusters, months)
    generate_plot(cluster_smooth_values, cluster_smooth_values_mean, months_numbers_linspace, clusters,
                  cluster_group_1, months, cluster_colors, 'cluster_group_1')
    generate_plot(cluster_smooth_values, cluster_smooth_values_mean, months_numbers_linspace, clusters,
                  cluster_group_2, months, cluster_colors, 'cluster_group_2')


if __name__ == '__main__':
    main()

