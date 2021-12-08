# author_name = 'Preetham Ganesh'
# project_title = 'Forecast of Rainfall Quantity and its Variation using Environmental Features'
# email = 'preetham.ganesh2015@gmail.com'
# doi = 'https://ieeexplore.ieee.org/document/8960026'


import pandas as pd
import os


def data_combiner(district_names: list):
    directory_path = '../data/combined_data'
    combined_dataframe = pd.read_csv('{}/{}'.format(directory_path, district_names[0]), sep='\t')
    for i in range(1, len(district_names)):
        district_data = pd.read_csv('{}/{}'.format(directory_path, district_names[i]), sep='\t')
        combined_dataframe = combined_dataframe.append(district_data)
    return combined_dataframe


def main():
    district_names = os.listdir('../data/combined_data')
    district_names.sort()
    data_combiner(district_names)


if __name__ == '__main__':
    main()