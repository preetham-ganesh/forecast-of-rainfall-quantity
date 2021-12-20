# Forecast of Rainfall Quantity and its Variation using Environmental Features

[Preetham Ganesh](https://www.linkedin.com/in/preethamganesh/), [Harsha Vardhini Vasu](https://www.linkedin.com/in/harshavardhini1/), [Dayanand Vinod*](https://in.linkedin.com/in/dayanand-vinod).

Paper link: [[IEEE]](https://ieeexplore.ieee.org/document/8960026)

## Contents

- [Setup](https://github.com/preetham-ganesh/forecast-of-rainfall-quantity#setup)
- [Data Preprocessing](https://github.com/preetham-ganesh/forecast-of-rainfall-quantity#data-preprocessing)

## Description

- It is an application for predicting rainfall in districts belonging to Tamil Nadu, India, using regression methods which captures sudden fluctuations.
- The project aims at developing three models which predict monthly rainfall for all districts in Tamil Nadu, India and also drawing a district-wise comparison among them to find the best model for prediction.
- The models developed are as follows:
	- **District-Specific Model**:
		- It trains on data from a particular district
	- **Cluster-Based Model**:
		- It groups districts based on the climatic conditions and trains on data from a particular cluster
	- **Generic-Regression Model**:
		- It trains on combined data from all the districts in Tamil Nadu, India.
- The project also aims at finding the monthly variation of rainfall across geographical regions.

## Usage

### Requirements Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements

Requires: Python 3.6.

```bash
# Clone this repository
git clone https://github.com/preetham-ganesh/forecast-of-rainfall-quantity.git
cd forecast-of-rainfall-quantity

# Create a Conda environment with dependencies
conda env create -f environment.yml
conda activate forq_env
pip install -r requirements.txt
```

### Files Execution

```bash
python3 data_preprocessing.py
python3 elbow_method.py
python3 clustering_and_data_splitting.py
python3 models.py
python3 variation_analysis.py
```

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Citation

If you use this code, please cite the following:

```
@INPROCEEDINGS{8960026,
  author={Ganesh, Preetham and Vasu, Harsha Vardhini and Vinod, Dayanand},
  booktitle={2019 Innovations in Power and Advanced Computing Technologies (i-PACT)}, 
  title={Forecast of Rainfall Quantity and its Variation using Environmental Features}, 
  year={2019},
  volume={1},
  number={},
  pages={1-8},
  doi={10.1109/i-PACT44901.2019.8960026}}
```
